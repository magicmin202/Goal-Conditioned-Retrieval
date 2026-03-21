"""Stage 2: Enhanced Retrieval Pipeline with full RAG.

Goal → Query Understanding → LLM Query Expansion (Gemini)
→ Hybrid Candidate Retrieval → Goal-Conditioned Reranking
→ Relevance Filtering → Diversity-aware Top-K Selection (adaptive)
→ Anchor-based Local Expansion (with goal relevance gate)
→ Temporal/Semantic Compression → LLM Analysis (Gemini)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.compression.local_expansion import LocalExpander
from app.compression.temporal_semantic_compressor import TemporalSemanticCompressor
from app.config import AdaptiveMode, AdaptivePolicyConfig, Stage2Config
from app.llm.analysis import GoalAnalyzer
from app.llm.llm_client import get_llm_client
from app.retrieval.candidate_retrieval import CandidateRetriever, RetrievalMode
from app.retrieval.diversity_selector import DiversitySelector
from app.retrieval.query_expansion import expand_goal_query
from app.retrieval.query_understanding import build_query
from app.retrieval.reranker import GoalConditionedReranker
from app.schemas import (
    CandidateLog, CompressedEvidenceUnit, RankedLog, ResearchGoal, ResearchLog,
)

logger = logging.getLogger(__name__)


def _detect_adaptive_mode(
    logs: list[ResearchLog],
    policy: AdaptivePolicyConfig | None = None,
) -> AdaptiveMode:
    """Infer retrieval intensity mode from corpus size and date span."""
    p = policy or AdaptivePolicyConfig()
    n = len(logs)
    if n == 0:
        return AdaptiveMode.SMALL

    from datetime import date as _date
    dates = sorted(log.date for log in logs if log.date)
    if len(dates) >= 2:
        try:
            span = (_date.fromisoformat(dates[-1]) - _date.fromisoformat(dates[0])).days
        except (ValueError, TypeError):
            span = 0
    else:
        span = 0

    if n < p.small_log_threshold or span < p.small_span_threshold:
        mode = AdaptiveMode.SMALL
    elif n >= p.large_log_threshold and span >= p.large_span_threshold:
        mode = AdaptiveMode.LARGE
    else:
        mode = AdaptiveMode.STANDARD

    density = round(n / max(span, 1), 2)
    logger.info(
        "AdaptiveMode=%s  logs=%d  span_days=%d  density=%.2f logs/day",
        mode, n, span, density,
    )
    return mode


@dataclass
class Stage2Result:
    goal: ResearchGoal
    candidates: list[CandidateLog]
    ranked_logs: list[RankedLog]
    selected_logs: list[RankedLog]
    evidence_units: list[CompressedEvidenceUnit]
    llm_analysis: str
    query_text: str
    expanded_terms: list[str] = field(default_factory=list)
    negative_terms: list[str] = field(default_factory=list)
    priority_terms: list[str] = field(default_factory=list)
    adaptive_mode: str = AdaptiveMode.STANDARD
    metadata: dict = field(default_factory=dict)


class Stage2Pipeline:
    """Full RAG pipeline for Stage 2 (Gemini API as default execution path)."""

    def __init__(
        self,
        config: Stage2Config | None = None,
        use_mock_llm: bool = False,   # False = use Gemini by default
    ) -> None:
        self.config = config or Stage2Config()
        self._retriever = CandidateRetriever(
            mode=RetrievalMode.HYBRID_EXPANDED,
            config=self.config.retrieval,
            vocab_boost_config=self.config.vocab_boost,
        )
        self._reranker = GoalConditionedReranker(config=self.config.ranker)
        self._selector = DiversitySelector(config=self.config.diversity)
        self._expander = LocalExpander(
            semantic_threshold=0.45,
            anchor_relevance_threshold=self.config.diversity.relevance_threshold,
            neighbor_goal_relevance_min=0.02,
        )
        self._compressor = TemporalSemanticCompressor()
        self._analyzer = GoalAnalyzer(llm=get_llm_client(mock=use_mock_llm))
        self._all_logs: list[ResearchLog] = []
        self._adaptive_mode: AdaptiveMode = AdaptiveMode.STANDARD
        self._indexed = False

    def index(self, logs: list[ResearchLog]) -> None:
        self._retriever.index(logs)
        self._all_logs = logs
        self._adaptive_mode = _detect_adaptive_mode(logs)
        self._indexed = True

    def run(self, goal: ResearchGoal) -> Stage2Result:
        if not self._indexed:
            raise RuntimeError("Call index() before run().")

        mode = self._adaptive_mode
        logger.info("Stage2 pipeline  goal=%s  mode=%s", goal.goal_id, mode)

        # 1. Query Understanding
        query_obj = build_query(goal)

        # 2. LLM Query Expansion (Gemini)
        expanded = expand_goal_query(
            goal, query_obj,
            max_terms=self.config.query_expansion.max_terms,
            mode=self.config.query_expansion.mode,
            use_mock_fallback=self.config.query_expansion.use_mock_fallback,
        )
        logger.info(
            "Stage2 expansion  goal=%s\n"
            "  priority=%s\n"
            "  evidence=%s\n"
            "  related=%s\n"
            "  negative=%s",
            goal.goal_id, expanded.priority_terms, expanded.expanded_terms,
            expanded.related_terms, expanded.negative_terms,
        )

        # 3. Hybrid Candidate Retrieval (+ vocab boost applied inside)
        candidates = self._retriever.retrieve(
            expanded, top_n=self.config.retrieval.candidate_size
        )
        logger.info("Stage2: %d candidates retrieved  goal=%s", len(candidates), goal.goal_id)

        # 4. Goal-Conditioned Reranking (3-tier goal_focus + phrase matching)
        ranked = self._reranker.rank(
            goal, candidates,
            expanded_terms=expanded.expanded_terms,
            negative_terms=expanded.negative_terms,
            priority_terms=expanded.priority_terms,
            related_terms=expanded.related_terms,
        )

        # 5. Relevance Filtering
        top_k = self.config.retrieval.top_k
        threshold = self.config.diversity.relevance_threshold
        above = [r for r in ranked if r.final_score >= threshold]
        pool = above if len(above) >= top_k else ranked[: top_k * 2]
        logger.info(
            "Stage2 relevance filter: %d→%d (threshold=%.3f)  mode=%s",
            len(ranked), len(pool), threshold, mode,
        )

        # 6. Diversity-aware Top-K Selection (adaptive lambda)
        selected = self._selector.select(goal, pool, top_k=top_k, adaptive_mode=mode)

        # 7. Anchor-based Local Expansion (with goal relevance gate on neighbors)
        expansion_map = self._expander.expand(
            selected, self._all_logs, expanded_terms=expanded.expanded_terms
        )

        # 8. Temporal-Semantic Compression
        evidence_units = self._compressor.compress(selected, expansion_map)
        logger.info("Stage2: %d evidence units  goal=%s", len(evidence_units), goal.goal_id)

        # 9. LLM Analysis (Gemini)
        analysis = self._analyzer.analyze(goal, evidence_units)

        return Stage2Result(
            goal=goal,
            candidates=candidates,
            ranked_logs=ranked,
            selected_logs=selected,
            evidence_units=evidence_units,
            llm_analysis=analysis,
            query_text=expanded.full_text,
            expanded_terms=expanded.expanded_terms,
            negative_terms=expanded.negative_terms,
            priority_terms=expanded.priority_terms,
            adaptive_mode=mode,
            metadata={
                "adaptive_mode": mode,
                "priority_terms": expanded.priority_terms,
                "evidence_terms": expanded.expanded_terms,
                "related_terms": expanded.related_terms,
                "negative_terms": expanded.negative_terms,
                "candidate_size": len(candidates),
                "after_filter": len(pool),
                "top_k": len(selected),
                "evidence_units": len(evidence_units),
            },
        )
