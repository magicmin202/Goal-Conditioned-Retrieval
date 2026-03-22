"""Stage 2: Anchor-centered Evidence Consolidation Pipeline.

Architecture:
  Goal
  → Query Understanding
  → LLM Query Expansion (Gemini)
  → Hybrid Candidate Retrieval          [Stage 1 equivalent: recall]
  → Goal-Conditioned Reranking          [Stage 1 equivalent: precision]
  → Strict Admission Filter             ← key change: no fill-to-k fallback
  → Top Admitted Anchors
  → Anchor-centered Local Expansion     [temporal window ± N days]
  → Neighbor Re-admission Check         [same reranker gate]
  → Admitted Cluster Formation
  → Cluster Summarization (compressor)  [summarization only]
  → LLM Analysis

Stage 2 invariants:
  - Stage 2 does NOT do new retrieval after admission.
  - Non-admitted logs are NEVER passed to the compressor.
  - allow_fewer_than_k=True: fewer correct > full noisy.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.compression.local_expansion import LocalExpander
from app.compression.temporal_semantic_compressor import TemporalSemanticCompressor
from app.config import AdaptiveMode, AdaptivePolicyConfig, ConsolidationConfig, Stage2Config
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


def _temporal_window(mode: AdaptiveMode, cfg: ConsolidationConfig) -> int:
    """Pick local expansion window based on corpus density."""
    if mode == AdaptiveMode.SMALL:
        return cfg.local_expansion_window_small
    if mode == AdaptiveMode.LARGE:
        return cfg.local_expansion_window_large
    return cfg.local_expansion_window_standard


@dataclass
class Stage2Result:
    goal: ResearchGoal
    candidates: list[CandidateLog]
    ranked_logs: list[RankedLog]
    selected_logs: list[RankedLog]          # = admitted anchors
    evidence_units: list[CompressedEvidenceUnit]
    llm_analysis: str
    query_text: str
    expanded_terms: list[str] = field(default_factory=list)
    negative_terms: list[str] = field(default_factory=list)
    priority_terms: list[str] = field(default_factory=list)
    adaptive_mode: str = AdaptiveMode.STANDARD
    metadata: dict = field(default_factory=dict)


class Stage2Pipeline:
    """Anchor-centered Evidence Consolidation Pipeline (Stage 2)."""

    def __init__(
        self,
        config: Stage2Config | None = None,
        use_mock_llm: bool = False,
        use_real_embeddings: bool = False,
    ) -> None:
        from app.retrieval.embedding_provider import get_embedding_provider
        self.config = config or Stage2Config()
        embed_provider = get_embedding_provider(real=use_real_embeddings)
        self._retriever = CandidateRetriever(
            mode=RetrievalMode.HYBRID_EXPANDED,
            config=self.config.retrieval,
            candidate_config=self.config.candidate,
            vocab_boost_config=self.config.vocab_boost,
            embedding_provider=embed_provider,
        )
        self._reranker = GoalConditionedReranker(config=self.config.ranker)
        self._selector = DiversitySelector(config=self.config.diversity)
        self._expander = LocalExpander(
            anchor_relevance_threshold=self.config.consolidation.anchor_admission_threshold,
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
        cfg = self.config.consolidation
        top_k = self.config.retrieval.top_k
        logger.info("Stage2 pipeline  goal=%s  mode=%s  consolidation_mode=%s",
                    goal.goal_id, mode, cfg.consolidation_mode)

        # ── 1. Query Understanding ────────────────────────────────────────────
        query_obj = build_query(goal)

        # ── 2. LLM Query Expansion (Gemini) ───────────────────────────────────
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

        # ── 3. Hybrid Candidate Retrieval ─────────────────────────────────────
        candidates = self._retriever.retrieve(
            expanded, top_n=self.config.retrieval.candidate_size
        )
        logger.info("Stage2: %d candidates retrieved  goal=%s", len(candidates), goal.goal_id)

        # ── 4. Goal-Conditioned Reranking ─────────────────────────────────────
        ranked = self._reranker.rank(
            goal, candidates,
            expanded_terms=expanded.expanded_terms,
            negative_terms=expanded.negative_terms,
            priority_terms=expanded.priority_terms,
            related_terms=expanded.related_terms,
        )

        # ── 5. Strict Admission Filter ────────────────────────────────────────
        #   Rule: only logs above anchor_admission_threshold become anchors.
        #   We do NOT fall back to threshold-failing logs to fill top_k.
        #   "fewer but correct" > "full but noisy"
        threshold = cfg.anchor_admission_threshold
        admitted = [r for r in ranked if r.final_score >= threshold]
        logger.info(
            "Stage2 admission: %d/%d logs admitted  threshold=%.3f  mode=%s",
            len(admitted), len(ranked), threshold, mode,
        )

        # Diversity selection within admitted set only (when >top_k candidates)
        if len(admitted) > top_k:
            anchors = self._selector.select(goal, admitted, top_k=top_k, adaptive_mode=mode)
            logger.info(
                "Stage2 diversity selection: %d → %d admitted anchors",
                len(admitted), len(anchors),
            )
        else:
            anchors = admitted
            if len(anchors) < top_k:
                logger.info(
                    "Stage2: fewer than top_k admitted (%d < %d) — keeping all"
                    " (allow_fewer_than_k=%s)",
                    len(anchors), top_k, cfg.allow_fewer_than_k,
                )

        # Log each anchor
        logger.info("Stage2 admitted anchors: %d", len(anchors))
        for a in anchors:
            logger.info(
                "  Anchor %s  date=%s  score=%.4f  [%s]",
                a.log_id, a.log.date, a.final_score, a.log.title,
            )

        # ── 6. Anchor-centered Local Expansion ───────────────────────────────
        #   - temporal window based on corpus density (AdaptiveMode)
        #   - neighbors must pass reranker re-admission
        #   - non-admitted neighbors are blocked from compressor
        window = _temporal_window(mode, cfg)
        expansion_map = self._expander.expand(
            anchors,
            self._all_logs,
            goal=goal,
            expanded_terms=expanded.expanded_terms,
            negative_terms=expanded.negative_terms,
            priority_terms=expanded.priority_terms,
            related_terms=expanded.related_terms,
            reranker=self._reranker,
            temporal_window=window,
            neighbor_admission_threshold=cfg.neighbor_admission_threshold,
            max_neighbors=cfg.max_neighbors_per_anchor,
        )

        # ── 7. Cluster Summarization (compressor = summarization only) ────────
        evidence_units = self._compressor.compress(anchors, expansion_map)
        logger.info("Stage2: %d evidence units  goal=%s", len(evidence_units), goal.goal_id)

        # ── 8. LLM Analysis ───────────────────────────────────────────────────
        analysis = self._analyzer.analyze(goal, evidence_units)

        return Stage2Result(
            goal=goal,
            candidates=candidates,
            ranked_logs=ranked,
            selected_logs=anchors,
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
                "admitted_anchors": len(anchors),
                "top_k_target": top_k,
                "anchor_admission_threshold": threshold,
                "temporal_window_days": window,
                "evidence_units": len(evidence_units),
            },
        )
