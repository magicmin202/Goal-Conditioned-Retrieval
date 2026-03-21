"""Stage 1: Candidate Retrieval → Reranking → Relevance Filtering → Diversity Selection.

Core method: raw goal (no expansion).
Expansion variant: set use_expansion=True (optional baseline).
expanded_terms and negative_terms are passed to the reranker in both cases.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.config import Stage1Config
from app.retrieval.candidate_retrieval import CandidateRetriever, RetrievalMode
from app.retrieval.diversity_selector import DiversitySelector
from app.retrieval.query_expansion import expand_goal_query
from app.retrieval.query_understanding import build_query
from app.retrieval.reranker import GoalConditionedReranker
from app.schemas import CandidateLog, RankedLog, ResearchGoal, ResearchLog

logger = logging.getLogger(__name__)


@dataclass
class Stage1Result:
    goal: ResearchGoal
    candidates: list[CandidateLog]
    ranked_logs: list[RankedLog]
    selected_logs: list[RankedLog]
    query_text: str
    used_expansion: bool = False
    expanded_terms: list[str] = field(default_factory=list)
    negative_terms: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class Stage1Pipeline:
    """Goal → Candidate Retrieval → Reranking → Relevance Filtering → Diversity."""

    def __init__(self, config: Stage1Config | None = None) -> None:
        self.config = config or Stage1Config()
        self._retriever = CandidateRetriever(
            mode=RetrievalMode.HYBRID,
            config=self.config.retrieval,
            vocab_boost_config=self.config.vocab_boost,
        )
        self._reranker = GoalConditionedReranker(config=self.config.ranker)
        self._selector = DiversitySelector(config=self.config.diversity)
        self._indexed = False

    def index(self, logs: list[ResearchLog]) -> None:
        self._retriever.index(logs)
        self._indexed = True

    def run(
        self,
        goal: ResearchGoal,
        use_expansion: bool | None = None,
    ) -> Stage1Result:
        if not self._indexed:
            raise RuntimeError("Call index() before run().")

        expand = (
            use_expansion if use_expansion is not None
            else self.config.query_expansion.enabled
        )

        # 1. Query Understanding
        query_obj = build_query(goal)

        # 2. Optional Query Expansion
        expanded_terms: list[str] = []
        priority_terms: list[str] = []
        related_terms: list[str] = []
        negative_terms: list[str] = []

        if expand:
            expanded = expand_goal_query(
                goal, query_obj,
                max_terms=self.config.query_expansion.max_terms,
                mode=self.config.query_expansion.mode,
                use_mock_fallback=self.config.query_expansion.use_mock_fallback,
            )
            active_query = expanded
            expanded_terms = expanded.expanded_terms
            priority_terms = expanded.priority_terms
            related_terms = expanded.related_terms
            negative_terms = expanded.negative_terms
            logger.info(
                "Stage1 expansion  goal=%s\n"
                "  priority=%s\n  evidence=%s\n  related=%s\n  negative=%s",
                goal.goal_id, priority_terms, expanded_terms, related_terms, negative_terms,
            )
        else:
            # Even without expansion, use heuristic terms for reranker scoring
            from app.retrieval.query_expansion import _heuristic_expansion
            heuristic = _heuristic_expansion(goal, max_terms=15)
            expanded_terms = heuristic.get("evidence_terms", [])
            priority_terms = heuristic.get("priority_terms", expanded_terms[:5])
            related_terms = heuristic.get("related_terms", [])
            negative_terms = heuristic.get("negative_terms", [])
            active_query = query_obj

        # 3. Candidate Retrieval (+ vocab boost if ExpandedQuery)
        candidates = self._retriever.retrieve(
            active_query, top_n=self.config.retrieval.candidate_size
        )
        logger.info("Stage1: %d candidates  goal=%s", len(candidates), goal.goal_id)

        # 4. Goal-Conditioned Reranking (3-tier goal_focus)
        ranked = self._reranker.rank(
            goal, candidates,
            expanded_terms=expanded_terms,
            negative_terms=negative_terms,
            priority_terms=priority_terms,
            related_terms=related_terms,
        )

        # 5. Relevance Filtering → cap pool before diversity
        top_k = self.config.retrieval.top_k
        threshold = self.config.diversity.relevance_threshold
        above = [r for r in ranked if r.final_score >= threshold]
        pool = above if len(above) >= top_k else ranked[: top_k * 2]
        logger.info(
            "Stage1 relevance filter: %d→%d (threshold=%.3f)",
            len(ranked), len(pool), threshold,
        )

        # 6. Diversity-aware Top-K Selection
        selected = self._selector.select(goal, pool, top_k=top_k)

        query_text = (
            active_query.full_text
            if hasattr(active_query, "full_text")
            else active_query.canonical_text
        )

        return Stage1Result(
            goal=goal,
            candidates=candidates,
            ranked_logs=ranked,
            selected_logs=selected,
            query_text=query_text,
            used_expansion=expand,
            expanded_terms=expanded_terms,
            negative_terms=negative_terms,
            metadata={
                "candidate_size": len(candidates),
                "after_filter": len(pool),
                "top_k": len(selected),
                "priority_terms": priority_terms,
                "related_terms": related_terms,
            },
        )
