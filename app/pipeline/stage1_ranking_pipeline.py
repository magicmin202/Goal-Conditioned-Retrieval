"""Stage 1: Candidate Retrieval -> Goal-Conditioned Ranking
                                -> Relevance Filtering
                                -> Diversity-aware Top-K Selection.

Core method: raw goal (no expansion).
Optional variant: --expand flag enables query expansion baseline.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.config import Stage1Config
from app.retrieval.candidate_retrieval import CandidateRetriever, RetrievalMode
from app.retrieval.diversity_selector import DiversitySelector
from app.retrieval.query_expansion import ExpandedQuery, expand_goal_query
from app.retrieval.query_understanding import QueryObject, build_query
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
    metadata: dict = field(default_factory=dict)


class Stage1Pipeline:
    """Goal -> Candidate Retrieval -> Reranking -> Relevance Filter -> Diversity Selection."""

    def __init__(self, config: Stage1Config | None = None) -> None:
        self.config = config or Stage1Config()
        self._retriever = CandidateRetriever(
            mode=RetrievalMode.HYBRID, config=self.config.retrieval
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
        expanded_terms: list[str] = []
        active_query: QueryObject | ExpandedQuery

        if expand:
            expanded_q = expand_goal_query(
                goal, query_obj,
                max_terms=self.config.query_expansion.max_terms,
                mode=self.config.query_expansion.mode,
            )
            active_query = expanded_q
            expanded_terms = expanded_q.all_expansion_terms
            logger.info(
                "Stage1: query expansion applied | goal=%s | terms=%s",
                goal.goal_id, expanded_terms[:5],
            )
        else:
            active_query = query_obj

        # 2. Candidate Retrieval (top-N pruning enforced by candidate_size)
        candidates = self._retriever.retrieve(
            active_query, top_n=self.config.retrieval.candidate_size
        )
        logger.info(
            "Stage1: %d candidates retrieved (corpus via index) for goal=%s",
            len(candidates), goal.goal_id,
        )

        # 3. Goal-Conditioned Reranking (pass expanded_terms for better goal_focus)
        ranked = self._reranker.rank(goal, candidates, expanded_terms=expanded_terms or None)

        # 4. Relevance Filtering + Diversity-aware Top-K Selection
        selected = self._selector.select(
            goal, ranked, top_k=self.config.retrieval.top_k,
            relevance_threshold=self.config.diversity.relevance_threshold,
        )

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
            metadata={
                "candidate_size": len(candidates),
                "ranked_size": len(ranked),
                "top_k": len(selected),
            },
        )
