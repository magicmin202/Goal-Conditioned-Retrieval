"""Stage 1: Candidate Retrieval → Goal-Conditioned Ranking → Diversity Selection.

Core method uses raw goal (no expansion).
Optional variant: set use_expansion=True for query expansion baseline.
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
    metadata: dict = field(default_factory=dict)


class Stage1Pipeline:
    """Goal → Candidate Retrieval → Reranking → Diversity Selection."""

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

        query_obj = build_query(goal)

        if expand:
            active_query = expand_goal_query(
                goal, query_obj,
                max_terms=self.config.query_expansion.max_terms,
                mode=self.config.query_expansion.mode,
            )
            logger.info("Stage1: query expansion applied for goal=%s", goal.goal_id)
        else:
            active_query = query_obj

        candidates = self._retriever.retrieve(
            active_query, top_n=self.config.retrieval.candidate_size
        )
        logger.info("Stage1: %d candidates for goal=%s", len(candidates), goal.goal_id)

        ranked = self._reranker.rank(goal, candidates)
        selected = self._selector.select(goal, ranked, top_k=self.config.retrieval.top_k)

        query_text = (
            active_query.full_text if hasattr(active_query, "full_text")
            else active_query.canonical_text
        )

        return Stage1Result(
            goal=goal,
            candidates=candidates,
            ranked_logs=ranked,
            selected_logs=selected,
            query_text=query_text,
            used_expansion=expand,
            metadata={"candidate_size": len(candidates), "top_k": len(selected)},
        )
