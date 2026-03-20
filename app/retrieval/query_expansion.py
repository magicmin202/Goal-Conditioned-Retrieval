"""LLM query expansion module.

Used in Stage 2 and as an optional variant in Stage 1.
TODO: Replace _call_llm_mock() with a real Gemini call.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from app.retrieval.query_understanding import QueryObject
from app.schemas import ResearchGoal

logger = logging.getLogger(__name__)

# ── MOCK expansion table ─────────────────────────────────────────────────────
_MOCK_EXPANSIONS: dict[str, list[str]] = {
    "ai 개발": ["수학", "선형대수", "확률통계", "머신러닝", "딥러닝", "논문 읽기", "모델 구현"],
    "역량": ["학습", "실습", "구현", "복습", "정리"],
    "default": ["학습", "실행", "복습", "정리", "계획"],
}
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ExpandedQuery:
    base_query: QueryObject
    expanded_terms: list[str] = field(default_factory=list)
    mode: str = "simple"

    @property
    def canonical_text(self) -> str:
        return self.base_query.canonical_text

    @property
    def full_text(self) -> str:
        return f"{self.base_query.canonical_text} {' '.join(self.expanded_terms)}".strip()

    @property
    def goal_id(self) -> str:
        return self.base_query.goal_id


def _call_llm_mock(goal: ResearchGoal, max_terms: int) -> list[str]:
    """Mock expansion.

    TODO: Replace with:
        from app.llm.llm_client import get_llm_client
        client = get_llm_client(mock=False)
        # prompt LLM to return sub-concepts for goal.query_text
    """
    key = goal.title.lower()
    for k, terms in _MOCK_EXPANSIONS.items():
        if k in key:
            return terms[:max_terms]
    return _MOCK_EXPANSIONS["default"][:max_terms]


def expand_goal_query(
    goal: ResearchGoal,
    base_query: QueryObject,
    max_terms: int = 7,
    mode: str = "simple",
) -> ExpandedQuery:
    """Expand goal into sub-concepts for retrieval."""
    terms = _call_llm_mock(goal, max_terms)
    logger.debug("Expanded query for goal=%s: %s", goal.goal_id, terms)
    return ExpandedQuery(base_query=base_query, expanded_terms=terms, mode=mode)
