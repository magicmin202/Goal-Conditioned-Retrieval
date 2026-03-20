"""Normalize goal information into a retrieval QueryObject."""
from __future__ import annotations
import re
from dataclasses import dataclass
from app.schemas import ResearchGoal


@dataclass
class QueryObject:
    raw_text: str
    canonical_text: str
    goal_id: str
    expanded_terms: list[str] | None = None


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def build_query(goal: ResearchGoal) -> QueryObject:
    """Build canonical QueryObject from a ResearchGoal.

    Stage 1 core: uses raw goal text, no expansion.
    """
    raw = goal.query_text
    canonical = _normalize(
        goal.goal_embedding_text if goal.goal_embedding_text else raw
    )
    return QueryObject(raw_text=raw, canonical_text=canonical, goal_id=goal.goal_id)
