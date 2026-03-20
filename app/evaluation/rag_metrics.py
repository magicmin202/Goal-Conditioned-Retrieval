"""Stage 2 RAG evaluation metrics."""
from __future__ import annotations
import logging
import re
from app.schemas import CompressedEvidenceUnit, ResearchGoal, ResearchLog

logger = logging.getLogger(__name__)


def _token_count(text: str) -> int:
    return len(text.split())


def goal_alignment_score(
    goal: ResearchGoal, evidence_units: list[CompressedEvidenceUnit]
) -> float:
    """Mock rubric: keyword overlap between goal and evidence summaries.

    TODO: Replace with LLM judge: judge_fn(goal, unit) -> float in [0, 1]
    """
    if not evidence_units:
        return 0.0
    goal_kw = set(re.findall(r"[\w가-힣]+", goal.query_text.lower()))
    scores = []
    for unit in evidence_units:
        unit_kw = set(re.findall(r"[\w가-힣]+", unit.summary.lower()))
        scores.append(min(len(goal_kw & unit_kw) / max(len(goal_kw), 1), 1.0))
    return sum(scores) / len(scores)


def actionability_score(evidence_units: list[CompressedEvidenceUnit]) -> float:
    """Fraction of units with non-empty temporal_progression."""
    if not evidence_units:
        return 0.0
    return sum(1 for u in evidence_units if u.temporal_progression) / len(evidence_units)


def token_reduction_rate(
    original_logs: list[ResearchLog], evidence_units: list[CompressedEvidenceUnit]
) -> float:
    original = sum(_token_count(log.full_text) for log in original_logs)
    compressed = sum(_token_count(u.summary) for u in evidence_units)
    return max(0.0, 1.0 - compressed / original) if original > 0 else 0.0


def redundancy_reduction(
    original_logs: list[ResearchLog], evidence_units: list[CompressedEvidenceUnit]
) -> float:
    if not evidence_units or not original_logs:
        return 0.0
    return sum(u.log_count for u in evidence_units) / len(original_logs)


def compute_rag_metrics(
    goal: ResearchGoal,
    original_logs: list[ResearchLog],
    evidence_units: list[CompressedEvidenceUnit],
) -> dict[str, float]:
    return {
        "goal_alignment_score": goal_alignment_score(goal, evidence_units),
        "actionability_score": actionability_score(evidence_units),
        "token_reduction_rate": token_reduction_rate(original_logs, evidence_units),
        "redundancy_reduction": redundancy_reduction(original_logs, evidence_units),
    }
