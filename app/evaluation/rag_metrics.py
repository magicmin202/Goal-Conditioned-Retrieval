"""Stage 2 RAG evaluation metrics."""
from __future__ import annotations
import logging
import re
from app.schemas import CompressedEvidenceUnit, ResearchGoal, ResearchLog

logger = logging.getLogger(__name__)

# Korean grammatical suffixes to strip when extracting goal keywords.
# Longer/more-specific patterns must come before shorter ones so the first match
# absorbs the full suffix rather than leaving a partial one.
# TODO: Replace this heuristic with a proper Korean morphological analyzer
#       (e.g., KoNLPy / Mecab) for accurate morpheme decomposition.  Current
#       suffix-stripping is a best-effort approximation and may produce wrong
#       stems for a small set of word forms.  The real fix is an LLM judge:
#       judge_fn(goal, evidence_unit) -> float in [0, 1] — see P1 tech-debt.
_KR_VERB_ENDINGS = ("하고", "하여", "하기", "한다", "온다", "합니다", "됩니다", "으로")
_KR_NOUN_PARTICLES = ("과", "와", "을", "를", "은", "는", "에서")


def _strip_kr_suffix(token: str) -> str:
    """Strip one level of the most common Korean grammatical suffix from *token*.

    Only strips if the remaining stem has at least 2 characters.
    Verb endings are tried first (longer patterns) before short noun particles.
    Short noun particles are only stripped from tokens of length ≥ 3 to reduce
    false-positive stripping (e.g. avoid turning '이' standalone into '').
    """
    for sfx in _KR_VERB_ENDINGS:
        if token.endswith(sfx) and len(token) - len(sfx) >= 2:
            return token[:-len(sfx)]
    if len(token) >= 3:
        for sfx in _KR_NOUN_PARTICLES:
            if token.endswith(sfx) and len(token) - len(sfx) >= 2:
                return token[:-len(sfx)]
    return token


def _extract_goal_keywords(text: str) -> set[str]:
    """Extract Korean content keywords from goal text.

    Strips common grammatical suffixes so that inflected forms in the goal
    description (e.g. '항공권과', '숙소를', '예약하고') match the base forms
    that appear in CEU summaries ('항공권', '숙소', '예약').
    """
    tokens = re.findall(r"[가-힣]{2,}", text.lower())
    return {_strip_kr_suffix(tok) for tok in tokens}


def _token_count(text: str) -> int:
    return len(text.split())


def goal_alignment_score(
    goal: ResearchGoal, evidence_units: list[CompressedEvidenceUnit]
) -> float:
    """Keyword overlap between goal content words and evidence summaries.

    Uses suffix-stripped Korean tokens so inflected goal description words
    (e.g. '항공권과' → '항공권') match base forms in CEU summaries.

    Known limitation: pure keyword overlap undercounts semantic alignment for
    paraphrased or domain-specific evidence. The ceiling with this approach is
    roughly 0.3–0.5 for typical goals; reaching 0.7+ requires an LLM judge.
    TODO: Replace with LLM judge: judge_fn(goal, unit) -> float in [0, 1]
    """
    if not evidence_units:
        return 0.0
    goal_kw = _extract_goal_keywords(goal.query_text)
    scores = []
    for unit in evidence_units:
        unit_kw = set(re.findall(r"[가-힣]{2,}", unit.summary.lower()))
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


def coverage_at_k(
    evidence_units: list[CompressedEvidenceUnit],
    labels: dict[str, float],
    k: int | None = None,
) -> float:
    """Coverage@k = |covered_relevant| / |relevant|.

    Measures what fraction of relevant raw logs are referenced by
    the top-k evidence units via anchor_log_ids.

    Parameters
    ----------
    evidence_units:
        CompressedEvidenceUnit list (already sorted by relevance).
    labels:
        {log_id: relevance_score} dict.  Logs with score >= 1 are
        considered relevant.
    k:
        Number of evidence units to consider.  None = all units.
    """
    if not labels:
        return 0.0
    relevant = {lid for lid, score in labels.items() if score >= 1}
    if not relevant:
        return 0.0
    units = evidence_units[:k] if k is not None else evidence_units
    covered: set[str] = set()
    for unit in units:
        covered.update(getattr(unit, "anchor_log_ids", []))
    return len(covered & relevant) / len(relevant)


def llm_judge_score(
    goal: ResearchGoal,
    unit: CompressedEvidenceUnit,
    judge_fn=None,
) -> float:
    """LLM-as-judge: goal–CEU relevance via an external judge function.

    Placeholder implementation — returns 0.0 until judge_fn is wired.

    Parameters
    ----------
    goal:      ResearchGoal
    unit:      CompressedEvidenceUnit
    judge_fn:  callable(goal, unit) -> float in [1.0, 5.0]
               None means not yet implemented → returns 0.0.

    Returns
    -------
    float: judge score in [1.0, 5.0], or 0.0 if judge_fn is None.
    """
    if judge_fn is None:
        return 0.0
    return judge_fn(goal, unit)


def compute_rag_metrics(
    goal: ResearchGoal,
    original_logs: list[ResearchLog],
    evidence_units: list[CompressedEvidenceUnit],
    labels: dict[str, float] | None = None,
    k: int | None = None,
) -> dict[str, float]:
    """Compute all Stage 2 RAG metrics.

    Parameters
    ----------
    labels:
        Optional {log_id: relevance_score} dict for coverage@k.
        Defaults to {} (coverage will be 0.0).
    k:
        Number of evidence units for coverage@k.  None = all.
    """
    lbl = labels or {}
    return {
        "goal_alignment_score": goal_alignment_score(goal, evidence_units),
        "actionability_score": actionability_score(evidence_units),
        "token_reduction_rate": token_reduction_rate(original_logs, evidence_units),
        "redundancy_reduction": redundancy_reduction(original_logs, evidence_units),
        "coverage@k": coverage_at_k(evidence_units, lbl, k),
        "evidence_unit_count": len(evidence_units),
    }
