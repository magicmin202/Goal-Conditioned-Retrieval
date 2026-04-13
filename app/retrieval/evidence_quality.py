"""Evidence Quality Scoring — Stage1 admission quality upgrade.

Separates *relevance* (goal signal match) from *evidence quality* (analysis value).

Problem:
  "Generic but relevant" logs (여행 준비물 쇼핑 ×3, 짐 준비, 조사해봄) can pass
  all relevance gates yet contribute little to goal analysis. This scorer
  penalises low-information evidence so only high-value anchors rise to top-k.

Components (all in [0, 1]):
  specificity     — how concrete/measurable? (numbers, metrics, domain nouns)
  actionability   — real execution/completion vs browsing/planning?
  goal_progress   — category-level value prior (booking > research, etc.)
  domain_consist  — goal-domain-aware activity type consistency

Redundancy is NOT scored here — it requires comparing across the admitted set.
Apply redundancy penalty in Stage1Pipeline after initial ranking.

Usage (inside reranker):
    qs = _quality_scorer.score(log, log_category, goal_domain, cat_result)
    quality_score = qs.total          # additive component in final formula

Usage (redundancy, in Stage1Pipeline):
    penalty, reason = compute_redundancy_penalty(log, admitted_logs)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.schemas import ResearchLog
from app.retrieval.schema_category import (
    classify_log_activity_type,
    get_activity_type_quality_prior,
)

# ── Completion / execution action vocabulary ──────────────────────────────────

_ACTION_COMPLETION: frozenset[str] = frozenset({
    # Korean completion/execution verbs
    "예약", "완료", "구매", "제출", "작성", "구현", "배포", "확정",
    "결제", "등록", "신청", "달성", "해결", "풀었", "풀이", "오답",
    "정리완료", "제출완료", "작성완료",
    # Korean verbs for concrete output
    "만들었", "완성", "성공", "획득", "합격", "통과", "완수",
    "발표", "실습완료", "수행완료",
    # English
    "booked", "reserved", "purchased", "submitted", "completed",
    "implemented", "deployed", "achieved", "finished", "solved",
    "paid", "registered",
})

# Generic browse/search verbs — signal low actionability
_GENERIC_BROWSE: frozenset[str] = frozenset({
    "조사", "검색", "찾아봄", "알아봄", "살펴봄", "봄", "구경",
    "스크롤", "탐색", "찾기", "읽어봄", "찾았", "찾아",
    "browse", "scroll", "searched", "looked",
})

# Generic planning verbs — some value but not concrete
_GENERIC_PLANNING: frozenset[str] = frozenset({
    "준비", "계획", "고민", "생각", "정리",
})

# Metric patterns (high specificity signal)
_METRIC_RE = re.compile(
    r"\d+\s*("
    r"kg|km|원|만원|천원|회|세트|분|시간|점|개|번|일|박|박\s*\d*일"
    r"|달러|\$|€|문제|장|권|페이지"
    r")",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"\d{2,}")   # 2+ digit numbers = likely meaningful

# _CATEGORY_VALUE_PRIORS removed — domain schema deprecated.
# Evidence quality is now derived from activity-type priors only.
# See get_activity_type_quality_prior() in schema_category.py.


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class EvidenceQualityScore:
    specificity:    float = 0.0
    actionability:  float = 0.0
    goal_progress:  float = 0.0   # category value prior
    domain_consist: float = 0.0   # goal-domain-aware activity type
    total:          float = 0.0   # weighted combination

    # Trace fields
    has_numbers:     bool = False
    has_metrics:     bool = False
    action_hits:     list[str] = field(default_factory=list)
    browse_hits:     list[str] = field(default_factory=list)


# ── Scorer ────────────────────────────────────────────────────────────────────

class EvidenceQualityScorer:
    """Score a single log on evidence quality (not goal relevance).

    Weights within quality_score:
      specificity    0.25
      actionability  0.35
      goal_progress  0.25
      domain_consist 0.15
    """

    _W_SPEC   = 0.25
    _W_ACTION = 0.35
    _W_PROG   = 0.25
    _W_DOM    = 0.15

    def score(
        self,
        log: ResearchLog,
        activity_type: str = "unknown",   # inferred from log text, replaces cat_relevance
    ) -> EvidenceQualityScore:
        spec, spec_trace = self._specificity(log)
        action, action_trace = self._actionability(log)
        progress = self._goal_progress(log, activity_type)
        domain = self._domain_consistency(activity_type)

        total = round(
            self._W_SPEC   * spec
            + self._W_ACTION * action
            + self._W_PROG   * progress
            + self._W_DOM    * domain,
            4,
        )

        return EvidenceQualityScore(
            specificity=round(spec, 4),
            actionability=round(action, 4),
            goal_progress=round(progress, 4),
            domain_consist=round(domain, 4),
            total=total,
            has_numbers=spec_trace["has_numbers"],
            has_metrics=spec_trace["has_metrics"],
            action_hits=action_trace["hits"],
            browse_hits=action_trace["browse"],
        )

    # ── Sub-components ────────────────────────────────────────────────────────

    def _specificity(self, log: ResearchLog) -> tuple[float, dict]:
        """How concrete/measurable is this log?

        Signals:
          +0.5  metric (숫자 + 단위): 항공권 30만원, 벤치 80kg 5세트
          +0.3  bare numbers (2+ digits) in content
          +0.2  title length ≥ 12 chars (descriptive title)
          +0.1  title length ≥ 8 chars
          −0.2  title consists entirely of generic verbs
        """
        text = log.full_text
        title = log.title.strip()

        has_metrics = bool(_METRIC_RE.search(text))
        has_numbers = bool(_NUMBER_RE.search(text))

        score = 0.0
        if has_metrics:
            score += 0.50
        elif has_numbers:
            score += 0.30

        tlen = len(title)
        if tlen >= 12:
            score += 0.20
        elif tlen >= 8:
            score += 0.10

        # Generic-title penalty
        title_toks = set(re.findall(r"[\w가-힣]+", title.lower()))
        if title_toks and title_toks.issubset(_GENERIC_BROWSE | _GENERIC_PLANNING):
            score -= 0.20

        return min(max(score, 0.0), 1.0), {
            "has_numbers": has_numbers,
            "has_metrics": has_metrics,
        }

    def _actionability(self, log: ResearchLog) -> tuple[float, dict]:
        """Is this log execution/completion rather than browsing/planning?

        Signals:
          +0.4 per completion verb (up to 0.80)
          +0.2  productive activity_type
          −0.2  browse-only with no action hits
          −0.1  daily activity_type
        """
        text = log.full_text.lower()
        toks = set(re.findall(r"[\w가-힣]+", text))

        action_hits = list(toks & _ACTION_COMPLETION)
        browse_hits = list(toks & _GENERIC_BROWSE)

        score = 0.0
        if action_hits:
            score += min(len(action_hits) * 0.40, 0.80)

        if browse_hits and not action_hits:
            score -= 0.20

        act = log.activity_type or ""
        if act in {"implementation", "coding", "execution", "exercise"}:
            score += 0.20
        elif act == "planning":
            score += 0.05
        elif act == "daily":
            score -= 0.10

        return min(max(score, 0.0), 1.0), {
            "hits": action_hits,
            "browse": browse_hits,
        }

    def _goal_progress(self, log: ResearchLog, activity_type: str) -> float:
        """Activity-type based goal progress prior.

        Evidence value is tied to *how* the log was done (activity_type),
        not domain taxonomy.  creative/execution types score highest (concrete
        output), lifestyle types score lowest.
        """
        weights = get_activity_type_quality_prior(activity_type)
        return weights["progression"]

    def _domain_consistency(self, activity_type: str) -> float:
        """Activity-type based domain consistency.

        Domain schema removed — consistency is now measured by how
        'productive' the activity type is, independent of goal domain.
        creative and execution types indicate concrete progress (high score);
        lifestyle indicates routine activity (low score).
        """
        weights = get_activity_type_quality_prior(activity_type)
        # Use actionability as a proxy for domain consistency:
        # high actionability types (execution/creative) map to high consistency.
        return weights["actionability"]


# ── Redundancy (applied in Stage1Pipeline, not inside reranker) ───────────────

def compute_redundancy_penalty(
    log: ResearchLog,
    admitted_logs: list[ResearchLog],
    exact_penalty: float = 0.30,
    similar_penalty: float = 0.15,
    similarity_threshold: float = 0.60,
) -> tuple[float, str]:
    """Return (penalty, reason) based on similarity to already-admitted logs.

    Exact title match  → exact_penalty
    High token overlap (≥ threshold) → similar_penalty

    Applied AFTER initial ranking so top-score logs enter admitted set first.
    """
    if not admitted_logs:
        return 0.0, ""

    title_norm = re.sub(r"\s+", "", log.title.lower())
    log_toks = frozenset(re.findall(r"[\w가-힣]{2,}", log.title.lower()))

    for prev in admitted_logs:
        prev_norm = re.sub(r"\s+", "", prev.title.lower())
        if title_norm == prev_norm:
            return exact_penalty, f"exact_dup:{prev.log_id}"

        prev_toks = frozenset(re.findall(r"[\w가-힣]{2,}", prev.title.lower()))
        if log_toks and prev_toks:
            union = log_toks | prev_toks
            overlap = len(log_toks & prev_toks) / len(union) if union else 0.0
            if overlap >= similarity_threshold:
                return similar_penalty, f"similar({overlap:.2f}):{prev.log_id}"

    return 0.0, ""


# Module-level singleton
_quality_scorer = EvidenceQualityScorer()
