"""Centralized phrase/token/title-aware matching utilities.

Match levels (priority order):
  phrase – full multi-token term appears as substring in text
  token  – at least one individual token found in document token set

Title is a stronger signal than body content.
All title matches receive a title_weight_multiplier bonus.

Used by both candidate_retrieval (vocab boost) and reranker (goal_focus, domain_mismatch).

Priority/evidence term scoring uses `score_priority_terms` which applies
WEAK_TOKEN filtering: generic action/state tokens ("완료", "정리", …) are
never allowed to trigger a positive match on their own.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)

MatchLevel = Literal["phrase", "token", "none"]

# ── Weak/generic tokens ───────────────────────────────────────────────────────
# These tokens must NOT trigger a positive match on their own for priority /
# evidence terms.  They are only allowed as *part* of an exact phrase match,
# or when at least one non-weak core token also matches.
WEAK_TOKENS: frozenset[str] = frozenset({
    "완료", "시작", "정리", "계획", "준비",
    "공부", "하기", "수행", "진행", "실행",
    "관리", "확인", "검토", "작성",
})


@dataclass
class TermMatch:
    term: str
    level: MatchLevel
    in_title: bool


@dataclass
class PriorityTermMatch:
    """Rich match result used by score_priority_terms."""
    term: str
    mode: str           # "exact_phrase" | "core_token" | "weak_token_only" | "none"
    score: float        # 0.0 for weak_token_only / none
    core_hits: list[str] = field(default_factory=list)
    weak_hits_only: list[str] = field(default_factory=list)
    in_title: bool = False


def _tok(text: str) -> list[str]:
    return re.findall(r"[\w가-힣]+", text.lower())


def _tok_set(text: str) -> set[str]:
    return set(_tok(text))


def match_term(
    term: str,
    text_lower: str,
    text_tokens: set[str],
    title_lower: str = "",
    title_tokens: set[str] | None = None,
) -> TermMatch:
    """Match a single term against (text, title).

    Priority: phrase > token > none.
    in_title=True if the match is found within the title string/tokens.
    """
    term_lower = term.lower()
    term_toks = set(_tok(term_lower))
    if not term_toks:
        return TermMatch(term=term, level="none", in_title=False)

    is_multi = len(term_toks) > 1

    # --- phrase match (full term as substring) ---
    if is_multi and term_lower in text_lower:
        in_title = bool(title_lower and term_lower in title_lower)
        return TermMatch(term=term, level="phrase", in_title=in_title)

    # --- token match (any term token in doc tokens) ---
    if any(t in text_tokens for t in term_toks):
        in_title = bool(
            title_tokens and any(t in title_tokens for t in term_toks)
        )
        return TermMatch(term=term, level="token", in_title=in_title)

    return TermMatch(term=term, level="none", in_title=False)


def score_terms(
    terms: list[str],
    text: str,
    title: str = "",
    phrase_weight: float = 1.5,
    token_weight: float = 1.0,
    title_multiplier: float = 1.5,
) -> tuple[float, list[TermMatch]]:
    """Normalized [0, 1] match score for a list of terms.

    Returns (score, matches) so callers can log which terms matched and how.

    Normalization: max_possible = len(terms) * phrase_weight * title_multiplier
    This means every term matching as phrase-in-title gives 1.0.
    """
    if not terms:
        return 0.0, []

    text_lower = text.lower()
    text_tokens = _tok_set(text_lower)
    title_lower = title.lower() if title else ""
    title_tokens = _tok_set(title_lower) if title else set()

    matches = [
        match_term(t, text_lower, text_tokens, title_lower, title_tokens)
        for t in terms
    ]

    max_possible = len(terms) * phrase_weight * title_multiplier
    total = 0.0
    for m in matches:
        if m.level == "none":
            continue
        base = phrase_weight if m.level == "phrase" else token_weight
        total += base * (title_multiplier if m.in_title else 1.0)

    return min(total / max_possible, 1.0), matches


def match_priority_phrase(
    term: str,
    text_lower: str,
    text_tokens: set[str],
    title_lower: str = "",
    title_tokens: set[str] | None = None,
) -> PriorityTermMatch:
    """Phrase-first matching with weak-token filtering.

    Rules:
      (A) Exact phrase match (multi-token)  → mode="exact_phrase",  score=1.0
      (B) Core (non-weak) token hit         → mode="core_token",    score=0.4
      (C) Weak token hit only               → mode="weak_token_only", score=0.0
      (D) No match                          → mode="none",           score=0.0

    Example:
      term="독서 완료", text="숙소 예약 완료"
        → no exact phrase, core=["독서"] misses, weak=["완료"] hits
        → weak_token_only, score=0.0   ← NOT a positive signal
    """
    term_lower = term.lower()
    tokens = _tok(term_lower)

    if not tokens:
        return PriorityTermMatch(term=term, mode="none", score=0.0)

    # (A) exact phrase for multi-token terms
    if len(tokens) > 1 and term_lower in text_lower:
        in_title = bool(title_lower and term_lower in title_lower)
        return PriorityTermMatch(
            term=term, mode="exact_phrase", score=1.0, in_title=in_title,
        )

    core_toks = [t for t in tokens if t not in WEAK_TOKENS]
    weak_toks  = [t for t in tokens if t in WEAK_TOKENS]

    core_hits = [t for t in core_toks if t in text_tokens]
    weak_hits  = [t for t in weak_toks  if t in text_tokens]

    if core_hits:
        in_title = bool(
            title_tokens and any(t in title_tokens for t in core_hits)
        )
        return PriorityTermMatch(
            term=term, mode="core_token", score=0.4,
            core_hits=core_hits, in_title=in_title,
        )

    if weak_hits:
        return PriorityTermMatch(
            term=term, mode="weak_token_only", score=0.0,
            weak_hits_only=weak_hits,
        )

    return PriorityTermMatch(term=term, mode="none", score=0.0)


def score_priority_terms(
    terms: list[str],
    text: str,
    title: str = "",
    phrase_weight: float = 1.5,
    token_weight: float = 0.4,       # reduced vs score_terms (core-token only)
    title_multiplier: float = 1.5,
) -> tuple[float, list[PriorityTermMatch]]:
    """Normalized [0, 1] score using weak-token-filtered phrase matching.

    Replaces score_terms() for priority/evidence term scoring in the reranker.
    Weak-only hits (e.g. "완료" matching "숙소 예약 완료") contribute 0 score.
    """
    if not terms:
        return 0.0, []

    text_lower = text.lower()
    text_tokens = _tok_set(text_lower)
    title_lower = title.lower() if title else ""
    title_tokens = _tok_set(title_lower) if title else set()

    matches = [
        match_priority_phrase(t, text_lower, text_tokens, title_lower, title_tokens)
        for t in terms
    ]

    max_possible = len(terms) * phrase_weight * title_multiplier
    total = 0.0
    for m in matches:
        if m.score == 0.0:
            continue
        base = phrase_weight if m.mode == "exact_phrase" else token_weight
        total += base * (title_multiplier if m.in_title else 1.0)

    return min(total / max_possible, 1.0), matches


def penalty_score(
    terms: list[str],
    text: str,
    title: str = "",
    phrase_penalty: float = 0.70,
    token_penalty: float = 0.40,
    title_extra_penalty: float = 0.30,
) -> tuple[float, list[str]]:
    """Compute penalty score from negative term matches.

    Returns (raw_penalty_sum, matched_descriptions).
    Caller is responsible for capping at 1.0.

    Phrase match in title → phrase_penalty + title_extra_penalty.
    Phrase match in body  → phrase_penalty.
    Token match in title  → token_penalty + title_extra_penalty.
    Token match in body   → token_penalty.
    """
    if not terms:
        return 0.0, []

    text_lower = text.lower()
    text_tokens = _tok_set(text_lower)
    title_lower = title.lower() if title else ""
    title_tokens = _tok_set(title_lower) if title else set()

    total = 0.0
    matched: list[str] = []

    for term in terms:
        m = match_term(term, text_lower, text_tokens, title_lower, title_tokens)
        if m.level == "none":
            continue
        base = phrase_penalty if m.level == "phrase" else token_penalty
        extra = title_extra_penalty if m.in_title else 0.0
        total += base + extra
        tag = f"{'phrase' if m.level == 'phrase' else 'token'}{'_title' if m.in_title else ''}:{term}"
        matched.append(tag)

    return total, matched
