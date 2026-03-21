"""Centralized phrase/token/title-aware matching utilities.

Match levels (priority order):
  phrase – full multi-token term appears as substring in text
  token  – at least one individual token found in document token set

Title is a stronger signal than body content.
All title matches receive a title_weight_multiplier bonus.

Used by both candidate_retrieval (vocab boost) and reranker (goal_focus, domain_mismatch).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

MatchLevel = Literal["phrase", "token", "none"]


@dataclass
class TermMatch:
    term: str
    level: MatchLevel
    in_title: bool


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
