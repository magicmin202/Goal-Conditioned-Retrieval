"""Goal-Conditioned Evidence Ranker.

score = semantic_weight  * semantic_relevance
      + goal_focus_weight * goal_focus          ← primary signal
      + evidence_value_weight * evidence_value
      + priority_boost                          ← additive if matches priority_terms
      - negative_term_penalty * domain_mismatch

goal_focus uses 3-tier vocabulary:
  priority_focus_weight  * priority_term_match  (phrase + token)
  + evidence_focus_weight * evidence_term_match
  + related_focus_weight  * related_term_match

domain_mismatch uses phrase + token matching against negative_terms.
priority_boost gives additive reward when log matches high-signal terms.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from app.config import RankerConfig
from app.retrieval.dense_retriever import DenseRetriever, cosine
from app.schemas import CandidateLog, RankedLog, ResearchGoal

logger = logging.getLogger(__name__)

_PRODUCTIVE_TYPES = {
    "study", "implementation", "reading", "exercise",
    "planning", "execution", "reflection", "coding", "social",
}
_NOISE_TYPES = {"daily"}

_EVIDENCE_KEYWORDS = {
    "완료", "완성", "제출", "발표", "구현", "해결", "달성", "성공",
    "finished", "completed", "implemented", "solved", "achieved",
    "공부", "연습", "작성", "읽기", "풀이", "실습", "수행",
}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[\w가-힣]+", text.lower())


def _token_set(text: str) -> set[str]:
    return set(_tokenize(text))


def _overlap_ratio(query_tokens: set[str], doc_tokens: set[str]) -> float:
    if not query_tokens:
        return 0.0
    return len(query_tokens & doc_tokens) / len(query_tokens)


def _phrase_overlap(terms: list[str], text: str, phrase_multiplier: float = 1.5) -> float:
    """Match score with phrase bonus.

    - Phrase match (multi-token term appears as substring): weight = phrase_multiplier
    - Token match (any token of term found): weight = 1.0
    Normalized to [0, 1] against max possible score.
    """
    if not terms:
        return 0.0
    text_lower = text.lower()
    text_tokens = _token_set(text_lower)

    matched = 0.0
    max_possible = len(terms) * phrase_multiplier

    for term in terms:
        term_lower = term.lower()
        term_tokens = _tokenize(term_lower)
        if len(term_tokens) > 1 and term_lower in text_lower:
            matched += phrase_multiplier      # phrase match → full credit
        elif term_tokens and any(t in text_tokens for t in term_tokens):
            matched += 1.0                    # token match → partial credit

    return min(matched / max_possible, 1.0)


class GoalConditionedReranker:
    """Multi-component scorer: semantic + goal_focus + evidence - penalty."""

    def __init__(
        self,
        config: RankerConfig | None = None,
        dense_retriever: DenseRetriever | None = None,
        negative_term_penalty: Optional[float] = None,
    ) -> None:
        self.config = config or RankerConfig()
        self._dense = dense_retriever or DenseRetriever()
        self.negative_term_penalty = (
            negative_term_penalty if negative_term_penalty is not None
            else self.config.negative_term_penalty
        )

    def _semantic_relevance(self, goal: ResearchGoal, log_text: str) -> float:
        g_emb = self._dense.embed(goal.query_text)
        l_emb = self._dense.embed(log_text)
        return max(0.0, cosine(g_emb, l_emb))

    def _goal_focus(
        self,
        goal: ResearchGoal,
        log_text: str,
        priority_terms: list[str],
        evidence_terms: list[str],
        related_terms: list[str],
    ) -> float:
        """3-tier goal relevance score.

        priority_focus_weight  * priority_term_match   (strongest signal)
        + evidence_focus_weight * evidence_term_match
        + related_focus_weight  * related_term_match
        + small base goal anchor
        """
        cfg = self.config
        priority_score = _phrase_overlap(priority_terms, log_text)
        evidence_score = _phrase_overlap(evidence_terms, log_text)
        related_score = _phrase_overlap(related_terms, log_text)

        # Base anchor: raw goal text overlap
        base_tokens = _token_set(goal.query_text + " " + goal.goal_embedding_text)
        base_score = _overlap_ratio(base_tokens, _token_set(log_text))

        return (
            cfg.priority_focus_weight * priority_score
            + cfg.evidence_focus_weight * evidence_score
            + cfg.related_focus_weight * related_score
            + 0.10 * base_score  # small anchor, not dominant
        )

    def _evidence_value(self, candidate: CandidateLog) -> float:
        tokens = _token_set(candidate.log.full_text)
        keyword_score = min(len(tokens & _EVIDENCE_KEYWORDS) / 3.0, 1.0)
        specificity = min(len(tokens) / 50.0, 0.4)
        act_bonus = 0.2 if candidate.log.activity_type in _PRODUCTIVE_TYPES else 0.0
        strength = candidate.log.metadata.get("evidence_strength", "medium")
        strength_bonus = {"high": 0.2, "medium": 0.1, "low": 0.0}.get(strength, 0.0)
        return min(keyword_score + specificity + act_bonus + strength_bonus, 1.0)

    def _domain_mismatch(
        self,
        candidate: CandidateLog,
        negative_terms: list[str],
    ) -> float:
        """Penalty: 0.0 (clean) to 1.0 (strong mismatch).

        Phrase match → 0.6 per term (stronger signal).
        Token match  → 0.3 per term (moderate signal).
        Daily activity_type adds 0.5 noise penalty.
        """
        log_text = candidate.log.full_text
        text_lower = log_text.lower()
        text_tokens = _token_set(text_lower)

        match_score = 0.0
        matched_terms: list[str] = []

        for term in negative_terms:
            term_lower = term.lower()
            term_tokens = _tokenize(term_lower)
            if len(term_tokens) > 1 and term_lower in text_lower:
                match_score += 0.6
                matched_terms.append(f"phrase:{term}")
            elif term_tokens and any(t in text_tokens for t in term_tokens):
                match_score += 0.3
                matched_terms.append(f"token:{term}")

        term_penalty = min(match_score, 1.0)
        noise_penalty = 0.5 if candidate.log.activity_type in _NOISE_TYPES else 0.0
        total = min(term_penalty + noise_penalty, 1.0)

        if total > 0.2:
            logger.debug(
                "NegMatch  log=%s  penalty=%.2f  matched=%s  [%s]",
                candidate.log.log_id, total, matched_terms, candidate.log.title,
            )
        return total

    def _priority_boost(
        self,
        log_text: str,
        priority_terms: list[str],
    ) -> float:
        """Additive boost when log matches priority (high-signal) terms.

        Uses phrase-aware matching: phrase hits count more than token hits.
        """
        if not priority_terms:
            return 0.0
        overlap = _phrase_overlap(priority_terms, log_text)
        if overlap > 0.0:
            boost = min(self.config.priority_term_boost * overlap * 2.0,
                        self.config.priority_term_boost)
            if boost > 0.0:
                logger.debug(
                    "PriorityBoost  overlap=%.3f  boost=%.4f  [%s]",
                    overlap, boost, log_text[:60],
                )
            return boost
        return 0.0

    def score(
        self,
        goal: ResearchGoal,
        candidate: CandidateLog,
        expanded_terms: list[str] | None = None,    # evidence_terms
        negative_terms: list[str] | None = None,
        priority_terms: list[str] | None = None,
        related_terms: list[str] | None = None,
    ) -> RankedLog:
        ev_terms = expanded_terms or []
        neg_terms = negative_terms or []
        pri_terms = priority_terms or []
        rel_terms = related_terms or []

        log_text = candidate.log.full_text
        sr = self._semantic_relevance(goal, log_text)
        gf = self._goal_focus(goal, log_text, pri_terms, ev_terms, rel_terms)
        ev = self._evidence_value(candidate)
        dm = self._domain_mismatch(candidate, neg_terms)
        pb = self._priority_boost(log_text, pri_terms)

        w = self.config
        final = max(0.0, round(
            w.semantic_weight * sr
            + w.goal_focus_weight * gf
            + w.evidence_value_weight * ev
            + pb
            - self.negative_term_penalty * dm,
            4,
        ))

        logger.debug(
            "Score  log=%s  sr=%.3f  gf=%.3f  ev=%.3f  dm=%.3f  pb=%.4f  → %.4f  [%s]",
            candidate.log.log_id, sr, gf, ev, dm, pb, final, candidate.log.title,
        )

        return RankedLog(
            log=candidate.log,
            semantic_relevance=round(sr, 4),
            goal_focus=round(gf, 4),
            evidence_value=round(ev, 4),
            final_score=final,
        )

    def rank(
        self,
        goal: ResearchGoal,
        candidates: list[CandidateLog],
        expanded_terms: list[str] | None = None,
        negative_terms: list[str] | None = None,
        priority_terms: list[str] | None = None,
        related_terms: list[str] | None = None,
    ) -> list[RankedLog]:
        ranked = [
            self.score(
                goal, c,
                expanded_terms=expanded_terms,
                negative_terms=negative_terms,
                priority_terms=priority_terms,
                related_terms=related_terms,
            )
            for c in candidates
        ]
        ranked.sort(key=lambda x: x.final_score, reverse=True)
        for i, r in enumerate(ranked):
            r.rank = i + 1
        logger.debug("Reranked %d candidates  goal=%s", len(ranked), goal.goal_id)
        return ranked
