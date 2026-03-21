"""Goal-Conditioned Evidence Ranker.

score = 0.40 * semantic_relevance
      + 0.45 * goal_focus          <- primary signal
      + 0.15 * evidence_value
      - NEGATIVE_TERM_PENALTY * domain_mismatch

goal_focus uses expanded evidence_terms for direct overlap scoring.
domain_mismatch penalises logs that contain negative_terms or are
entirely unrelated daily activities.
"""
from __future__ import annotations

import logging
import re

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
    "공부", "학습", "정리", "연습", "작성", "읽기", "review", "풀이",
    "실습", "수행", "진행", "실행",
}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[\w가-힣]+", text.lower())


def _token_set(text: str) -> set[str]:
    return set(_tokenize(text))


def _overlap_ratio(query_tokens: set[str], doc_tokens: set[str]) -> float:
    """Fraction of query_tokens found in doc_tokens."""
    if not query_tokens:
        return 0.0
    return len(query_tokens & doc_tokens) / len(query_tokens)


class GoalConditionedReranker:
    """Multi-component scorer: semantic + goal_focus + evidence - penalty."""

    def __init__(
        self,
        config: RankerConfig | None = None,
        dense_retriever: DenseRetriever | None = None,
        negative_term_penalty: float = 0.30,
    ) -> None:
        self.config = config or RankerConfig()
        self._dense = dense_retriever or DenseRetriever()
        self.negative_term_penalty = negative_term_penalty

    def _semantic_relevance(self, goal: ResearchGoal, log_text: str) -> float:
        g_emb = self._dense.embed(goal.query_text)
        l_emb = self._dense.embed(log_text)
        return max(0.0, cosine(g_emb, l_emb))

    def _goal_focus(
        self,
        goal: ResearchGoal,
        log_text: str,
        expanded_terms: list[str],
    ) -> float:
        """Goal relevance: 70% expanded_terms overlap + 30% base goal overlap."""
        log_tokens = _token_set(log_text)

        # Build expanded token set
        expanded_tokens: set[str] = set()
        for term in expanded_terms:
            expanded_tokens.update(_tokenize(term))
        expanded_overlap = _overlap_ratio(expanded_tokens, log_tokens) if expanded_tokens else 0.0

        # Base goal token overlap
        goal_tokens = _token_set(goal.query_text + " " + goal.goal_embedding_text)
        base_overlap = _overlap_ratio(goal_tokens, log_tokens) if goal_tokens else 0.0

        return 0.7 * expanded_overlap + 0.3 * base_overlap

    def _evidence_value(self, candidate: CandidateLog) -> float:
        """Concrete evidence heuristic: keywords + activity_type + metadata strength."""
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
        """Penalty: 0.0 (no mismatch) to 1.0 (strong mismatch)."""
        log_tokens = _token_set(candidate.log.full_text)

        neg_tokens: set[str] = set()
        for term in negative_terms:
            neg_tokens.update(_tokenize(term))
        neg_overlap = _overlap_ratio(neg_tokens, log_tokens) if neg_tokens else 0.0

        noise_penalty = 0.5 if candidate.log.activity_type in _NOISE_TYPES else 0.0

        return min(neg_overlap + noise_penalty, 1.0)

    def score(
        self,
        goal: ResearchGoal,
        candidate: CandidateLog,
        expanded_terms: list[str] | None = None,
        negative_terms: list[str] | None = None,
    ) -> RankedLog:
        exp_terms = expanded_terms or []
        neg_terms = negative_terms or []

        log_text = candidate.log.full_text
        sr = self._semantic_relevance(goal, log_text)
        gf = self._goal_focus(goal, log_text, exp_terms)
        ev = self._evidence_value(candidate)
        dm = self._domain_mismatch(candidate, neg_terms)

        w = self.config
        final = max(0.0, round(
            w.semantic_weight * sr
            + w.goal_focus_weight * gf
            + w.evidence_value_weight * ev
            - self.negative_term_penalty * dm,
            4,
        ))

        if dm > 0.3:
            logger.debug(
                "Penalty  log=%s  dm=%.2f  final=%.4f  [%s]",
                candidate.log.log_id, dm, final, candidate.log.title,
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
    ) -> list[RankedLog]:
        ranked = [
            self.score(goal, c, expanded_terms=expanded_terms, negative_terms=negative_terms)
            for c in candidates
        ]
        ranked.sort(key=lambda x: x.final_score, reverse=True)
        for i, r in enumerate(ranked):
            r.rank = i + 1
        logger.debug("Reranked %d candidates  goal=%s", len(ranked), goal.goal_id)
        return ranked
