"""Goal-Conditioned Evidence Ranker.

score = 0.45 * semantic_relevance
      + 0.40 * goal_focus        <- raised; uses expanded terms for precision
      + 0.15 * evidence_value

goal_focus now uses expanded_terms overlap so that goal-specific
vocabulary (e.g. "코딩 테스트", "알고리즘") boosts relevant logs
rather than generic activity words.
"""
from __future__ import annotations

import logging
import re

from app.config import RankerConfig
from app.retrieval.dense_retriever import DenseRetriever, cosine
from app.schemas import CandidateLog, RankedLog, ResearchGoal

logger = logging.getLogger(__name__)

_EVIDENCE_KEYWORDS = {
    "완료", "완성", "제출", "발표", "구현", "해결", "달성",
    "finished", "completed", "implemented", "solved", "achieved",
    "문제 풀이", "오답 정리", "결과 정리", "제출 완료",
}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[\w\uAC00-\uD7A3]+", text.lower())


def _token_overlap_score(query_tokens: list[str], log_tokens: list[str]) -> float:
    """Soft overlap: fraction of query tokens that appear in log."""
    if not query_tokens or not log_tokens:
        return 0.0
    log_set = set(log_tokens)
    hits = sum(1 for t in query_tokens if t in log_set)
    return hits / len(query_tokens)


class GoalConditionedReranker:
    """Multi-component evidence scorer.

    Accepts optional expanded_terms to strengthen goal_focus signal.
    """

    def __init__(
        self,
        config: RankerConfig | None = None,
        dense_retriever: DenseRetriever | None = None,
    ) -> None:
        self.config = config or RankerConfig()
        self._dense = dense_retriever or DenseRetriever()

    def _semantic_relevance(self, goal: ResearchGoal, log_text: str) -> float:
        g_emb = self._dense.embed(goal.query_text)
        l_emb = self._dense.embed(log_text)
        return max(0.0, cosine(g_emb, l_emb))

    def _goal_focus(
        self,
        goal: ResearchGoal,
        log_text: str,
        expanded_terms: list[str] | None = None,
    ) -> float:
        """Overlap-based goal relevance.

        Uses expanded_terms (goal-specific vocabulary) when available,
        falling back to raw goal tokens.
        High-precision: checks how much of the goal vocab appears in the log,
        not the reverse (avoids noisy matches from short logs).
        """
        log_tokens = _tokenize(log_text)

        if expanded_terms:
            # Score against expanded vocabulary
            exp_tokens = [t for term in expanded_terms for t in _tokenize(term)]
            score_exp = _token_overlap_score(exp_tokens, log_tokens)

            # Also score against raw goal text
            goal_tokens = _tokenize(goal.query_text)
            score_raw = _token_overlap_score(goal_tokens, log_tokens)

            # Weighted combination: expanded terms are more precise
            return min(0.7 * score_exp + 0.3 * score_raw, 1.0)
        else:
            goal_tokens = _tokenize(goal.query_text)
            return _token_overlap_score(goal_tokens, log_tokens)

    def _evidence_value(self, log_text: str) -> float:
        tokens_set = set(_tokenize(log_text))
        # Direct keyword hits
        keyword_score = min(
            sum(1 for kw in _EVIDENCE_KEYWORDS if kw in log_text.lower()) / 3.0,
            1.0,
        )
        # Specificity bonus for longer, content-rich logs
        specificity = min(len(tokens_set) / 40.0, 0.4)
        return min(keyword_score + specificity, 1.0)

    def score(
        self,
        goal: ResearchGoal,
        candidate: CandidateLog,
        expanded_terms: list[str] | None = None,
    ) -> RankedLog:
        log_text = candidate.log.full_text
        sr = self._semantic_relevance(goal, log_text)
        gf = self._goal_focus(goal, log_text, expanded_terms)
        ev = self._evidence_value(log_text)
        w = self.config
        final = (
            w.semantic_weight * sr
            + w.goal_focus_weight * gf
            + w.evidence_value_weight * ev
        )
        return RankedLog(
            log=candidate.log,
            semantic_relevance=round(sr, 4),
            goal_focus=round(gf, 4),
            evidence_value=round(ev, 4),
            final_score=round(final, 4),
        )

    def rank(
        self,
        goal: ResearchGoal,
        candidates: list[CandidateLog],
        expanded_terms: list[str] | None = None,
    ) -> list[RankedLog]:
        """Score and rank candidates. Pass expanded_terms for better goal_focus."""
        ranked = [self.score(goal, c, expanded_terms) for c in candidates]
        ranked.sort(key=lambda x: x.final_score, reverse=True)
        for i, r in enumerate(ranked):
            r.rank = i + 1
        logger.debug(
            "Reranked %d candidates for goal=%s (top score=%.4f)",
            len(ranked), goal.goal_id, ranked[0].final_score if ranked else 0,
        )
        return ranked
