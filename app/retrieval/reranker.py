"""Goal-Conditioned Evidence Ranker.

score = 0.5 * semantic_relevance + 0.3 * goal_focus + 0.2 * evidence_value

All components are rule-based / lexical by default.
TODO: replace semantic_relevance with real embedding cosine similarity.
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
    "공부", "학습", "정리", "연습", "작성", "읽기", "review",
}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[\w가-힣]+", text.lower())


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


class GoalConditionedReranker:
    """Compute a multi-component evidence score for each candidate log."""

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

    def _goal_focus(self, goal: ResearchGoal, log_text: str) -> float:
        return _jaccard(_tokenize(goal.query_text), _tokenize(log_text))

    def _evidence_value(self, log_text: str) -> float:
        tokens = set(_tokenize(log_text))
        overlap = tokens & _EVIDENCE_KEYWORDS
        score = min(len(overlap) / 3.0, 1.0)
        specificity = min(len(tokens) / 50.0, 0.5)
        return min(score + specificity, 1.0)

    def score(self, goal: ResearchGoal, candidate: CandidateLog) -> RankedLog:
        log_text = candidate.log.full_text
        sr = self._semantic_relevance(goal, log_text)
        gf = self._goal_focus(goal, log_text)
        ev = self._evidence_value(log_text)
        w = self.config
        final = w.semantic_weight * sr + w.goal_focus_weight * gf + w.evidence_value_weight * ev
        return RankedLog(
            log=candidate.log,
            semantic_relevance=round(sr, 4),
            goal_focus=round(gf, 4),
            evidence_value=round(ev, 4),
            final_score=round(final, 4),
        )

    def rank(self, goal: ResearchGoal, candidates: list[CandidateLog]) -> list[RankedLog]:
        ranked = [self.score(goal, c) for c in candidates]
        ranked.sort(key=lambda x: x.final_score, reverse=True)
        for i, r in enumerate(ranked):
            r.rank = i + 1
        logger.debug("Reranked %d candidates for goal=%s", len(ranked), goal.goal_id)
        return ranked
