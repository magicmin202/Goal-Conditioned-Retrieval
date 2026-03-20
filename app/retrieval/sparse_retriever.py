"""BM25-based sparse retriever.

Falls back to simple TF scoring if rank_bm25 is not installed.
"""
from __future__ import annotations
import logging
import re
from collections import Counter
from app.schemas import CandidateLog, ResearchLog

logger = logging.getLogger(__name__)

try:
    from rank_bm25 import BM25Okapi  # type: ignore
    _HAS_BM25 = True
except ImportError:
    _HAS_BM25 = False
    logger.warning("rank_bm25 not installed; using simple TF scoring.")


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[\w가-힣]+", text.lower())


def _tf_score(query_tokens: list[str], doc_tokens: list[str]) -> float:
    if not doc_tokens:
        return 0.0
    counter = Counter(doc_tokens)
    total = len(doc_tokens)
    return sum(counter.get(t, 0) / total for t in query_tokens)


class SparseRetriever:
    """BM25 retriever over a corpus of ResearchLog objects."""

    def __init__(self) -> None:
        self._corpus: list[ResearchLog] = []
        self._tokenized: list[list[str]] = []
        self._bm25 = None

    def index(self, logs: list[ResearchLog]) -> None:
        self._corpus = logs
        self._tokenized = [_tokenize(log.full_text) for log in logs]
        if _HAS_BM25:
            self._bm25 = BM25Okapi(self._tokenized)
        logger.debug("SparseRetriever indexed %d docs", len(logs))

    def retrieve(self, query: str, top_n: int = 30) -> list[CandidateLog]:
        if not self._corpus:
            return []
        query_tokens = _tokenize(query)
        if _HAS_BM25 and self._bm25 is not None:
            scores = list(self._bm25.get_scores(query_tokens))
        else:
            scores = [_tf_score(query_tokens, doc) for doc in self._tokenized]

        max_s = max(scores) if scores else 1.0
        if max_s == 0:
            max_s = 1.0

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_n]
        return [
            CandidateLog(log=self._corpus[i], sparse_score=round(s / max_s, 6))
            for i, s in ranked
        ]
