"""BM25-based sparse retriever.

Uses a pure-Python BM25Okapi implementation (no numpy dependency).
"""
from __future__ import annotations
import logging
import math
import re
from collections import Counter
from app.schemas import CandidateLog, ResearchLog

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[\w가-힣]+", text.lower())


class _BM25Okapi:
    """Pure-Python BM25Okapi — no numpy required."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.avgdl = sum(len(d) for d in corpus) / max(self.corpus_size, 1)
        self.doc_freqs: list[Counter] = [Counter(doc) for doc in corpus]
        self.doc_lens: list[int] = [len(doc) for doc in corpus]

        # document frequency per term
        df: Counter = Counter()
        for freq in self.doc_freqs:
            for term in freq:
                df[term] += 1
        self.idf: dict[str, float] = {
            term: math.log((self.corpus_size - n + 0.5) / (n + 0.5) + 1)
            for term, n in df.items()
        }

    def get_scores(self, query: list[str]) -> list[float]:
        scores = []
        for i, freq in enumerate(self.doc_freqs):
            dl = self.doc_lens[i]
            score = 0.0
            for term in query:
                if term not in freq:
                    continue
                tf = freq[term]
                idf = self.idf.get(term, 0.0)
                score += idf * (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                )
            scores.append(score)
        return scores


class SparseRetriever:
    """BM25 retriever over a corpus of ResearchLog objects."""

    def __init__(self) -> None:
        self._corpus: list[ResearchLog] = []
        self._bm25: _BM25Okapi | None = None

    def index(self, logs: list[ResearchLog]) -> None:
        self._corpus = logs
        tokenized = [_tokenize(log.full_text) for log in logs]
        self._bm25 = _BM25Okapi(tokenized)
        logger.debug("SparseRetriever indexed %d docs (BM25Okapi)", len(logs))

    def retrieve(self, query: str, top_n: int = 30) -> list[CandidateLog]:
        if not self._corpus or self._bm25 is None:
            return []
        query_tokens = _tokenize(query)
        scores = self._bm25.get_scores(query_tokens)

        max_s = max(scores) if scores else 1.0
        if max_s == 0:
            max_s = 1.0

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_n]
        return [
            CandidateLog(log=self._corpus[i], sparse_score=round(s / max_s, 6))
            for i, s in ranked
        ]
