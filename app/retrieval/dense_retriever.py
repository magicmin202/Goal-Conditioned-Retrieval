"""Dense embedding-based retriever.

Uses EmbeddingProvider for swappable backends.
Default: MockEmbeddingProvider (hash-based, no deps).
Real: SentenceTransformerProvider (multilingual, install sentence-transformers).

In the dual-space retrieval architecture:
  Stage 1 (candidate): dense similarity = 0.45 weight (recall-focused)
  Stage 2 (reranker):  dense similarity = 0.05 weight (tie-breaker only)
"""
from __future__ import annotations

import logging
import math

from app.retrieval.embedding_provider import (
    EmbeddingProvider,
    MockEmbeddingProvider,
)
from app.schemas import CandidateLog, ResearchLog

logger = logging.getLogger(__name__)


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x ** 2 for x in a)) or 1.0
    nb = math.sqrt(sum(x ** 2 for x in b)) or 1.0
    return dot / (na * nb)


class DenseRetriever:
    """Embedding-based retriever backed by an EmbeddingProvider."""

    def __init__(self, provider: EmbeddingProvider | None = None) -> None:
        self._provider: EmbeddingProvider = provider or MockEmbeddingProvider()
        self._corpus: list[ResearchLog] = []
        self._embeddings: list[list[float]] = []

    # ── Indexing ──────────────────────────────────────────────────────────────

    def index(self, logs: list[ResearchLog]) -> None:
        self._corpus = logs
        self._embeddings = self._provider.encode_batch(
            [log.full_text for log in logs]
        )
        logger.debug(
            "DenseRetriever indexed %d docs [provider=%s]",
            len(logs), self._provider.name,
        )

    # ── Embedding helpers ─────────────────────────────────────────────────────

    def embed(self, text: str) -> list[float]:
        """Encode a single text (used by reranker for semantic similarity)."""
        return self._provider.encode(text)

    def score_all(self, query: str) -> list[tuple[ResearchLog, float]]:
        """Return (log, normalized_cosine_score) for every indexed document.

        Normalization: divide by max score so the best match = 1.0.
        Used by HybridRetriever for score-based combination.
        """
        if not self._corpus:
            return []
        q_emb = self._provider.encode(query)
        raw = [cosine(q_emb, e) for e in self._embeddings]
        max_s = max(raw) if raw else 1.0
        if max_s <= 0:
            max_s = 1.0
        return [(log, r / max_s) for log, r in zip(self._corpus, raw)]

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_n: int = 30) -> list[CandidateLog]:
        """Return top-N candidates by dense similarity (normalized to [0,1])."""
        pairs = self.score_all(query)
        if not pairs:
            return []
        ranked = sorted(pairs, key=lambda x: x[1], reverse=True)[:top_n]
        return [
            CandidateLog(log=log, dense_score=round(score, 6))
            for log, score in ranked
        ]
