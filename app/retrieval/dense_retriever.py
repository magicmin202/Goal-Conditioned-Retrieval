"""Dense embedding-based retriever.

Default: GeminiEmbeddingProvider (gemini-embedding-001) when GEMINI_API_KEY is set.
Fallback: MockEmbeddingProvider (hash-based, no semantic meaning).

doc / query task_type 분리:
  - indexing logs   → RETRIEVAL_DOCUMENT
  - encoding query  → RETRIEVAL_QUERY
"""
from __future__ import annotations

import logging
import math
import os

from app.retrieval.embedding_provider import (
    EmbeddingProvider,
    GeminiEmbeddingProvider,
    MockEmbeddingProvider,
)
from app.schemas import CandidateLog, ResearchLog

logger = logging.getLogger(__name__)

_GEMINI_MODEL = "models/gemini-embedding-001"
_CACHE_DIR = ".cache/embeddings"


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x ** 2 for x in a)) or 1.0
    nb = math.sqrt(sum(x ** 2 for x in b)) or 1.0
    return dot / (na * nb)


def _build_gemini_providers(
    api_key: str,
) -> tuple[EmbeddingProvider, EmbeddingProvider]:
    """doc provider (RETRIEVAL_DOCUMENT) + query provider (RETRIEVAL_QUERY)."""
    doc = GeminiEmbeddingProvider(
        api_key=api_key,
        model=_GEMINI_MODEL,
        task_type="RETRIEVAL_DOCUMENT",
        cache_dir=_CACHE_DIR,
    )
    qry = GeminiEmbeddingProvider(
        api_key=api_key,
        model=_GEMINI_MODEL,
        task_type="RETRIEVAL_QUERY",
        cache_dir=_CACHE_DIR,
    )
    logger.info(
        "DenseRetriever: gemini-embedding-001 "
        "[doc_cache=%d  query_cache=%d]",
        len(doc._cache), len(qry._cache),
    )
    return doc, qry


class DenseRetriever:
    """Embedding-based retriever backed by EmbeddingProvider.

    Uses gemini-embedding-001 by default (RETRIEVAL_DOCUMENT for indexing,
    RETRIEVAL_QUERY for query encoding). Falls back to mock when no API key.

    In the dual-space retrieval architecture:
      Stage 1 (candidate): dense similarity = 0.45 weight (recall-focused)
      Stage 2 (reranker):  dense similarity = 0.05 weight (tie-breaker only)
    """

    def __init__(
        self,
        doc_provider: EmbeddingProvider | None = None,
        query_provider: EmbeddingProvider | None = None,
    ) -> None:
        # Auto-detect providers unless explicitly passed
        if doc_provider is None and query_provider is None:
            api_key = (
                os.environ.get("GEMINI_API_KEY")
                or os.environ.get("GOOGLE_API_KEY")
            )
            if api_key:
                try:
                    doc_provider, query_provider = _build_gemini_providers(api_key)
                except Exception as exc:
                    logger.warning(
                        "GeminiEmbeddingProvider init failed (%s) → mock fallback", exc
                    )

        if doc_provider is None:
            logger.warning(
                "DenseRetriever: MockEmbeddingProvider "
                "(no GEMINI_API_KEY — dense has NO semantic meaning)"
            )
            doc_provider = MockEmbeddingProvider()

        self._doc_provider: EmbeddingProvider = doc_provider
        self._query_provider: EmbeddingProvider = query_provider or doc_provider

        self._corpus: list[ResearchLog] = []
        self._embeddings: list[list[float]] = []

    @property
    def is_real(self) -> bool:
        return self._doc_provider.name != "mock"

    # ── Indexing ──────────────────────────────────────────────────────────────

    def index(self, logs: list[ResearchLog]) -> None:
        self._corpus = logs
        self._embeddings = self._doc_provider.encode_batch(
            [log.full_text for log in logs]
        )
        logger.info(
            "DenseRetriever indexed %d docs [provider=%s]",
            len(logs), self._doc_provider.name,
        )

    # ── Embedding helpers ─────────────────────────────────────────────────────

    def embed(self, text: str) -> list[float]:
        """Encode text with doc provider (used by reranker for semantic score)."""
        return self._doc_provider.encode(text)

    def embed_query(self, text: str) -> list[float]:
        """Encode query/goal with query-optimised provider."""
        return self._query_provider.encode(text)

    def score_all(self, query: str) -> list[tuple[ResearchLog, float]]:
        """Return (log, normalized_cosine) for every indexed doc."""
        if not self._corpus:
            return []
        q_emb = self._query_provider.encode(query)
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
