"""Dense embedding-based retriever.

Uses mock cosine similarity (hash-based embeddings) by default.
TODO: Replace _mock_embed with sentence-transformers or Gemini Embeddings.
"""
from __future__ import annotations
import logging
import math
import random
from typing import Callable
from app.schemas import CandidateLog, ResearchLog

logger = logging.getLogger(__name__)
EmbedFn = Callable[[str], list[float]]
EMBEDDING_DIM = 128


def _mock_embed(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    """Deterministic mock embedding based on text hash.

    TODO: Replace with:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        return model.encode(text).tolist()
    """
    rng = random.Random(hash(text) % (2**32))
    vec = [rng.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(v**2 for v in vec)) or 1.0
    return [v / norm for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x**2 for x in a)) or 1.0
    nb = math.sqrt(sum(x**2 for x in b)) or 1.0
    return dot / (na * nb)


class DenseRetriever:
    """Embedding-based retriever."""

    def __init__(self, embed_fn: EmbedFn | None = None) -> None:
        self._embed_fn: EmbedFn = embed_fn or _mock_embed
        self._corpus: list[ResearchLog] = []
        self._embeddings: list[list[float]] = []

    def index(self, logs: list[ResearchLog]) -> None:
        self._corpus = logs
        self._embeddings = [self._embed_fn(log.full_text) for log in logs]
        logger.debug("DenseRetriever indexed %d docs", len(logs))

    def embed(self, text: str) -> list[float]:
        return self._embed_fn(text)

    def retrieve(self, query: str, top_n: int = 30) -> list[CandidateLog]:
        if not self._corpus:
            return []
        q_emb = self._embed_fn(query)
        scores = [cosine(q_emb, emb) for emb in self._embeddings]
        max_s = max(scores) if scores else 1.0
        if max_s <= 0:
            max_s = 1.0
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_n]
        return [
            CandidateLog(log=self._corpus[i], dense_score=round(s / max_s, 6))
            for i, s in ranked
        ]
