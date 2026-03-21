"""Embedding provider abstraction.

Allows swapping embedding backends without changing retrieval code.

Available providers:
  MockEmbeddingProvider           – deterministic hash-based (default, zero deps)
  GeminiEmbeddingProvider         – Google Gemini text-embedding-004 (REST, no grpc)
  SentenceTransformerProvider     – local multilingual model (requires sentence-transformers)

Priority (get_embedding_provider):
  1. GeminiEmbeddingProvider   if GEMINI_API_KEY set and real=True
  2. SentenceTransformerProvider if installed and real=True
  3. MockEmbeddingProvider     (always available fallback)

Usage:
    provider = get_embedding_provider()           # mock (safe default)
    provider = get_embedding_provider(real=True)  # Gemini or ST if available
    provider = GeminiEmbeddingProvider(api_key)   # explicit Gemini
"""
from __future__ import annotations

import logging
import math
import os
import random
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 128
_ST_DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
_GEMINI_EMBED_MODEL = "models/gemini-embedding-001"  # 3072-dim, multilingual


# ── Abstract base ─────────────────────────────────────────────────────────────

class EmbeddingProvider(ABC):
    """Common interface for all embedding backends."""

    @abstractmethod
    def encode(self, text: str) -> list[float]:
        """Return a normalized embedding vector for text."""
        ...

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """Encode multiple texts. Default implementation: iterate."""
        return [self.encode(t) for t in texts]

    @property
    @abstractmethod
    def dim(self) -> int:
        """Embedding dimension."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name for logging."""
        ...


# ── Mock provider (always available) ─────────────────────────────────────────

class MockEmbeddingProvider(EmbeddingProvider):
    """Deterministic hash-based mock embeddings. No external dependencies.

    Produces meaningful token-overlap-like similarity for short texts with
    shared vocabulary, but does NOT capture true semantic similarity.
    Use as a stand-in until a real model is installed.
    """

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self._dim = dim

    def encode(self, text: str) -> list[float]:
        rng = random.Random(hash(text) % (2 ** 32))
        vec = [rng.gauss(0, 1) for _ in range(self._dim)]
        norm = math.sqrt(sum(v ** 2 for v in vec)) or 1.0
        return [v / norm for v in vec]

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return "mock"


# ── SentenceTransformer provider (optional) ───────────────────────────────────

class SentenceTransformerProvider(EmbeddingProvider):
    """sentence-transformers based multilingual embeddings.

    Install: pip install sentence-transformers
    Default model: paraphrase-multilingual-MiniLM-L12-v2
      – supports Korean, ~118M params, fast on CPU, dim=384
    """

    def __init__(self, model_name: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer  # type: ignore
        self._model_name = model_name or _ST_DEFAULT_MODEL
        self._model = SentenceTransformer(self._model_name)
        self._dim_size: int = self._model.get_sentence_embedding_dimension()
        logger.info(
            "SentenceTransformerProvider loaded: %s  dim=%d",
            self._model_name, self._dim_size,
        )

    def encode(self, text: str) -> list[float]:
        return self._model.encode(text, normalize_embeddings=True).tolist()

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()

    @property
    def dim(self) -> int:
        return self._dim_size

    @property
    def name(self) -> str:
        return f"sentence-transformers/{self._model_name}"


# ── Gemini Embedding provider ─────────────────────────────────────────────────

class GeminiEmbeddingProvider(EmbeddingProvider):
    """Google Gemini text-embedding-004 via google-genai SDK (REST, no gRPC).

    Model: models/text-embedding-004
      - 768-dim output, normalized
      - Multilingual (Korean supported)
      - Task type: RETRIEVAL_DOCUMENT for logs, RETRIEVAL_QUERY for goal

    Includes an in-memory LRU-style cache to avoid redundant API calls
    when the same text (e.g., a goal query) is embedded multiple times.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = _GEMINI_EMBED_MODEL,
        task_type: str = "RETRIEVAL_DOCUMENT",
        cache_size: int = 512,
    ) -> None:
        from google import genai  # type: ignore

        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY not set; cannot use GeminiEmbeddingProvider")

        self._client = genai.Client(api_key=key)
        self._model = model
        self._task_type = task_type
        self._cache: dict[str, list[float]] = {}
        self._cache_size = cache_size
        self._dim_size: int | None = None
        logger.info("GeminiEmbeddingProvider initialized [model=%s]", model)

    def _api_encode(self, text: str) -> list[float]:
        from google.genai import types  # type: ignore

        result = self._client.models.embed_content(
            model=self._model,
            contents=text,
            config=types.EmbedContentConfig(task_type=self._task_type),
        )
        return list(result.embeddings[0].values)

    def encode(self, text: str) -> list[float]:
        if text not in self._cache:
            if len(self._cache) >= self._cache_size:
                # Evict oldest entry
                self._cache.pop(next(iter(self._cache)))
            vec = self._api_encode(text)
            self._cache[text] = vec
            if self._dim_size is None:
                self._dim_size = len(vec)
        return self._cache[text]

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """Encode batch, using cache to skip already-embedded texts."""
        results = []
        for text in texts:
            results.append(self.encode(text))
        return results

    @property
    def dim(self) -> int:
        return self._dim_size or 768

    @property
    def name(self) -> str:
        return f"gemini/{self._model}"


# ── Factory ───────────────────────────────────────────────────────────────────

def get_embedding_provider(real: bool = False) -> EmbeddingProvider:
    """Return the best available embedding provider.

    When real=True, priority order:
      1. GeminiEmbeddingProvider  (if GEMINI_API_KEY is set — no extra install needed)
      2. SentenceTransformerProvider (if sentence-transformers installed)
      3. MockEmbeddingProvider (fallback)
    """
    if real:
        # Try Gemini first (already installed via google-genai)
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if api_key:
            try:
                provider = GeminiEmbeddingProvider(api_key=api_key)
                logger.info("EmbeddingProvider: %s", provider.name)
                return provider
            except Exception as exc:
                logger.warning("GeminiEmbeddingProvider failed (%s) → trying ST", exc)

        # Try SentenceTransformers
        try:
            provider = SentenceTransformerProvider()
            logger.info("EmbeddingProvider: %s", provider.name)
            return provider
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Options: (1) set GEMINI_API_KEY, or (2) pip install sentence-transformers"
            )
        except Exception as exc:
            logger.warning("SentenceTransformerProvider failed (%s)", exc)

    logger.info("EmbeddingProvider: mock (hash-based, no semantic similarity)")
    return MockEmbeddingProvider()