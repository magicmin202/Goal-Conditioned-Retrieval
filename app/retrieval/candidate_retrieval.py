"""Candidate Retrieval — Dense-only recall-focused retrieval.

Architecture (Stage 1):
  candidate_score = dense_similarity (Gemini embedding-001 cosine similarity)

Dense retrieval alone achieves candidate_recall=1.00 with Gemini embedding-001,
making BM25 and vocabulary boost redundant. Precision control is handled entirely
by the reranker (Stage 2).
"""
from __future__ import annotations

import logging

from app.config import RetrievalConfig
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.embedding_provider import EmbeddingProvider
from app.retrieval.query_expansion import ExpandedQuery
from app.retrieval.query_understanding import QueryObject
from app.schemas import CandidateLog, ResearchLog

logger = logging.getLogger(__name__)


def _dynamic_candidate_size(corpus_size: int) -> int:
    """Corpus 크기에 비례한 candidate_size 동적 계산.

    corpus_size   candidate_size
    ────────────────────────────
    ≤  60         30  (최소 보장)
       70         35
      100         50
      200        100  (상한 적용)
     1000        100  (상한 적용)
    """
    return min(max(30, int(corpus_size * 0.50)), 100)


class CandidateRetriever:
    """Dense-only recall-focused candidate retriever."""

    def __init__(
        self,
        config: RetrievalConfig | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        **kwargs,  # absorb removed params (mode, candidate_config, vocab_boost_config)
    ) -> None:
        self.config = config or RetrievalConfig()
        self._dense = DenseRetriever(doc_provider=embedding_provider)
        self._indexed = False

    def index(self, logs: list[ResearchLog]) -> None:
        self._dense.index(logs)
        self._corpus_size = len(logs)
        self._indexed = True
        logger.info(
            "CandidateRetriever indexed %d logs [dense-only]",
            len(logs),
        )

    def retrieve(
        self,
        query: QueryObject | ExpandedQuery,
        top_n: int | None = None,
    ) -> list[CandidateLog]:
        if not self._indexed:
            raise RuntimeError("Call index() before retrieve().")

        if top_n is not None:
            n = top_n
        else:
            corpus_size = getattr(self, "_corpus_size", None) or self.config.candidate_size
            n = _dynamic_candidate_size(corpus_size)

        if isinstance(query, ExpandedQuery):
            dense_text = query.dense_query
        else:
            dense_text = query.canonical_text

        logger.debug("CandidateRetriever  dense_q=%s  top_n=%d", dense_text[:80], n)

        candidates = self._dense.retrieve(dense_text, top_n=n)

        logger.debug("Stage1 candidates: %d  dense_q=%s", len(candidates), dense_text[:60])

        return candidates
