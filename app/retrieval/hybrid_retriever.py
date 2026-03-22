"""Hybrid retriever: BM25 (sparse) + Dense embedding — dual-space candidate scoring.

Architecture (Stage 1 = recall-focused):

  candidate_score = bm25_weight * bm25_score
                  + dense_weight * dense_score

Both scores are normalized to [0, 1] before combination.
The vocabulary boost (0.15 weight) is applied on top in CandidateRetriever.

Default weights: BM25=0.40, Dense=0.45  (sum=0.85, leaves room for vocab=0.15)

Rationale:
  - BM25 is precise for exact keyword matches
  - Dense similarity recalls paraphrase / semantically related logs
  - Together they cover both lexical and semantic recall
"""
from __future__ import annotations

import logging

from app.config import RetrievalConfig
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.embedding_provider import EmbeddingProvider
from app.retrieval.sparse_retriever import SparseRetriever
from app.schemas import CandidateLog, ResearchLog

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Dual-space candidate retriever: BM25 × weight + Dense × weight."""

    def __init__(
        self,
        config: RetrievalConfig | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self.config = config or RetrievalConfig()
        self.sparse = SparseRetriever()
        self.dense = DenseRetriever(provider=embedding_provider)

    def index(self, logs: list[ResearchLog]) -> None:
        self.sparse.index(logs)
        self.dense.index(logs)
        logger.debug(
            "HybridRetriever indexed %d docs [bm25=%.2f dense=%.2f]",
            len(logs), self.config.sparse_weight, self.config.dense_weight,
        )

    def retrieve(
        self,
        query: str,
        top_n: int | None = None,
        dense_query: str | None = None,
    ) -> list[CandidateLog]:
        """Score-based dual-space combination (not rank-based RRF).

        Each log gets:
          hybrid_score = sparse_weight * bm25_score + dense_weight * dense_score

        sparse (BM25) uses `query`; dense uses `dense_query` if provided.
        Keeping them separate avoids semantic drift caused by injecting full
        lexical expansion into the dense embedding query.
        """
        n = top_n or self.config.candidate_size
        cfg = self.config

        # Retrieve from both spaces (get full corpus coverage)
        fetch_n = max(n * 4, 50)
        sparse_res = self.sparse.retrieve(query, top_n=fetch_n)
        dense_q = dense_query if dense_query is not None else query
        if dense_query and dense_query != query:
            logger.debug(
                "HybridRetriever: bm25_q=%s  dense_q=%s",
                query[:50], dense_q[:50],
            )
        dense_res = self.dense.retrieve(dense_q, top_n=fetch_n)

        # Build per-log score maps
        sparse_map: dict[str, float] = {c.log_id: c.sparse_score for c in sparse_res}
        dense_map: dict[str, float] = {c.log_id: c.dense_score for c in dense_res}
        log_map: dict[str, CandidateLog] = {}
        for c in sparse_res:
            log_map[c.log_id] = c
        for c in dense_res:
            if c.log_id not in log_map:
                log_map[c.log_id] = c

        # Combine
        results: list[CandidateLog] = []
        for lid in set(sparse_map) | set(dense_map):
            s = sparse_map.get(lid, 0.0)
            d = dense_map.get(lid, 0.0)
            hybrid = round(cfg.sparse_weight * s + cfg.dense_weight * d, 6)
            base = log_map[lid]
            results.append(CandidateLog(
                log=base.log,
                sparse_score=s,
                dense_score=d,
                hybrid_score=hybrid,
            ))

        results.sort(key=lambda x: x.hybrid_score, reverse=True)

        if results:
            top = results[0]
            logger.debug(
                "HybridRetriever top-1: bm25=%.3f dense=%.3f hybrid=%.4f [%s]",
                top.sparse_score, top.dense_score, top.hybrid_score, top.log.title,
            )

        return results[:n]
