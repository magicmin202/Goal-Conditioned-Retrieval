"""Hybrid retriever: sparse + dense via Reciprocal Rank Fusion (RRF)."""
from __future__ import annotations
import logging
from app.config import RetrievalConfig
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.sparse_retriever import SparseRetriever
from app.schemas import CandidateLog, ResearchLog

logger = logging.getLogger(__name__)


def _rrf_score(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank + 1)


def _reciprocal_rank_fusion(
    sparse_results: list[CandidateLog],
    dense_results: list[CandidateLog],
    k: int = 60,
) -> list[CandidateLog]:
    scores: dict[str, float] = {}
    sparse_map: dict[str, CandidateLog] = {}
    dense_map: dict[str, CandidateLog] = {}

    for rank, c in enumerate(sparse_results):
        scores[c.log_id] = scores.get(c.log_id, 0.0) + _rrf_score(rank, k)
        sparse_map[c.log_id] = c
    for rank, c in enumerate(dense_results):
        scores[c.log_id] = scores.get(c.log_id, 0.0) + _rrf_score(rank, k)
        dense_map[c.log_id] = c

    merged = []
    for log_id, hybrid_score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        base = sparse_map.get(log_id) or dense_map[log_id]
        empty = CandidateLog(log=base.log)
        merged.append(CandidateLog(
            log=base.log,
            sparse_score=sparse_map.get(log_id, empty).sparse_score,
            dense_score=dense_map.get(log_id, empty).dense_score,
            hybrid_score=round(hybrid_score, 6),
        ))
    return merged


class HybridRetriever:
    """Combines SparseRetriever and DenseRetriever via RRF."""

    def __init__(self, config: RetrievalConfig | None = None) -> None:
        self.config = config or RetrievalConfig()
        self.sparse = SparseRetriever()
        self.dense = DenseRetriever()

    def index(self, logs: list[ResearchLog]) -> None:
        self.sparse.index(logs)
        self.dense.index(logs)
        logger.debug("HybridRetriever indexed %d docs", len(logs))

    def retrieve(self, query: str, top_n: int | None = None) -> list[CandidateLog]:
        n = top_n or self.config.candidate_size
        sparse_res = self.sparse.retrieve(query, top_n=n * 2)
        dense_res = self.dense.retrieve(query, top_n=n * 2)
        fused = _reciprocal_rank_fusion(sparse_res, dense_res, k=self.config.rrf_k)
        return fused[:n]
