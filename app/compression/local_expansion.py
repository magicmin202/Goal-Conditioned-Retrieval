"""Anchor-based Local Expansion.

For each top-k anchor log, finds semantically similar neighbor logs
from the full candidate pool.
"""
from __future__ import annotations
import logging
from app.retrieval.dense_retriever import DenseRetriever, cosine
from app.schemas import RankedLog, ResearchLog

logger = logging.getLogger(__name__)


class LocalExpander:
    """Expand anchor logs with semantically similar neighborhood logs."""

    def __init__(
        self,
        similarity_threshold: float = 0.7,
        dense_retriever: DenseRetriever | None = None,
    ) -> None:
        self.threshold = similarity_threshold
        self._dense = dense_retriever or DenseRetriever()

    def expand(
        self,
        anchor_logs: list[RankedLog],
        full_log_pool: list[ResearchLog],
        max_neighbors: int = 3,
    ) -> dict[str, list[ResearchLog]]:
        """Return anchor_log_id → neighbor logs mapping."""
        anchor_ids = {r.log_id for r in anchor_logs}
        anchor_embeddings = {r.log_id: self._dense.embed(r.log.full_text) for r in anchor_logs}
        pool_embeddings = {
            log.log_id: self._dense.embed(log.full_text)
            for log in full_log_pool
            if log.log_id not in anchor_ids
        }

        result: dict[str, list[ResearchLog]] = {}
        for anchor in anchor_logs:
            a_emb = anchor_embeddings[anchor.log_id]
            neighbors: list[tuple[float, ResearchLog]] = []
            for log in full_log_pool:
                if log.log_id in anchor_ids:
                    continue
                sim = cosine(a_emb, pool_embeddings[log.log_id])
                activity_bonus = 0.05 if log.activity_type == anchor.log.activity_type else 0.0
                if sim + activity_bonus >= self.threshold:
                    neighbors.append((sim + activity_bonus, log))
            neighbors.sort(key=lambda x: x[0], reverse=True)
            result[anchor.log_id] = [log for _, log in neighbors[:max_neighbors]]

        logger.debug("LocalExpander: %d anchors expanded", len(anchor_logs))
        return result
