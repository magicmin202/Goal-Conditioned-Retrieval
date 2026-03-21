"""Anchor-based Local Expansion with multi-criteria clustering.

Cluster criteria (any 2+ triggers inclusion):
  1. activity_type match
  2. metadata.topic match   ← most reliable for synthetic data
  3. title keyword overlap  (≥1 shared token)
  4. semantic similarity    (embedding cosine ≥ 0.45)

This avoids relying solely on mock embeddings, which are hash-based
and not semantically meaningful.
"""
from __future__ import annotations

import logging
import re

from app.retrieval.dense_retriever import DenseRetriever, cosine
from app.schemas import RankedLog, ResearchLog

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[\w가-힣]{2,}", text.lower()))


def _criteria_met(
    anchor: RankedLog,
    log: ResearchLog,
    anchor_emb: list[float],
    log_emb: list[float],
    semantic_threshold: float,
) -> int:
    """Return number of cluster criteria satisfied (0-4)."""
    count = 0

    # 1. Activity type match
    if log.activity_type == anchor.log.activity_type:
        count += 1

    # 2. Topic metadata match (exact)
    a_topic = anchor.log.metadata.get("topic", "")
    l_topic = log.metadata.get("topic", "")
    if a_topic and l_topic and a_topic == l_topic:
        count += 1

    # 3. Title keyword overlap
    a_tokens = _tokenize(anchor.log.title)
    l_tokens = _tokenize(log.title)
    if a_tokens and l_tokens and a_tokens & l_tokens:
        count += 1

    # 4. Semantic similarity
    if cosine(anchor_emb, log_emb) >= semantic_threshold:
        count += 1

    return count


class LocalExpander:
    """Expand each anchor log into a cluster of related logs."""

    def __init__(
        self,
        min_criteria: int = 2,
        semantic_threshold: float = 0.45,
        dense_retriever: DenseRetriever | None = None,
    ) -> None:
        self.min_criteria = min_criteria
        self.semantic_threshold = semantic_threshold
        self._dense = dense_retriever or DenseRetriever()

    def expand(
        self,
        anchor_logs: list[RankedLog],
        full_log_pool: list[ResearchLog],
        max_neighbors: int = 5,
    ) -> dict[str, list[ResearchLog]]:
        """Return anchor_log_id → list of neighbor logs.

        Neighbors satisfy at least min_criteria clustering criteria.
        Already-selected anchors are excluded from each other's pools.
        """
        anchor_ids = {r.log_id for r in anchor_logs}

        # Pre-compute embeddings once
        anchor_embeddings = {
            r.log_id: self._dense.embed(r.log.full_text) for r in anchor_logs
        }
        pool_logs = [log for log in full_log_pool if log.log_id not in anchor_ids]
        pool_embeddings = {
            log.log_id: self._dense.embed(log.full_text) for log in pool_logs
        }

        result: dict[str, list[ResearchLog]] = {}

        for anchor in anchor_logs:
            a_emb = anchor_embeddings[anchor.log_id]
            scored: list[tuple[int, float, ResearchLog]] = []

            for log in pool_logs:
                l_emb = pool_embeddings[log.log_id]
                n_criteria = _criteria_met(
                    anchor, log, a_emb, l_emb, self.semantic_threshold
                )
                if n_criteria >= self.min_criteria:
                    sim = cosine(a_emb, l_emb)
                    scored.append((n_criteria, sim, log))

            # Sort: more criteria first, then by similarity
            scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
            neighbors = [log for _, _, log in scored[:max_neighbors]]
            result[anchor.log_id] = neighbors

            logger.debug(
                "Anchor %s (%s) → %d neighbors  topic=%s",
                anchor.log_id, anchor.log.title,
                len(neighbors),
                anchor.log.metadata.get("topic", "-"),
            )

        total_clustered = sum(len(v) for v in result.values())
        logger.info(
            "LocalExpander: %d anchors, %d total neighbor logs clustered",
            len(anchor_logs), total_clustered,
        )
        return result
