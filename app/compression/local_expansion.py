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


def _goal_relevance_score(log_text: str, expanded_terms: list[str]) -> float:
    """Simple goal relevance: fraction of expanded term tokens found in log."""
    if not expanded_terms:
        return 1.0  # no gate possible → pass through
    log_tokens = set(re.findall(r"[\w가-힣]+", log_text.lower()))
    exp_tokens: set[str] = set()
    for term in expanded_terms:
        exp_tokens.update(re.findall(r"[\w가-힣]+", term.lower()))
    if not exp_tokens:
        return 1.0
    return len(exp_tokens & log_tokens) / len(exp_tokens)


class LocalExpander:
    """Expand each anchor log into a cluster of related logs.

    anchor_relevance_threshold: anchors below this final_score are skipped.
    neighbor_goal_relevance_min: neighbors must have ≥ this goal relevance score.
    """

    def __init__(
        self,
        min_criteria: int = 2,
        semantic_threshold: float = 0.45,
        anchor_relevance_threshold: float = 0.05,
        neighbor_goal_relevance_min: float = 0.0,
        dense_retriever: DenseRetriever | None = None,
    ) -> None:
        self.min_criteria = min_criteria
        self.semantic_threshold = semantic_threshold
        self.anchor_relevance_threshold = anchor_relevance_threshold
        self.neighbor_goal_relevance_min = neighbor_goal_relevance_min
        self._dense = dense_retriever or DenseRetriever()

    def expand(
        self,
        anchor_logs: list[RankedLog],
        full_log_pool: list[ResearchLog],
        max_neighbors: int = 5,
        expanded_terms: list[str] | None = None,
    ) -> dict[str, list[ResearchLog]]:
        """Return anchor_log_id → list of neighbor logs.

        Anchors below anchor_relevance_threshold are skipped entirely.
        Neighbors must meet:
          - min_criteria cluster criteria
          - goal relevance ≥ neighbor_goal_relevance_min (when expanded_terms provided)
        Already-selected anchors are excluded from each other's pools.
        """
        exp_terms = expanded_terms or []

        # Filter anchors by relevance threshold
        valid_anchors = [
            a for a in anchor_logs if a.final_score >= self.anchor_relevance_threshold
        ]
        skipped = len(anchor_logs) - len(valid_anchors)
        if skipped:
            logger.debug(
                "LocalExpander: skipped %d anchors below threshold %.3f",
                skipped, self.anchor_relevance_threshold,
            )

        anchor_ids = {r.log_id for r in anchor_logs}  # exclude all anchors from pool

        # Pre-compute embeddings once
        anchor_embeddings = {
            r.log_id: self._dense.embed(r.log.full_text) for r in valid_anchors
        }
        pool_logs = [log for log in full_log_pool if log.log_id not in anchor_ids]
        pool_embeddings = {
            log.log_id: self._dense.embed(log.full_text) for log in pool_logs
        }

        result: dict[str, list[ResearchLog]] = {}

        for anchor in valid_anchors:
            a_emb = anchor_embeddings[anchor.log_id]
            scored: list[tuple[int, float, ResearchLog]] = []

            for log in pool_logs:
                l_emb = pool_embeddings[log.log_id]
                n_criteria = _criteria_met(
                    anchor, log, a_emb, l_emb, self.semantic_threshold
                )
                if n_criteria < self.min_criteria:
                    continue

                # Goal relevance gate for neighbors
                if self.neighbor_goal_relevance_min > 0.0 and exp_terms:
                    rel = _goal_relevance_score(log.full_text, exp_terms)
                    if rel < self.neighbor_goal_relevance_min:
                        logger.debug(
                            "Neighbor gate: skip %s (rel=%.3f < %.3f)",
                            log.log_id, rel, self.neighbor_goal_relevance_min,
                        )
                        continue

                sim = cosine(a_emb, l_emb)
                scored.append((n_criteria, sim, log))

            # Sort: more criteria first, then by similarity
            scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
            neighbors = [log for _, _, log in scored[:max_neighbors]]
            result[anchor.log_id] = neighbors

            logger.debug(
                "Anchor %s (%s) → %d neighbors  topic=%s  score=%.4f",
                anchor.log_id, anchor.log.title,
                len(neighbors),
                anchor.log.metadata.get("topic", "-"),
                anchor.final_score,
            )

        # Fill empty entries for skipped anchors (so pipeline doesn't break)
        for anchor in anchor_logs:
            if anchor.log_id not in result:
                result[anchor.log_id] = []

        total_clustered = sum(len(v) for v in result.values())
        logger.info(
            "LocalExpander: %d/%d anchors active, %d total neighbor logs clustered",
            len(valid_anchors), len(anchor_logs), total_clustered,
        )
        return result
