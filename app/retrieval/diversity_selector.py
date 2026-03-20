"""Diversity-aware Top-K selection via MMR.

Pipeline order (enforced here):
  reranked logs
  -> relevance_threshold filtering   <- NEW: irrelevant logs dropped first
  -> MMR diversity selection
"""
from __future__ import annotations

import logging

from app.config import DiversityConfig
from app.retrieval.dense_retriever import DenseRetriever, cosine
from app.schemas import RankedLog, ResearchGoal

logger = logging.getLogger(__name__)


class DiversitySelector:
    """Select top-k logs via MMR with a relevance pre-filter."""

    def __init__(
        self,
        config: DiversityConfig | None = None,
        dense_retriever: DenseRetriever | None = None,
    ) -> None:
        self.config = config or DiversityConfig()
        self._dense = dense_retriever or DenseRetriever()

    def select(
        self,
        goal: ResearchGoal,
        ranked_logs: list[RankedLog],
        top_k: int | None = None,
        relevance_threshold: float | None = None,
    ) -> list[RankedLog]:
        """Apply relevance filtering then MMR.

        Args:
            goal: Target goal (used for embedding comparison).
            ranked_logs: Reranked candidates (highest score first).
            top_k: Number of logs to select.
            relevance_threshold: Override config threshold.
        """
        k = top_k or self.config.top_k
        lam = self.config.mmr_lambda
        threshold = (
            relevance_threshold
            if relevance_threshold is not None
            else self.config.relevance_threshold
        )

        if not ranked_logs:
            return []

        # Step 1: relevance threshold filtering
        filtered = [r for r in ranked_logs if r.final_score >= threshold]
        if not filtered:
            # Safety: if everything is below threshold, keep top-k by score
            logger.warning(
                "All %d logs below relevance_threshold=%.3f. Keeping top-%d by score.",
                len(ranked_logs), threshold, k,
            )
            filtered = ranked_logs[:k]

        dropped = len(ranked_logs) - len(filtered)
        if dropped > 0:
            logger.debug(
                "Relevance filter: dropped %d/%d logs (threshold=%.3f)",
                dropped, len(ranked_logs), threshold,
            )

        if len(filtered) <= k:
            for i, r in enumerate(filtered):
                r.diversity_score = r.final_score
            return filtered

        # Step 2: MMR on filtered candidates
        embeddings = {r.log_id: self._dense.embed(r.log.full_text) for r in filtered}
        selected: list[RankedLog] = []
        remaining = list(filtered)

        while remaining and len(selected) < k:
            best_mmr = -float("inf")
            best = None
            for candidate in remaining:
                rel = candidate.final_score
                sim_to_selected = (
                    max(cosine(embeddings[candidate.log_id], embeddings[s.log_id]) for s in selected)
                    if selected else 0.0
                )
                mmr = lam * rel - (1 - lam) * sim_to_selected
                if mmr > best_mmr:
                    best_mmr, best = mmr, candidate
            if best is None:
                break
            best.diversity_score = round(best_mmr, 4)
            selected.append(best)
            remaining.remove(best)

        logger.debug(
            "DiversitySelector: selected %d from %d filtered logs",
            len(selected), len(filtered),
        )
        return selected
