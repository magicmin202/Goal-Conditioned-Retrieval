"""Diversity-aware Top-K selection.

Pipeline:
  1. Relevance filtering: drop logs below relevance_threshold
  2. Pre-MMR cap: keep top (k * pre_mmr_multiplier) to reduce compute
  3. MMR selection on the filtered pool (lambda adapted to corpus mode)
"""
from __future__ import annotations

import logging

from app.config import AdaptiveMode, DiversityConfig
from app.retrieval.dense_retriever import DenseRetriever, cosine
from app.schemas import RankedLog, ResearchGoal

logger = logging.getLogger(__name__)


class DiversitySelector:
    """Select top-k logs: relevance filter → MMR."""

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
        adaptive_mode: AdaptiveMode = AdaptiveMode.STANDARD,
    ) -> list[RankedLog]:
        """Apply relevance threshold → pre-MMR cap → MMR.

        score = λ * relevance - (1-λ) * max_sim_to_selected
        λ adapts to corpus mode: SMALL uses higher λ (precision), LARGE uses lower (diversity).
        """
        k = top_k or self.config.top_k
        threshold = self.config.relevance_threshold

        # Adaptive lambda
        if adaptive_mode == AdaptiveMode.SMALL:
            lam = self.config.mmr_lambda_small
        elif adaptive_mode == AdaptiveMode.LARGE:
            lam = self.config.mmr_lambda_large
        else:
            lam = self.config.mmr_lambda

        if not ranked_logs:
            return []

        # ── Step 1: relevance filtering ──────────────────────────────────────
        above = [r for r in ranked_logs if r.final_score >= threshold]
        if len(above) < k:
            # Not enough pass threshold → keep top (k * 2) regardless
            filtered = ranked_logs[: k * 2]
            logger.debug(
                "Relevance filter: only %d/%d above threshold %.3f → using top %d",
                len(above), len(ranked_logs), threshold, len(filtered),
            )
        else:
            filtered = above
            logger.debug(
                "Relevance filter: %d/%d logs above threshold %.3f",
                len(filtered), len(ranked_logs), threshold,
            )

        # ── Step 2: pre-MMR cap ───────────────────────────────────────────────
        cap = k * self.config.pre_mmr_multiplier
        pool = filtered[:cap]
        if len(filtered) > cap:
            logger.debug("Pre-MMR cap: %d→%d (k=%d × %d)", len(filtered), cap, k, self.config.pre_mmr_multiplier)

        logger.debug("DiversitySelector mode=%s lambda=%.2f pool=%d", adaptive_mode, lam, len(pool))

        # ── Step 3: MMR ───────────────────────────────────────────────────────
        embeddings = {r.log_id: self._dense.embed(r.log.full_text) for r in pool}
        selected: list[RankedLog] = []
        remaining = list(pool)

        while remaining and len(selected) < k:
            best_mmr = -float("inf")
            best = None
            for candidate in remaining:
                rel = candidate.final_score
                sim_to_selected = (
                    max(
                        cosine(embeddings[candidate.log_id], embeddings[s.log_id])
                        for s in selected
                    )
                    if selected
                    else 0.0
                )
                mmr = lam * rel - (1 - lam) * sim_to_selected
                if mmr > best_mmr:
                    best_mmr, best = mmr, candidate
            if best is None:
                break
            best.diversity_score = round(best_mmr, 4)
            selected.append(best)
            remaining.remove(best)

        logger.debug("DiversitySelector: %d selected from pool of %d", len(selected), len(pool))
        return selected
