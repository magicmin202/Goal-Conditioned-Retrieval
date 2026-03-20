"""Diversity-aware Top-K selection using Maximal Marginal Relevance (MMR)."""
from __future__ import annotations
import logging
from app.config import DiversityConfig
from app.retrieval.dense_retriever import DenseRetriever, cosine
from app.schemas import RankedLog, ResearchGoal

logger = logging.getLogger(__name__)


class DiversitySelector:
    """Select top-k logs via MMR to balance relevance and diversity."""

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
    ) -> list[RankedLog]:
        """Apply MMR: score = λ * relevance - (1-λ) * max_sim_to_selected."""
        k = top_k or self.config.top_k
        lam = self.config.mmr_lambda
        if not ranked_logs:
            return []

        embeddings = {r.log_id: self._dense.embed(r.log.full_text) for r in ranked_logs}
        selected: list[RankedLog] = []
        remaining = list(ranked_logs)

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

        logger.debug("DiversitySelector selected %d/%d logs", len(selected), len(ranked_logs))
        return selected
