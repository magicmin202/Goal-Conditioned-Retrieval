"""Stage 1 evaluation metrics: Recall@k, Precision@k, MRR, nDCG@k, Diversity Coverage."""
from __future__ import annotations
import logging
import math
from app.schemas import GoalLogLabel, RankedLog

logger = logging.getLogger(__name__)


def _relevant_ids(labels: list[GoalLogLabel]) -> set[str]:
    return {lbl.log_id for lbl in labels if lbl.label == "relevant"}


def recall_at_k(ranked: list[RankedLog], labels: list[GoalLogLabel], k: int) -> float:
    relevant = _relevant_ids(labels)
    if not relevant:
        return 0.0
    retrieved = {r.log_id for r in ranked[:k]}
    return len(retrieved & relevant) / len(relevant)


def precision_at_k(ranked: list[RankedLog], labels: list[GoalLogLabel], k: int) -> float:
    relevant = _relevant_ids(labels)
    retrieved = [r.log_id for r in ranked[:k]]
    if not retrieved:
        return 0.0
    return sum(1 for rid in retrieved if rid in relevant) / k


def mrr(ranked: list[RankedLog], labels: list[GoalLogLabel]) -> float:
    relevant = _relevant_ids(labels)
    for i, r in enumerate(ranked):
        if r.log_id in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked: list[RankedLog], labels: list[GoalLogLabel], k: int) -> float:
    label_map = {lbl.log_id: lbl.relevance_score for lbl in labels}

    def dcg(logs: list[RankedLog], n: int) -> float:
        return sum(
            label_map.get(r.log_id, 0.0) / math.log2(i + 2)
            for i, r in enumerate(logs[:n])
        )

    actual_dcg = dcg(ranked, k)
    ideal_scores = sorted([lbl.relevance_score for lbl in labels], reverse=True)
    ideal_dcg = sum(s / math.log2(i + 2) for i, s in enumerate(ideal_scores[:k]))
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def diversity_coverage(selected: list[RankedLog], total_activity_types: set[str]) -> float:
    if not total_activity_types:
        return 0.0
    selected_types = {r.log.activity_type for r in selected}
    return len(selected_types & total_activity_types) / len(total_activity_types)


def compute_all_metrics(
    ranked: list[RankedLog],
    labels: list[GoalLogLabel],
    k: int = 10,
    all_activity_types: set[str] | None = None,
) -> dict[str, float]:
    return {
        f"recall@{k}": recall_at_k(ranked, labels, k),
        f"precision@{k}": precision_at_k(ranked, labels, k),
        "mrr": mrr(ranked, labels),
        f"ndcg@{k}": ndcg_at_k(ranked, labels, k),
        "diversity_coverage": diversity_coverage(
            ranked[:k], all_activity_types or set()
        ),
    }
