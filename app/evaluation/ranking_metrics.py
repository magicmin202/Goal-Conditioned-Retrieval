"""Stage 1 evaluation metrics: Recall@k, Precision@k, MRR, nDCG@k, Diversity Coverage.

Two evaluation layers:
  1. Candidate layer  — measures recall/precision of the top-N candidate pool
                        (before reranking).  Tells you whether retrieval missed
                        any relevant logs at all.
  2. Selected layer   — measures recall/precision of the final top-K selection
                        (after reranking + diversity).  Tells you whether
                        ranking/filtering made the right precision decisions.

Compare the two to locate the bottleneck:
  - candidate_recall low  → retrieval problem (BM25/dense/boost)
  - selected_precision low → reranker/filter problem
"""
from __future__ import annotations
import logging
import math
from app.schemas import CandidateLog, GoalLogLabel, RankedLog

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


def selected_precision(
    selected_logs: list,
    labels: list[GoalLogLabel],
    relevance_threshold: float = 0.5,
) -> float:
    """selected_precision = relevant_selected / selected_count.

    Unlike precision_at_k (denominator = k), this uses the actual number of
    admitted logs as the denominator.  Useful when selected_count < top_k.
    """
    if not selected_logs:
        return 0.0
    relevant = _relevant_ids(labels)
    log_ids = [
        (r.log_id if hasattr(r, "log_id") else r)
        for r in selected_logs
    ]
    relevant_count = sum(1 for lid in log_ids if lid in relevant)
    return relevant_count / len(log_ids)


def f1_at_k(ranked: list[RankedLog], labels: list[GoalLogLabel], k: int) -> float:
    """F1@k = 2 * precision@k * recall@k / (precision@k + recall@k).

    Returns 0.0 when both precision and recall are 0.
    Signature is identical to precision_at_k / recall_at_k.
    """
    p = precision_at_k(ranked, labels, k)
    r = recall_at_k(ranked, labels, k)
    if p + r == 0.0:
        return 0.0
    return 2 * p * r / (p + r)


def false_positive_rate(
    selected_logs: list,
    labels: list[GoalLogLabel],
) -> float:
    """false_positive_rate = irrelevant_selected / selected_count.

    Defined over admitted (selected) logs, not over fixed k.
    Equivalent to 1 - selected_precision.
    Returns 0.0 when no logs are selected.
    """
    if not selected_logs:
        return 0.0
    return 1.0 - selected_precision(selected_logs, labels)


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


def candidate_recall(
    candidates: list[CandidateLog],
    labels: list[GoalLogLabel],
) -> float:
    """Fraction of relevant logs that made it into the candidate pool.

    candidate_recall = |relevant ∩ candidates| / |relevant|

    A low value means retrieval itself dropped relevant logs before
    reranking even had a chance to see them.
    """
    relevant = _relevant_ids(labels)
    if not relevant:
        return 0.0
    cand_ids = {c.log_id for c in candidates}
    return len(cand_ids & relevant) / len(relevant)


def candidate_precision(
    candidates: list[CandidateLog],
    labels: list[GoalLogLabel],
) -> float:
    """Fraction of candidate pool that is relevant.

    candidate_precision = |relevant ∩ candidates| / |candidates|

    A low value means many irrelevant logs entered the pool.
    """
    if not candidates:
        return 0.0
    relevant = _relevant_ids(labels)
    cand_ids = [c.log_id for c in candidates]
    return sum(1 for lid in cand_ids if lid in relevant) / len(cand_ids)


def candidate_f1(
    candidates: list[CandidateLog],
    labels: list[GoalLogLabel],
) -> float:
    cr = candidate_recall(candidates, labels)
    cp = candidate_precision(candidates, labels)
    if cr + cp == 0.0:
        return 0.0
    return 2 * cr * cp / (cr + cp)


def compute_candidate_metrics(
    candidates: list[CandidateLog],
    labels: list[GoalLogLabel],
) -> dict[str, float]:
    """Metrics for the candidate retrieval pool (before reranking).

    Returns
    -------
    dict with keys:
        candidate_recall    – recall over relevant set
        candidate_precision – precision over candidate pool
        candidate_f1        – harmonic mean of the two
        candidate_size      – number of candidates
        relevant_in_pool    – count of relevant logs in pool
        relevant_total      – total relevant logs in label set
    """
    relevant = _relevant_ids(labels)
    cand_ids = {c.log_id for c in candidates}
    hit = len(cand_ids & relevant)
    cr = hit / len(relevant) if relevant else 0.0
    cp = hit / len(candidates) if candidates else 0.0
    cf1 = 2 * cr * cp / (cr + cp) if (cr + cp) > 0 else 0.0
    return {
        "candidate_recall":    round(cr, 4),
        "candidate_precision": round(cp, 4),
        "candidate_f1":        round(cf1, 4),
        "candidate_size":      len(candidates),
        "relevant_in_pool":    hit,
        "relevant_total":      len(relevant),
    }


def compute_all_metrics(
    ranked: list[RankedLog],
    labels: list[GoalLogLabel],
    k: int = 10,
    all_activity_types: set[str] | None = None,
    selected_logs: list | None = None,
) -> dict[str, float]:
    """Compute all Stage 1 retrieval metrics.

    Parameters
    ----------
    selected_logs:
        Actual admitted logs (may differ from ranked[:k] when admission
        gates reduce the count).  Used for selected_precision,
        false_positive_rate, and selected_count.
        Defaults to ranked[:k] when not provided.
    """
    admitted = selected_logs if selected_logs is not None else ranked[:k]
    return {
        f"recall@{k}": recall_at_k(ranked, labels, k),
        f"precision@{k}": precision_at_k(ranked, labels, k),
        f"f1@{k}": f1_at_k(ranked, labels, k),
        "mrr": mrr(ranked, labels),
        f"ndcg@{k}": ndcg_at_k(ranked, labels, k),
        "diversity_coverage": diversity_coverage(
            ranked[:k], all_activity_types or set()
        ),
        "selected_precision": selected_precision(admitted, labels),
        "false_positive_rate": false_positive_rate(admitted, labels),
        "selected_count": len(admitted),
    }
