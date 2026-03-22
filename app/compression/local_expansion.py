"""Anchor-centered Local Expansion — temporal window + neighbor re-admission.

Stage 2 = consolidation only (NOT new retrieval).

For each admitted anchor:
  1. Temporal filter  — only logs within anchor.date ± window_days
  2. Topical filter   — activity_type / topic / title keyword overlap (≥ min_criteria)
  3. Neighbor re-admission — same reranker gate as anchors
     → logs that fail are NEVER sent to the compressor

Invariants:
  - Non-admitted logs do NOT enter the compressor.
  - Fewer logs in cluster is acceptable; correctness > coverage.
"""
from __future__ import annotations

import logging
import re
from datetime import date as _date, timedelta

from app.schemas import RankedLog, ResearchGoal, ResearchLog

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[\w가-힣]{2,}", text.lower()))


def _parse_date(s: str | None) -> _date | None:
    if not s:
        return None
    try:
        return _date.fromisoformat(s)
    except (ValueError, TypeError):
        return None


def _topical_criteria(anchor: RankedLog, log: ResearchLog) -> int:
    """Count topical cluster criteria satisfied (0–3; no semantic needed)."""
    count = 0
    if log.activity_type == anchor.log.activity_type:
        count += 1
    a_topic = anchor.log.metadata.get("topic", "")
    l_topic = log.metadata.get("topic", "")
    if a_topic and l_topic and a_topic == l_topic:
        count += 1
    a_tokens = _tokenize(anchor.log.title)
    l_tokens = _tokenize(log.title)
    if a_tokens and l_tokens and a_tokens & l_tokens:
        count += 1
    return count


class LocalExpander:
    """Expand each admitted anchor into a cluster of temporally-close, admitted neighbors.

    Parameters
    ----------
    min_topical_criteria:
        Minimum number of topical criteria a neighbor must meet (default 1).
    anchor_relevance_threshold:
        Anchors below this final_score are skipped entirely.
    """

    def __init__(
        self,
        min_topical_criteria: int = 1,
        semantic_threshold: float = 0.45,   # kept for API compat, unused
        anchor_relevance_threshold: float = 0.05,
        neighbor_goal_relevance_min: float = 0.0,  # kept for API compat, unused
        dense_retriever=None,               # kept for API compat, unused
    ) -> None:
        self.min_topical_criteria = min_topical_criteria
        self.anchor_relevance_threshold = anchor_relevance_threshold

    def expand(
        self,
        anchor_logs: list[RankedLog],
        full_log_pool: list[ResearchLog],
        goal: ResearchGoal | None = None,
        expanded_terms: list[str] | None = None,
        negative_terms: list[str] | None = None,
        priority_terms: list[str] | None = None,
        related_terms: list[str] | None = None,
        reranker=None,
        temporal_window: int = 3,
        neighbor_admission_threshold: float = 0.08,
        max_neighbors: int = 5,
        # Back-compat keyword (ignored; use temporal_window instead)
        max_neighbors_compat: int = 5,
    ) -> dict[str, list[ResearchLog]]:
        """Return anchor_log_id → list of admitted neighbor logs.

        Steps per anchor:
          1. Skip anchor if below anchor_relevance_threshold
          2. Temporal filter: pool → logs within ±temporal_window days
          3. Topical filter: ≥ min_topical_criteria
          4. Reranker re-admission check (if reranker + goal provided)
          5. Admit top max_neighbors
        """
        exp_terms = expanded_terms or []
        neg_terms = negative_terms or []
        pri_terms = priority_terms or []
        rel_terms = related_terms or []

        # Exclude all anchors from neighbor pool
        anchor_ids = {r.log_id for r in anchor_logs}
        pool_logs = [log for log in full_log_pool if log.log_id not in anchor_ids]

        valid_anchors = [
            a for a in anchor_logs if a.final_score >= self.anchor_relevance_threshold
        ]
        skipped_anchors = len(anchor_logs) - len(valid_anchors)
        if skipped_anchors:
            logger.debug(
                "LocalExpander: %d anchors skipped (below threshold %.3f)",
                skipped_anchors, self.anchor_relevance_threshold,
            )

        result: dict[str, list[ResearchLog]] = {}

        for anchor in valid_anchors:
            anchor_date = _parse_date(anchor.log.date)
            if anchor_date:
                window_start = anchor_date - timedelta(days=temporal_window)
                window_end = anchor_date + timedelta(days=temporal_window)
            else:
                window_start = window_end = None

            # ── Step 1: Temporal filter ───────────────────────────────────────
            temporal_candidates: list[ResearchLog] = []
            for log in pool_logs:
                if window_start and window_end:
                    log_date = _parse_date(log.date)
                    if log_date is None or not (window_start <= log_date <= window_end):
                        continue
                temporal_candidates.append(log)

            # ── Step 2: Topical similarity filter ─────────────────────────────
            topical_candidates: list[ResearchLog] = []
            for log in temporal_candidates:
                if _topical_criteria(anchor, log) >= self.min_topical_criteria:
                    topical_candidates.append(log)

            # ── Step 3: Neighbor re-admission via reranker ────────────────────
            admitted: list[ResearchLog] = []
            rejected: list[tuple[str, float]] = []

            for log in topical_candidates:
                if reranker is not None and goal is not None:
                    score = reranker.score_log(
                        goal, log,
                        expanded_terms=exp_terms,
                        negative_terms=neg_terms,
                        priority_terms=pri_terms,
                        related_terms=rel_terms,
                    )
                    if score < neighbor_admission_threshold:
                        rejected.append((log.log_id, score))
                        logger.debug(
                            "  Neighbor REJECTED %s (score=%.4f < %.4f): %s",
                            log.log_id, score, neighbor_admission_threshold, log.title,
                        )
                        continue
                    logger.debug(
                        "  Neighbor ADMITTED %s (score=%.4f): %s",
                        log.log_id, score, log.title,
                    )
                admitted.append(log)

            neighbors = admitted[:max_neighbors]
            result[anchor.log_id] = neighbors

            logger.info(
                "[Anchor Consolidation]\n"
                "  anchor=%s  date=%s\n"
                "  window=±%dd  [%s → %s]\n"
                "  temporal_candidates=%d  topical=%d  admitted=%d  → cluster_size=%d"
                "  (rejected=%d)",
                anchor.log_id, anchor.log.date,
                temporal_window,
                window_start.isoformat() if window_start else "?",
                window_end.isoformat() if window_end else "?",
                len(temporal_candidates), len(topical_candidates),
                len(admitted), len(neighbors),
                len(rejected),
            )
            if rejected:
                logger.debug(
                    "  Rejected from cluster: %s",
                    [(lid, f"{s:.4f}") for lid, s in rejected],
                )

        # Fill empty entries for anchors that were skipped (below threshold)
        for anchor in anchor_logs:
            if anchor.log_id not in result:
                result[anchor.log_id] = []

        total_in_clusters = sum(len(v) for v in result.values())
        logger.info(
            "LocalExpander: %d/%d anchors active  %d total neighbor logs in clusters",
            len(valid_anchors), len(anchor_logs), total_in_clusters,
        )
        return result
