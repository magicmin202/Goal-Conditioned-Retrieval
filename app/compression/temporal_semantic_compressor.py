"""Temporal-Semantic Compressor — progression-aware summarisation.

For each anchor+neighbor cluster:
  1. Sort logs by date
  2. Detect activity progression (start → develop → complete)
  3. Build a summary that includes period, repetition count, and progression arc
"""
from __future__ import annotations

import logging
from collections import defaultdict

from app.schemas import CompressedEvidenceUnit, RankedLog, ResearchLog

logger = logging.getLogger(__name__)


def _date_range(logs: list[ResearchLog]) -> str:
    dates = sorted(log.date for log in logs if log.date)
    if not dates:
        return "unknown"
    return dates[0] if len(dates) == 1 else f"{dates[0]} ~ {dates[-1]}"


def _detect_progression(logs: list[ResearchLog]) -> str:
    """Build a short progression arc from sorted log titles.

    Pattern: start → [middle...] → end
    """
    if len(logs) < 2:
        return logs[0].title if logs else ""

    sorted_logs = sorted(logs, key=lambda l: l.date)

    # Deduplicate consecutive identical titles
    seen: list[str] = []
    for log in sorted_logs:
        if not seen or log.title != seen[-1]:
            seen.append(log.title)

    if len(seen) == 1:
        return seen[0]
    if len(seen) == 2:
        return f"{seen[0]} → {seen[1]}"
    # Compress middle: show start, one midpoint, end
    mid = seen[len(seen) // 2]
    return f"{seen[0]} → {mid} → {seen[-1]}"


def _build_progression_summary(anchor: RankedLog, neighbors: list[ResearchLog]) -> str:
    """Build a human-readable progression summary for the cluster."""
    all_logs = sorted([anchor.log] + neighbors, key=lambda l: l.date)
    date_range = _date_range(all_logs)
    topic = anchor.log.metadata.get("topic") or anchor.log.activity_type
    n = len(all_logs)

    # Count evidence strengths
    strengths = [l.metadata.get("evidence_strength", "medium") for l in all_logs]
    high_count = strengths.count("high")
    low_count = strengths.count("low")

    # Progression arc
    progression = _detect_progression(all_logs)

    # Build summary
    if n == 1:
        summary = f"{date_range}: '{topic}' 활동 수행 ({anchor.log.title})"
    else:
        summary = (
            f"{date_range} 동안 '{topic}' 관련 활동 {n}회 수행. "
            f"진행: {progression}"
        )
        if high_count > 0:
            summary += f" (구체적 실행 {high_count}회 포함)"
        if high_count == 0 and low_count == n:
            summary += " (계획 단계 위주)"

    return summary


class TemporalSemanticCompressor:
    """Compress anchor+neighbor clusters into progression-aware CompressedEvidenceUnits."""

    def compress(
        self,
        anchor_logs: list[RankedLog],
        expansion_map: dict[str, list[ResearchLog]],
    ) -> list[CompressedEvidenceUnit]:
        """One CompressedEvidenceUnit per anchor cluster.

        Clusters with same activity_type and topic are merged when
        they share anchor-neighbor overlap (deduplication).
        """
        units: list[CompressedEvidenceUnit] = []
        seen_log_ids: set[str] = set()

        for i, anchor in enumerate(anchor_logs):
            neighbors = expansion_map.get(anchor.log_id, [])

            # Deduplicate: skip neighbors already consumed by a previous unit
            fresh_neighbors = [n for n in neighbors if n.log_id not in seen_log_ids]
            all_logs = [anchor.log] + fresh_neighbors

            # Mark consumed
            for log in all_logs:
                seen_log_ids.add(log.log_id)

            summary = _build_progression_summary(anchor, fresh_neighbors)
            progression = _detect_progression(all_logs)

            unit = CompressedEvidenceUnit(
                unit_id=f"CEU-{i+1:03d}",
                anchor_log_ids=[anchor.log_id] + [n.log_id for n in fresh_neighbors],
                summary=summary,
                date_range=_date_range(all_logs),
                activity_cluster=anchor.log.activity_type,
                log_count=len(all_logs),
                temporal_progression=progression,
            )
            units.append(unit)
            logger.info(
                "CEU-%03d  topic=%s  logs=%d  range=%s  | %s",
                i + 1,
                anchor.log.metadata.get("topic", "-"),
                unit.log_count,
                unit.date_range,
                unit.temporal_progression,
            )

        return units
