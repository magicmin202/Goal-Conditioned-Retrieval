"""Temporal-Semantic Compressor.

Summarizes clusters of related logs into CompressedEvidenceUnits
capturing temporal progression and semantic consistency.
"""
from __future__ import annotations
import logging
from app.schemas import CompressedEvidenceUnit, RankedLog, ResearchLog

logger = logging.getLogger(__name__)


def _date_range(logs: list[ResearchLog]) -> str:
    dates = sorted(log.date for log in logs if log.date)
    if not dates:
        return "unknown"
    return dates[0] if len(dates) == 1 else f"{dates[0]} ~ {dates[-1]}"


def _build_summary(anchor: RankedLog, neighbors: list[ResearchLog]) -> str:
    all_logs = [anchor.log] + neighbors
    date_range = _date_range(all_logs)
    titles = [log.title for log in all_logs if log.title][:5]
    return f"{date_range} 동안 진행된 활동: {', '.join(titles)}"


def _detect_progression(logs: list[ResearchLog]) -> str:
    if len(logs) < 2:
        return ""
    stages = [log.title for log in sorted(logs, key=lambda x: x.date)]
    return " → ".join(stages[:4])


class TemporalSemanticCompressor:
    """Compress anchor+neighbor clusters into CompressedEvidenceUnits."""

    def compress(
        self,
        anchor_logs: list[RankedLog],
        expansion_map: dict[str, list[ResearchLog]],
    ) -> list[CompressedEvidenceUnit]:
        units: list[CompressedEvidenceUnit] = []
        for i, anchor in enumerate(anchor_logs):
            neighbors = expansion_map.get(anchor.log_id, [])
            all_logs = [anchor.log] + neighbors
            unit = CompressedEvidenceUnit(
                unit_id=f"CEU-{i+1:03d}",
                anchor_log_ids=[anchor.log_id] + [n.log_id for n in neighbors],
                summary=_build_summary(anchor, neighbors),
                date_range=_date_range(all_logs),
                activity_cluster=anchor.log.activity_type,
                log_count=len(all_logs),
                temporal_progression=_detect_progression(all_logs),
            )
            units.append(unit)
            logger.debug("Compressed unit %s: %d logs", unit.unit_id, unit.log_count)
        return units
