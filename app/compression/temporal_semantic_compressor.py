"""Temporal-Semantic Compressor — progression-aware summarisation.

Compressor role: SUMMARIZATION ONLY.
  - Does NOT judge relevance.
  - Does NOT add new logs to clusters.
  - Does NOT re-retrieve from corpus.

Category hard-drop (Stage 2 safety):
  - Anchors with category_hit_strength="none" are dropped before clustering.
  - This is a final safety check — anchors should never reach here with "none"
    because the reranker category gate fires first, but this prevents any
    edge-case contamination of evidence units.

Neighbors arrive as RankedLog (from LocalExpander), preserving
category_hit_strength for cluster integrity checks.
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


def _detect_progression(logs: list[ResearchLog]) -> str:
    """Build a short progression arc from sorted log titles."""
    if len(logs) < 2:
        return logs[0].title if logs else ""

    sorted_logs = sorted(logs, key=lambda l: l.date)
    seen: list[str] = []
    for log in sorted_logs:
        if not seen or log.title != seen[-1]:
            seen.append(log.title)

    if len(seen) == 1:
        return seen[0]
    if len(seen) == 2:
        return f"{seen[0]} → {seen[1]}"
    mid = seen[len(seen) // 2]
    return f"{seen[0]} → {mid} → {seen[-1]}"


def _build_progression_summary(anchor: RankedLog, neighbors: list[RankedLog]) -> str:
    """Build a human-readable progression summary for the cluster."""
    neighbor_logs = [n.log for n in neighbors]
    all_logs = sorted([anchor.log] + neighbor_logs, key=lambda l: l.date)
    date_range = _date_range(all_logs)
    topic = anchor.log.metadata.get("topic") or anchor.log.activity_type
    n = len(all_logs)

    strengths = [l.metadata.get("evidence_strength", "medium") for l in all_logs]
    high_count = strengths.count("high")
    low_count = strengths.count("low")

    progression = _detect_progression(all_logs)

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
    """Compress anchor+neighbor clusters into progression-aware CompressedEvidenceUnits.

    expansion_map type: dict[anchor_log_id → list[RankedLog]]
    Neighbors carry category_hit_strength so we can enforce category whitelist.
    """

    def compress(
        self,
        anchor_logs: list[RankedLog],
        expansion_map: dict[str, list[RankedLog]],
    ) -> list[CompressedEvidenceUnit]:
        """Summarise each admitted anchor+neighbor cluster.

        Stage 2 category hard-drop:
          Anchors with category_hit_strength="none" are dropped before clustering.
          This is a final safety net — should not fire if reranker gate is working.
        """
        total_neighbor_logs = sum(len(v) for v in expansion_map.values())
        logger.info(
            "Compressor input: %d admitted anchors  %d total neighbor logs"
            "  (summarization only — no relevance re-judgment)",
            len(anchor_logs), total_neighbor_logs,
        )

        # ── Category hard-drop (Stage 2 safety whitelist) ─────────────────────
        valid_anchors: list[RankedLog] = []
        hard_dropped = 0
        for anchor in anchor_logs:
            if anchor.category_hit_strength == "none":
                logger.warning(
                    "HARD DROP anchor %s  cat=%s  strength=none  "
                    "[%s]  (should not happen — reranker gate should have caught this)",
                    anchor.log_id, anchor.schema_category, anchor.log.title,
                )
                hard_dropped += 1
            else:
                valid_anchors.append(anchor)

        if hard_dropped:
            logger.warning(
                "Stage2 hard-dropped %d anchor(s) with no schema category",
                hard_dropped,
            )

        units: list[CompressedEvidenceUnit] = []
        seen_log_ids: set[str] = set()

        for i, anchor in enumerate(valid_anchors):
            raw_neighbors = expansion_map.get(anchor.log_id, [])

            # Deduplicate: skip neighbors already consumed by a previous unit
            fresh_neighbors = [n for n in raw_neighbors if n.log_id not in seen_log_ids]

            # Safety: also drop any neighbor that slipped through with no category
            fresh_neighbors = [
                n for n in fresh_neighbors if n.category_hit_strength != "none"
            ]

            all_log_objs = [anchor.log] + [n.log for n in fresh_neighbors]

            # Mark consumed
            for log in all_log_objs:
                seen_log_ids.add(log.log_id)

            summary = _build_progression_summary(anchor, fresh_neighbors)
            progression = _detect_progression(all_log_objs)

            unit = CompressedEvidenceUnit(
                unit_id=f"CEU-{i+1:03d}",
                anchor_log_ids=[anchor.log_id] + [n.log_id for n in fresh_neighbors],
                summary=summary,
                date_range=_date_range(all_log_objs),
                activity_cluster=anchor.schema_category or anchor.log.activity_type,
                log_count=len(all_log_objs),
                temporal_progression=progression,
            )
            units.append(unit)
            logger.info(
                "CEU-%03d  topic=%s  cat=%s(%s)  logs=%d  range=%s  | %s",
                i + 1,
                anchor.log.metadata.get("topic", "-"),
                anchor.schema_category, anchor.category_hit_strength,
                unit.log_count,
                unit.date_range,
                unit.temporal_progression,
            )

        return units
