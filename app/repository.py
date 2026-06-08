"""Repository layer — JSON file 기반 dataset 접근."""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any

from app.schemas import GoalLogLabel, ResearchGoal, ResearchLog

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "synthetic_v2"


def _to_research_log(data: dict[str, Any]) -> ResearchLog:
    return ResearchLog(
        log_id=data.get("log_id", ""),
        user_id=data.get("user_id", ""),
        date=data.get("date", ""),
        title=data.get("title", ""),
        content=data.get("content", ""),
        activity_type=data.get("activity_type", "study"),
        metadata=data.get("metadata", {}),
        timestamp=data.get("timestamp", ""),
        created_at=data.get("created_at", ""),
    )


def _to_research_goal(data: dict[str, Any]) -> ResearchGoal:
    return ResearchGoal(
        goal_id=data.get("goal_id", ""),
        user_id=data.get("user_id", ""),
        domain=data.get("domain", ""),
        long_term=data.get("long_term", ""),
        mid_term=data.get("mid_term", ""),
        short_term=data.get("short_term", ""),
        status=data.get("status", "active"),
        noise_ratio=float(data.get("noise_ratio", 0.0)),
        created_at=data.get("created_at", ""),
    )


class DatasetRepository:
    """data/synthetic_v2/ JSON 파일에서 데이터 로드."""

    def __init__(self, data_dir: Path = _DATA_DIR) -> None:
        self._dir = data_dir
        self._goals: list[ResearchGoal] | None = None
        self._logs: list[ResearchLog] | None = None
        self._labels: list[GoalLogLabel] | None = None

    def _load(self, name: str) -> list[dict]:
        path = self._dir / f"{name}.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def goals(self) -> list[ResearchGoal]:
        if self._goals is None:
            self._goals = [_to_research_goal(d) for d in self._load("goals")]
        return self._goals

    def logs(self) -> list[ResearchLog]:
        if self._logs is None:
            self._logs = [_to_research_log(d) for d in self._load("logs")]
        return self._logs

    def labels(self) -> list[GoalLogLabel]:
        if self._labels is None:
            self._labels = [
                GoalLogLabel(
                    label_id=d.get("label_id", ""),
                    user_id=d.get("user_id", ""),
                    goal_id=d.get("goal_id", ""),
                    log_id=d.get("log_id", ""),
                    label=d.get("label", "relevant"),
                    relevance_score=float(d.get("relevance_score", 1.0)),
                    label_source=d.get("label_source", "synthetic_rule"),
                )
                for d in self._load("labels")
            ]
        return self._labels
