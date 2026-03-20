"""Export and import dataset to/from local JSON files."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from typing import Any

from app.schemas import GoalLogLabel, ResearchGoal, ResearchLog, ResearchUser

logger = logging.getLogger(__name__)


def _to_dict(obj: Any) -> dict:
    return asdict(obj)


def export_dataset_to_json(
    users: list[ResearchUser],
    goals: list[ResearchGoal],
    logs: list[ResearchLog],
    labels: list[GoalLogLabel],
    output_dir: str = "data/synthetic",
) -> None:
    """Export dataset to JSON files under output_dir.

    Creates:
        output_dir/users.json
        output_dir/goals.json
        output_dir/logs.json
        output_dir/labels.json
    """
    os.makedirs(output_dir, exist_ok=True)
    files = {
        "users.json":  [_to_dict(u) for u in users],
        "goals.json":  [_to_dict(g) for g in goals],
        "logs.json":   [_to_dict(l) for l in logs],
        "labels.json": [_to_dict(lb) for lb in labels],
    }
    for filename, data in files.items():
        path = os.path.join(output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("Exported %d records → %s", len(data), path)


def load_dataset_from_json(
    input_dir: str = "data/synthetic",
) -> tuple[list[ResearchUser], list[ResearchGoal], list[ResearchLog], list[GoalLogLabel]]:
    """Load dataset from JSON files."""

    def _load(filename: str) -> list[dict]:
        path = os.path.join(input_dir, filename)
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    users = [ResearchUser(**d) for d in _load("users.json")]
    goals = [ResearchGoal(**d) for d in _load("goals.json")]
    logs  = [ResearchLog(**d)  for d in _load("logs.json")]
    labels = [GoalLogLabel(**d) for d in _load("labels.json")]

    logger.info(
        "Loaded: %d users, %d goals, %d logs, %d labels from %s",
        len(users), len(goals), len(logs), len(labels), input_dir,
    )
    return users, goals, logs, labels


def upload_dataset_to_firestore(
    users: list[ResearchUser],
    goals: list[ResearchGoal],
    logs: list[ResearchLog],
    labels: list[GoalLogLabel],
    config=None,
) -> None:
    """Upload dataset to Firestore research collections.

    Requires serviceAccountKey.json or GOOGLE_APPLICATION_CREDENTIALS.
    """
    from app.firestore_loader import get_firestore_client, write_doc
    from app.config import DEFAULT_CONFIG

    cfg = config or DEFAULT_CONFIG
    client = get_firestore_client()
    cols = cfg.collections

    for u in users:
        write_doc(client, cols.research_users, u.user_id, _to_dict(u))
    logger.info("Uploaded %d users", len(users))

    for g in goals:
        write_doc(client, cols.research_goals, g.goal_id, _to_dict(g))
    logger.info("Uploaded %d goals", len(goals))

    for l in logs:
        write_doc(client, cols.research_logs, l.log_id, _to_dict(l))
    logger.info("Uploaded %d logs", len(logs))

    for lb in labels:
        write_doc(client, cols.research_goal_log_labels, lb.label_id, _to_dict(lb))
    logger.info("Uploaded %d labels", len(labels))
