"""Top-level dataset builder.

Orchestrates user → goal → log skeleton → rendered log → label generation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

from app.data_generation.goal_generator import generate_goals_for_user
from app.data_generation.label_generator import generate_goal_log_labels
from app.data_generation.log_skeleton_generator import generate_log_skeletons_for_user
from app.data_generation.log_text_renderer import render_log_text
from app.data_generation.user_generator import generate_users
from app.schemas import GoalLogLabel, ResearchGoal, ResearchLog, ResearchUser

logger = logging.getLogger(__name__)


@dataclass
class SyntheticDataset:
    users: list[ResearchUser] = field(default_factory=list)
    goals: list[ResearchGoal] = field(default_factory=list)
    logs: list[ResearchLog] = field(default_factory=list)
    labels: list[GoalLogLabel] = field(default_factory=list)

    def stats(self) -> dict:
        label_counts: dict[str, int] = {}
        for lb in self.labels:
            label_counts[lb.label] = label_counts.get(lb.label, 0) + 1
        return {
            "users": len(self.users),
            "goals": len(self.goals),
            "logs": len(self.logs),
            "labels": len(self.labels),
            "label_distribution": label_counts,
        }


def build_dataset(
    num_users: int = 100,
    start_date: date = date(2026, 3, 1),
    end_date: date = date(2026, 3, 31),
    seed: int = 42,
    small_mode: bool = False,
) -> SyntheticDataset:
    """Build a complete synthetic dataset.

    Args:
        num_users: Total number of users (overridden to 3 in small_mode).
        start_date: Start of the log period.
        end_date: End of the log period.
        seed: Master random seed.
        small_mode: If True, generate a small dataset for quick testing
                    (3 users, 3 goals each, 25-40 logs each).
    """
    if small_mode:
        num_users = 3

    min_logs, max_logs = (25, 40) if small_mode else (80, 150)
    min_goals, max_goals = (3, 3) if small_mode else (3, 5)

    dataset = SyntheticDataset()
    users = generate_users(num_users, seed=seed)
    dataset.users = users

    for i, user in enumerate(users):
        goals = generate_goals_for_user(
            user.user_id,
            user_index=i,
            min_goals=min_goals,
            max_goals=max_goals,
            seed=seed,
        )
        dataset.goals.extend(goals)

        skeletons = generate_log_skeletons_for_user(
            user_id=user.user_id,
            user_index=i,
            goals=goals,
            start_date=start_date,
            end_date=end_date,
            min_logs=min_logs,
            max_logs=max_logs,
            seed=seed,
        )

        logs = [render_log_text(sk) for sk in skeletons]
        dataset.logs.extend(logs)

        labels = generate_goal_log_labels(goals, skeletons, seed=seed + i)
        dataset.labels.extend(labels)

        logger.info(
            "User %s: %d goals, %d logs, %d labels",
            user.user_id, len(goals), len(logs), len(labels),
        )

    s = dataset.stats()
    logger.info(
        "Dataset built: %d users / %d goals / %d logs / %d labels",
        s["users"], s["goals"], s["logs"], s["labels"],
    )
    logger.info("Label distribution: %s", s["label_distribution"])
    return dataset
