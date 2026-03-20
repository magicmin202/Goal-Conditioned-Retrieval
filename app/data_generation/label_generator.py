"""Generate goal-log relevance labels.

Label assignment rules:
- relevant  (1.0): log was generated for this goal.
- partial   (0.5): log was generated for another goal but shares domain overlap,
                   or has weak indirect connection.
- irrelevant(0.0): noise log or log from a completely different domain.

Target distribution: relevant ~30%, partial ~30%, irrelevant ~40%.
"""
from __future__ import annotations

import random
from app.data_generation.log_skeleton_generator import LogSkeleton, _infer_category
from app.schemas import GoalLogLabel, ResearchGoal


# Domain similarity matrix — categories that can create partial relevance
_PARTIAL_OVERLAP: dict[str, set[str]] = {
    "career":       {"education"},
    "education":    {"career", "habit"},
    "health":       {"habit"},
    "habit":        {"health", "education"},
    "relationship": set(),
    "travel":       {"finance"},
    "finance":      {"travel", "career"},
    "hobby":        {"habit"},
}


def _is_partial(goal_cat: str, log_goal_cat: str) -> bool:
    return log_goal_cat in _PARTIAL_OVERLAP.get(goal_cat, set())


def generate_goal_log_labels(
    goals: list[ResearchGoal],
    skeletons: list[LogSkeleton],
    seed: int = 42,
) -> list[GoalLogLabel]:
    """Generate GoalLogLabel for every (goal, log) pair.

    Args:
        goals: List of goals for the user.
        skeletons: List of log skeletons (used for goal_id hint and category).
        seed: Random seed for tie-breaking.
    """
    rng = random.Random(seed)
    labels: list[GoalLogLabel] = []
    label_counter = 1

    goal_categories = {g.goal_id: _infer_category(g) for g in goals}

    for skeleton in skeletons:
        for goal in goals:
            goal_cat = goal_categories[goal.goal_id]
            log_goal_id = skeleton.goal_id
            log_goal_cat = goal_categories.get(log_goal_id, "noise") if log_goal_id else "noise"

            # Determine label
            if log_goal_id == goal.goal_id:
                # Direct match
                label = "relevant"
                score = round(rng.uniform(0.8, 1.0), 2)
            elif log_goal_id is None:
                # Noise log
                label = "irrelevant"
                score = 0.0
            elif _is_partial(goal_cat, log_goal_cat):
                # Partial domain overlap
                label = "partial"
                score = round(rng.uniform(0.3, 0.6), 2)
            else:
                label = "irrelevant"
                score = 0.0

            labels.append(
                GoalLogLabel(
                    label_id=f"GL-{skeleton.user_id}-{label_counter:05d}",
                    user_id=skeleton.user_id,
                    goal_id=goal.goal_id,
                    log_id=skeleton.log_id,
                    label=label,
                    relevance_score=score,
                    label_source="synthetic_rule",
                )
            )
            label_counter += 1

    return labels
