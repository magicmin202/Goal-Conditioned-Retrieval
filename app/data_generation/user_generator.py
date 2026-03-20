"""Generate synthetic ResearchUser objects."""
from __future__ import annotations

import random
from app.schemas import ResearchUser

_ROLES = ["college_student", "graduate_student", "developer", "job_seeker", "researcher"]
_DOMAINS = ["ai_engineering", "software", "biology", "finance", "design", "education"]
_PLANNING_STYLES = ["structured", "flexible", "reactive"]


def generate_users(num_users: int, seed: int = 42) -> list[ResearchUser]:
    """Generate a list of synthetic ResearchUser objects.

    Args:
        num_users: Number of users to generate.
        seed: Random seed for reproducibility.
    """
    rng = random.Random(seed)
    users = []
    for i in range(1, num_users + 1):
        uid = f"U{i:04d}"
        users.append(
            ResearchUser(
                user_id=uid,
                profile={
                    "role": rng.choice(_ROLES),
                    "domain": rng.choice(_DOMAINS),
                    "planning_style": rng.choice(_PLANNING_STYLES),
                },
                dataset_version="v1",
                created_at="2026-03-01",
            )
        )
    return users
