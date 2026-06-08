"""Dataset schemas."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ResearchUser:
    user_id: str
    profile: dict[str, Any] = field(default_factory=dict)
    dataset_version: str = "v1"
    created_at: str = ""


@dataclass
class ResearchGoal:
    goal_id: str
    user_id: str
    domain: str = ""
    long_term: str = ""
    mid_term: str = ""
    short_term: str = ""
    status: str = "active"
    noise_ratio: float = 0.0
    created_at: str = ""


@dataclass
class ResearchLog:
    log_id: str
    user_id: str
    date: str
    title: str
    content: str = ""
    activity_type: str = "study"
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    created_at: str = ""

    @property
    def embedding_text(self) -> str:
        topic = self.metadata.get("topic", "")
        parts = [f"title: {self.title}"]
        if self.activity_type and self.activity_type != "unknown":
            parts.append(f"activity_type: {self.activity_type}")
        if topic:
            parts.append(f"topic: {topic}")
        if self.content:
            parts.append(f"content: {self.content}")
        return "\n".join(parts).strip()


@dataclass
class GoalLogLabel:
    label_id: str
    user_id: str
    goal_id: str
    log_id: str
    label: str  # "relevant" | "irrelevant"
    relevance_score: float = 1.0
    label_source: str = "synthetic_rule"
