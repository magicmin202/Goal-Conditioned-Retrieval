"""Repository layer — wraps Firestore access for research collections."""
from __future__ import annotations
import logging
from typing import Any

from app.config import AppConfig, DEFAULT_CONFIG
from app.firestore_loader import batch_get_docs, get_firestore_client
from app.schemas import GoalLogLabel, ResearchGoal, ResearchLog

logger = logging.getLogger(__name__)


def _to_research_log(data: dict[str, Any]) -> ResearchLog:
    return ResearchLog(
        log_id=data.get("log_id") or data.get("_doc_id", ""),
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
        goal_id=data.get("goal_id") or data.get("_doc_id", ""),
        user_id=data.get("user_id", ""),
        title=data.get("title", ""),
        description=data.get("description", ""),
        time_horizon=data.get("time_horizon", "mid_term"),
        status=data.get("status", "active"),
        goal_embedding_text=data.get("goal_embedding_text", ""),
        created_at=data.get("created_at", ""),
    )


class ResearchRepository:
    def __init__(self, config: AppConfig = DEFAULT_CONFIG) -> None:
        self.config = config
        self.collections = config.collections
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = get_firestore_client()
        return self._client

    def get_user_goals(self, user_id: str) -> list[ResearchGoal]:
        docs = batch_get_docs(
            self.client, self.collections.research_goals, {"user_id": user_id}
        )
        return [_to_research_goal(d) for d in docs]

    def get_goal(self, goal_id: str) -> ResearchGoal | None:
        doc = (
            self.client.collection(self.collections.research_goals)
            .document(goal_id)
            .get()
        )
        if not doc.exists:
            return None
        data = doc.to_dict() or {}
        data["_doc_id"] = doc.id
        return _to_research_goal(data)

    def get_user_logs(self, user_id: str) -> list[ResearchLog]:
        docs = batch_get_docs(
            self.client, self.collections.research_logs, {"user_id": user_id}
        )
        return [_to_research_log(d) for d in docs]

    def get_logs_by_date_range(
        self, user_id: str, start_date: str, end_date: str
    ) -> list[ResearchLog]:
        all_logs = self.get_user_logs(user_id)
        return [l for l in all_logs if start_date <= l.date <= end_date]

    def get_goal_log_labels(
        self, user_id: str, goal_id: str
    ) -> list[GoalLogLabel]:
        docs = batch_get_docs(
            self.client,
            self.collections.research_goal_log_labels,
            {"user_id": user_id, "goal_id": goal_id},
        )
        return [
            GoalLogLabel(
                label_id=d.get("label_id") or d.get("_doc_id", ""),
                user_id=d.get("user_id", ""),
                goal_id=d.get("goal_id", ""),
                log_id=d.get("log_id", ""),
                label=d.get("label", "relevant"),
                relevance_score=float(d.get("relevance_score", 1.0)),
                label_source=d.get("label_source", "synthetic_rule"),
            )
            for d in docs
        ]

    # compat aliases
    def get_user_goal_projects(self, user_id: str) -> list[ResearchGoal]:
        return self.get_user_goals(user_id)

    def get_project_logs(self, user_id: str, project_id: str) -> list[ResearchLog]:
        labels = self.get_goal_log_labels(user_id, project_id)
        labeled_ids = {lbl.log_id for lbl in labels}
        return [l for l in self.get_user_logs(user_id) if l.log_id in labeled_ids]
