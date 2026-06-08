"""이진 label 생성 (v2): relevant / irrelevant만 사용.

partial 제거 이유:
  - partial은 도메인 overlap matrix 기반의 주관적 판단
  - 평가 지표를 단순화하고 retrieval precision 측정을 명확히 하기 위해
  - 실제 Prologue 서비스에서도 사용자 목표에 직접 관련된 로그만이 relevant

Label 규칙:
  relevant  : log.metadata.goal_id_hint == goal.goal_id
  irrelevant: 그 외 모두 (다른 goal의 로그 + noise 로그)
"""
from __future__ import annotations
from app.schemas import GoalLogLabel, ResearchGoal, ResearchLog


def generate_labels_v2(
    goals: list[ResearchGoal],
    logs: list[ResearchLog],
    user_id: str,
) -> list[GoalLogLabel]:
    labels: list[GoalLogLabel] = []
    label_counter = 1

    for log in logs:
        hint = log.metadata.get("goal_id_hint")
        for goal in goals:
            if hint == goal.goal_id:
                label = "relevant"
                score = 1.0
            else:
                label = "irrelevant"
                score = 0.0

            labels.append(GoalLogLabel(
                label_id=f"GL-{user_id}-{label_counter:05d}",
                user_id=user_id,
                goal_id=goal.goal_id,
                log_id=log.log_id,
                label=label,
                relevance_score=score,
                label_source="synthetic_v2_rule",
            ))
            label_counter += 1

    return labels
