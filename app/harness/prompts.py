"""Harness 분석 프롬프트 템플릿."""
from __future__ import annotations

from app.schemas import ResearchGoal, ResearchLog

_LOG_ITEM = "[{date}] {title}\n{content}"


def _format_logs(logs: list[ResearchLog]) -> str:
    """로그 목록을 LLM 입력용 텍스트로 변환 (title, content, time)."""
    parts = []
    for log in sorted(logs, key=lambda l: l.date):
        content = (log.content or "").strip()
        parts.append(_LOG_ITEM.format(
            date    = log.date or "날짜 미상",
            title   = log.title,
            content = content if content else "(내용 없음)",
        ))
    return "\n\n".join(parts)


def build_initial_prompt(goal: ResearchGoal, logs: list[ResearchLog]) -> str:
    """첫 분석 프롬프트 — 전체 로그 기반 초기 분석."""
    logs_text = _format_logs(logs)
    return f"""당신은 사용자의 목표 달성을 분석하는 AI 코치입니다.

[목표]
제목: {goal.title}
설명: {goal.description}

[사용자 활동 로그 — 총 {len(logs)}개]
{logs_text}

---
위 활동 로그를 바탕으로 다음을 분석해주세요:
1. 목표 대비 현재 진행 상황 요약
2. 잘 되고 있는 점
3. 부족하거나 보완이 필요한 점
4. 다음 단계 권장 액션

분석은 구체적이고 실질적으로 작성해주세요."""


def build_update_prompt(
    goal: ResearchGoal,
    previous_summary: str,
    new_logs: list[ResearchLog],
) -> str:
    """업데이트 프롬프트 — 이전 분석 + 신규 로그만 전달."""
    new_logs_text = _format_logs(new_logs)
    return f"""당신은 사용자의 목표 달성을 분석하는 AI 코치입니다.

[목표]
제목: {goal.title}
설명: {goal.description}

[이전 분석 요약]
{previous_summary}

[새로 추가된 활동 로그 — {len(new_logs)}개]
{new_logs_text}

---
새로운 활동들을 반영하여 분석을 업데이트해주세요:
1. 이전 분석 이후 달라진 점 (진전 또는 변화)
2. 현재 진행 상황 요약 (전체 기준)
3. 다음 단계 권장 액션

이전 분석과 새 활동을 통합하여 일관성 있게 작성해주세요."""
