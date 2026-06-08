"""LLM 기반 목표-로그 분석 모듈.

두 가지 방식 지원:
  full : 사용자 전체 로그를 LLM에 전달
  rag  : embedding 필터링 후 관련 로그만 전달
"""
from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3.5-flash"


@dataclass
class AnalysisResult:
    mode: str               # "full" | "rag"
    goal_id: str
    user_id: str
    noise_ratio: float
    n_logs_total: int       # 사용자 전체 로그 수
    n_logs_sent: int        # LLM에 실제로 보낸 로그 수
    input_tokens: int
    output_tokens: int
    analysis: str           # LLM 응답 전문
    model: str = DEFAULT_MODEL


def _build_goal_text(goal: dict) -> str:
    return (
        f"[목표 정보]\n"
        f"- 장기 목표(Long-term): {goal.get('long_term', '')}\n"
        f"- 중기 목표(Mid-term): {goal.get('mid_term', '')}\n"
        f"- 단기 목표(Short-term): {goal.get('short_term', '')}\n"
        f"- 목표 설정일: {goal.get('created_at', '')}"
    )


def _build_logs_text(logs: list[dict]) -> str:
    lines = ["[활동 로그 (시간순)]"]
    for log in sorted(logs, key=lambda x: x.get("date", "")):
        lines.append(
            f"- [{log.get('date','')}] {log.get('title','')} | {log.get('content','')}"
        )
    return "\n".join(lines)


def _build_prompt(goal: dict, logs: list[dict]) -> str:
    goal_text = _build_goal_text(goal)
    logs_text = _build_logs_text(logs)

    return f"""당신은 개인 목표 달성 코치입니다. 아래 목표와 활동 로그를 분석하여 피드백을 제공해주세요.

{goal_text}

{logs_text}

---
위 내용을 바탕으로 다음 항목을 분석해주세요:

1. **현재 진행 상태 분석**: 목표 설정일 이후 사용자의 활동 패턴을 시계열로 파악하고, 현재 목표 달성을 향해 얼마나 진행되었는지 평가하세요.

2. **목표 적합성**: 사용자의 활동들이 설정한 장기/중기/단기 목표와 얼마나 부합하는지 평가하세요. (매우 적합 / 적합 / 보통 / 부적합 중 선택 후 근거 설명)

3. **다음 행동 추천**: 목표 달성을 위해 지금 당장 실행할 수 있는 구체적인 행동 3가지를 추천하세요.

4. **종합 의견**: 전반적인 평가와 응원 메시지를 작성하세요.

응답은 한국어로 작성하고, 각 항목을 명확히 구분하여 500자 이내로 간결하게 작성하세요."""


def analyze(
    goal: dict,
    logs: list[dict],
    mode: str,
    n_logs_total: int,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    call_interval: float = 13.0,   # 분당 5회 제한 → 12초 간격 + 여유
) -> AnalysisResult:
    """LLM에 goal + logs를 전달하여 분석 결과를 반환."""
    import time
    from google import genai
    from google.genai import types

    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise ValueError("GEMINI_API_KEY not set")

    client = genai.Client(api_key=key)
    prompt = _build_prompt(goal, logs)

    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=800,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            break
        except Exception as exc:
            if "429" in str(exc):
                logger.warning("429 rate limit — 65s 대기 후 재시도 (%d/5)", attempt + 1)
                time.sleep(65)
            else:
                raise
    else:
        raise RuntimeError("LLM 호출 5회 모두 실패")

    text       = response.text or ""
    in_tokens  = response.usage_metadata.prompt_token_count     or 0
    out_tokens = response.usage_metadata.candidates_token_count or 0

    # 다음 호출 전 대기 (rate limit 준수)
    time.sleep(call_interval)

    return AnalysisResult(
        mode=mode,
        goal_id=goal["goal_id"],
        user_id=goal["user_id"],
        noise_ratio=goal.get("noise_ratio", 0.0),
        n_logs_total=n_logs_total,
        n_logs_sent=len(logs),
        input_tokens=in_tokens,
        output_tokens=out_tokens,
        analysis=text,
        model=model,
    )
