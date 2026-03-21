"""LLM-based query expansion.

Default path: Gemini API → structured JSON with evidence_terms + negative_terms.
Fallback: domain-specific heuristic table (goal-specific, NOT generic words).

Negative terms are used by the reranker to penalise unrelated logs.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from app.retrieval.query_understanding import QueryObject
from app.schemas import ResearchGoal

logger = logging.getLogger(__name__)

# ── Heuristic expansion table (fallback) ─────────────────────────────────────
_HEURISTIC: dict[str, dict[str, list[str]]] = {
    "개발자": {
        "evidence_terms": [
            "코딩 테스트", "알고리즘", "자료구조", "포트폴리오", "프로젝트 구현",
            "기술 면접", "Python", "백엔드", "CS 공부", "문제 풀이", "코딩",
            "프로그래밍", "깃허브", "소프트웨어",
        ],
        "negative_terms": ["주식", "투자", "ETF", "여행", "연애", "소개팅", "운동", "헬스", "요리"],
    },
    "AI 엔지니어": {
        "evidence_terms": [
            "머신러닝", "딥러닝", "선형대수", "확률통계", "논문", "모델 구현",
            "PyTorch", "데이터", "실험", "수학", "알고리즘", "AI", "모델",
        ],
        "negative_terms": ["주식", "투자", "여행", "연애", "소개팅", "운동", "요리", "청소"],
    },
    "대학원": {
        "evidence_terms": [
            "연구계획서", "논문 읽기", "연구실 탐색", "지도교수", "GRE", "영어 성적",
            "연구 주제", "세미나", "학술", "대학원 준비",
        ],
        "negative_terms": ["주식", "투자", "여행", "운동", "요리", "게임"],
    },
    "토익": {
        "evidence_terms": [
            "리스닝", "리딩", "어휘 암기", "문법", "RC", "LC", "실전 문제", "영어 공부",
        ],
        "negative_terms": ["주식", "운동", "여행", "연애", "요리", "코딩"],
    },
    "운동": {
        "evidence_terms": [
            "헬스장", "러닝", "근력 운동", "유산소", "스트레칭", "홈트", "운동 루틴",
            "벤치프레스", "스쿼트", "체중",
        ],
        "negative_terms": ["주식", "투자", "코딩", "공부", "연애", "요리"],
    },
    "식단": {
        "evidence_terms": ["칼로리", "채소", "단백질", "다이어트", "건강식", "외식", "식재료"],
        "negative_terms": ["주식", "코딩", "알고리즘", "여행", "연애"],
    },
    "연애": {
        "evidence_terms": ["소개팅", "데이트", "이성 만남", "대화", "관계", "연락"],
        "negative_terms": ["주식", "코딩", "알고리즘", "운동", "요리"],
    },
    "친구": {
        "evidence_terms": ["모임", "약속", "네트워킹", "사교", "인맥"],
        "negative_terms": ["주식", "코딩", "공부", "운동"],
    },
    "여행": {
        "evidence_terms": [
            "항공권", "숙소 예약", "여행 계획", "예산", "짐 준비", "여행지", "배낭",
        ],
        "negative_terms": ["주식", "코딩", "알고리즘", "운동", "연애"],
    },
    "저축": {
        "evidence_terms": ["가계부", "지출 관리", "적금", "예산", "절약", "수입"],
        "negative_terms": ["코딩", "알고리즘", "운동", "여행", "연애"],
    },
    "투자": {
        "evidence_terms": ["주식", "ETF", "부동산", "재테크", "경제", "포트폴리오", "배당"],
        "negative_terms": ["코딩", "알고리즘", "운동", "여행", "연애"],
    },
    "독서": {
        "evidence_terms": ["책 읽기", "독후감", "서평", "독서 목록", "지식", "도서"],
        "negative_terms": ["주식", "운동", "여행", "연애", "요리"],
    },
    "기상": {
        "evidence_terms": ["아침 루틴", "수면", "기상 시간", "생산성", "오전"],
        "negative_terms": ["주식", "코딩", "여행", "연애"],
    },
    "사진": {
        "evidence_terms": ["촬영", "카메라", "편집", "구도", "라이트룸", "포토샵"],
        "negative_terms": ["주식", "코딩", "알고리즘", "운동", "연애"],
    },
    "요리": {
        "evidence_terms": ["레시피", "요리 실습", "식재료", "쿠킹", "홈쿡", "조리"],
        "negative_terms": ["주식", "코딩", "알고리즘", "운동", "여행"],
    },
}

# ── Gemini expansion prompt ───────────────────────────────────────────────────
_EXPANSION_PROMPT = """당신은 정보 검색(Information Retrieval) 전문가입니다.

사용자의 목표가 주어졌을 때, 그 목표와 관련된 행동 로그를 검색하기 위한 vocabulary를 생성하세요.

[목표]
제목: {title}
설명: {description}

[규칙]
- evidence_terms는 반드시 이 목표의 domain에 특화된 구체적 단어여야 합니다
- "학습", "실행", "정리", "계획" 같은 범용 단어는 절대 포함하지 마세요
- negative_terms는 이 목표와 무관한 다른 domain의 특징적 단어입니다
- JSON 형식으로만 답하고 다른 설명은 작성하지 마세요

[출력 형식]
{{
  "goal_summary": "목표를 한 문장으로 요약",
  "core_intents": ["핵심 하위 목표 (3-5개)"],
  "evidence_terms": ["이 목표와 직접 연관된 구체적 활동/개념/키워드 (10-15개)"],
  "negative_terms": ["이 목표와 전혀 무관한 다른 domain 단어 (5-8개)"]
}}"""


# ── Data class ────────────────────────────────────────────────────────────────
@dataclass
class ExpandedQuery:
    base_query: QueryObject
    expanded_terms: list[str] = field(default_factory=list)
    negative_terms: list[str] = field(default_factory=list)
    core_intents: list[str] = field(default_factory=list)
    goal_summary: str = ""
    mode: str = "structured"

    @property
    def canonical_text(self) -> str:
        return self.base_query.canonical_text

    @property
    def full_text(self) -> str:
        return f"{self.base_query.canonical_text} {' '.join(self.expanded_terms)}".strip()

    @property
    def goal_id(self) -> str:
        return self.base_query.goal_id


# ── Heuristic fallback ────────────────────────────────────────────────────────
def _heuristic_expansion(goal: ResearchGoal, max_terms: int) -> dict:
    key = goal.title
    for kw, data in _HEURISTIC.items():
        if kw in key:
            return {
                "evidence_terms": data["evidence_terms"][:max_terms],
                "negative_terms": data["negative_terms"],
                "core_intents": [],
                "goal_summary": key,
            }
    # Last resort: use embedding text tokens (still better than "학습/실행")
    tokens = goal.goal_embedding_text.split()
    return {
        "evidence_terms": tokens[:max_terms],
        "negative_terms": [],
        "core_intents": [],
        "goal_summary": goal.title,
    }


# ── Gemini call ───────────────────────────────────────────────────────────────
def _call_gemini(goal: ResearchGoal, max_terms: int, gemini_config=None) -> dict:
    from app.llm.llm_client import get_llm_client

    llm = get_llm_client(mock=False, config=gemini_config)
    prompt = _EXPANSION_PROMPT.format(
        title=goal.title,
        description=goal.description or goal.goal_embedding_text,
    )
    response_text = llm.generate(prompt)

    match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON in Gemini response: {response_text[:200]}")

    parsed = json.loads(match.group())
    parsed["evidence_terms"] = parsed.get("evidence_terms", [])[:max_terms]
    parsed.setdefault("negative_terms", [])
    parsed.setdefault("core_intents", [])
    parsed.setdefault("goal_summary", "")
    return parsed


# ── Public API ────────────────────────────────────────────────────────────────
def expand_goal_query(
    goal: ResearchGoal,
    base_query: QueryObject,
    max_terms: int = 10,
    mode: str = "structured",
    use_mock_fallback: bool = True,
    gemini_config=None,
) -> ExpandedQuery:
    """Expand goal into retrieval evidence vocabulary via Gemini API.

    Falls back to heuristic table if API is unavailable or fails.
    """
    parsed: dict | None = None

    try:
        parsed = _call_gemini(goal, max_terms, gemini_config=gemini_config)
        logger.info(
            "Gemini expansion  goal=%s | evidence=%s | negative=%s",
            goal.goal_id, parsed["evidence_terms"], parsed["negative_terms"],
        )
    except Exception as exc:
        if use_mock_fallback:
            logger.warning(
                "Gemini expansion failed (%s) → heuristic fallback  goal=%s",
                exc, goal.goal_id,
            )
            parsed = _heuristic_expansion(goal, max_terms)
        else:
            raise

    return ExpandedQuery(
        base_query=base_query,
        expanded_terms=parsed.get("evidence_terms", []),
        negative_terms=parsed.get("negative_terms", []),
        core_intents=parsed.get("core_intents", []),
        goal_summary=parsed.get("goal_summary", ""),
        mode=mode,
    )
