"""LLM-based query expansion.

Default path: Gemini API -> goal-specific evidence vocabulary.
Fallback: heuristic lookup table with goal-specific terms.

Key rule: generate SPECIFIC evidence terms for the goal domain,
NOT generic productivity words like "학습", "실행", "정리".
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from app.retrieval.query_understanding import QueryObject
from app.schemas import ResearchGoal

logger = logging.getLogger(__name__)

_HEURISTIC_EXPANSIONS: dict[str, list[str]] = {
    "개발자": [
        "코딩 테스트", "알고리즘", "자료구조", "포트폴리오",
        "프로젝트 구현", "기술 면접", "Python", "백엔드", "CS 공부", "문제 풀이",
    ],
    "AI 엔지니어": [
        "머신러닝", "딥러닝", "선형대수", "확률통계",
        "논문 읽기", "모델 구현", "PyTorch", "데이터 전처리", "실험 설계", "코드 구현",
    ],
    "취업": [
        "자기소개서", "이력서", "면접 준비", "포트폴리오",
        "코딩 테스트", "직무 분석", "기업 조사", "지원서 작성",
    ],
    "대학원": [
        "연구계획서", "논문 읽기", "연구실 탐색", "지도교수",
        "GRE", "영어 성적", "연구 주제", "세미나", "학술 자료",
    ],
    "토익": [
        "영어 리스닝", "영어 리딩", "어휘 암기", "LC RC",
        "실전 문제", "오답 정리", "파트별 전략", "모의고사",
    ],
    "운동": [
        "헬스장", "러닝", "근력 운동", "유산소",
        "스트레칭", "홈트", "운동 루틴", "세트 반복", "체중 관리",
    ],
    "식단": [
        "칼로리", "채소", "단백질", "다이어트",
        "건강식", "외식 줄이기", "식단 일지", "영양소",
    ],
    "연애": [
        "소개팅", "데이트", "이성 만남", "대화 주제",
        "관계 발전", "자기 관리", "외모 관리",
    ],
    "친구": [
        "모임", "약속", "네트워킹", "사교 활동",
        "동호회", "커뮤니티", "인맥",
    ],
    "여행": [
        "항공권", "숙소 예약", "여행 계획", "예산 관리",
        "짐 준비", "여행지 조사", "비자", "환전",
    ],
    "저축": [
        "가계부", "지출 관리", "적금", "예산",
        "절약", "고정비 줄이기", "소비 패턴",
    ],
    "투자": [
        "주식", "ETF", "부동산", "재테크",
        "경제 공부", "투자 포트폴리오", "배당", "리밸런싱",
    ],
    "독서": [
        "책 읽기", "독후감", "서평", "독서 목록",
        "밑줄 긋기", "요약 정리", "책 추천",
    ],
    "기상": [
        "아침 루틴", "수면 패턴", "기상 시간",
        "생산성", "조기 기상", "오전 루틴", "기상 성공",
    ],
    "사진": [
        "촬영 외출", "카메라 설정", "구도",
        "라이트룸 편집", "포토샵", "사진 업로드",
    ],
    "요리": [
        "레시피", "요리 실습", "식재료 준비",
        "쿠킹 클래스", "새 요리 시도", "홈쿡",
    ],
}

_EXPANSION_PROMPT = (
    "당신은 개인 목표 관리 시스템의 정보 검색 전문가입니다.\n\n"
    "사용자의 목표가 주어졌을 때, 그 목표와 직접 관련된 행동 로그를 검색하기 위한\n"
    "retrieval evidence vocabulary를 생성하세요.\n\n"
    "## 목표\n"
    "- 제목: {title}\n"
    "- 설명: {description}\n\n"
    "## 규칙\n"
    "- '학습', '실행', '정리', '복습', '계획' 같은 일반적인 단어는 절대 생성하지 마세요.\n"
    "- 이 목표 도메인에 특화된 구체적인 하위 행동, 하위 개념, 증거 키워드만 생성하세요.\n"
    "- 예: '개발자로 취업하기' -> '코딩 테스트', '알고리즘', '포트폴리오 구현', '기술 면접'\n\n"
    "## 출력 형식 (JSON만 출력)\n"
    "{{\n"
    '  "goal_summary": "한 문장 요약",\n'
    '  "subgoals": ["세부목표1", "세부목표2"],\n'
    '  "evidence_terms": ["증거키워드1", "증거키워드2", "증거키워드3"],\n'
    '  "related_actions": ["관련행동1", "관련행동2", "관련행동3"]\n'
    "}}"
)


@dataclass
class ExpandedQuery:
    base_query: QueryObject
    expanded_terms: list[str] = field(default_factory=list)
    mode: str = "structured"
    goal_summary: str = ""
    subgoals: list[str] = field(default_factory=list)
    related_actions: list[str] = field(default_factory=list)

    @property
    def canonical_text(self) -> str:
        return self.base_query.canonical_text

    @property
    def full_text(self) -> str:
        all_terms = self.expanded_terms + self.subgoals + self.related_actions
        unique = list(dict.fromkeys(all_terms))
        return f"{self.base_query.canonical_text} {' '.join(unique)}".strip()

    @property
    def goal_id(self) -> str:
        return self.base_query.goal_id

    @property
    def all_expansion_terms(self) -> list[str]:
        """All unique expansion terms used by reranker for goal_focus scoring."""
        all_terms = self.expanded_terms + self.subgoals + self.related_actions
        return list(dict.fromkeys(all_terms))


def _heuristic_expand(goal: ResearchGoal, max_terms: int) -> ExpandedQuery:
    """Fallback: keyword-based heuristic expansion with goal-specific terms."""
    base_query = QueryObject(
        raw_text=goal.query_text,
        canonical_text=goal.query_text.lower(),
        goal_id=goal.goal_id,
    )
    title_lower = goal.title.lower()
    terms: list[str] = []
    for kw, kw_terms in _HEURISTIC_EXPANSIONS.items():
        if kw in title_lower:
            terms.extend(kw_terms)
            if len(terms) >= max_terms:
                break

    if not terms:
        emb_tokens = re.findall(r"[\w\uAC00-\uD7A3]{2,}", goal.goal_embedding_text or goal.title)
        terms = emb_tokens[:max_terms]

    terms = list(dict.fromkeys(terms))[:max_terms]
    logger.debug("Heuristic expansion for goal=%s: %s", goal.goal_id, terms)
    return ExpandedQuery(base_query=base_query, expanded_terms=terms, mode="heuristic")


def _gemini_expand(goal: ResearchGoal, max_terms: int, config=None) -> ExpandedQuery:
    """Call Gemini API to generate goal-specific expansion."""
    from app.llm.llm_client import GeminiLLMClient
    from app.config import DEFAULT_CONFIG

    cfg = config or DEFAULT_CONFIG.gemini
    client = GeminiLLMClient(
        api_key=cfg.api_key,
        model_name=cfg.model_name,
        temperature=cfg.temperature,
        max_output_tokens=cfg.max_output_tokens,
    )

    prompt = _EXPANSION_PROMPT.format(title=goal.title, description=goal.description)
    raw = client.generate(prompt)

    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not json_match:
        raise ValueError(f"No JSON found in Gemini response: {raw[:300]}")

    parsed = json.loads(json_match.group())
    base_query = QueryObject(
        raw_text=goal.query_text,
        canonical_text=goal.query_text.lower(),
        goal_id=goal.goal_id,
    )

    evidence_terms = parsed.get("evidence_terms", [])[:max_terms]
    subgoals = parsed.get("subgoals", [])[:5]
    related_actions = parsed.get("related_actions", [])[:5]

    logger.info("Gemini expansion for goal=%s | evidence_terms=%s", goal.goal_id, evidence_terms)
    return ExpandedQuery(
        base_query=base_query,
        expanded_terms=evidence_terms,
        subgoals=subgoals,
        related_actions=related_actions,
        goal_summary=parsed.get("goal_summary", ""),
        mode="structured",
    )


def expand_goal_query(
    goal: ResearchGoal,
    base_query: QueryObject,
    max_terms: int = 10,
    mode: str = "structured",
    config=None,
) -> ExpandedQuery:
    """Expand goal into goal-specific evidence vocabulary.

    Flow:
      1. mode='simple' -> heuristic only.
      2. Gemini API key present -> call Gemini.
      3. On failure or no key -> heuristic fallback.
    """
    from app.config import DEFAULT_CONFIG
    cfg = config or DEFAULT_CONFIG.gemini

    if mode == "simple":
        return _heuristic_expand(goal, max_terms)

    if cfg.api_key:
        try:
            return _gemini_expand(goal, max_terms, config=cfg)
        except Exception as exc:
            if cfg.use_mock_fallback:
                logger.warning(
                    "Gemini expansion failed for goal=%s (%s). Using heuristic fallback.",
                    goal.goal_id, exc,
                )
            else:
                raise

    return _heuristic_expand(goal, max_terms)
