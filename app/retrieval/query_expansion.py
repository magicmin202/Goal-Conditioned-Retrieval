"""LLM-based query expansion.

Default path: Gemini API → structured JSON with 6 fields:
  goal_summary, core_intents, evidence_terms, priority_terms, related_terms, negative_terms

Fallback: domain-specific heuristic table.

Post-processing:
- Remove generic/vague terms (학습, 실행, 정리, 계획, ...)
- Deduplicate, normalize, length-cap
- Phrase-level terms preserved for phrase-match scoring in reranker/retrieval
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from app.retrieval.query_understanding import QueryObject
from app.schemas import ResearchGoal

logger = logging.getLogger(__name__)

# ── Generic terms to strip from evidence/priority/related ────────────────────
_GENERIC_TERMS: set[str] = {
    "학습", "실행", "정리", "계획", "복습", "준비", "활동", "작업",
    "수행", "진행", "완료", "시작", "관련", "내용", "방법", "과정",
    "노력", "결과", "목표", "생각", "시간", "하루", "오늘", "일상",
    "review", "study", "work", "task", "plan", "done", "기록", "확인",
}


def _normalize_term(term: str) -> str:
    return re.sub(r"\s+", " ", term.strip().lower())


def _remove_generic(terms: list[str]) -> list[str]:
    """Drop terms whose every token is in the generic set."""
    result = []
    for term in terms:
        tokens = set(re.findall(r"[\w가-힣]+", term.lower()))
        if tokens and not tokens.issubset(_GENERIC_TERMS):
            result.append(term)
    return result


def _postprocess(terms: list[str], min_count: int, max_count: int, remove_generic: bool = True) -> list[str]:
    normalized = [_normalize_term(t) for t in terms if t.strip()]
    # dedup (preserve order)
    seen: set[str] = set()
    unique = [t for t in normalized if not (t in seen or seen.add(t))]  # type: ignore[func-returns-value]
    filtered = _remove_generic(unique) if remove_generic else unique
    return filtered[:max_count]


# ── Heuristic expansion table (fallback) ─────────────────────────────────────
_HEURISTIC: dict[str, dict[str, list[str]]] = {
    "개발자": {
        "evidence_terms": [
            "코딩 테스트", "알고리즘", "자료구조", "포트폴리오", "프로젝트 구현",
            "기술 면접", "CS 공부", "문제 풀이", "코딩", "프로그래밍",
            "깃허브", "소프트웨어 개발", "백엔드 개발",
        ],
        "priority_terms": [
            "코딩 문제 풀이", "알고리즘 문제 풀이", "프로젝트 구현",
            "포트폴리오 작업", "기술 면접 준비",
        ],
        "related_terms": [
            "Python 실습", "Java 공부", "웹 개발", "API 개발",
            "디버깅", "코드 리뷰", "오픈소스",
        ],
        "negative_terms": [
            "주식 공부", "투자 전략", "ETF", "적금 계획", "재테크",
            "홈트레이닝", "산책", "여행 준비", "소개팅", "지인과 통화",
            "낮잠", "식비 정리",
        ],
    },
    "AI 엔지니어": {
        "evidence_terms": [
            "머신러닝", "딥러닝", "선형대수", "확률통계", "논문 읽기",
            "모델 구현", "PyTorch", "데이터 전처리", "실험 설계", "AI 모델",
        ],
        "priority_terms": [
            "논문 구현", "모델 구현", "딥러닝 실습", "데이터 분석", "실험 결과",
        ],
        "related_terms": [
            "Python 코딩", "수학 공부", "모델 학습", "데이터셋 구축", "코드 작성",
        ],
        "negative_terms": [
            "주식 공부", "투자 전략", "여행 준비", "소개팅", "홈트레이닝",
            "요리 실습", "낮잠", "지인 약속",
        ],
    },
    "대학원": {
        "evidence_terms": [
            "연구계획서", "논문 읽기", "연구실 탐색", "지도교수 연락", "GRE",
            "영어 성적", "연구 주제 선정", "세미나 참석", "학술 논문",
        ],
        "priority_terms": [
            "연구계획서 작성", "지도교수 컨택", "논문 리뷰", "연구 주제 탐색",
        ],
        "related_terms": [
            "영어 공부", "독서", "학교 자료 조사", "발표 자료 작성",
        ],
        "negative_terms": [
            "주식 공부", "투자 전략", "여행 준비", "홈트레이닝", "요리 실습",
            "게임", "낮잠", "소개팅",
        ],
    },
    "토익": {
        "evidence_terms": [
            "리스닝 연습", "리딩 문제", "어휘 암기", "문법 공부", "RC 풀이",
            "LC 풀이", "실전 모의고사", "영어 단어",
        ],
        "priority_terms": [
            "토익 모의고사", "LC 청취 연습", "RC 독해 연습", "영어 단어 암기",
        ],
        "related_terms": [
            "영어 회화", "영어 작문", "영어 문법 정리",
        ],
        "negative_terms": [
            "주식 공부", "홈트레이닝", "여행 준비", "소개팅", "요리 실습",
            "코딩 테스트", "낮잠", "지인 통화",
        ],
    },
    "운동": {
        "evidence_terms": [
            "헬스장 방문", "러닝", "근력 운동", "유산소 운동", "스트레칭",
            "홈트레이닝", "운동 루틴", "벤치프레스", "스쿼트", "체중 측정",
        ],
        "priority_terms": [
            "헬스장 운동", "근력 운동 세션", "러닝 완료", "운동 루틴 수행",
        ],
        "related_terms": [
            "식단 관리", "단백질 섭취", "체중 기록", "운동 일지",
        ],
        "negative_terms": [
            "주식 공부", "투자 전략", "코딩 테스트", "논문 읽기",
            "소개팅", "낮잠", "지인 통화",
        ],
    },
    "식단": {
        "evidence_terms": [
            "칼로리 계산", "채소 식단", "단백질 식품", "다이어트 식단",
            "건강식 요리", "식재료 준비", "식단 기록",
        ],
        "priority_terms": [
            "식단 기록", "칼로리 계산", "건강식 준비", "다이어트 식단 유지",
        ],
        "related_terms": [
            "요리 실습", "외식 기록", "영양 정보 확인",
        ],
        "negative_terms": [
            "주식 공부", "코딩 테스트", "여행 준비", "소개팅", "낮잠",
        ],
    },
    "연애": {
        "evidence_terms": [
            "소개팅", "데이트 계획", "이성 만남", "대화 연습", "관계 개선",
            "연락 시도", "마음 표현",
        ],
        "priority_terms": [
            "소개팅 참석", "데이트 계획 수립", "이성과 연락",
        ],
        "related_terms": [
            "자기 개발", "외모 관리", "취미 공유",
        ],
        "negative_terms": [
            "주식 공부", "코딩 테스트", "운동 루틴", "논문 읽기", "낮잠",
        ],
    },
    "여행": {
        "evidence_terms": [
            "항공권 예약", "숙소 예약", "여행 일정 작성", "여행 예산 계획",
            "짐 준비", "여행지 조사", "환전",
        ],
        "priority_terms": [
            "항공권 예약 완료", "숙소 예약 완료", "여행 일정 확정",
        ],
        "related_terms": [
            "여행 후기 작성", "여행 사진 정리", "지도 검색",
        ],
        "negative_terms": [
            "주식 공부", "코딩 테스트", "알고리즘 풀이", "홈트레이닝",
            "논문 읽기", "낮잠",
        ],
    },
    "저축": {
        "evidence_terms": [
            "가계부 작성", "지출 기록", "적금 납입", "예산 설정",
            "절약 실천", "수입 정리", "소비 패턴 분석",
        ],
        "priority_terms": [
            "가계부 작성", "적금 납입", "월 예산 설정", "지출 절약",
        ],
        "related_terms": [
            "재테크 공부", "소비 습관 점검", "신용카드 사용 기록",
        ],
        "negative_terms": [
            "코딩 테스트", "알고리즘 풀이", "여행 준비", "소개팅", "낮잠",
        ],
    },
    "투자": {
        "evidence_terms": [
            "주식 분석", "ETF 조사", "부동산 시장 조사", "재테크 공부",
            "경제 뉴스", "포트폴리오 구성", "배당주 분석",
        ],
        "priority_terms": [
            "주식 매수 검토", "ETF 포트폴리오 점검", "경제 공부", "투자 전략 수립",
        ],
        "related_terms": [
            "금융 지식 습득", "절약 실천", "가계부 작성",
        ],
        "negative_terms": [
            "코딩 테스트", "알고리즘 풀이", "여행 준비", "소개팅",
            "운동 루틴", "낮잠",
        ],
    },
    "독서": {
        "evidence_terms": [
            "책 읽기", "독후감 작성", "서평", "독서 목록 관리",
            "도서 구입", "책 요약",
        ],
        "priority_terms": [
            "독서 완료", "독후감 작성", "책 내용 정리",
        ],
        "related_terms": [
            "글쓰기", "지식 메모", "인용구 기록",
        ],
        "negative_terms": [
            "주식 공부", "홈트레이닝", "여행 준비", "소개팅", "낮잠",
        ],
    },
    "기상": {
        "evidence_terms": [
            "아침 루틴", "수면 기록", "기상 시간 기록", "모닝 루틴",
            "오전 활동 시작",
        ],
        "priority_terms": [
            "목표 기상 시간 달성", "아침 루틴 완료", "일찍 기상 성공",
        ],
        "related_terms": [
            "수면 시간 단축", "오전 운동", "아침 식사",
        ],
        "negative_terms": [
            "주식 공부", "코딩 테스트", "여행 준비", "소개팅",
        ],
    },
    "사진": {
        "evidence_terms": [
            "촬영 세션", "카메라 사용", "사진 편집", "구도 연습",
            "라이트룸 편집", "포토샵 작업",
        ],
        "priority_terms": [
            "촬영 완료", "사진 편집 완료", "포트폴리오 사진 선정",
        ],
        "related_terms": [
            "카메라 공부", "편집 기술 습득", "사진 인화",
        ],
        "negative_terms": [
            "주식 공부", "코딩 테스트", "운동 루틴", "소개팅", "낮잠",
        ],
    },
    "요리": {
        "evidence_terms": [
            "레시피 실습", "요리 완성", "식재료 손질", "홈쿡 세션",
            "조리 기술 연습", "요리 기록",
        ],
        "priority_terms": [
            "새 레시피 도전", "요리 완성", "홈쿡 완료",
        ],
        "related_terms": [
            "식재료 구입", "레시피 검색", "음식 사진 촬영",
        ],
        "negative_terms": [
            "주식 공부", "코딩 테스트", "알고리즘 풀이", "여행 준비", "낮잠",
        ],
    },
}

# ── Gemini expansion prompt ───────────────────────────────────────────────────
_EXPANSION_PROMPT = """당신은 개인 목표 기반 정보 검색(Goal-Conditioned IR) 전문가입니다.

사용자의 목표가 주어졌을 때, 실제 행동 로그(diary/log entries)에서 관련 기록을 찾기 위한 vocabulary를 생성하세요.

[목표]
제목: {title}
설명: {description}

[생성 원칙]
1. 이 목표를 위해 실제로 할 수 있는 구체적 하위 행동을 떠올려라
2. 실제 사용자 로그의 title/content에 등장할 법한 표현으로 작성하라
3. "학습", "실행", "정리", "계획", "복습" 같은 지나치게 일반적인 단어는 절대 포함하지 마라
4. priority_terms는 이 목표를 가장 강하게 식별하는 핵심 표현 (4-8개)
5. evidence_terms는 직접 관련된 구체적 활동/개념/키워드 (8-15개)
6. related_terms는 간접 관련 또는 유사 표현 (5-10개)
7. negative_terms는 이 목표와 무관한 다른 생활 도메인의 로그 표현 (8-15개)
   - 금융(주식, 투자, 적금), 건강/운동, 여행, 연애/사교, 일상 잡무 등 포함
   - 단어보다 phrase 수준 표현 권장 (예: "주식 시장 공부", "헬스장 운동")

[중요] JSON 형식으로만 답하고 다른 설명은 작성하지 마세요.

[출력 형식]
{{
  "goal_summary": "목표를 한 문장으로 요약",
  "core_intents": ["핵심 하위 목표 (3-5개)"],
  "evidence_terms": ["이 목표와 직접 연관된 구체적 활동/개념/키워드 (8-15개)"],
  "priority_terms": ["top-k retrieval에서 강하게 우선시해야 할 핵심 표현 (4-8개)"],
  "related_terms": ["유사 표현, 확장 표현, 간접 관련 표현 (5-10개)"],
  "negative_terms": ["이 목표와 무관한 다른 생활 도메인의 로그 표현 (8-15개)"]
}}"""


# ── Data class ────────────────────────────────────────────────────────────────
@dataclass
class ExpandedQuery:
    base_query: QueryObject
    expanded_terms: list[str] = field(default_factory=list)   # evidence_terms
    priority_terms: list[str] = field(default_factory=list)
    related_terms: list[str] = field(default_factory=list)
    negative_terms: list[str] = field(default_factory=list)
    core_intents: list[str] = field(default_factory=list)
    goal_summary: str = ""
    mode: str = "structured"

    @property
    def canonical_text(self) -> str:
        return self.base_query.canonical_text

    @property
    def full_text(self) -> str:
        all_terms = self.priority_terms + self.expanded_terms + self.related_terms
        return f"{self.base_query.canonical_text} {' '.join(all_terms)}".strip()

    @property
    def goal_id(self) -> str:
        return self.base_query.goal_id


# ── Heuristic fallback ────────────────────────────────────────────────────────
def _heuristic_expansion(goal: ResearchGoal, max_terms: int) -> dict:
    key = goal.title
    for kw, data in _HEURISTIC.items():
        if kw in key:
            evidence = _postprocess(data["evidence_terms"], 0, max_terms)
            priority = _postprocess(data.get("priority_terms", evidence[:5]), 0, 8)
            related = _postprocess(data.get("related_terms", []), 0, 10)
            negative = _postprocess(data.get("negative_terms", []), 0, 15, remove_generic=False)
            return {
                "evidence_terms": evidence,
                "priority_terms": priority,
                "related_terms": related,
                "negative_terms": negative,
                "core_intents": [],
                "goal_summary": key,
            }
    # Last resort: use embedding text tokens
    tokens = goal.goal_embedding_text.split()
    evidence = _postprocess(tokens, 0, max_terms)
    return {
        "evidence_terms": evidence,
        "priority_terms": evidence[:5],
        "related_terms": [],
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

    # Post-process all term lists
    evidence = _postprocess(parsed.get("evidence_terms", []), 0, max_terms)
    priority = _postprocess(parsed.get("priority_terms", evidence[:5]), 0, 8)
    related = _postprocess(parsed.get("related_terms", []), 0, 10)
    # Negative terms: keep phrases but still normalize/dedup
    negative = _postprocess(
        parsed.get("negative_terms", []), 0, 15, remove_generic=False
    )

    removed_evidence = [t for t in parsed.get("evidence_terms", []) if _normalize_term(t) not in evidence]
    removed_priority = [t for t in parsed.get("priority_terms", []) if _normalize_term(t) not in priority]
    if removed_evidence or removed_priority:
        logger.debug(
            "Generic terms removed  evidence_dropped=%s  priority_dropped=%s",
            removed_evidence, removed_priority,
        )

    return {
        "evidence_terms": evidence,
        "priority_terms": priority,
        "related_terms": related,
        "negative_terms": negative,
        "core_intents": parsed.get("core_intents", []),
        "goal_summary": parsed.get("goal_summary", ""),
    }


# ── Public API ────────────────────────────────────────────────────────────────
def expand_goal_query(
    goal: ResearchGoal,
    base_query: QueryObject,
    max_terms: int = 15,
    mode: str = "structured",
    use_mock_fallback: bool = True,
    gemini_config=None,
) -> ExpandedQuery:
    """Expand goal into structured retrieval vocabulary via Gemini API.

    Returns ExpandedQuery with priority_terms / evidence_terms / related_terms
    / negative_terms for use in candidate retrieval and reranking.
    """
    parsed: dict | None = None

    try:
        parsed = _call_gemini(goal, max_terms, gemini_config=gemini_config)
        logger.info(
            "Gemini expansion  goal=%s\n"
            "  priority=%s\n"
            "  evidence=%s\n"
            "  related=%s\n"
            "  negative=%s",
            goal.goal_id,
            parsed["priority_terms"],
            parsed["evidence_terms"],
            parsed["related_terms"],
            parsed["negative_terms"],
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
        priority_terms=parsed.get("priority_terms", parsed.get("evidence_terms", [])[:5]),
        related_terms=parsed.get("related_terms", []),
        negative_terms=parsed.get("negative_terms", []),
        core_intents=parsed.get("core_intents", []),
        goal_summary=parsed.get("goal_summary", ""),
        mode=mode,
    )
