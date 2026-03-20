"""Generate synthetic ResearchGoal objects per user.

Goals span diverse life domains so that the retrieval problem
involves discriminating between unrelated goal contexts.
"""
from __future__ import annotations

import random
from app.schemas import ResearchGoal

# Each entry: (category, title, description, embedding_text, time_horizon)
_GOAL_POOL: list[tuple[str, str, str, str, str]] = [
    # career
    (
        "career",
        "개발자로 취업하기",
        "소프트웨어 개발 역량을 쌓아 IT 기업에 취업한다.",
        "개발자 취업 코딩 알고리즘 포트폴리오 기술 면접",
        "mid_term",
    ),
    (
        "career",
        "AI 엔지니어 역량 강화",
        "수학, 머신러닝, 논문 읽기, 구현 역량을 강화한다.",
        "AI 개발 역량 수학 머신러닝 딥러닝 논문 구현",
        "mid_term",
    ),
    # education
    (
        "education",
        "대학원 진학 준비",
        "석사 과정 입학을 위해 연구실 탐색, 연구계획서 작성, 영어 공부를 한다.",
        "대학원 석사 연구 지도교수 연구계획서 GRE TOEFL",
        "long_term",
    ),
    (
        "education",
        "토익 900점 달성",
        "취업과 대학원 지원을 위해 토익 900점을 목표로 공부한다.",
        "토익 영어 리스닝 리딩 어휘 문법",
        "short_term",
    ),
    # health
    (
        "health",
        "규칙적으로 운동하기",
        "매주 4회 이상 운동하여 체력을 키우고 건강을 관리한다.",
        "운동 헬스 유산소 근력 스트레칭 체력",
        "long_term",
    ),
    (
        "health",
        "식단 관리하기",
        "건강한 식습관을 유지하고 불필요한 외식을 줄인다.",
        "식단 다이어트 건강식 채소 단백질 칼로리",
        "mid_term",
    ),
    # relationship
    (
        "relationship",
        "연애 시작하기",
        "자기계발을 병행하며 만남의 기회를 늘려 좋은 관계를 만든다.",
        "연애 이성 만남 소개팅 관계 소통",
        "mid_term",
    ),
    (
        "relationship",
        "친구 관계 넓히기",
        "다양한 모임과 활동을 통해 인맥을 넓히고 사회성을 키운다.",
        "친구 인맥 모임 네트워킹 사교 활동",
        "mid_term",
    ),
    # travel
    (
        "travel",
        "저비용 해외여행 계획하기",
        "항공권과 숙소를 미리 예약하고 여행 예산을 관리하여 저비용으로 해외여행을 다녀온다.",
        "여행 해외 항공권 숙소 예산 배낭 관광",
        "short_term",
    ),
    (
        "travel",
        "국내 여행 루틴 만들기",
        "매달 1회 국내 여행을 다니며 새로운 장소를 경험한다.",
        "국내여행 당일치기 숙소 카페 맛집 사진",
        "long_term",
    ),
    # finance
    (
        "finance",
        "월 저축 50만원 달성하기",
        "지출을 줄이고 수입을 관리하여 매달 50만원 이상 저축한다.",
        "저축 절약 지출 가계부 재테크 적금",
        "mid_term",
    ),
    (
        "finance",
        "투자 공부 시작하기",
        "주식, ETF, 부동산 등 기본 투자 지식을 익히고 소액 투자를 시작한다.",
        "투자 주식 ETF 부동산 재테크 경제 공부",
        "long_term",
    ),
    # habit
    (
        "habit",
        "독서 습관 만들기",
        "매일 30분 이상 독서를 하여 지식을 넓히고 사고력을 키운다.",
        "독서 책 습관 자기계발 지식 독후감",
        "long_term",
    ),
    (
        "habit",
        "일찍 일어나는 습관 만들기",
        "오전 6시 기상을 목표로 수면 패턴을 개선한다.",
        "기상 아침 루틴 수면 습관 생산성",
        "mid_term",
    ),
    # hobby
    (
        "hobby",
        "사진 촬영 실력 키우기",
        "카메라 사용법을 익히고 매주 외출하여 사진을 찍고 편집한다.",
        "사진 카메라 촬영 편집 포토샵 구도",
        "long_term",
    ),
    (
        "hobby",
        "요리 레퍼토리 늘리기",
        "매주 새로운 요리를 시도하여 요리 실력을 키운다.",
        "요리 레시피 요리법 식재료 쿠킹 홈쿡",
        "mid_term",
    ),
]

# Map category → indices in _GOAL_POOL
_CATEGORY_INDEX: dict[str, list[int]] = {}
for _i, (_cat, *_) in enumerate(_GOAL_POOL):
    _CATEGORY_INDEX.setdefault(_cat, []).append(_i)

_ALL_CATEGORIES = list(_CATEGORY_INDEX.keys())


def generate_goals_for_user(
    user_id: str,
    user_index: int,
    min_goals: int = 3,
    max_goals: int = 5,
    seed: int = 42,
) -> list[ResearchGoal]:
    """Generate 3-5 goals for a user, each from a different category.

    Args:
        user_id: Firestore uid string.
        user_index: Numeric index used to offset seeds.
        min_goals: Minimum goals per user.
        max_goals: Maximum goals per user.
        seed: Base random seed.
    """
    rng = random.Random(seed + user_index * 97)
    num_goals = rng.randint(min_goals, max_goals)
    chosen_categories = rng.sample(_ALL_CATEGORIES, k=min(num_goals, len(_ALL_CATEGORIES)))

    goals: list[ResearchGoal] = []
    for j, cat in enumerate(chosen_categories):
        candidate_indices = _CATEGORY_INDEX[cat]
        pool_entry = _GOAL_POOL[rng.choice(candidate_indices)]
        _, title, desc, emb_text, horizon = pool_entry
        gid = f"G-{user_id}-{j+1:02d}"
        goals.append(
            ResearchGoal(
                goal_id=gid,
                user_id=user_id,
                title=title,
                description=desc,
                time_horizon=horizon,
                status="active",
                goal_embedding_text=emb_text,
                created_at="2026-03-01",
            )
        )
    return goals
