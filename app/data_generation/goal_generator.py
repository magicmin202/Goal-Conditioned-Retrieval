"""Generate synthetic ResearchGoal objects per user.

Goals span 15 life domains. A global domain_counter enforces a max-per-domain
cap across the entire dataset so no single category dominates.
"""
from __future__ import annotations

import random
from app.schemas import ResearchGoal

# ── Goal pool: (category, title, description, time_horizon) ──────────────────
# 15 domains × 2-3 goals each = 37 goals total
_GOAL_POOL: list[tuple[str, str, str, str]] = [
    # career (5 slots max)
    ("career", "개발자로 취업하기",
     "소프트웨어 개발 역량을 쌓아 IT 기업에 취업한다.", "mid_term"),
    ("career", "AI 엔지니어 역량 강화",
     "수학, 머신러닝, 논문 읽기, 구현 역량을 강화한다.", "mid_term"),
    ("career", "백엔드 개발 실력 키우기",
     "Spring Boot와 DB 설계 역량을 강화해 백엔드 개발자로 성장한다.", "mid_term"),

    # education
    ("education", "대학원 진학 준비",
     "석사 과정 입학을 위해 연구실 탐색, 연구계획서 작성, 영어 공부를 한다.", "long_term"),
    ("education", "토익 900점 달성",
     "취업과 대학원 지원을 위해 토익 900점을 목표로 공부한다.", "short_term"),
    ("education", "공무원 시험 준비",
     "9급 행정직 공무원 시험을 위해 과목별 학습 계획을 수립하고 실행한다.", "long_term"),

    # health
    ("health", "규칙적으로 운동하기",
     "매주 4회 이상 운동하여 체력을 키우고 건강을 관리한다.", "long_term"),
    ("health", "식단 관리하기",
     "건강한 식습관을 유지하고 불필요한 외식을 줄인다.", "mid_term"),
    ("health", "10km 달리기 완주하기",
     "3개월 안에 10km 달리기를 완주할 수 있는 체력을 만든다.", "mid_term"),

    # relationship
    ("relationship", "연애 시작하기",
     "자기계발을 병행하며 만남의 기회를 늘려 좋은 관계를 만든다.", "mid_term"),
    ("relationship", "친구 관계 넓히기",
     "다양한 모임과 활동을 통해 인맥을 넓히고 사회성을 키운다.", "mid_term"),

    # travel
    ("travel", "저비용 해외여행 계획하기",
     "항공권과 숙소를 미리 예약하고 여행 예산을 관리하여 저비용으로 해외여행을 다녀온다.", "short_term"),
    ("travel", "국내 여행 루틴 만들기",
     "매달 1회 국내 여행을 다니며 새로운 장소를 경험한다.", "long_term"),
    ("travel", "일본 배낭여행 준비하기",
     "오사카, 도쿄, 교토를 배낭여행으로 여행하기 위한 모든 준비를 한다.", "short_term"),

    # finance
    ("finance", "월 저축 50만원 달성하기",
     "지출을 줄이고 수입을 관리하여 매달 50만원 이상 저축한다.", "mid_term"),
    ("finance", "투자 공부 시작하기",
     "주식, ETF, 부동산 등 기본 투자 지식을 익히고 소액 투자를 시작한다.", "long_term"),

    # habit
    ("habit", "독서 습관 만들기",
     "매일 30분 이상 독서를 하여 지식을 넓히고 사고력을 키운다.", "long_term"),
    ("habit", "일찍 일어나는 습관 만들기",
     "오전 6시 기상을 목표로 수면 패턴을 개선한다.", "mid_term"),
    ("habit", "명상과 마음챙김 루틴 만들기",
     "매일 10분 명상을 통해 집중력과 정서 안정을 기른다.", "long_term"),

    # hobby
    ("hobby", "사진 촬영 실력 키우기",
     "카메라 사용법을 익히고 매주 외출하여 사진을 찍고 편집한다.", "long_term"),
    ("hobby", "요리 레퍼토리 늘리기",
     "매주 새로운 요리를 시도하여 요리 실력을 키운다.", "mid_term"),

    # language
    ("language", "일본어 JLPT N3 취득하기",
     "히라가나, 가타카나, 문법, 어휘를 학습하여 JLPT N3를 합격한다.", "long_term"),
    ("language", "영어 회화 실력 높이기",
     "원어민과 자유롭게 대화할 수 있도록 스피킹과 리스닝을 강화한다.", "mid_term"),
    ("language", "스페인어 기초 마스터하기",
     "스페인어 알파벳부터 기초 회화까지 독학으로 익힌다.", "long_term"),

    # music
    ("music", "기타 코드 마스터하기",
     "기초 코드 10가지를 완벽히 익히고 좋아하는 곡을 연주할 수 있게 된다.", "mid_term"),
    ("music", "피아노 바이엘 완성하기",
     "피아노 바이엘 교본을 완성하고 간단한 곡을 칠 수 있게 된다.", "long_term"),
    ("music", "우쿨렐레 독학하기",
     "유튜브를 통해 우쿨렐레를 독학하며 3곡을 연주할 수 있게 된다.", "mid_term"),

    # art
    ("art", "수채화 기초 익히기",
     "수채화 도구 사용법과 기초 기법을 익혀 풍경화를 그릴 수 있게 된다.", "mid_term"),
    ("art", "디지털 드로잉 실력 키우기",
     "iPad와 Procreate를 활용해 캐릭터 드로잉 실력을 키운다.", "long_term"),
    ("art", "스케치 데일리 챌린지",
     "매일 하나씩 스케치하며 관찰력과 드로잉 기초를 쌓는다.", "mid_term"),

    # writing
    ("writing", "기술 블로그 꾸준히 운영하기",
     "배운 개발 지식을 주 1회 이상 블로그에 정리하여 공유한다.", "long_term"),
    ("writing", "독서 기록 일기 쓰기",
     "읽은 책마다 독후감을 써서 생각을 정리하고 지식을 내면화한다.", "long_term"),
    ("writing", "단편 소설 완성하기",
     "아이디어를 구체화하여 5000자 이상의 단편 소설을 완성한다.", "mid_term"),

    # sports
    ("sports", "클라이밍 5급 달성하기",
     "실내 클라이밍 센터에서 꾸준히 훈련하여 5급 루트를 완등한다.", "mid_term"),
    ("sports", "수영 자유형 마스터하기",
     "25m 자유형을 쉬지 않고 완주하며 기초 영법을 완성한다.", "mid_term"),
    ("sports", "테니스 기초 배우기",
     "테니스 레슨을 받으며 포핸드, 백핸드, 서브 기초를 익힌다.", "short_term"),

    # certification
    ("certification", "정보처리기사 취득하기",
     "필기와 실기 시험을 준비하여 정보처리기사 자격증을 취득한다.", "mid_term"),
    ("certification", "AWS 자격증 취득하기",
     "AWS Solutions Architect Associate 자격증을 취득하여 클라우드 역량을 증명한다.", "mid_term"),
]

# Map category → indices in _GOAL_POOL
_CATEGORY_INDEX: dict[str, list[int]] = {}
for _i, (_cat, *_) in enumerate(_GOAL_POOL):
    _CATEGORY_INDEX.setdefault(_cat, []).append(_i)

_ALL_CATEGORIES = list(_CATEGORY_INDEX.keys())

# Max goals allowed per domain across the entire dataset
DOMAIN_CAP = 5


def generate_goals_for_user(
    user_id: str,
    user_index: int,
    min_goals: int = 3,
    max_goals: int = 4,
    seed: int = 42,
    domain_counter: dict[str, int] | None = None,
) -> list[ResearchGoal]:
    """Generate goals for one user, enforcing global domain_counter cap.

    Args:
        user_id: Firestore uid string.
        user_index: Numeric index for seed offset.
        min_goals: Minimum goals per user.
        max_goals: Maximum goals per user.
        seed: Base random seed.
        domain_counter: Mutable dict tracking how many times each domain has
                        been used across all users. Updated in-place.
                        If None, no cap is enforced.
    """
    rng = random.Random(seed + user_index * 97)

    # Available categories: exclude those at cap
    available = [
        cat for cat in _ALL_CATEGORIES
        if domain_counter is None or domain_counter.get(cat, 0) < DOMAIN_CAP
    ]
    if not available:
        available = _ALL_CATEGORIES[:]  # fallback: ignore cap

    num_goals = rng.randint(min_goals, min(max_goals, len(available)))
    chosen_categories = rng.sample(available, k=num_goals)

    goals: list[ResearchGoal] = []
    for j, cat in enumerate(chosen_categories):
        candidate_indices = _CATEGORY_INDEX[cat]
        # Pick a goal not yet used too frequently (rotate through pool)
        pool_entry = _GOAL_POOL[rng.choice(candidate_indices)]
        _, title, desc, horizon = pool_entry
        gid = f"G-{user_id}-{j+1:02d}"
        goals.append(
            ResearchGoal(
                goal_id=gid,
                user_id=user_id,
                title=title,
                description=desc,
                time_horizon=horizon,
                status="active",
                created_at="2026-01-01",
            )
        )
        if domain_counter is not None:
            domain_counter[cat] = domain_counter.get(cat, 0) + 1

    return goals
