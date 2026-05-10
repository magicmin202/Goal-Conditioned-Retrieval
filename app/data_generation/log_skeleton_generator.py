"""Generate day-level log skeletons for a user.

Key design decisions:
- Logs are day-level records.
- The monthly log stream mixes activities tied to different goals + noise logs.
- Repeated activity blocks simulate realistic learning bursts.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date, timedelta

from app.schemas import ResearchGoal


@dataclass
class LogSkeleton:
    """Intermediate representation before text rendering."""
    log_id: str
    user_id: str
    date: str
    activity_type: str
    goal_id: str | None
    topic: str
    evidence_strength: str
    metadata: dict = field(default_factory=dict)


# ── Activity templates per goal category ─────────────────────────────────────
# Each entry: (activity_type, topic, evidence_strength)
_GOAL_ACTIVITIES: dict[str, list[tuple[str, str, str]]] = {
    "career": [
        ("study",          "알고리즘 문제 풀이",       "high"),
        ("implementation", "포트폴리오 프로젝트 구현",  "high"),
        ("study",          "자료구조 개념 정리",        "medium"),
        ("planning",       "취업 계획 수립",            "low"),
        ("reading",        "개발 기술 블로그 읽기",     "medium"),
        ("study",          "코딩 테스트 연습",          "high"),
        ("implementation", "백엔드 API 구현",           "high"),
        ("study",          "시스템 설계 학습",          "medium"),
    ],
    "education": [
        ("study",    "논문 읽기",              "high"),
        ("planning", "연구계획서 작성",        "high"),
        ("study",    "영어 단어 암기",         "medium"),
        ("reading",  "교수님 연구 자료 정리",  "medium"),
        ("planning", "연구실 탐색",            "low"),
        ("study",    "GRE 문제 풀이",          "high"),
        ("study",    "공무원 과목 학습",       "high"),
        ("study",    "기출문제 풀이",          "high"),
        ("planning", "시험 스케줄 관리",       "medium"),
    ],
    "health": [
        ("exercise", "헬스장 운동",     "high"),
        ("exercise", "러닝",            "high"),
        ("exercise", "스트레칭",        "medium"),
        ("planning", "운동 계획 작성",  "low"),
        ("exercise", "홈트레이닝",      "medium"),
        ("exercise", "수영 연습",       "high"),
        ("exercise", "야외 달리기",     "high"),
    ],
    "relationship": [
        ("social",   "소개팅",           "high"),
        ("social",   "친구 모임",        "medium"),
        ("planning", "대화 주제 정리",   "low"),
        ("social",   "데이트 계획 수립", "medium"),
        ("social",   "네트워킹 모임 참석", "medium"),
    ],
    "travel": [
        ("planning",  "항공권 검색",      "high"),
        ("planning",  "숙소 예약",        "high"),
        ("reading",   "여행 후기 조사",   "medium"),
        ("planning",  "여행 예산 정리",   "medium"),
        ("execution", "여행 준비물 구매", "high"),
        ("reading",   "현지 맛집 조사",   "medium"),
        ("planning",  "여행 일정 계획",   "high"),
    ],
    "finance": [
        ("study",     "가계부 정리",      "high"),
        ("study",     "ETF 공부",         "medium"),
        ("planning",  "저축 계획 작성",   "low"),
        ("study",     "주식 기초 학습",   "medium"),
        ("execution", "소액 투자 실행",   "high"),
        ("study",     "재무제표 읽기",    "medium"),
        ("planning",  "월간 예산 설정",   "medium"),
    ],
    "habit": [
        ("reading",   "독서",             "high"),
        ("planning",  "독서 목록 작성",   "low"),
        ("execution", "아침 루틴 실행",   "high"),
        ("reflection","하루 반성 일지",   "medium"),
        ("execution", "명상 실행",        "high"),
        ("reflection","감사 일기 작성",   "medium"),
    ],
    "hobby": [
        ("execution", "촬영 외출",        "high"),
        ("study",     "사진 편집 연습",   "medium"),
        ("execution", "요리 실습",        "high"),
        ("reading",   "레시피 조사",      "medium"),
        ("execution", "그림 그리기",      "high"),
    ],
    "language": [
        ("study",     "외국어 단어 암기",  "high"),
        ("study",     "문법 학습",         "medium"),
        ("execution", "회화 연습",         "high"),
        ("study",     "교재 학습",         "medium"),
        ("execution", "언어 교환 모임",    "high"),
        ("study",     "어휘 복습",         "medium"),
        ("execution", "듣기 연습",         "medium"),
    ],
    "music": [
        ("execution", "악기 연습",         "high"),
        ("study",     "악보 학습",         "medium"),
        ("execution", "곡 연주 연습",      "high"),
        ("study",     "음악 이론 공부",    "low"),
        ("execution", "레슨 수강",         "high"),
        ("execution", "합주 연습",         "medium"),
    ],
    "art": [
        ("execution", "그림 그리기 연습",  "high"),
        ("study",     "드로잉 기법 학습",  "medium"),
        ("execution", "야외 스케치",       "high"),
        ("reading",   "미술 서적 읽기",    "low"),
        ("execution", "디지털 드로잉",     "high"),
        ("study",     "색채 이론 공부",    "medium"),
    ],
    "writing": [
        ("execution", "블로그 글 작성",    "high"),
        ("execution", "독후감 작성",       "high"),
        ("study",     "글쓰기 기법 학습",  "medium"),
        ("execution", "소설 집필",         "high"),
        ("planning",  "글쓰기 주제 구상",  "low"),
        ("reflection","글쓰기 피드백 반영","medium"),
    ],
    "sports": [
        ("execution", "스포츠 훈련",       "high"),
        ("execution", "레슨 참여",         "high"),
        ("study",     "스포츠 기술 학습",  "medium"),
        ("execution", "실전 연습",         "high"),
        ("planning",  "훈련 계획 작성",    "low"),
    ],
    "certification": [
        ("study",     "자격증 과목 학습",  "high"),
        ("study",     "기출문제 풀이",     "high"),
        ("planning",  "시험 일정 관리",    "low"),
        ("study",     "오답 노트 정리",    "medium"),
        ("execution", "모의고사 응시",     "high"),
    ],
}

_NOISE_ACTIVITIES: list[tuple[str, str]] = [
    ("daily", "점심 식사"),
    ("daily", "카페 방문"),
    ("daily", "친구와 통화"),
    ("daily", "유튜브 시청"),
    ("daily", "산책"),
    ("daily", "청소"),
    ("daily", "마트 장보기"),
    ("daily", "드라마 시청"),
    ("daily", "낮잠"),
    ("daily", "SNS 확인"),
    ("daily", "배달 주문"),
    ("daily", "외식"),
    ("daily", "게임"),
    ("daily", "빨래"),
    ("daily", "친구 만남"),
]

# Category lookup for goals
_CATEGORY_KEYWORDS: dict[str, str] = {
    "개발자": "career", "AI 엔지니어": "career", "취업": "career", "백엔드": "career",
    "대학원": "education", "토익": "education", "공무원": "education",
    "운동": "health", "식단": "health", "달리기": "health",
    "연애": "relationship", "친구 관계": "relationship",
    "여행": "travel", "배낭여행": "travel", "일본": "travel",
    "저축": "finance", "투자": "finance",
    "독서": "habit", "기상": "habit", "명상": "habit",
    "사진": "hobby", "요리": "hobby",
    "일본어": "language", "영어 회화": "language", "스페인어": "language",
    "기타": "music", "피아노": "music", "우쿨렐레": "music",
    "수채화": "art", "드로잉": "art", "스케치": "art",
    "블로그": "writing", "소설": "writing", "독서 기록": "writing",
    "클라이밍": "sports", "수영": "sports", "테니스": "sports",
    "정보처리기사": "certification", "AWS": "certification",
}


def _infer_category(goal: ResearchGoal) -> str:
    for kw, cat in _CATEGORY_KEYWORDS.items():
        if kw in goal.title:
            return cat
    return "career"


def _date_range(start: date, end: date) -> list[date]:
    days = (end - start).days + 1
    return [start + timedelta(days=i) for i in range(days)]


def generate_log_skeletons_for_user(
    user_id: str,
    user_index: int,
    goals: list[ResearchGoal],
    start_date: date,
    end_date: date,
    min_logs: int = 60,
    max_logs: int = 80,
    seed: int = 42,
) -> list[LogSkeleton]:
    """Generate day-level log skeletons for one user."""
    rng = random.Random(seed + user_index * 131)
    all_dates = _date_range(start_date, end_date)
    target_count = rng.randint(min_logs, max_logs)

    date_cursor = list(all_dates)
    rng.shuffle(date_cursor)

    goal_log_budget = int(target_count * 0.60)
    noise_budget = target_count - goal_log_budget

    goal_skeletons: list[LogSkeleton] = []
    for goal in goals:
        cat = _infer_category(goal)
        activities = _GOAL_ACTIVITIES.get(cat, _GOAL_ACTIVITIES["career"])
        per_goal = max(2, goal_log_budget // len(goals))

        days_used: list[date] = sorted(
            rng.sample(date_cursor[: len(date_cursor) // 2], k=min(per_goal, len(date_cursor) // 3))
        )
        block_start = 0
        while block_start < len(days_used):
            act_type, topic, strength = rng.choice(activities)
            block_len = rng.randint(2, 4)
            for d in days_used[block_start: block_start + block_len]:
                goal_skeletons.append(LogSkeleton(
                    log_id="",
                    user_id=user_id,
                    date=d.isoformat(),
                    activity_type=act_type,
                    goal_id=goal.goal_id,
                    topic=topic,
                    evidence_strength=strength,
                ))
            block_start += block_len

    rng.shuffle(goal_skeletons)
    goal_skeletons = goal_skeletons[:goal_log_budget]

    noise_skeletons: list[LogSkeleton] = []
    noise_dates = sorted(rng.sample(date_cursor, k=min(noise_budget, len(date_cursor))))
    for d in noise_dates:
        act_type, topic = rng.choice(_NOISE_ACTIVITIES)
        noise_skeletons.append(LogSkeleton(
            log_id="",
            user_id=user_id,
            date=d.isoformat(),
            activity_type=act_type,
            goal_id=None,
            topic=topic,
            evidence_strength="low",
        ))

    all_skeletons = goal_skeletons + noise_skeletons
    all_skeletons.sort(key=lambda s: s.date)
    for idx, sk in enumerate(all_skeletons):
        sk.log_id = f"L-{user_id}-{idx+1:04d}"

    return all_skeletons
