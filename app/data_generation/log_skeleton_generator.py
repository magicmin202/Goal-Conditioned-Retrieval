"""Generate day-level log skeletons for a user.

Key design decisions:
- Logs are day-level records (no hour/minute/second).
- The monthly log stream is a mix of activities tied to different goals
  plus noise logs unrelated to any goal.
- Repeated activity blocks simulate realistic learning bursts,
  enabling Stage 2 compression testing.
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
    activity_type: str          # study | implementation | reading | exercise | planning | noise | ...
    goal_id: str | None         # None = noise log
    topic: str                  # abstract topic keyword used by renderer
    evidence_strength: str      # high | medium | low
    metadata: dict = field(default_factory=dict)


# ── Activity templates per goal category ────────────────────────────────────
# Each entry: (activity_type, topic, evidence_strength)
_GOAL_ACTIVITIES: dict[str, list[tuple[str, str, str]]] = {
    "career": [
        ("study",          "알고리즘 문제 풀이",       "high"),
        ("implementation", "포트폴리오 프로젝트 구현",  "high"),
        ("study",          "자료구조 개념 정리",        "medium"),
        ("planning",       "취업 계획 수립",            "low"),
        ("reading",        "개발 기술 블로그 읽기",     "medium"),
        ("study",          "코딩 테스트 연습",          "high"),
    ],
    "education": [
        ("study",    "논문 읽기",              "high"),
        ("planning", "연구계획서 작성",        "high"),
        ("study",    "영어 단어 암기",         "medium"),
        ("reading",  "교수님 연구 자료 정리",  "medium"),
        ("planning", "연구실 탐색",            "low"),
        ("study",    "GRE 문제 풀이",          "high"),
    ],
    "health": [
        ("exercise", "헬스장 운동",     "high"),
        ("exercise", "러닝",            "high"),
        ("exercise", "스트레칭",        "medium"),
        ("planning", "운동 계획 작성",  "low"),
        ("exercise", "홈트레이닝",      "medium"),
    ],
    "relationship": [
        ("social",   "소개팅",           "high"),
        ("social",   "친구 모임",        "medium"),
        ("planning", "대화 주제 정리",   "low"),
        ("social",   "데이트 계획 수립", "medium"),
    ],
    "travel": [
        ("planning",  "항공권 검색",      "high"),
        ("planning",  "숙소 예약",        "high"),
        ("reading",   "여행 후기 조사",   "medium"),
        ("planning",  "여행 예산 정리",   "medium"),
        ("execution", "여행 준비물 구매", "high"),
    ],
    "finance": [
        ("study",    "가계부 정리",      "high"),
        ("study",    "ETF 공부",         "medium"),
        ("planning", "저축 계획 작성",   "low"),
        ("study",    "주식 기초 학습",   "medium"),
        ("execution","소액 투자 실행",   "high"),
    ],
    "habit": [
        ("reading",   "독서",             "high"),
        ("planning",  "독서 목록 작성",   "low"),
        ("execution", "아침 루틴 실행",   "high"),
        ("reflection","하루 반성 일지",   "medium"),
    ],
    "hobby": [
        ("execution", "촬영 외출",        "high"),
        ("study",     "사진 편집 연습",   "medium"),
        ("execution", "요리 실습",        "high"),
        ("reading",   "레시피 조사",      "medium"),
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
]

# Category lookup for goals
_CATEGORY_KEYWORDS: dict[str, str] = {
    "개발자": "career", "AI 엔지니어": "career", "취업": "career",
    "대학원": "education", "토익": "education",
    "운동": "health", "식단": "health",
    "연애": "relationship", "친구": "relationship",
    "여행": "travel",
    "저축": "finance", "투자": "finance",
    "독서": "habit", "기상": "habit",
    "사진": "hobby", "요리": "hobby",
}


def _infer_category(goal: ResearchGoal) -> str:
    for kw, cat in _CATEGORY_KEYWORDS.items():
        if kw in goal.title:
            return cat
    return "career"  # fallback


def _date_range(start: date, end: date) -> list[date]:
    days = (end - start).days + 1
    return [start + timedelta(days=i) for i in range(days)]


def generate_log_skeletons_for_user(
    user_id: str,
    user_index: int,
    goals: list[ResearchGoal],
    start_date: date,
    end_date: date,
    min_logs: int = 25,
    max_logs: int = 40,
    seed: int = 42,
) -> list[LogSkeleton]:
    """Generate day-level log skeletons for one user.

    Strategy:
    1. Build repeated activity blocks per goal (3–5 day bursts).
    2. Fill remaining slots with noise logs.
    3. Shuffle by date and assign log IDs.
    """
    rng = random.Random(seed + user_index * 131)
    all_dates = _date_range(start_date, end_date)
    target_count = rng.randint(min_logs, max_logs)

    # Assign dates to activity blocks
    skeletons: list[LogSkeleton] = []
    date_cursor = list(all_dates)
    rng.shuffle(date_cursor)

    # Allocate ~60% of logs to goal-related activities
    goal_log_budget = int(target_count * 0.60)
    noise_budget = target_count - goal_log_budget

    goal_skeletons: list[LogSkeleton] = []
    for goal in goals:
        cat = _infer_category(goal)
        activities = _GOAL_ACTIVITIES.get(cat, _GOAL_ACTIVITIES["career"])
        per_goal = max(2, goal_log_budget // len(goals))

        # Build repeated blocks (2–4 days per activity)
        days_used: list[date] = sorted(
            rng.sample(date_cursor[: len(date_cursor) // 2], k=min(per_goal, len(date_cursor) // 3))
        )
        block_start = 0
        while block_start < len(days_used):
            act_type, topic, strength = rng.choice(activities)
            block_len = rng.randint(2, 4)
            for d in days_used[block_start : block_start + block_len]:
                goal_skeletons.append(
                    LogSkeleton(
                        log_id="",  # assigned later
                        user_id=user_id,
                        date=d.isoformat(),
                        activity_type=act_type,
                        goal_id=goal.goal_id,
                        topic=topic,
                        evidence_strength=strength,
                    )
                )
            block_start += block_len

    # Trim to budget
    rng.shuffle(goal_skeletons)
    goal_skeletons = goal_skeletons[:goal_log_budget]

    # Noise logs
    noise_skeletons: list[LogSkeleton] = []
    noise_dates = sorted(rng.sample(date_cursor, k=min(noise_budget, len(date_cursor))))
    for d in noise_dates:
        act_type, topic = rng.choice(_NOISE_ACTIVITIES)
        noise_skeletons.append(
            LogSkeleton(
                log_id="",
                user_id=user_id,
                date=d.isoformat(),
                activity_type=act_type,
                goal_id=None,
                topic=topic,
                evidence_strength="low",
            )
        )

    all_skeletons = goal_skeletons + noise_skeletons
    all_skeletons.sort(key=lambda s: s.date)

    for idx, sk in enumerate(all_skeletons):
        sk.log_id = f"L-{user_id}-{idx+1:04d}"

    return all_skeletons
