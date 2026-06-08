"""noise_ratio 기반 로그 생성 (v2) — 30-domain 체계.

noise_ratio=0.0 → 전체 로그가 goal 관련 (engaged user)
noise_ratio=1.0 → 전체 로그가 noise (disengaged user)

persona가 제공되면 hobbies/skills/occupation에서 도메인별
persona-specific activity topic을 우선 사용.
"""
from __future__ import annotations

import random
from datetime import date, timedelta

from app.schemas import ResearchGoal, ResearchLog

# ── 30-domain goal activity 템플릿 ────────────────────────────────────────────
_GOAL_ACTIVITIES: dict[str, list[tuple[str, str, str]]] = {
    "romance": [
        ("social",      "소개팅 참여",            "high"),
        ("planning",    "데이트 계획 세우기",      "medium"),
        ("reflection",  "연애 감정 일지 작성",     "medium"),
        ("social",      "이성 친구 모임 참석",     "medium"),
        ("reflection",  "자기 매력 점검",          "low"),
    ],
    "marriage": [
        ("planning",    "예식장 방문 및 견적 확인", "high"),
        ("planning",    "혼수 목록 작성",           "high"),
        ("execution",   "배우자와 데이트",          "high"),
        ("planning",    "결혼 예산 계획 수립",      "high"),
        ("execution",   "신혼집 탐색",              "medium"),
    ],
    "family": [
        ("social",      "가족 식사 모임",           "high"),
        ("execution",   "부모님 방문",              "high"),
        ("planning",    "가족 여행 계획",           "medium"),
        ("social",      "자녀와 활동 공유",         "high"),
        ("execution",   "가족 행사 준비",           "medium"),
    ],
    "appearance": [
        ("execution",   "스킨케어 루틴 실행",       "high"),
        ("study",       "뷰티/패션 트렌드 공부",    "medium"),
        ("execution",   "헤어 관리",               "medium"),
        ("planning",    "다이어트 식단 계획",        "medium"),
        ("execution",   "옷장 정리 및 코디 연습",   "medium"),
    ],
    "health": [
        ("exercise",    "헬스장 운동",              "high"),
        ("exercise",    "달리기",                   "high"),
        ("exercise",    "스트레칭",                 "medium"),
        ("planning",    "건강 식단 준비",           "medium"),
        ("exercise",    "홈트레이닝",              "medium"),
        ("execution",   "건강 검진 예약",           "medium"),
    ],
    "social": [
        ("social",      "동호회/모임 참석",         "high"),
        ("social",      "직장 동료와 점심",         "medium"),
        ("planning",    "네트워킹 이벤트 참가 계획", "medium"),
        ("social",      "새로운 사람과 대화",       "medium"),
        ("execution",   "커뮤니티 활동 참여",       "high"),
    ],
    "friendship": [
        ("social",      "친구와 만남",              "high"),
        ("social",      "오래된 친구에게 연락",     "high"),
        ("social",      "친구 생일 챙기기",         "medium"),
        ("social",      "친구와 취미 활동 공유",    "medium"),
        ("reflection",  "우정 일지 작성",           "low"),
    ],
    "receiving": [
        ("reflection",  "감사 받은 경험 일지 작성", "medium"),
        ("execution",   "도움 요청 연습",           "high"),
        ("reflection",  "감정 패턴 분석",           "medium"),
        ("social",      "주변에 도움 요청하기",     "high"),
        ("reflection",  "칭찬 수용 연습",           "medium"),
    ],
    "boundaries": [
        ("reflection",  "감정 경계 일지 작성",      "high"),
        ("study",       "자기주장 훈련 학습",        "medium"),
        ("reflection",  "거절 경험 회고",           "medium"),
        ("execution",   "명확한 의사 표현 연습",    "high"),
        ("reflection",  "감정 소진 패턴 점검",      "medium"),
    ],
    "prosocial": [
        ("social",      "적극적 경청 연습",         "high"),
        ("execution",   "주변 사람 칭찬하기",       "medium"),
        ("reflection",  "공감 일지 작성",           "medium"),
        ("social",      "봉사활동 참여",            "high"),
        ("execution",   "격려 메시지 보내기",       "medium"),
    ],
    "teaching": [
        ("execution",   "재능 기부 활동",           "high"),
        ("execution",   "후배 멘토링 세션",         "high"),
        ("planning",    "강의/수업 자료 준비",      "medium"),
        ("social",      "커뮤니티 교육 참여",       "medium"),
        ("execution",   "지식 나눔 콘텐츠 작성",   "high"),
    ],
    "leadership": [
        ("study",       "리더십 서적 읽기",         "high"),
        ("execution",   "팀 프로젝트 주도",         "high"),
        ("planning",    "조직 개선 계획 수립",      "medium"),
        ("execution",   "발표 및 의사 결정 연습",   "high"),
        ("reflection",  "리더십 피드백 정리",       "medium"),
    ],
    "religion": [
        ("execution",   "예배/법회 참석",           "high"),
        ("reflection",  "기도/명상 실행",           "high"),
        ("reading",     "경전/종교 서적 읽기",      "medium"),
        ("social",      "종교 공동체 활동 참여",    "medium"),
        ("reflection",  "신앙 일지 작성",           "medium"),
    ],
    "awareness": [
        ("reading",     "신문/뉴스 읽기",           "high"),
        ("study",       "사회 이슈 자료 조사",      "high"),
        ("reflection",  "시사 이슈 정리 노트",      "medium"),
        ("social",      "시사 토론 참여",           "medium"),
        ("study",       "다큐멘터리 시청",          "medium"),
    ],
    "ethics": [
        ("reflection",  "가치관 일지 작성",         "high"),
        ("study",       "윤리/철학 서적 읽기",      "medium"),
        ("execution",   "윤리적 소비 실천",         "medium"),
        ("reflection",  "삶의 방향 점검",           "high"),
        ("execution",   "환경 실천 행동",           "medium"),
    ],
    "freedom": [
        ("study",       "부업/창업 공부",           "high"),
        ("execution",   "사이드 프로젝트 진행",     "high"),
        ("planning",    "경제적 자유 로드맵 작성",  "medium"),
        ("study",       "재테크/투자 공부",         "medium"),
        ("execution",   "원격 근무 역량 개발",      "high"),
    ],
    "aesthetics": [
        ("execution",   "미술 전시 관람",           "high"),
        ("study",       "예술사/미학 공부",         "medium"),
        ("execution",   "인테리어 소품 구매/배치",  "medium"),
        ("reflection",  "예술 감상 일지 작성",      "medium"),
        ("execution",   "사진/영상 구도 연습",      "high"),
    ],
    "creativity": [
        ("execution",   "창작 프로젝트 작업",       "high"),
        ("execution",   "드로잉/스케치 연습",       "high"),
        ("execution",   "악기 연주 연습",           "high"),
        ("study",       "창의성 개발 기법 학습",    "medium"),
        ("execution",   "아이디어 노트 작성",       "medium"),
    ],
    "openness": [
        ("execution",   "처음 하는 활동 도전",      "high"),
        ("social",      "다양한 배경 사람들과 대화", "medium"),
        ("execution",   "낯선 장소 탐방",           "medium"),
        ("reflection",  "새로운 경험 기록",         "medium"),
        ("execution",   "새로운 음식/문화 체험",    "high"),
    ],
    "entertainment": [
        ("execution",   "취미 활동 참여",           "high"),
        ("execution",   "공연/영화 관람",           "high"),
        ("execution",   "게임/오락 활동",           "medium"),
        ("social",      "취미 동호회 참석",         "medium"),
        ("execution",   "스포츠 관람",              "medium"),
    ],
    "wellbeing": [
        ("reflection",  "감정 일지 작성",           "high"),
        ("execution",   "명상/호흡 연습",           "high"),
        ("execution",   "심리 상담 참여",           "medium"),
        ("execution",   "충분한 휴식 취하기",       "medium"),
        ("execution",   "디지털 디톡스 실행",       "medium"),
    ],
    "stability": [
        ("planning",    "재정 현황 점검",           "high"),
        ("execution",   "비상금 자동 저축 설정",    "high"),
        ("planning",    "생활 안정화 계획 수립",    "medium"),
        ("execution",   "보험/연금 확인",           "medium"),
        ("planning",    "주거 안정 계획 작성",      "medium"),
    ],
    "meaning": [
        ("reflection",  "삶의 의미 탐색 일지",      "high"),
        ("reading",     "인생 철학/에세이 읽기",    "medium"),
        ("planning",    "버킷리스트 작성/업데이트", "high"),
        ("reflection",  "가치관 명료화 작업",       "medium"),
        ("reflection",  "죽음과 유산에 대한 성찰",  "low"),
    ],
    "growth": [
        ("reading",     "자기계발 서적 읽기",       "high"),
        ("study",       "온라인 강의 수강",         "high"),
        ("reflection",  "성장 일지 작성",           "medium"),
        ("execution",   "강점 기반 프로젝트 진행",  "medium"),
        ("reflection",  "피드백 반영 및 개선",      "high"),
    ],
    "order": [
        ("planning",    "일일 할 일 목록 작성",     "high"),
        ("execution",   "공간 정리 및 청소",        "high"),
        ("planning",    "주간 일정 계획",           "high"),
        ("execution",   "파일/자료 정리",           "medium"),
        ("execution",   "디지털 파일 정리",         "medium"),
    ],
    "achievement": [
        ("planning",    "목표 세분화 및 계획 수립", "high"),
        ("study",       "자격증/시험 공부",         "high"),
        ("execution",   "모의 테스트 응시",         "high"),
        ("reflection",  "진행 상황 점검",           "medium"),
        ("execution",   "챌린지 실행",              "high"),
    ],
    "autonomy": [
        ("planning",    "독립 계획 수립",           "high"),
        ("execution",   "프리랜서 프로젝트 시작",   "high"),
        ("reflection",  "자기결정 일지 작성",       "medium"),
        ("study",       "자립 관련 공부",           "medium"),
        ("execution",   "자기 주도 프로젝트 진행",  "high"),
    ],
    "career": [
        ("study",       "직무 관련 공부",           "high"),
        ("execution",   "포트폴리오 작업",          "high"),
        ("planning",    "커리어 로드맵 작성",       "medium"),
        ("execution",   "직무 역량 실습",           "high"),
        ("study",       "업계 트렌드 파악",         "medium"),
        ("execution",   "이력서/자기소개서 작성",   "high"),
    ],
    "education": [
        ("study",       "강의/교재 학습",           "high"),
        ("study",       "기출문제 풀이",            "high"),
        ("planning",    "학습 계획 수립",           "medium"),
        ("reflection",  "오답 노트 정리",           "medium"),
        ("study",       "논문/자료 읽기",           "high"),
    ],
    "finance": [
        ("planning",    "가계부 정리",              "high"),
        ("study",       "재테크/투자 공부",         "medium"),
        ("planning",    "저축 계획 작성",           "medium"),
        ("execution",   "소액 투자 실행",           "high"),
        ("planning",    "월간 예산 설정",           "medium"),
    ],
}

# goal title → domain 역매핑
_TITLE_CAT: dict[str, str] = {
    # romance
    "좋은 연인 만나기": "romance", "연인과의 관계 깊어지기": "romance", "연애 자신감 키우기": "romance",
    # marriage
    "결혼 준비하기": "marriage", "배우자와의 유대 강화하기": "marriage", "결혼 생활 안정화하기": "marriage",
    # family
    "가족과의 시간 늘리기": "family", "부모님 건강 챙기기": "family", "자녀와의 유대감 높이기": "family",
    # appearance
    "피부 관리 루틴 만들기": "appearance", "체형 관리하기": "appearance", "패션 감각 키우기": "appearance",
    # health
    "규칙적으로 운동하기": "health", "건강한 식습관 만들기": "health", "10km 달리기 완주하기": "health",
    # social
    "새로운 커뮤니티 참여하기": "social", "직장 내 인간관계 개선하기": "social", "네트워킹 역량 키우기": "social",
    # friendship
    "오래된 친구와 연락 이어가기": "friendship", "진정한 우정 쌓기": "friendship", "친구 관계 넓히기": "friendship",
    # receiving
    "도움 요청하는 법 배우기": "receiving", "감사함 받아들이기": "receiving", "취약함 인정하기": "receiving",
    # boundaries
    "감정 경계 설정하기": "boundaries", "거절 잘 하기": "boundaries", "자기 보호 능력 키우기": "boundaries",
    # prosocial
    "공감 능력 키우기": "prosocial", "긍정적인 에너지 나누기": "prosocial", "경청 습관 만들기": "prosocial",
    # teaching
    "재능 기부 시작하기": "teaching", "멘토링 활동하기": "teaching", "지식 나눔 콘텐츠 만들기": "teaching",
    # leadership
    "리더십 역량 키우기": "leadership", "조직 내 영향력 높이기": "leadership", "의사 결정 능력 강화하기": "leadership",
    # religion
    "신앙 생활 충실히 하기": "religion", "영적 성장 추구하기": "religion", "종교 공동체 참여 늘리기": "religion",
    # awareness
    "시사 공부 습관 만들기": "awareness", "사회 문제 공부하기": "awareness", "미디어 리터러시 키우기": "awareness",
    # ethics
    "가치관 정립하기": "ethics", "윤리적 소비 실천하기": "ethics", "정직하고 일관된 삶 살기": "ethics",
    # freedom
    "경제적 자유 준비하기": "freedom", "부업/사이드 프로젝트 시작하기": "freedom", "디지털 노마드 준비하기": "freedom",
    # aesthetics
    "예술 감상 습관 만들기": "aesthetics", "공간 미학 가꾸기": "aesthetics", "사진/영상 미학 익히기": "aesthetics",
    # creativity
    "창의적인 프로젝트 시작하기": "creativity", "그림/드로잉 습관 만들기": "creativity", "악기 연주 실력 키우기": "creativity",
    # openness
    "새로운 경험에 도전하기": "openness", "다양한 문화 체험하기": "openness", "낯선 취미 시작하기": "openness",
    # entertainment
    "취미 활동 꾸준히 즐기기": "entertainment", "문화 생활 풍요롭게 하기": "entertainment", "게임/스포츠 즐기기": "entertainment",
    # wellbeing
    "정서 건강 관리하기": "wellbeing", "번아웃 예방 루틴 만들기": "wellbeing", "수면 습관 개선하기": "wellbeing",
    # stability
    "비상금 마련하기": "stability", "안정적인 생활 기반 만들기": "stability", "장기 보험/연금 정비하기": "stability",
    # meaning
    "삶의 목적 찾기": "meaning", "버킷리스트 완성하기": "meaning", "유산과 영향력 생각하기": "meaning",
    # growth
    "자기계발 습관 만들기": "growth", "강점 발견하고 키우기": "growth", "피드백 수용 능력 키우기": "growth",
    # order
    "시간 관리 시스템 만들기": "order", "공간 정리 습관 만들기": "order", "디지털 정리 하기": "order",
    # achievement
    "목표 달성 루틴 만들기": "achievement", "자격증/시험 도전하기": "achievement", "100일 챌린지 완주하기": "achievement",
    # autonomy
    "독립적인 의사결정 훈련하기": "autonomy", "프리랜서/부업 시작하기": "autonomy", "자기 삶의 주도권 갖기": "autonomy",
    # career
    "이직/취업 성공하기": "career", "전문성 강화하기": "career", "성과로 인정받기": "career",
    # education
    "대학원 진학 준비하기": "education", "자격 시험 준비하기": "education", "독학으로 지식 쌓기": "education",
    # finance
    "월 저축 목표 달성하기": "finance", "투자 공부 시작하기": "finance", "가계부 작성 습관 만들기": "finance",
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
    ("daily", "빨래"),
    ("daily", "친구 만남"),
    ("daily", "게임"),
]

# 도메인별 취미 키워드 (페르소나 topic 추출용)
_DOMAIN_HOBBY_KEYWORDS: dict[str, list[str]] = {
    "health":      ["헬스", "운동", "달리기", "수영", "요가", "등산", "필라테스", "자전거", "산책"],
    "appearance":  ["피부", "화장", "패션", "뷰티", "스타일"],
    "creativity":  ["그림", "드로잉", "수채화", "스케치", "공예", "기타", "피아노", "노래", "사진", "촬영"],
    "aesthetics":  ["전시", "미술관", "인테리어", "디자인", "예술"],
    "entertainment":["영화", "드라마", "공연", "게임", "유튜브", "스포츠 관람"],
    "openness":    ["여행", "배낭여행", "트레킹", "요리", "베이킹", "역사", "탐방"],
    "social":      ["동호회", "네트워킹", "모임", "커뮤니티"],
    "wellbeing":   ["명상", "요가", "사우나", "힐링"],
    "growth":      ["독서", "자기계발", "강의"],
    "teaching":    ["봉사", "재능기부", "멘토링"],
    "religion":    ["기도", "예배", "법회"],
    "finance":     ["투자", "주식", "저축", "재테크", "ETF"],
}


def _persona_topics(domain: str, persona) -> list[tuple[str, str, str]]:
    """페르소나의 hobbies/skills에서 domain 관련 activity topic 추출."""
    results: list[tuple[str, str, str]] = []
    kws = _DOMAIN_HOBBY_KEYWORDS.get(domain, [])

    for hobby in getattr(persona, "hobbies", []):
        if any(kw in hobby for kw in kws):
            results.append(("execution", hobby, "high"))

    if domain == "career":
        for skill in getattr(persona, "skills", [])[:2]:
            results.append(("study", f"{skill} 학습", "high"))
            results.append(("execution", f"{skill} 실습", "medium"))

    if domain in ("education", "achievement"):
        occ = getattr(persona, "occupation", "")
        if occ:
            results.append(("study", f"{occ} 관련 자격 공부", "medium"))

    return results


def _render_text(topic: str, activity_type: str) -> tuple[str, str]:
    """(title, content) 텍스트 생성."""
    filler = {
        "study":       "공부했다. 오늘 학습한 내용을 정리하고 복습했다.",
        "execution":   "실행했다. 계획한 대로 진행하며 결과물을 만들었다.",
        "planning":    "계획을 세웠다. 목표를 구체화하고 일정을 잡았다.",
        "reading":     "읽었다. 핵심 내용을 메모하며 꼼꼼히 읽었다.",
        "exercise":    "운동했다. 목표 횟수를 채우며 체력을 키웠다.",
        "social":      "참여했다. 사람들을 만나 좋은 시간을 보냈다.",
        "reflection":  "돌아봤다. 오늘 하루를 되짚으며 개선점을 찾았다.",
        "daily":       "했다.",
    }.get(activity_type, "진행했다.")
    return topic, f"오늘 {topic}을(를) {filler}"


def _date_range(start: date, end: date) -> list[date]:
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def generate_logs_v2(
    user_id: str,
    user_index: int,
    goals: list[ResearchGoal],
    start_date: date,
    end_date: date,
    min_logs: int = 61,
    max_logs: int = 81,
    seed: int = 42,
    persona=None,
    noise_ratio: float | None = None,  # deprecated, ignored (kept for compat)
) -> list[ResearchLog]:
    """goal별 noise_ratio에 따라 relevant 로그와 noise 로그를 혼합 생성.

    각 goal의 goal.noise_ratio가 해당 목표의 참여도를 결정:
      0.0 → relevant 로그 가득  /  1.0 → relevant 로그 없음

    총 로그 수(total)는 61~81로 고정.
    noise 로그는 날짜 중복을 허용(rng.choices)해 총 수를 맞춤.
    """
    rng = random.Random(seed + user_index * 131)
    all_dates = _date_range(start_date, end_date)
    total = rng.randint(min_logs, max_logs)
    n_goals = len(goals) if goals else 1
    per_goal_budget = max(5, total // n_goals)

    logs: list[ResearchLog] = []
    log_idx = 1
    total_relevant = 0

    # ── Goal별 relevant 로그 생성 ───────────────────────────────────────────
    for goal in goals:
        relevant_count = max(0, round(per_goal_budget * (1.0 - goal.noise_ratio)))
        if relevant_count == 0:
            continue
        total_relevant += relevant_count

        cat = getattr(goal, "domain", None) or "growth"
        base_activities = _GOAL_ACTIVITIES.get(cat, _GOAL_ACTIVITIES["growth"])

        if persona is not None:
            extra = _persona_topics(cat, persona)
            activities = extra + base_activities if extra else base_activities
        else:
            activities = base_activities

        n = min(relevant_count, len(all_dates))
        days = sorted(rng.sample(all_dates, k=n))

        cursor = 0
        while cursor < len(days):
            act_type, topic, strength = rng.choice(activities)
            block = rng.randint(1, 3)
            for d in days[cursor: cursor + block]:
                title, content = _render_text(topic, act_type)
                logs.append(ResearchLog(
                    log_id=f"L-{user_id}-{log_idx:04d}",
                    user_id=user_id,
                    date=d.isoformat(),
                    title=title,
                    content=content,
                    activity_type=act_type,
                    metadata={
                        "topic": topic,
                        "evidence_strength": strength,
                        "goal_id_hint": goal.goal_id,
                        "is_paraphrase": False,
                    },
                    timestamp="",
                    created_at=d.isoformat(),
                ))
                log_idx += 1
            cursor += block

    # ── Noise 로그 생성 (날짜 중복 허용) ───────────────────────────────────
    noise_budget = max(0, total - total_relevant)
    if noise_budget > 0:
        noise_dates = sorted(rng.choices(all_dates, k=noise_budget))
        for d in noise_dates:
            act_type, topic = rng.choice(_NOISE_ACTIVITIES)
            title, content = _render_text(topic, act_type)
            logs.append(ResearchLog(
                log_id=f"L-{user_id}-{log_idx:04d}",
                user_id=user_id,
                date=d.isoformat(),
                title=title,
                content=content,
                activity_type=act_type,
                metadata={
                    "topic": topic,
                    "evidence_strength": "low",
                    "goal_id_hint": None,
                    "is_paraphrase": False,
                },
                timestamp="",
                created_at=d.isoformat(),
            ))
            log_idx += 1

    logs.sort(key=lambda l: l.date)
    for i, log in enumerate(logs):
        log.log_id = f"L-{user_id}-{i+1:04d}"

    return logs
