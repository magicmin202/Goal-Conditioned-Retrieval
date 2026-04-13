"""Schema-based evidence category mapper.

Architecture role:
  - Category gate   → hard admission filter in reranker (Stage1/Stage2)
  - Schema signals  → tiny weak boost at candidate stage ONLY
  - NOT in reranker scoring formula (goal lexical remains primary)

Goal domain detection: inferred from goal title/description keywords.
Log category detection: inferred from activity_type + content keywords.

Category relevance to goal:
  "core"       → direct evidence for this goal domain → admit with any goal signal
  "supporting" → contextual evidence → admit only with explicit goal lexical signal
  "none"       → unrelated to this goal domain → always reject (hard gate)

Initial domains (4 — expand carefully):
  fitness_muscle_gain
  fitness_fat_loss
  productivity_development
  learning_coding
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.schemas import ResearchGoal, ResearchLog


def _tok(text: str) -> set[str]:
    return set(re.findall(r"[\w가-힣]{2,}", text.lower()))


# ── Log-level evidence categories ─────────────────────────────────────────────

@dataclass(frozen=True)
class _LogCategoryDef:
    name: str
    activity_types: frozenset[str]   # activity_type exact matches (strong signal)
    keywords: frozenset[str]         # content/title keyword matches


_LOG_CATEGORIES: list[_LogCategoryDef] = [
    # ── Fitness ───────────────────────────────────────────────────────────────
    _LogCategoryDef(
        "training",
        frozenset(["exercise"]),
        frozenset([
            "운동", "헬스", "근력", "스쿼트", "데드리프트", "벤치프레스",
            "런닝", "달리기", "유산소", "웨이트", "홈트", "필라테스",
            "풀업", "딥스", "오버헤드프레스", "바벨", "덤벨",
        ]),
    ),
    _LogCategoryDef(
        "nutrition",
        frozenset([]),
        frozenset([
            "식단", "단백질", "칼로리", "식사", "음식", "영양",
            "닭가슴살", "탄수화물", "지방", "끼니", "채소", "샐러드",
        ]),
    ),
    _LogCategoryDef(
        "body_metrics",
        frozenset([]),
        frozenset([
            "체중", "체지방", "몸무게", "체성분", "인바디",
            "허리둘레", "근육량", "감량", "증량",
        ]),
    ),
    _LogCategoryDef(
        "recovery",
        frozenset([]),
        frozenset(["스트레칭", "수면", "휴식", "회복", "피로", "마사지", "foam"]),
    ),

    # ── Development / Coding ──────────────────────────────────────────────────
    _LogCategoryDef(
        "implementation",
        frozenset(["implementation", "coding"]),
        frozenset([
            "구현", "개발", "코딩", "작성", "프로젝트", "포트폴리오",
            "배포", "빌드", "앱", "서버", "웹", "api", "프론트엔드",
            "백엔드", "풀스택", "커밋", "깃허브",
        ]),
    ),
    _LogCategoryDef(
        "debugging",
        frozenset([]),
        frozenset(["디버깅", "에러", "버그", "오류", "수정", "트러블슈팅"]),
    ),
    _LogCategoryDef(
        "study_progress",
        frozenset(["study", "reading"]),
        frozenset([
            "공부", "학습", "강의", "독서", "책", "수강", "노트",
            "개념", "이론", "cs", "자료구조",
        ]),
    ),
    _LogCategoryDef(
        "problem_solving",
        frozenset(),  # removed "execution" — that activity_type is too broad and
                      # caused travel-prep logs (act=execution) to be misclassified as
                      # problem_solving, triggering a category_mismatch reject under
                      # travel_planning domain.  Problem-solving logs are identified
                      # reliably via keywords alone (알고리즘/코테/백준/프로그래머스…).
        frozenset([
            "문제", "알고리즘", "코테", "풀이", "풀었", "코딩테스트",
            "리트코드", "백준", "프로그래머스",
        ]),
    ),
    _LogCategoryDef(
        "planning",
        frozenset(["planning"]),
        frozenset(["계획", "목표", "전략", "로드맵", "일정"]),
    ),

    # ── Language Exam ─────────────────────────────────────────────────────────
    _LogCategoryDef(
        "language_exam_study",
        frozenset(["study", "reading"]),
        frozenset([
            "토익", "toeic", "영어 공부", "영어 학습", "영어 시험",
            "teps", "ielts", "toefl", "영어 자격증",
        ]),
    ),
    _LogCategoryDef(
        "vocab_building",
        frozenset([]),
        frozenset([
            "단어 암기", "어휘", "vocabulary", "단어장", "영단어",
            "단어 외우기", "word", "voca",
        ]),
    ),
    _LogCategoryDef(
        "listening_practice",
        frozenset([]),
        frozenset([
            "리스닝", "lc", "lc 풀이", "청취", "듣기 연습",
            "listening", "lc 연습", "받아쓰기",
        ]),
    ),
    _LogCategoryDef(
        "reading_practice",
        frozenset([]),
        frozenset([
            "리딩", "rc", "rc 풀이", "독해", "독해 연습",
            "reading", "rc 연습", "지문 풀이",
        ]),
    ),
    _LogCategoryDef(
        "mock_test",
        frozenset(),  # removed "execution" — same reason as problem_solving above;
                      # mock-test logs are unambiguously identified by their keywords.
        frozenset([
            "모의고사", "실전 문제", "기출 문제", "mock", "실전 연습",
            "파트 풀이", "전체 풀이", "시험 풀이",
        ]),
    ),
    _LogCategoryDef(
        "graduate_admission_prep",
        frozenset(["study"]),
        frozenset([
            "대학원 준비", "대학원 지원", "gre", "지도교수 컨택", "지원서",
            "연구계획서", "sop", "추천서", "대학원 서류",
        ]),
    ),
    _LogCategoryDef(
        "career_support",
        frozenset(["planning"]),
        frozenset([
            "취업 준비", "취업 스펙", "영어 자격증 준비", "이력서", "자소서",
            "자기소개서", "채용 공고", "job", "resume",
        ]),
    ),

    # ── Travel ────────────────────────────────────────────────────────────────
    _LogCategoryDef(
        "booking",
        frozenset([]),
        frozenset([
            "예약", "항공권", "비행기", "숙소", "호텔", "에어비앤비",
            "결제", "확정", "티켓", "예매", "booking", "reserved",
        ]),
    ),
    _LogCategoryDef(
        "budgeting",
        frozenset([]),
        frozenset([
            "예산", "비용", "환전", "가계부", "지출", "저비용", "경비",
            "절약", "할인", "가격", "저렴", "budget",
        ]),
    ),
    _LogCategoryDef(
        "logistics",
        frozenset([]),
        frozenset([
            "짐", "준비물", "패킹", "챙기기", "이동", "교통",
            "필요한것", "체크리스트", "packing", "luggage",
        ]),
    ),
    _LogCategoryDef(
        "travel_research",
        frozenset([]),
        frozenset([
            "여행지", "관광", "명소", "후기", "리뷰", "조사",
            "여행 정보", "블로그", "검색", "destination",
        ]),
    ),
]

_CAT_BY_NAME: dict[str, _LogCategoryDef] = {c.name: c for c in _LOG_CATEGORIES}


# ── Activity-type definitions ──────────────────────────────────────────────────
# Five coarse activity buckets that describe *how* a log was done (not *what*).
# Used for compatibility gating: goal's expected activity types vs log's inferred type.

ACTIVITY_TYPE_DEFINITIONS: dict[str, dict] = {
    "learning": {
        "keywords": [
            "공부", "학습", "조사", "탐색", "읽기", "정리", "개념", "이해",
            "study", "learn", "research", "review", "정독", "분석", "탐구",
            "강의", "수강", "시청", "청취", "독서", "습득",
        ],
        "examples": ["ETF 투자 학습", "토익 공부", "여행지 조사", "알고리즘 공부"],
    },
    "execution": {
        "keywords": [
            "구매", "예약", "등록", "개설", "결제", "신청", "매수", "예매",
            "완료", "실행", "시작", "book", "buy", "register", "purchase",
            "가입", "계약", "체결", "투자", "납부", "환전", "송금",
        ],
        "examples": ["항공권 예약", "헬스장 등록", "주식 매수", "보험 가입"],
    },
    "lifestyle": {
        "keywords": [
            "운동", "식사", "수면", "휴식", "루틴", "건강", "스트레칭",
            "산책", "낮잠", "요리", "식단", "러닝", "헬스", "요가",
            "명상", "목욕", "취침",
        ],
        "examples": ["아침 러닝", "식사 기록", "스트레칭 루틴", "낮잠"],
    },
    "planning": {
        "keywords": [
            "계획", "기록", "예산", "목표", "일정", "정리", "관리",
            "plan", "budget", "schedule", "review", "점검", "설정",
            "세우기", "수립", "작성", "검토",
        ],
        "examples": ["여행 예산 계획", "월간 목표 정리", "포트폴리오 점검"],
    },
    "creative": {
        "keywords": [
            "제작", "창작", "글쓰기", "개발", "구현", "제출", "작성",
            "만들기", "build", "create", "implement", "write",
            "코딩", "프로그래밍", "디자인", "편집", "촬영", "포트폴리오",
        ],
        "examples": ["앱 개발", "포트폴리오 작성", "블로그 포스팅"],
    },
}


def classify_log_activity_type(log_text: str) -> str:
    """Classify a log's activity type from its text via keyword matching.

    Returns one of the ACTIVITY_TYPE_DEFINITIONS keys, or "unknown" if no
    keyword fires.  No LLM call — pure rule-based.
    """
    text_lower = log_text.lower()
    scores: dict[str, int] = {at: 0 for at in ACTIVITY_TYPE_DEFINITIONS}

    for activity_type, defn in ACTIVITY_TYPE_DEFINITIONS.items():
        for kw in defn["keywords"]:
            if kw in text_lower:
                scores[activity_type] += 1

    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] > 0 else "unknown"


def get_goal_expected_activity_types(
    goal_title: str,
    goal_description: str = "",
) -> list[str]:
    """Infer the activity types expected for a goal (up to 2).

    Scans goal title + description for activity-type keywords and returns the
    top-scoring types.  Ties broken by definition order (learning first).
    Falls back to ["learning", "execution", "planning"] when nothing matches.
    """
    combined = (goal_title + " " + goal_description).lower()
    scores: dict[str, int] = {at: 0 for at in ACTIVITY_TYPE_DEFINITIONS}

    for activity_type, defn in ACTIVITY_TYPE_DEFINITIONS.items():
        for kw in defn["keywords"]:
            if kw in combined:
                scores[activity_type] += 1

    ranked = sorted(
        [(at, sc) for at, sc in scores.items() if sc > 0],
        key=lambda x: x[1],
        reverse=True,
    )
    result = [at for at, _ in ranked[:2]]
    return result if result else ["learning", "execution", "planning"]


# Pairs (goal_activity_type, log_activity_type) that are fundamentally incompatible.
# Only pairs that are *completely* unrelated are listed — false negatives are
# preferable to false positives (better to admit a borderline log than reject a
# relevant one; the lexical gate will handle precision).
_INCOMPATIBLE_PAIRS: set[tuple[str, str]] = {
    ("learning",  "lifestyle"),   # pure study goal ← lifestyle routine log
    ("execution", "lifestyle"),   # action/purchase goal ← lifestyle routine log
    ("lifestyle", "creative"),    # habit/fitness goal ← software build log
}


def is_activity_type_compatible(
    log_activity_type: str,
    goal_activity_types: list[str],
) -> bool:
    """Return True if log_activity_type is compatible with goal_activity_types.

    A log is compatible unless ALL goal types mark it as incompatible.
    "unknown" is always compatible (conservative — let lexical gate decide).
    """
    if log_activity_type == "unknown":
        return True

    for goal_at in goal_activity_types:
        if log_activity_type == goal_at:
            return True  # direct match
        if (goal_at, log_activity_type) not in _INCOMPATIBLE_PAIRS:
            return True  # not marked incompatible → allow

    return False


def get_activity_type_quality_prior(activity_type: str) -> dict:
    """Return quality-component weights for the given activity type.

    Weights (specificity + actionability + progression = 1.0) are used by
    EvidenceQualityScorer (next Step) to adjust per-component emphasis.
    """
    _PRIORS: dict[str, dict[str, float]] = {
        "learning": {
            "specificity":   0.30,
            "actionability": 0.20,
            "progression":   0.50,
        },
        "execution": {
            "specificity":   0.20,
            "actionability": 0.60,
            "progression":   0.20,
        },
        "lifestyle": {
            "specificity":   0.30,
            "actionability": 0.40,
            "progression":   0.30,
        },
        "planning": {
            "specificity":   0.50,
            "actionability": 0.20,
            "progression":   0.30,
        },
        "creative": {
            "specificity":   0.30,
            "actionability": 0.50,
            "progression":   0.20,
        },
        "unknown": {
            "specificity":   0.33,
            "actionability": 0.33,
            "progression":   0.34,
        },
    }
    return _PRIORS.get(activity_type, _PRIORS["unknown"])


# ── Goal-level domains (DEPRECATED — removed to support open-domain goals) ─────
# _GOAL_DOMAINS and _DOMAIN_BY_NAME have been removed.
# Hard-coded domain schema caused all logs to be rejected as cat=unknown
# for goals outside the 6 predefined domains (e.g. "친구 관계 넓히기").
# Precision control is now handled by: activity-type gate + lexical gate + negative veto.

# Stub for any remaining internal references
_GOAL_DOMAINS: list = []
_DOMAIN_BY_NAME: dict = {}


# ── Result ─────────────────────────────────────────────────────────────────────

@dataclass
class CategoryScore:
    log_category: str       # assigned log evidence category ("training", "unknown", …)
    goal_domain: str        # inferred goal domain ("productivity_development", …)
    relevance: str          # "core" | "supporting" | "none"
    reason: str = ""        # rejection / admission reason string (for debug)

    @property
    def is_relevant(self) -> bool:
        return self.relevance != "none"

    @property
    def strength(self) -> str:
        return self.relevance


# ── SchemaMapper ───────────────────────────────────────────────────────────────

class SchemaMapper:
    """Map (log, goal) → CategoryScore for category-first admission gating.

    Schema is a WEAK HINT system:
      - Category gate:  hard filter — "none" → reject immediately
      - Score boost:    small additive at candidate stage only (not in reranker formula)
      - Goal lexical:   remains primary evidence signal in reranker

    Thread-safe: stateless, can be used as module-level singleton.
    """

    def detect_goal_domain(self, goal: ResearchGoal) -> str:
        """Infer goal domain from title + description. Returns domain name or 'unknown'."""
        goal_text = (goal.query_text + " " + goal.goal_embedding_text).lower()
        goal_toks = _tok(goal_text)

        best_domain = "unknown"
        best_hits = 0
        for domain in _GOAL_DOMAINS:
            hits = len(goal_toks & domain.detection_keywords)
            if hits > best_hits:
                best_hits = hits
                best_domain = domain.name

        return best_domain

    def detect_log_category(self, log: ResearchLog) -> str | None:
        """Map log to evidence category. Returns category name or None if no signal."""
        log_text = log.full_text.lower()
        log_toks = _tok(log_text)
        act_type = log.activity_type or ""

        best_cat: str | None = None
        best_score = 0

        for cat in _LOG_CATEGORIES:
            score = 0
            if act_type in cat.activity_types:
                score += 2          # activity_type match is a stronger signal
            kw_hits = len(log_toks & cat.keywords)
            score += kw_hits

            if score > best_score:
                best_score = score
                best_cat = cat.name

        return best_cat if best_score >= 1 else None

    def support_context_hit(
        self, log: ResearchLog, goal_domain: str
    ) -> tuple[bool, list[str]]:
        """DEPRECATED — domain schema removed. Always returns (False, [])."""
        return False, []

    def is_subdomain_consistent(self, log_category: str, goal_domain: str) -> bool:
        """DEPRECATED — domain schema removed. Always returns False."""
        return False

    def evaluate(self, log: ResearchGoal, goal: ResearchGoal) -> "CategoryScore":
        """DEPRECATED — Schema Category Gate has been removed.

        Always returns relevance='unknown' so that the old
        ``cat_result.relevance == 'none'`` branch in reranker.py never fires.
        Precision control is now handled by the independent activity-type gate,
        lexical gate, and negative veto inside GoalConditionedReranker.score().
        """
        return CategoryScore(
            log_category="unknown",
            goal_domain="open",
            relevance="unknown",
            reason="schema_gate_disabled",
        )


# Module-level singleton
_schema_mapper = SchemaMapper()


def evaluate(log: ResearchLog, goal: ResearchGoal) -> CategoryScore:
    """Module-level convenience wrapper around SchemaMapper.evaluate()."""
    return _schema_mapper.evaluate(log, goal)
