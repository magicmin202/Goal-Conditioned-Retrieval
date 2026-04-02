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
        frozenset(["execution"]),
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


# ── Goal-level domains ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _GoalDomainDef:
    name: str
    detection_keywords: frozenset[str]       # infer domain from goal text
    core_categories: frozenset[str]          # direct evidence (strong gate)
    supporting_categories: frozenset[str]    # contextual evidence (relaxed gate)


_GOAL_DOMAINS: list[_GoalDomainDef] = [
    _GoalDomainDef(
        "fitness_muscle_gain",
        frozenset(["근육", "벌크", "웨이트", "증량", "체중증가", "헬스장"]),
        frozenset(["training", "body_metrics"]),
        frozenset(["nutrition", "recovery"]),
    ),
    _GoalDomainDef(
        "fitness_fat_loss",
        frozenset(["체중감량", "다이어트", "살빼기", "칼로리", "유산소", "체지방", "감량"]),
        frozenset(["training", "nutrition", "body_metrics"]),
        frozenset(["recovery"]),
    ),
    _GoalDomainDef(
        "productivity_development",
        frozenset(["개발", "취업", "포트폴리오", "프로그래밍", "엔지니어", "개발자", "코딩"]),
        frozenset(["implementation", "problem_solving", "debugging"]),
        frozenset(["study_progress", "planning"]),
    ),
    _GoalDomainDef(
        "learning_coding",
        frozenset(["알고리즘", "코테", "코딩테스트", "학습", "공부", "강의", "스터디"]),
        frozenset(["study_progress", "problem_solving"]),
        frozenset(["implementation", "planning"]),
    ),
    _GoalDomainDef(
        "travel_planning",
        frozenset(["여행", "해외", "여행지", "배낭", "숙소", "항공", "해외여행", "저비용"]),
        frozenset(["booking", "budgeting", "logistics"]),
        frozenset(["travel_research", "planning"]),
    ),
]

_DOMAIN_BY_NAME: dict[str, _GoalDomainDef] = {d.name: d for d in _GOAL_DOMAINS}


# ── Result ─────────────────────────────────────────────────────────────────────

@dataclass
class CategoryScore:
    log_category: str       # assigned log evidence category ("training", "unknown", …)
    goal_domain: str        # inferred goal domain ("productivity_development", …)
    relevance: str          # "core" | "supporting" | "none"

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

    def evaluate(self, log: ResearchLog, goal: ResearchGoal) -> CategoryScore:
        """Evaluate log's evidence category relevance to goal domain.

        Returns CategoryScore where:
          .relevance == "none"       → reject (hard gate)
          .relevance == "core"       → admit with any goal signal
          .relevance == "supporting" → admit with explicit goal lexical signal
        """
        goal_domain = self.detect_goal_domain(goal)
        log_cat = self.detect_log_category(log)

        if log_cat is None:
            return CategoryScore(
                log_category="unknown",
                goal_domain=goal_domain,
                relevance="none",
            )

        if goal_domain == "unknown":
            # Unknown goal domain → cannot determine relevance → allow with "supporting"
            # (better recall for edge cases; goal lexical gate will still filter)
            return CategoryScore(
                log_category=log_cat,
                goal_domain=goal_domain,
                relevance="supporting",
            )

        domain_def = _DOMAIN_BY_NAME.get(goal_domain)
        if domain_def is None:
            return CategoryScore(
                log_category=log_cat, goal_domain=goal_domain, relevance="supporting"
            )

        if log_cat in domain_def.core_categories:
            relevance = "core"
        elif log_cat in domain_def.supporting_categories:
            relevance = "supporting"
        else:
            relevance = "none"

        return CategoryScore(
            log_category=log_cat,
            goal_domain=goal_domain,
            relevance=relevance,
        )


# Module-level singleton
_schema_mapper = SchemaMapper()


def evaluate(log: ResearchLog, goal: ResearchGoal) -> CategoryScore:
    """Module-level convenience wrapper around SchemaMapper.evaluate()."""
    return _schema_mapper.evaluate(log, goal)
