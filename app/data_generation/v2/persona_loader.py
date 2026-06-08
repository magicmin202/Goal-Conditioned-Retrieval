"""Nemotron-Personas-Korea 데이터셋 로더.

Persona Hub (Chan et al., 2024) 방법론에 따라 다양한 한국인 페르소나를
goal 도메인 선택 및 로그 생성의 기반으로 활용.

Reference:
  - Chan et al. (2024). Scaling Synthetic Data Creation with 1,000,000,000 Personas.
  - nvidia/Nemotron-Personas-Korea (HuggingFace)
"""
from __future__ import annotations

import ast
import random
from dataclasses import dataclass, field


@dataclass
class PersonaProfile:
    uuid: str
    name: str
    age: int
    sex: str
    occupation: str
    district: str
    province: str
    education_level: str
    marital_status: str
    family_type: str
    hobbies: list[str]         # parsed from hobbies_and_interests_list
    skills: list[str]          # parsed from skills_and_expertise_list
    career_goals: str          # career_goals_and_ambitions (원문)
    persona_text: str          # persona (서사형 소개)
    inferred_domains: list[str] = field(default_factory=list)


# 취미/직업 → goal 도메인 매핑 키워드
_HOBBY_DOMAIN_MAP: dict[str, str] = {
    # health (5)
    "수영": "health", "등산": "health", "클라이밍": "health", "테니스": "health",
    "배드민턴": "health", "축구": "health", "러닝": "health", "달리기": "health",
    "헬스": "health", "운동": "health", "필라테스": "health", "자전거": "health",
    "산책": "health", "조깅": "health", "농구": "health", "볼링": "health",
    # appearance (4)
    "피부": "appearance", "화장": "appearance", "패션": "appearance",
    "스타일": "appearance", "뷰티": "appearance", "헤어": "appearance",
    # creativity (18)
    "그림": "creativity", "드로잉": "creativity", "수채화": "creativity",
    "스케치": "creativity", "공예": "creativity", "기타": "creativity",
    "피아노": "creativity", "우쿨렐레": "creativity", "노래": "creativity",
    "악기": "creativity", "사진": "creativity", "촬영": "creativity",
    "글쓰기": "creativity", "소설": "creativity", "블로그": "creativity",
    # aesthetics (17)
    "전시": "aesthetics", "미술관": "aesthetics", "갤러리": "aesthetics",
    "인테리어": "aesthetics", "디자인": "aesthetics", "예술": "aesthetics",
    # entertainment (20)
    "영화": "entertainment", "드라마": "entertainment", "유튜브": "entertainment",
    "공연": "entertainment", "게임": "entertainment", "스포츠 관람": "entertainment",
    "트로트": "entertainment", "음악 감상": "entertainment",
    # openness (19)
    "여행": "openness", "배낭여행": "openness", "트레킹": "openness",
    "요리": "openness", "베이킹": "openness", "역사": "openness", "탐방": "openness",
    # social (6)
    "동호회": "social", "네트워킹": "social", "커뮤니티": "social",
    "모임": "social", "술자리": "social",
    # friendship (7)
    "친구 모임": "friendship",
    # family (3)
    "가족": "family",
    # romance (1)
    "연애": "romance", "소개팅": "romance", "데이트": "romance",
    # teaching (11)
    "봉사": "teaching", "재능기부": "teaching", "멘토링": "teaching",
    # wellbeing (21)
    "명상": "wellbeing", "사우나": "wellbeing", "힐링": "wellbeing",
    "요가": "wellbeing",
    # growth (24)
    "독서": "growth", "자기계발": "growth", "강의": "growth",
    # finance (30)
    "투자": "finance", "주식": "finance", "ETF": "finance",
    "저축": "finance", "재테크": "finance",
    # career (28)
    "개발": "career", "프로그래밍": "career", "코딩": "career", "연구": "career",
    # education (29)
    "공부": "education", "학습": "education",
    # religion (13)
    "기도": "religion", "예배": "religion", "법회": "religion",
    # awareness (14)
    "뉴스": "awareness", "시사": "awareness", "환경": "awareness",
    # order (25)
    "정리": "order", "미니멀": "order",
    # meaning (23)
    "철학": "meaning", "에세이": "meaning",
    # freedom (16)
    "프리랜서": "freedom", "부업": "freedom", "창업": "freedom",
    # autonomy (27)
    "독립": "autonomy",
    # achievement (26)
    "자격증": "achievement",
}

_OCCUPATION_DOMAIN_MAP: dict[str, str] = {
    "교사": "teaching", "교수": "teaching", "강사": "teaching",
    "개발": "career", "엔지니어": "career", "프로그래머": "career", "연구원": "career",
    "회계": "finance", "세무": "finance", "경리": "finance",
    "의사": "health", "간호": "health", "보건": "health",
    "작가": "creativity", "기자": "awareness", "편집": "creativity",
    "음악": "creativity", "예술": "aesthetics", "디자인": "aesthetics",
    "관리자": "leadership", "팀장": "leadership", "임원": "leadership", "대표": "leadership",
    "사회복지": "teaching", "복지사": "teaching",
    "승려": "religion", "목사": "religion", "신부": "religion",
    "무직": "stability", "경비": "stability",
    "자영업": "autonomy",
    "마케팅": "career", "기획": "career", "영업": "career",
    "물류": "order", "운송": "order",
}

_DEFAULT_DOMAINS = ["health", "growth", "social", "entertainment", "order"]


def _parse_list_field(s: str) -> list[str]:
    """'[\"항목1\", \"항목2\"]' 형태의 문자열을 파싱."""
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed]
    except Exception:
        pass
    return [s.strip()] if s.strip() else []


def _infer_domains(hobbies: list[str], occupation: str) -> list[str]:
    """취미와 직업에서 goal 도메인 추론 (최대 5개)."""
    domains: list[str] = []
    full_text = " ".join(hobbies) + " " + occupation

    for kw, domain in _HOBBY_DOMAIN_MAP.items():
        if kw in full_text and domain not in domains:
            domains.append(domain)
        if len(domains) >= 5:
            break

    if len(domains) < 2:
        for kw, domain in _OCCUPATION_DOMAIN_MAP.items():
            if kw in occupation and domain not in domains:
                domains.append(domain)

    # fallback
    if len(domains) < 2:
        domains.extend(d for d in _DEFAULT_DOMAINS if d not in domains)

    return domains[:5]


def _extract_name(persona_text: str) -> str:
    """페르소나 텍스트에서 이름 추출 ('홍길동 씨는...' → '홍길동')."""
    if " 씨는" in persona_text:
        return persona_text.split(" 씨는")[0].strip()
    if " 씨" in persona_text:
        return persona_text.split(" 씨")[0].strip()
    return "사용자"


def load_personas(
    n: int = 100,
    seed: int = 42,
    db_path: str | None = None,
) -> list[PersonaProfile]:
    """Nemotron-Personas-Korea에서 n개 페르소나 로드.

    우선순위:
      1. db_path SQLite 파일 (persona_mini.db 등)
      2. fallback 내장 페르소나
    """
    import logging
    log = logging.getLogger(__name__)

    if db_path is None:
        # 기본 경로 탐색
        from pathlib import Path
        candidates = [
            Path(__file__).resolve().parents[4] / "data" / "persona_mini.db",
            Path("data/persona_mini.db"),
        ]
        for p in candidates:
            if p.exists():
                db_path = str(p)
                break

    if db_path:
        try:
            profiles = _load_from_sqlite(db_path, n, seed)
            log.info("SQLite 페르소나 로드: %d개 (%s)", len(profiles), db_path)
            return profiles
        except Exception as e:
            log.warning("SQLite 로드 실패 (%s) → fallback 사용", e)

    log.warning("persona_mini.db 없음 → fallback 페르소나 사용")
    return _fallback_personas(n, seed)


def _load_from_sqlite(db_path: str, n: int, seed: int) -> list[PersonaProfile]:
    """SQLite DB에서 n개 페르소나를 다양성 있게 샘플링."""
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    total = conn.execute("SELECT COUNT(*) FROM persona").fetchone()[0]
    # RANDOM() 샘플링 (seed 적용 불가이므로 넉넉히 가져와 Python에서 shuffle)
    limit = min(n * 5, total)
    rows = conn.execute(
        "SELECT * FROM persona LIMIT ?", (limit,)
    ).fetchall()
    conn.close()

    rng = random.Random(seed)
    rng.shuffle(rows)
    selected = rows[:n]

    profiles: list[PersonaProfile] = []
    for row in selected:
        hobbies = _parse_list_field(row["hobbies_and_interests_list"])
        skills  = _parse_list_field(row["skills_and_expertise_list"])
        occupation = row["occupation"] or ""
        domains = _infer_domains(hobbies, occupation)
        name = _extract_name(row["persona"] or "")
        profiles.append(PersonaProfile(
            uuid=row["uuid"],
            name=name,
            age=row["age"],
            sex=row["sex"],
            occupation=occupation,
            district=row["district"],
            province=row["province"] or "",
            education_level=row["education_level"] or "",
            marital_status=row["marital_status"] or "",
            family_type=row["family_type"] or "",
            hobbies=hobbies,
            skills=skills,
            career_goals=row["career_goals_and_ambitions"] or "",
            persona_text=row["persona"] or "",
            inferred_domains=domains,
        ))
    return profiles


def _load_from_hub(n: int, seed: int) -> list[PersonaProfile]:
    from datasets import load_dataset
    ds = load_dataset("nvidia/Nemotron-Personas-Korea", split="train", streaming=True)

    rng = random.Random(seed)
    collected: list[dict] = []

    # 스트리밍으로 넉넉히 수집 후 샘플링
    buffer: list[dict] = []
    for row in ds:
        buffer.append(row)
        if len(buffer) >= n * 5:
            break

    rng.shuffle(buffer)
    selected = buffer[:n]

    profiles: list[PersonaProfile] = []
    for i, row in enumerate(selected):
        hobbies = _parse_list_field(row.get("hobbies_and_interests_list", "[]"))
        skills  = _parse_list_field(row.get("skills_and_expertise_list", "[]"))
        occupation = row.get("occupation", "")
        domains = _infer_domains(hobbies, occupation)
        profiles.append(PersonaProfile(
            uuid=row.get("uuid", f"fallback-{i}"),
            name=_extract_name(row.get("persona", "")),
            age=row.get("age", 30),
            sex=row.get("sex", ""),
            occupation=occupation,
            district=row.get("district", ""),
            province=row.get("province", ""),
            education_level=row.get("education_level", ""),
            marital_status=row.get("marital_status", ""),
            family_type=row.get("family_type", ""),
            hobbies=hobbies,
            skills=skills,
            career_goals=row.get("career_goals_and_ambitions", ""),
            persona_text=row.get("persona", ""),
            inferred_domains=domains,
        ))
    return profiles


def _fallback_personas(n: int, seed: int) -> list[PersonaProfile]:
    """HuggingFace 접근 불가 시 내장 fallback 페르소나 100개."""
    rng = random.Random(seed)

    TEMPLATES = [
        ("김민준", 28, "남자", "소프트웨어 개발자", "서울-강남구", "4년제 대학교",
         ["코딩", "독서", "러닝"], "개발자로 성장하고 싶다"),
        ("이서연", 25, "여자", "대학원생", "서울-관악구", "4년제 대학교",
         ["논문 읽기", "영어 공부", "수영"], "연구자가 되고 싶다"),
        ("박지호", 32, "남자", "회사원", "경기-성남시", "4년제 대학교",
         ["투자", "헬스", "기타"], "경제적 자유를 얻고 싶다"),
        ("최수아", 27, "여자", "디자이너", "서울-마포구", "4년제 대학교",
         ["그림 그리기", "사진", "요리"], "창작 활동을 계속하고 싶다"),
        ("정민혁", 30, "남자", "마케터", "서울-송파구", "4년제 대학교",
         ["블로그", "여행", "테니스"], "자기 브랜드를 만들고 싶다"),
        ("윤지은", 23, "여자", "대학생", "서울-서대문구", "재학 중",
         ["일본어", "독서", "피아노"], "일본어 JLPT N2를 따고 싶다"),
        ("강현우", 35, "남자", "프리랜서", "부산-해운대구", "4년제 대학교",
         ["클라이밍", "드로잉", "명상"], "건강한 라이프스타일을 유지하고 싶다"),
        ("임나경", 29, "여자", "간호사", "대구-수성구", "4년제 대학교",
         ["수채화", "달리기", "요가"], "10km 마라톤 완주가 목표다"),
        ("오승현", 31, "남자", "교사", "서울-노원구", "대학원",
         ["독서", "글쓰기", "배드민턴"], "교육 콘텐츠를 만들고 싶다"),
        ("한지민", 26, "여자", "회계사", "서울-중구", "4년제 대학교",
         ["저축", "요리", "영어"], "재무 독립을 이루고 싶다"),
    ]

    profiles: list[PersonaProfile] = []
    for i in range(n):
        tmpl = TEMPLATES[i % len(TEMPLATES)]
        name, age, sex, occ, dist, edu, hobbies, career = tmpl
        # 약간의 variation
        age_var = age + rng.randint(-2, 2)
        domains = _infer_domains(hobbies, occ)
        disp_name = f"{name}{i // len(TEMPLATES) + 1}" if i >= len(TEMPLATES) else name
        profiles.append(PersonaProfile(
            uuid=f"fallback-{i:04d}",
            name=disp_name,
            age=age_var,
            sex=sex,
            occupation=occ,
            district=dist,
            province=dist.split("-")[0] if "-" in dist else dist,
            education_level=edu,
            marital_status="",
            family_type="",
            hobbies=hobbies,
            skills=[],
            career_goals=career,
            persona_text=f"{disp_name} 씨는 {dist}에 사는 {age_var}세 {occ}입니다.",
            inferred_domains=domains,
        ))
    return profiles
