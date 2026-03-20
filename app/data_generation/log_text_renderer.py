"""Render log skeleton into natural language title + content.

The same topic is rendered with surface-level variation to simulate
realistic log diversity. Evidence strength controls specificity.
"""
from __future__ import annotations

import random
from app.data_generation.log_skeleton_generator import LogSkeleton
from app.schemas import ResearchLog

# ── Title templates ──────────────────────────────────────────────────────────
# {topic} is substituted in; multiple variants per topic for diversity.
_TITLE_VARIANTS: dict[str, list[str]] = {
    "알고리즘 문제 풀이":      ["알고리즘 문제 풀기", "코딩 문제 풀이", "알고리즘 연습"],
    "포트폴리오 프로젝트 구현": ["포트폴리오 작업", "프로젝트 구현", "개인 프로젝트 개발"],
    "자료구조 개념 정리":      ["자료구조 공부", "자료구조 복습", "CS 기초 정리"],
    "취업 계획 수립":          ["취업 계획 세우기", "커리어 로드맵 정리", "취업 준비 계획"],
    "개발 기술 블로그 읽기":   ["기술 블로그 정독", "개발 아티클 읽기", "기술 트렌드 조사"],
    "코딩 테스트 연습":        ["코딩 테스트 문제 풀기", "알고리즘 테스트 연습", "코테 대비"],
    "논문 읽기":               ["논문 정독", "논문 리뷰", "연구 논문 읽기"],
    "연구계획서 작성":         ["연구계획서 초안 작성", "연구 방향 정리", "대학원 연구계획 작성"],
    "영어 단어 암기":          ["영어 단어 학습", "토익 어휘 암기", "영어 공부"],
    "교수님 연구 자료 정리":   ["연구실 자료 정리", "교수님 논문 정리", "연구 자료 검토"],
    "연구실 탐색":             ["연구실 조사", "지도교수 탐색", "대학원 연구실 리서치"],
    "GRE 문제 풀이":           ["GRE 연습 문제 풀기", "GRE 수학 섹션 학습", "GRE 대비"],
    "헬스장 운동":             ["헬스장 방문", "웨이트 트레이닝", "근력 운동"],
    "러닝":                    ["조깅", "러닝 운동", "아침 달리기"],
    "스트레칭":                ["스트레칭 루틴", "유연성 운동", "스트레칭"],
    "운동 계획 작성":          ["운동 루틴 계획", "헬스 계획 수립", "운동 스케줄 정리"],
    "홈트레이닝":              ["홈트 실시", "집에서 운동", "홈 워크아웃"],
    "소개팅":                  ["소개팅 참석", "첫 만남", "소개팅 미팅"],
    "친구 모임":               ["친구들과 만남", "지인 모임", "친구 약속"],
    "대화 주제 정리":          ["대화 준비", "관심사 정리", "관계 계획"],
    "데이트 계획 수립":        ["데이트 코스 계획", "주말 데이트 준비", "만남 일정 조율"],
    "항공권 검색":             ["항공권 비교 검색", "비행 일정 조사", "항공편 탐색"],
    "숙소 예약":               ["숙소 예약 완료", "호텔 예약", "에어비앤비 예약"],
    "여행 후기 조사":          ["여행 블로그 읽기", "여행지 정보 조사", "여행 후기 탐색"],
    "여행 예산 정리":          ["여행 예산 계획", "여행 비용 정리", "예산 관리"],
    "여행 준비물 구매":        ["여행 준비물 쇼핑", "여행용품 구매", "짐 준비"],
    "가계부 정리":             ["가계부 작성", "지출 기록", "이번 달 지출 정리"],
    "ETF 공부":                ["ETF 투자 학습", "ETF 개념 정리", "인덱스 펀드 공부"],
    "저축 계획 작성":          ["저축 계획 수립", "재테크 계획", "적금 계획 작성"],
    "주식 기초 학습":          ["주식 공부", "주식 시장 공부", "주식 기초 정리"],
    "소액 투자 실행":          ["소액 투자 시작", "첫 주식 매수", "ETF 소액 투자"],
    "독서":                    ["책 읽기", "독서", "도서 읽기"],
    "독서 목록 작성":          ["읽을 책 목록 작성", "독서 계획 수립", "책 리스트 정리"],
    "아침 루틴 실행":          ["아침 루틴 완료", "조기 기상 성공", "오전 루틴 실행"],
    "하루 반성 일지":          ["하루 회고 작성", "일일 반성", "오늘 회고"],
    "촬영 외출":               ["사진 촬영 외출", "카메라 들고 외출", "포토 산책"],
    "사진 편집 연습":          ["라이트룸 편집", "사진 보정 연습", "포토샵 편집"],
    "요리 실습":               ["요리 만들기", "새 요리 시도", "쿠킹 실습"],
    "레시피 조사":             ["요리 레시피 찾기", "새 요리 레시피 조사", "요리 블로그 탐색"],
    # noise
    "점심 식사":  ["점심 먹기", "식사", "외식"],
    "카페 방문":  ["카페에서 시간 보내기", "카페 방문", "카페 작업"],
    "친구와 통화": ["친구 전화 통화", "지인과 통화", "카카오톡 통화"],
    "유튜브 시청": ["유튜브 보기", "영상 시청", "유튜브 서핑"],
    "산책":       ["산책하기", "동네 산책", "저녁 산책"],
    "청소":       ["집 청소", "청소", "방 청소"],
    "마트 장보기": ["장보기", "마트 방문", "식재료 구매"],
    "드라마 시청": ["드라마 보기", "넷플릭스 시청", "TV 드라마"],
    "낮잠":       ["낮잠 자기", "잠깐 휴식", "오후 낮잠"],
    "SNS 확인":   ["인스타그램 확인", "소셜 미디어 확인", "SNS 서핑"],
}

# ── Content templates by evidence strength ────────────────────────────────────
_CONTENT_HIGH = [
    "{topic}을(를) 완료했다. 주요 내용을 정리하고 결과를 기록했다.",
    "{topic} 작업을 수행하고 결과를 검토했다. 다음 단계로 넘어갈 준비가 됐다.",
    "{topic}을(를) 진행하며 구체적인 성과를 얻었다. 오답/실수를 정리했다.",
    "{topic}을(를) 마무리했다. 오늘 목표한 분량을 달성했다.",
]
_CONTENT_MEDIUM = [
    "{topic}을(를) 어느 정도 진행했다.",
    "{topic} 관련 내용을 복습하고 요점을 메모했다.",
    "{topic} 작업을 진행했다. 일부 내용은 내일 이어서 할 예정이다.",
    "{topic}을(를) 공부했다. 개념은 잡혔는데 연습이 더 필요하다.",
]
_CONTENT_LOW = [
    "{topic}을(를) 해야겠다고 생각했다.",
    "오늘 {topic}을(를) 잠깐 살펴봤다. 본격적으로 시작하지는 못했다.",
    "{topic}에 대해 막연하게 생각해봤다.",
    "{topic} 관련 내용을 흘깃 봤다.",
]
_CONTENT_NOISE = [
    "오늘 {topic}을(를) 했다.",
    "{topic}. 특별한 일은 없었다.",
    "{topic}을(를) 하며 하루를 보냈다.",
]


def render_log_text(skeleton: LogSkeleton, seed_offset: int = 0) -> ResearchLog:
    """Render a LogSkeleton into a ResearchLog with title + content.

    Surface variation is applied so identical topics produce different text.
    """
    rng = random.Random(hash(skeleton.log_id) + seed_offset)

    # Title
    variants = _TITLE_VARIANTS.get(skeleton.topic, [skeleton.topic])
    title = rng.choice(variants)

    # Content
    if skeleton.goal_id is None:
        tmpl = rng.choice(_CONTENT_NOISE)
    elif skeleton.evidence_strength == "high":
        tmpl = rng.choice(_CONTENT_HIGH)
    elif skeleton.evidence_strength == "medium":
        tmpl = rng.choice(_CONTENT_MEDIUM)
    else:
        tmpl = rng.choice(_CONTENT_LOW)

    content = tmpl.format(topic=skeleton.topic)

    return ResearchLog(
        log_id=skeleton.log_id,
        user_id=skeleton.user_id,
        date=skeleton.date,
        title=title,
        content=content,
        activity_type=skeleton.activity_type,
        metadata={
            "topic": skeleton.topic,
            "evidence_strength": skeleton.evidence_strength,
            "goal_id_hint": skeleton.goal_id,
        },
        created_at=skeleton.date,
    )
