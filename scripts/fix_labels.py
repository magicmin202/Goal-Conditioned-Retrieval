#!/usr/bin/env python3
"""Relevant 레이블 포스트 프로세싱 — hobby/sports 교차 오염 수정.

문제: hobby 카테고리 goal이 동일 카테고리 내 다른 도메인 로그를 relevant로 레이블링.
     예: "사진 촬영 실력 키우기" → "요리 만들기" = relevant (잘못됨)
         "수영 자유형 마스터하기" → "테니스 연습" = relevant (잘못됨)

수정 기준 (타겟 카테고리만):
  1. goal이 특정 도메인 키워드를 포함하고 (예: "사진")
  2. log가 그 도메인 키워드를 하나도 포함하지 않고
  3. log가 다른 도메인 키워드를 포함하면 (예: "요리")
  → relevant → partial 다운그레이드

Usage:
    cd /home/user/retrieval
    .venv/bin/python scripts/fix_labels.py [--dry_run]
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path("data/synthetic")

# ── 도메인별 키워드 매핑 ──────────────────────────────────────────────────────
# (goal_keywords, log_keywords) — goal이 goal_kws를 포함하면 log도 log_kws를 포함해야 함
_DOMAIN_RULES: list[tuple[frozenset[str], frozenset[str]]] = [
    # hobby — photo
    (
        frozenset({"사진", "촬영", "포토"}),
        frozenset({"사진", "촬영", "카메라", "렌즈", "편집", "보정", "포토", "구도", "이미지"}),
    ),
    # hobby — cooking
    (
        frozenset({"요리", "쿠킹", "베이킹"}),
        frozenset({"요리", "레시피", "식재료", "조리", "요리법", "쿡", "음식", "베이킹", "식사"}),
    ),
    # hobby — drawing/art (goal keywords)
    (
        frozenset({"드로잉", "수채화", "일러스트"}),
        frozenset({"그림", "드로잉", "스케치", "일러스트", "수채화", "아크릴", "페인팅", "채색",
                   "사생", "미술", "화구", "붓", "물감", "캔버스"}),
    ),
    # sports — swimming
    (
        frozenset({"수영"}),
        frozenset({"수영", "자유형", "평영", "배영", "접영", "수영장", "레인", "턴",
                   "스포츠 센터", "훈련", "레슨", "수업", "코치", "실전", "연습"}),
    ),
    # sports — tennis
    (
        frozenset({"테니스"}),
        frozenset({"테니스", "라켓", "서브", "포핸드", "백핸드", "코트", "발리"}),
    ),
    # sports — climbing
    (
        frozenset({"클라이밍"}),
        frozenset({"클라이밍", "암벽", "볼더링", "등반", "홀드", "루트",
                   "훈련", "루틴", "연습", "실전", "레슨", "수업"}),
    ),
    # language — Japanese
    (
        frozenset({"일본어"}),
        frozenset({"일본어", "일어", "히라가나", "가타카나", "한자", "일본"}),
    ),
    # language — English conversation
    (
        frozenset({"영어 회화", "영어회화"}),
        frozenset({"영어", "회화", "speaking", "english", "pronunciation", "발음",
                   "언어 교환", "language", "exchange", "어휘", "단어", "플래시카드",
                   "문법", "듣기", "리스닝", "원어민"}),
    ),
]


def _has_any(text: str, keywords: frozenset[str]) -> bool:
    return any(kw in text for kw in keywords)


def _goal_matches_domain(goal_title: str, goal_kws: frozenset[str]) -> bool:
    return _has_any(goal_title, goal_kws)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true",
                        help="변경 없이 분석만 출력")
    args = parser.parse_args()

    print("[데이터 로드]")
    labels = json.load(open(DATA_DIR / "labels.json", encoding="utf-8"))
    logs   = json.load(open(DATA_DIR / "logs.json",   encoding="utf-8"))
    goals  = json.load(open(DATA_DIR / "goals.json",  encoding="utf-8"))

    log_map  = {l["log_id"]:  l for l in logs}
    goal_map = {g["goal_id"]: g for g in goals}

    dist_before: dict[str, int] = defaultdict(int)
    for lb in labels:
        dist_before[lb["label"]] += 1
    print(f"  레이블: {len(labels)}개")
    print(f"  수정 전 분포: {dict(dist_before)}")

    changed = 0
    change_log = []

    for lb in labels:
        if lb["label"] != "relevant":
            continue

        goal = goal_map.get(lb["goal_id"])
        log  = log_map.get(lb["log_id"])
        if not goal or not log:
            continue

        goal_title = goal["title"].lower()
        log_txt    = (log.get("title", "") + " " + log.get("content", "")).lower()

        for goal_kws, log_kws in _DOMAIN_RULES:
            # goal이 이 도메인에 해당하는가?
            if not _has_any(goal_title, goal_kws):
                continue
            # log에 이 도메인 키워드가 있는가?
            if not _has_any(log_txt, log_kws):
                lb["label"] = "partial"
                changed += 1
                change_log.append({
                    "goal":     goal["title"],
                    "log":      log["title"],
                    "domain":   sorted(goal_kws)[0],
                    "log_text": log.get("title", ""),
                })
                break   # 한 번만 수정

    print(f"\n수정된 레이블: {changed}개 (relevant → partial)")

    # 도메인별 통계
    domain_counts: dict[str, int] = defaultdict(int)
    for r in change_log:
        domain_counts[r["domain"]] += 1
    print("\n[도메인별 수정 수]")
    for d, cnt in sorted(domain_counts.items(), key=lambda x: -x[1]):
        print(f"  {d}: {cnt}개")

    # 샘플
    print("\n[수정 샘플 (20개)]")
    for r in change_log[:20]:
        print(f"  [{r['domain']}] goal: {r['goal']:<28}  log: {r['log_text']}")

    dist_after: dict[str, int] = defaultdict(int)
    for lb in labels:
        dist_after[lb["label"]] += 1
    print(f"\n수정 후 분포: {dict(dist_after)}")

    if not args.dry_run:
        out = DATA_DIR / "labels.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(labels, f, ensure_ascii=False, indent=2)
        print(f"\n[저장] {out}")
    else:
        print("\n[dry_run — 저장하지 않음]")


if __name__ == "__main__":
    main()
