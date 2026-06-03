#!/usr/bin/env python3
"""Step 1: Negative Penalty 수치 근거 실험
============================================
Part A — 매칭 타입별 신호 강도 분석
  negative_term 매칭 타입(phrase / token / daily)이 실제로 irrelevant 로그를
  얼마나 잘 식별하는지 측정 → 각 페널티 값의 상한 근거 도출

  측정값: irrelevant_rate = n_irrelevant / (n_relevant + n_irrelevant)
    - 1.0에 가까울수록 해당 신호가 강한 rejection 근거
    - 0.5에 가까울수록 신호가 noise (random)와 다를 바 없음

Part B — 5가지 패널티 설정 비교
  (phrase_penalty, token_penalty, daily_noise)를 5가지로 달리해서
  selected_precision / recall@k / F1@k / FPR 측정
  → 현재 (0.70/0.40/0.20)가 최적인지, 더 나은 설정이 있는지 확인

Usage:
    cd /home/user/retrieval
    .venv/bin/python validation/step1_negative_penalty/run.py
    .venv/bin/python validation/step1_negative_penalty/run.py --sample 50
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

import logging
logging.basicConfig(level=logging.WARNING)

from app.config import Stage1Config
from app.data_generation.export_utils import load_dataset_from_json
from app.evaluation.ranking_metrics import compute_all_metrics, compute_candidate_metrics
from app.pipeline.stage1_ranking_pipeline import Stage1Pipeline
from app.retrieval.query_expansion import _heuristic_expansion
from app.schemas import GoalLogLabel, ResearchGoal, ResearchLog

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = ROOT / "data" / "synthetic"
TOP_K = 10

# ── 5가지 패널티 케이스 ───────────────────────────────────────────────────────
#
# 비교 설계:
#   current      → 기존 설정 (baseline)
#   strict       → 모든 값 상향: 더 공격적으로 reject
#   lenient      → 모든 값 하향: recall 보호 우선
#   phrase_dominant → phrase만 신뢰, token은 거의 무시
#   equal_weight → phrase와 token을 동등하게 취급
#
CASES = [
    {"name": "current",         "phrase": 0.70, "token": 0.40, "daily": 0.20},
    {"name": "strict",          "phrase": 0.90, "token": 0.60, "daily": 0.30},
    {"name": "lenient",         "phrase": 0.50, "token": 0.20, "daily": 0.10},
    {"name": "phrase_dominant", "phrase": 1.00, "token": 0.10, "daily": 0.10},
    {"name": "equal_weight",    "phrase": 0.50, "token": 0.50, "daily": 0.20},
]


# ── 공통 유틸 ─────────────────────────────────────────────────────────────────

def _tok(text: str) -> set[str]:
    return set(re.findall(r"[\w가-힣]+", text.lower()))


def classify_negative_match(
    negative_terms: list[str],
    full_text: str,
) -> str:
    """negative_terms 대비 로그 매칭 타입 반환.

    penalty_score()와 동일한 규칙 적용:
      multi-token term → phrase-only (substring)
      single-token term → token match

    Returns: "phrase" | "token" | "none"
    """
    if not negative_terms:
        return "none"

    text_l = full_text.lower()
    text_toks = _tok(text_l)
    has_phrase = False
    has_token = False

    for term in negative_terms:
        term_l = term.lower()
        term_toks = _tok(term_l)
        is_multi = len(term_toks) > 1

        if is_multi:
            if term_l in text_l:
                has_phrase = True
        else:
            if term_toks & text_toks:
                has_token = True

    if has_phrase:
        return "phrase"
    if has_token:
        return "token"
    return "none"


def make_config(case: dict, top_k: int = TOP_K) -> Stage1Config:
    cfg = Stage1Config()
    cfg.retrieval.top_k = top_k
    cfg.ranker.negative_penalty_phrase = case["phrase"]
    cfg.ranker.negative_penalty_token  = case["token"]
    cfg.ranker.negative_daily_penalty  = case["daily"]
    return cfg


# ── Part A: 매칭 타입별 신호 강도 분석 ───────────────────────────────────────

def run_part_a(
    goals: list[ResearchGoal],
    logs: list[ResearchLog],
    labels: list[GoalLogLabel],
) -> None:
    print("\n" + "=" * 66)
    print("PART A  매칭 타입별 신호 강도 분석")
    print("=" * 66)
    print("측정: irrelevant_rate = n_irrelevant / (n_relevant + n_irrelevant)")
    print("  → 1.0에 가까울수록 강한 rejection 신호\n")

    goal_map = {g.goal_id: g for g in goals}
    log_map  = {l.log_id:  l for l in logs}

    # bucket → {"relevant": int, "irrelevant": int}
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"relevant": 0, "irrelevant": 0})

    skipped = 0
    for lbl in labels:
        goal = goal_map.get(lbl.goal_id)
        log  = log_map.get(lbl.log_id)
        if not goal or not log:
            skipped += 1
            continue
        if lbl.label not in ("relevant", "irrelevant"):
            continue

        heuristic  = _heuristic_expansion(goal, max_terms=15)
        neg_terms  = heuristic.get("negative_terms", [])
        match_type = classify_negative_match(neg_terms, log.full_text)
        is_daily   = (log.activity_type == "daily")

        # 6개 bucket으로 분류
        if is_daily and match_type == "none":
            bucket = "daily_only"
        elif is_daily and match_type == "phrase":
            bucket = "phrase+daily"
        elif is_daily and match_type == "token":
            bucket = "token+daily"
        elif match_type == "phrase":
            bucket = "phrase_only"
        elif match_type == "token":
            bucket = "token_only"
        else:
            bucket = "no_signal"

        counts[bucket][lbl.label] += 1

    # ── 결과 출력 ─────────────────────────────────────────────────────────────
    bucket_order = [
        "phrase_only", "phrase+daily",
        "token_only",  "token+daily",
        "daily_only",  "no_signal",
    ]
    rows = []
    for bucket in bucket_order:
        c = counts[bucket]
        rel = c["relevant"]
        irr = c["irrelevant"]
        total = rel + irr
        if total == 0:
            continue
        irr_rate = irr / total
        strength = (
            "★★★ 강함"  if irr_rate >= 0.85 else
            "★★  중간"  if irr_rate >= 0.70 else
            "★   약함"  if irr_rate >= 0.55 else
            "    노이즈"
        )
        rows.append({
            "bucket":          bucket,
            "n_relevant":      rel,
            "n_irrelevant":    irr,
            "n_total":         total,
            "irrelevant_rate": round(irr_rate, 4),
            "signal_strength": strength.strip(),
        })

    hdr = (
        f"{'매칭 타입':<16} {'relevant':>9} {'irrelevant':>11}"
        f" {'전체':>6} {'irr_rate':>9} {'신호강도'}"
    )
    print(hdr)
    print("-" * 68)
    for r in rows:
        print(
            f"{r['bucket']:<16} {r['n_relevant']:>9} {r['n_irrelevant']:>11}"
            f" {r['n_total']:>6} {r['irrelevant_rate']:>9.4f}  {r['signal_strength']}"
        )

    # ── 해석 가이드 ───────────────────────────────────────────────────────────
    print("\n[해석]")
    for r in rows:
        rate = r["irrelevant_rate"]
        if r["bucket"] == "no_signal":
            print(f"  no_signal baseline: {rate:.4f}  ← 이 값 이하면 신호 무의미")
        elif r["bucket"] == "phrase_only":
            print(f"  phrase_only:        {rate:.4f}  ← phrase penalty 상한 근거")
        elif r["bucket"] == "token_only":
            print(f"  token_only:         {rate:.4f}  ← token penalty 상한 근거")
        elif r["bucket"] == "daily_only":
            print(f"  daily_only:         {rate:.4f}  ← daily noise 상한 근거")

    # ── CSV 저장 ──────────────────────────────────────────────────────────────
    out = RESULTS_DIR / "part_a_match_distribution.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[저장] {out}")
    if skipped:
        print(f"(skipped {skipped}: goal or log not found)")


# ── Part B: 5가지 패널티 설정 비교 ───────────────────────────────────────────

def run_part_b(
    goals: list[ResearchGoal],
    logs: list[ResearchLog],
    labels: list[GoalLogLabel],
) -> None:
    print("\n" + "=" * 66)
    print("PART B  5가지 패널티 설정 비교")
    print("=" * 66)

    # 유저별 그룹화
    user_goals: dict[str, list[ResearchGoal]] = defaultdict(list)
    for g in goals:
        user_goals[g.user_id].append(g)

    user_labels_map: dict[str, list[GoalLogLabel]] = defaultdict(list)
    for lb in labels:
        user_labels_map[lb.user_id].append(lb)

    case_summary: list[dict] = []
    detail_rows:  list[dict] = []

    for ci, case in enumerate(CASES):
        print(
            f"\n  [{ci+1}/{len(CASES)}] {case['name']:<16}"
            f"  phrase={case['phrase']}  token={case['token']}  daily={case['daily']}"
        )
        cfg      = make_config(case)
        pipeline = Stage1Pipeline(config=cfg, use_real_embeddings=True)

        goal_metrics: list[dict] = []
        n_vetoed_total   = 0
        n_zero_total     = 0
        n_admitted_total = 0

        for user_id, u_goals in user_goals.items():
            user_logs  = [l for l in logs   if l.user_id == user_id]
            u_labels   = user_labels_map[user_id]
            if not user_logs or not u_labels:
                continue

            pipeline.index(user_logs)

            for goal in u_goals:
                goal_labels = [lb for lb in u_labels if lb.goal_id == goal.goal_id]
                if not goal_labels:
                    continue

                try:
                    result = pipeline.run(goal, use_expansion=False)
                except Exception as e:
                    print(f"    ERROR {goal.goal_id}: {e}")
                    continue

                # veto / zero-score 집계
                n_vetoed = sum(
                    1 for r in result.ranked_logs
                    if r.rejection_reason and "domain_conflict_veto" in r.rejection_reason
                )
                n_zero   = sum(1 for r in result.ranked_logs if r.final_score == 0.0)
                n_vetoed_total   += n_vetoed
                n_zero_total     += n_zero
                n_admitted_total += len(result.selected_logs)

                all_types = {l.activity_type for l in user_logs}
                m = compute_all_metrics(
                    result.ranked_logs,
                    goal_labels,
                    k=TOP_K,
                    all_activity_types=all_types,
                    selected_logs=result.selected_logs,
                )
                cand_m = compute_candidate_metrics(result.candidates, goal_labels)
                goal_metrics.append({**m, **cand_m})

                detail_rows.append({
                    "case":               case["name"],
                    "goal_id":            goal.goal_id,
                    "recall@k":           m[f"recall@{TOP_K}"],
                    "selected_precision": m["selected_precision"],
                    "f1@k":               m[f"f1@{TOP_K}"],
                    "false_positive_rate":m["false_positive_rate"],
                    "selected_count":     m["selected_count"],
                    "candidate_recall":   cand_m["candidate_recall"],
                    "n_vetoed":           n_vetoed,
                    "n_zero_score":       n_zero,
                })

        if not goal_metrics:
            continue

        def mean(key: str) -> float:
            vals = [m[key] for m in goal_metrics if key in m]
            return round(sum(vals) / len(vals), 4) if vals else 0.0

        n = len(goal_metrics)
        summary = {
            "case":               case["name"],
            "phrase_penalty":     case["phrase"],
            "token_penalty":      case["token"],
            "daily_noise":        case["daily"],
            "n_goals":            n,
            "candidate_recall":   mean("candidate_recall"),
            "recall@k":           mean(f"recall@{TOP_K}"),
            "selected_precision": mean("selected_precision"),
            "f1@k":               mean(f"f1@{TOP_K}"),
            "false_positive_rate":mean("false_positive_rate"),
            "avg_selected_count": round(n_admitted_total / n, 2),
            "total_vetoed":       n_vetoed_total,
            "total_zero_score":   n_zero_total,
        }
        case_summary.append(summary)

        print(
            f"    recall@k={summary['recall@k']:.4f}"
            f"  sel_P={summary['selected_precision']:.4f}"
            f"  F1={summary['f1@k']:.4f}"
            f"  FPR={summary['false_positive_rate']:.4f}"
            f"  vetoed={n_vetoed_total}"
            f"  zero={n_zero_total}"
            f"  (n={n})"
        )

    # ── 최종 비교 테이블 ──────────────────────────────────────────────────────
    if not case_summary:
        print("결과 없음")
        return

    print("\n" + "=" * 80)
    print("최종 비교 테이블")
    print("=" * 80)
    hdr = (
        f"{'case':<16} {'phrase':>6} {'token':>6} {'daily':>6}"
        f" {'rec@k':>6} {'sel_P':>6} {'F1@k':>6} {'FPR':>6}"
        f" {'vetoed':>7} {'zero':>6}"
    )
    print(hdr)
    print("-" * 80)
    for s in case_summary:
        print(
            f"{s['case']:<16} {s['phrase_penalty']:>6.2f} {s['token_penalty']:>6.2f}"
            f" {s['daily_noise']:>6.2f}"
            f" {s['recall@k']:>6.4f} {s['selected_precision']:>6.4f}"
            f" {s['f1@k']:>6.4f} {s['false_positive_rate']:>6.4f}"
            f" {s['total_vetoed']:>7} {s['total_zero_score']:>6}"
        )

    # ── 최고 케이스 추천 ──────────────────────────────────────────────────────
    best_f1 = max(case_summary, key=lambda x: x["f1@k"])
    best_p  = max(case_summary, key=lambda x: x["selected_precision"])
    best_r  = max(case_summary, key=lambda x: x["recall@k"])
    print(f"\n  F1@k 최고:         {best_f1['case']}  ({best_f1['f1@k']:.4f})")
    print(f"  Precision 최고:    {best_p['case']}  ({best_p['selected_precision']:.4f})")
    print(f"  Recall 최고:       {best_r['case']}  ({best_r['recall@k']:.4f})")

    # ── CSV 저장 ──────────────────────────────────────────────────────────────
    out_summary = RESULTS_DIR / "part_b_case_metrics.csv"
    with open(out_summary, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(case_summary[0].keys()))
        writer.writeheader()
        writer.writerows(case_summary)
    print(f"\n[저장] {out_summary}")

    out_detail = RESULTS_DIR / "part_b_detail.csv"
    if detail_rows:
        with open(out_detail, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
            writer.writeheader()
            writer.writerows(detail_rows)
        print(f"[저장] {out_detail}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Step 1: Negative Penalty 수치 근거 실험")
    parser.add_argument(
        "--sample", type=int, default=None,
        help="무작위 N개 goal만 사용 (전체 실행 전 빠른 검증용)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--part", choices=["a", "b", "all"], default="all",
        help="실행할 파트 선택 (default: all)",
    )
    args = parser.parse_args()

    print("[데이터 로드]")
    _, goals, logs, labels = load_dataset_from_json(str(DATA_DIR))
    print(f"  goals={len(goals)}  logs={len(logs)}  labels={len(labels)}")

    if args.sample:
        import random
        random.seed(args.seed)
        goals = random.sample(goals, min(args.sample, len(goals)))
        sampled_ids = {g.goal_id for g in goals}
        labels = [lb for lb in labels if lb.goal_id in sampled_ids]
        print(f"  [샘플링] {len(goals)}개 goal 사용 (seed={args.seed})")

    if args.part in ("a", "all"):
        run_part_a(goals, logs, labels)

    if args.part in ("b", "all"):
        run_part_b(goals, logs, labels)


if __name__ == "__main__":
    main()
