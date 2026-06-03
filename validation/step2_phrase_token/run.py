#!/usr/bin/env python3
"""Step 2: Phrase / Token 가중치 비율 근거 실험
================================================
Part A — 신호 정밀도 분석
  priority_terms 매칭 타입별 P(relevant | match_type) 측정
  → phrase match가 token match보다 얼마나 더 신뢰할 수 있는 positive 신호인가
  → 이 비율이 phrase_weight / token_weight 의 이상적 비율 근거가 됨

Part B — 5가지 phrase:token 비율 비교
  phrase_weight=1.0 고정, token_weight를 0.0~1.0 변경
  priority / evidence / related 전부 동일 비율 적용
  → F1@k / selected_precision / recall@k 비교

평가 방법:
  - 전체 레이블 쌍(28233개)에 대해 재점수 계산 (임베딩 호출 없음)
  - 임계값(0.08) 이상 로그 중 top-10 선택 → precision/recall/F1 계산
  - sem=0.0 고정 (모든 케이스 동일 조건으로 비율만 비교)

Usage:
    cd /home/user/retrieval
    .venv/bin/python validation/step2_phrase_token/run.py
    .venv/bin/python validation/step2_phrase_token/run.py --part a
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

import logging
logging.basicConfig(level=logging.WARNING)

from app.config import DEFAULT_CONFIG
from app.data_generation.export_utils import load_dataset_from_json
from app.retrieval.query_expansion import _heuristic_expansion
from app.schemas import GoalLogLabel, ResearchGoal, ResearchLog
from app.utils.text_matching import (
    PriorityTermMatch, TermMatch,
    _tok_set, match_priority_phrase, match_term,
    penalty_score, score_priority_terms, score_terms,
)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = ROOT / "data" / "synthetic"

TOP_K       = 10
THRESHOLD   = 0.08   # Stage1Config diversity.relevance_threshold
CFG         = DEFAULT_CONFIG.stage1.ranker

# ── 5가지 케이스 ──────────────────────────────────────────────────────────────
#
# phrase_weight=1.0 고정, token_weight만 변경
# → max_possible = len(terms) × 1.0 으로 동일 → 순수 비율만 비교
#
CASES = [
    {"name": "equal",        "phrase": 1.0, "token": 1.0},   # 1:1   구분 없음
    {"name": "slight_gap",   "phrase": 1.0, "token": 0.7},   # 1.4:1
    {"name": "moderate_gap", "phrase": 1.0, "token": 0.5},   # 2:1
    {"name": "large_gap",    "phrase": 1.0, "token": 0.3},   # 3.3:1 ≈ 현재(1.5/0.4)
    {"name": "phrase_only",  "phrase": 1.0, "token": 0.0},   # ∞:1   phrase만 인정
]


# ── 스코어 재계산 헬퍼 ────────────────────────────────────────────────────────

def _pri_score(matches: list[PriorityTermMatch], terms: list[str],
               phrase_w: float, token_w: float) -> float:
    """PriorityTermMatch 결과로 score_priority_terms와 동일 계산."""
    if not terms:
        return 0.0
    max_p = len(terms) * phrase_w
    if max_p == 0:
        return 0.0
    total = sum(
        phrase_w if m.mode == "exact_phrase" else token_w
        for m in matches if m.score > 0.0
    )
    return min(total / max_p, 1.0)


def _rel_score(matches: list[TermMatch], terms: list[str],
               phrase_w: float, token_w: float) -> float:
    """TermMatch 결과로 score_terms와 동일 계산."""
    if not terms:
        return 0.0
    max_p = len(terms) * phrase_w
    if max_p == 0:
        return 0.0
    total = sum(
        phrase_w if m.level == "phrase" else token_w
        for m in matches if m.level != "none"
    )
    return min(total / max_p, 1.0)


def _final_score(
    pri_score: float, ev_score: float, rel_score: float,
    sem: float, base: float,
    raw_dm: float, capped: float,
    pri_score_for_veto: float,
) -> float:
    """relevance_score - capped 계산. veto 체크 포함."""
    # Negative Veto
    if (CFG.negative_veto_enabled
            and raw_dm >= CFG.negative_veto_dm_threshold
            and pri_score_for_veto < CFG.negative_veto_priority_min):
        return 0.0

    total_w = (CFG.priority_weight + CFG.evidence_weight + CFG.related_weight
               + CFG.semantic_weight + CFG.base_weight)
    scale = 1.0 / total_w
    rel = scale * (
        CFG.priority_weight * pri_score
        + CFG.evidence_weight * ev_score
        + CFG.related_weight * rel_score
        + CFG.semantic_weight * sem
        + CFG.base_weight * base
    )
    return max(0.0, round(rel - capped, 4))


# ── 사전 계산 ─────────────────────────────────────────────────────────────────

def precompute(
    goals: list[ResearchGoal],
    logs: list[ResearchLog],
    labels: list[GoalLogLabel],
) -> list[dict]:
    """(goal, log, label) 전체에 대해 매칭 데이터 사전 계산.

    임베딩 없이 순수 lexical 데이터만 수집.
    """
    goal_map = {g.goal_id: g for g in goals}
    log_map  = {l.log_id:  l for l in logs}

    # 목표별 heuristic 캐싱
    expansion_cache: dict[str, dict] = {}

    records = []
    skipped = 0

    for lbl in labels:
        if lbl.label not in ("relevant", "irrelevant"):
            continue
        goal = goal_map.get(lbl.goal_id)
        log  = log_map.get(lbl.log_id)
        if not goal or not log:
            skipped += 1
            continue

        if lbl.goal_id not in expansion_cache:
            expansion_cache[lbl.goal_id] = _heuristic_expansion(goal, max_terms=15)
        exp = expansion_cache[lbl.goal_id]

        pri_terms = exp.get("priority_terms", [])
        ev_terms  = exp.get("evidence_terms", [])
        rel_terms = exp.get("related_terms", [])
        neg_terms = exp.get("negative_terms", [])

        text  = log.full_text
        title = log.title
        tl    = title.lower()
        tt    = _tok_set(tl)
        txl   = text.lower()
        txt   = _tok_set(txl)

        # Priority/Evidence 매칭 (weak-token 필터 있음)
        pri_matches = [
            match_priority_phrase(t, txl, txt, tl, tt) for t in pri_terms
        ]
        ev_matches = [
            match_priority_phrase(t, txl, txt, tl, tt) for t in ev_terms
        ]

        # Related 매칭 (weak-token 필터 없음)
        rel_matches = [
            match_term(t, txl, txt, tl, tt) for t in rel_terms
        ]

        # Base goal overlap
        goal_toks = _tok_set(goal.query_text)
        base = len(goal_toks & txt) / len(goal_toks) if goal_toks else 0.0

        # Negative penalty
        raw_dm, _ = penalty_score(
            neg_terms, text, title,
            phrase_penalty=CFG.negative_penalty_phrase,
            token_penalty=CFG.negative_penalty_token,
        )
        noise  = CFG.negative_daily_penalty if log.activity_type == "daily" else 0.0
        capped = min(raw_dm + noise, 1.0)

        records.append({
            "goal_id":     lbl.goal_id,
            "log_id":      lbl.log_id,
            "label":       lbl.label,
            "pri_matches": pri_matches,
            "ev_matches":  ev_matches,
            "rel_matches": rel_matches,
            "sem":         0.0,   # sem 고정 (모든 케이스 동일 조건)
            "base":        base,
            "raw_dm":      raw_dm,
            "capped":      capped,
            "pri_terms":   pri_terms,
            "ev_terms":    ev_terms,
            "rel_terms":   rel_terms,
        })

    if skipped:
        print(f"  (skipped {skipped}: goal or log not found)")
    return records


# ── Part A ────────────────────────────────────────────────────────────────────

def run_part_a(records: list[dict]) -> None:
    print("\n" + "=" * 66)
    print("PART A  Priority Terms 매칭 타입별 신호 정밀도")
    print("=" * 66)
    print("측정: relevant_rate = n_relevant / (n_relevant + n_irrelevant)")
    print("  → 1.0에 가까울수록 강한 positive 신호\n")

    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"relevant": 0, "irrelevant": 0})

    for r in records:
        label = r["label"]
        has_phrase = any(m.mode == "exact_phrase" for m in r["pri_matches"])
        has_token  = any(m.mode == "core_token"   for m in r["pri_matches"])

        if has_phrase:
            bucket = "phrase_match"
        elif has_token:
            bucket = "token_only"
        else:
            bucket = "no_match"

        counts[bucket][label] += 1

    bucket_order = ["phrase_match", "token_only", "no_match"]
    rows = []
    no_match_rate = 0.0

    for bucket in bucket_order:
        c     = counts[bucket]
        rel   = c["relevant"]
        irr   = c["irrelevant"]
        total = rel + irr
        if total == 0:
            continue
        rel_rate = rel / total
        if bucket == "no_match":
            no_match_rate = rel_rate

        rows.append({
            "bucket":       bucket,
            "n_relevant":   rel,
            "n_irrelevant": irr,
            "n_total":      total,
            "relevant_rate": round(rel_rate, 4),
        })

    hdr = (
        f"{'매칭 타입':<16} {'relevant':>9} {'irrelevant':>11}"
        f" {'전체':>7} {'relevant_rate':>14}"
    )
    print(hdr)
    print("-" * 60)
    for r in rows:
        print(
            f"{r['bucket']:<16} {r['n_relevant']:>9} {r['n_irrelevant']:>11}"
            f" {r['n_total']:>7} {r['relevant_rate']:>14.4f}"
        )

    # 이상적 비율 계산
    phrase_rate = next((r["relevant_rate"] for r in rows if r["bucket"] == "phrase_match"), 0)
    token_rate  = next((r["relevant_rate"] for r in rows if r["bucket"] == "token_only"),  0)

    print("\n[해석]")
    print(f"  no_match baseline:     {no_match_rate:.4f}")
    print(f"  P(relevant | phrase):  {phrase_rate:.4f}")
    print(f"  P(relevant | token):   {token_rate:.4f}")
    if token_rate > 0:
        ideal_ratio = phrase_rate / token_rate
        print(f"\n  이상적 phrase:token 비율 = {phrase_rate:.4f} / {token_rate:.4f} = {ideal_ratio:.2f}:1")
        print(f"  현재 비율 (1.5/0.4)     = {1.5/0.4:.2f}:1")
    else:
        print("\n  token match relevant_rate = 0 → token signal 의미 없음")
        print("  → phrase_only (token_weight=0) 가능성 있음")

    out = RESULTS_DIR / "part_a_signal_precision.csv"
    if rows:
        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"\n[저장] {out}")


# ── Part B ────────────────────────────────────────────────────────────────────

def evaluate_case(
    records: list[dict],
    phrase_w: float,
    token_w: float,
) -> dict[str, float]:
    """한 케이스의 메트릭 계산.

    - 레이블 전체를 재점수 계산
    - goal별 top-k 선택 후 precision/recall/F1 집계
    """
    # goal별로 그룹화
    goal_records: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        goal_records[r["goal_id"]].append(r)

    goal_metrics = []

    for goal_id, grecs in goal_records.items():
        scored = []
        for r in grecs:
            ps = _pri_score(r["pri_matches"], r["pri_terms"], phrase_w, token_w)
            es = _pri_score(r["ev_matches"],  r["ev_terms"],  phrase_w, token_w)
            rs = _rel_score(r["rel_matches"], r["rel_terms"], phrase_w, token_w)

            fs = _final_score(
                ps, es, rs,
                r["sem"], r["base"],
                r["raw_dm"], r["capped"],
                pri_score_for_veto=ps,
            )
            scored.append((r["log_id"], r["label"], fs))

        # threshold 통과 후 top-k
        admitted = [(lid, lbl, fs) for lid, lbl, fs in scored if fs >= THRESHOLD]
        admitted.sort(key=lambda x: x[2], reverse=True)
        selected = admitted[:TOP_K]

        total_relevant = sum(1 for _, lbl, _ in scored if lbl == "relevant")
        if total_relevant == 0:
            continue

        tp = sum(1 for _, lbl, _ in selected if lbl == "relevant")
        sel_n = len(selected)

        precision = tp / sel_n       if sel_n > 0          else 0.0
        recall    = tp / total_relevant
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        fpr       = 1.0 - precision if sel_n > 0 else 0.0

        # token_only relevant 탈락 수 추적 (phrase_only 케이스에서 의미 있음)
        # relevant이면서 phrase match 없고 token match만 있는 로그 중
        # 최종 탈락한 것
        token_only_relevant_dropped = 0
        if token_w == 0.0:
            for r in grecs:
                if r["label"] != "relevant":
                    continue
                has_phrase = any(m.mode == "exact_phrase" for m in r["pri_matches"])
                has_token  = any(m.mode == "core_token"   for m in r["pri_matches"])
                if not has_phrase and has_token:
                    # phrase_only 케이스에서 이 로그의 점수를 재계산
                    fs = _final_score(
                        _pri_score(r["pri_matches"], r["pri_terms"], phrase_w, token_w),
                        _pri_score(r["ev_matches"],  r["ev_terms"],  phrase_w, token_w),
                        _rel_score(r["rel_matches"], r["rel_terms"], phrase_w, token_w),
                        r["sem"], r["base"], r["raw_dm"], r["capped"],
                        pri_score_for_veto=0.0,
                    )
                    if fs < THRESHOLD:
                        token_only_relevant_dropped += 1

        goal_metrics.append({
            "precision":    precision,
            "recall":       recall,
            "f1":           f1,
            "fpr":          fpr,
            "selected_n":   sel_n,
            "token_only_rel_dropped": token_only_relevant_dropped,
        })

    def mean(k):
        vals = [m[k] for m in goal_metrics]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    return {
        "n_goals":     len(goal_metrics),
        "precision":   mean("precision"),
        "recall":      mean("recall"),
        "f1":          mean("f1"),
        "fpr":         mean("fpr"),
        "avg_selected": mean("selected_n"),
        "token_only_rel_dropped": sum(m["token_only_rel_dropped"] for m in goal_metrics),
    }


def run_part_b(records: list[dict]) -> None:
    print("\n" + "=" * 66)
    print("PART B  5가지 phrase:token 비율 비교")
    print("=" * 66)
    print(f"  sem=0.0 고정 (모든 케이스 동일 조건)  threshold={THRESHOLD}  top_k={TOP_K}\n")

    summary = []
    for c in CASES:
        m = evaluate_case(records, c["phrase"], c["token"])
        row = {
            "case":    c["name"],
            "phrase":  c["phrase"],
            "token":   c["token"],
            "ratio":   f"{c['phrase']/c['token']:.1f}:1" if c["token"] > 0 else "∞:1",
            **m,
        }
        summary.append(row)
        print(
            f"  [{c['name']:<12}]  phrase={c['phrase']}  token={c['token']}"
            f"  ratio={row['ratio']:>6}"
            f"  precision={m['precision']:.4f}"
            f"  recall={m['recall']:.4f}"
            f"  F1={m['f1']:.4f}"
            f"  FPR={m['fpr']:.4f}"
            + (f"  token_rel_dropped={m['token_only_rel_dropped']}"
               if c["token"] == 0.0 else "")
        )

    # 최종 비교 테이블
    print("\n" + "=" * 72)
    print("비교 테이블")
    print("=" * 72)
    hdr = (
        f"{'case':<14} {'ratio':>6}"
        f" {'precision':>10} {'recall':>8} {'F1':>8} {'FPR':>8}"
        f" {'avg_sel':>8}"
    )
    print(hdr)
    print("-" * 72)
    for r in summary:
        print(
            f"{r['case']:<14} {r['ratio']:>6}"
            f" {r['precision']:>10.4f} {r['recall']:>8.4f}"
            f" {r['f1']:>8.4f} {r['fpr']:>8.4f}"
            f" {r['avg_selected']:>8.2f}"
        )

    # 추천
    best_f1 = max(summary, key=lambda x: x["f1"])
    best_p  = max(summary, key=lambda x: x["precision"])
    best_r  = max(summary, key=lambda x: x["recall"])
    print(f"\n  F1 최고:         {best_f1['case']}  ({best_f1['f1']:.4f})")
    print(f"  Precision 최고:  {best_p['case']}  ({best_p['precision']:.4f})")
    print(f"  Recall 최고:     {best_r['case']}  ({best_r['recall']:.4f})")

    # phrase_only에서 token_only relevant가 몇 개 탈락했는지
    phrase_only = next((r for r in summary if r["case"] == "phrase_only"), None)
    if phrase_only:
        dropped = phrase_only["token_only_rel_dropped"]
        print(f"\n  phrase_only 케이스에서 token-only relevant 로그 탈락: {dropped}개")
        if dropped == 0:
            print("  → token_weight 제거해도 recall 손실 없음 (구조 단순화 가능)")
        else:
            print("  → token_weight 완전 제거 시 recall 손실 발생")

    # CSV 저장
    out = RESULTS_DIR / "part_b_case_metrics.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    print(f"\n[저장] {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Step 2: Phrase/Token 가중치 비율 실험")
    parser.add_argument("--part", choices=["a", "b", "all"], default="all")
    args = parser.parse_args()

    print("[데이터 로드]")
    _, goals, logs, labels = load_dataset_from_json(str(DATA_DIR))
    print(f"  goals={len(goals)}  logs={len(logs)}  labels={len(labels)}")

    print("\n[사전 계산 — 매칭 데이터 수집]")
    records = precompute(goals, logs, labels)
    print(f"  처리 완료: {len(records)}개 (goal, log) 쌍")

    if args.part in ("a", "all"):
        run_part_a(records)

    if args.part in ("b", "all"):
        run_part_b(records)


if __name__ == "__main__":
    main()
