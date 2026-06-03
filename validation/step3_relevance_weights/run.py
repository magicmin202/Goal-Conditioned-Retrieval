#!/usr/bin/env python3
"""Step 3: Relevance Score 가중치 근거 실험
============================================
Phase 1 — Ablation
  컴포넌트를 하나씩 제거(weight=0)했을 때 F1 변화 측정
  → 각 컴포넌트의 실질 기여도 순위 도출

Phase 2 — Grid Search
  Ablation 결과 기반으로 중요 컴포넌트의 비율 탐색
  → F1 최고점 weight 조합 도출

평가 방법:
  - (goal, log, label) 전체 쌍에 대해 컴포넌트 점수 사전 계산 (임베딩 없음)
  - 각 weight 설정으로 final_score 재계산
  - threshold(0.08) 이상 → top-10 선택 → precision/recall/F1 계산
  - sem=0.0 고정 (query embedding 미포함, 상대 비교에 영향 없음)

Usage:
    cd /home/user/retrieval
    .venv/bin/python validation/step3_relevance_weights/run.py
    .venv/bin/python validation/step3_relevance_weights/run.py --phase 1
    .venv/bin/python validation/step3_relevance_weights/run.py --phase 2
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

import logging
logging.basicConfig(level=logging.WARNING)

from app.config import DEFAULT_CONFIG
from app.data_generation.export_utils import load_dataset_from_json
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.query_expansion import _heuristic_expansion
from app.schemas import GoalLogLabel, ResearchGoal, ResearchLog
from app.utils.text_matching import (
    _tok_set, match_priority_phrase, match_term, penalty_score,
)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = ROOT / "data" / "synthetic"

TOP_K     = 10
THRESHOLD = 0.08
CFG       = DEFAULT_CONFIG.stage1.ranker

# ── Phase 1: Ablation cases ───────────────────────────────────────────────────
#
# 하나씩 제거(0으로)해서 F1 변화량 측정
# → 변화가 클수록 해당 컴포넌트가 중요
#
ABLATION_CASES = [
    # 기존 ablation: 하나 제거 (나머지 고정, scale 보정됨)
    {"name": "baseline",    "pri": 0.35, "ev": 0.20, "rel": 0.10, "sem": 0.05, "base": 0.05},
    {"name": "no_priority", "pri": 0.00, "ev": 0.20, "rel": 0.10, "sem": 0.05, "base": 0.05},
    {"name": "no_evidence", "pri": 0.35, "ev": 0.00, "rel": 0.10, "sem": 0.05, "base": 0.05},
    {"name": "no_related",  "pri": 0.35, "ev": 0.20, "rel": 0.00, "sem": 0.05, "base": 0.05},
    {"name": "no_semantic", "pri": 0.35, "ev": 0.20, "rel": 0.10, "sem": 0.00, "base": 0.05},
    {"name": "no_base",     "pri": 0.35, "ev": 0.20, "rel": 0.10, "sem": 0.05, "base": 0.00},
    # 단독 실험: 하나만 활성화 (동일 조건에서 순수 기여도 비교)
    {"name": "only_priority", "pri": 1.00, "ev": 0.00, "rel": 0.00, "sem": 0.00, "base": 0.00},
    {"name": "only_evidence", "pri": 0.00, "ev": 1.00, "rel": 0.00, "sem": 0.00, "base": 0.00},
    {"name": "only_related",  "pri": 0.00, "ev": 0.00, "rel": 1.00, "sem": 0.00, "base": 0.00},
    {"name": "only_semantic", "pri": 0.00, "ev": 0.00, "rel": 0.00, "sem": 1.00, "base": 0.00},
    {"name": "only_base",     "pri": 0.00, "ev": 0.00, "rel": 0.00, "sem": 0.00, "base": 1.00},
]

# ── Phase 2: Grid Search cases ────────────────────────────────────────────────
#
# 설계 원칙:
#   - Ablation에서 중요하지 않은 컴포넌트는 0 또는 최소화
#   - priority:evidence 비율 탐색
#   - related 포함 여부 탐색
#   - 3가지 축: (1)priority 비중, (2)evidence 비중, (3)related 포함 여부
#
GRID_CASES = [
    # baseline
    {"name": "baseline",       "pri": 0.35, "ev": 0.20, "rel": 0.10, "sem": 0.05, "base": 0.05},
    # priority 비중 증가
    {"name": "pri_0.5",        "pri": 0.50, "ev": 0.20, "rel": 0.10, "sem": 0.05, "base": 0.05},
    {"name": "pri_0.6",        "pri": 0.60, "ev": 0.20, "rel": 0.10, "sem": 0.05, "base": 0.05},
    {"name": "pri_0.7",        "pri": 0.70, "ev": 0.15, "rel": 0.05, "sem": 0.05, "base": 0.05},
    # evidence 비중 증가
    {"name": "ev_0.35",        "pri": 0.35, "ev": 0.35, "rel": 0.10, "sem": 0.05, "base": 0.05},
    {"name": "ev_0.40",        "pri": 0.35, "ev": 0.40, "rel": 0.10, "sem": 0.05, "base": 0.05},
    # priority+evidence 균형
    {"name": "pri_ev_equal",   "pri": 0.40, "ev": 0.40, "rel": 0.10, "sem": 0.05, "base": 0.05},
    # related 제거
    {"name": "no_rel_pri_0.4", "pri": 0.40, "ev": 0.30, "rel": 0.00, "sem": 0.05, "base": 0.05},
    {"name": "no_rel_pri_0.5", "pri": 0.50, "ev": 0.35, "rel": 0.00, "sem": 0.05, "base": 0.05},
    # sem/base 제거 → 순수 lexical
    {"name": "lex_only",       "pri": 0.40, "ev": 0.30, "rel": 0.15, "sem": 0.00, "base": 0.00},
    {"name": "lex_pri_heavy",  "pri": 0.55, "ev": 0.30, "rel": 0.15, "sem": 0.00, "base": 0.00},
    # 2-component (priority + evidence만)
    {"name": "pri_ev_only",    "pri": 0.60, "ev": 0.40, "rel": 0.00, "sem": 0.00, "base": 0.00},
    # 1-component (priority만)
    {"name": "pri_only",       "pri": 1.00, "ev": 0.00, "rel": 0.00, "sem": 0.00, "base": 0.00},
]


# ── 사전 계산 ─────────────────────────────────────────────────────────────────

def precompute(
    goals: list[ResearchGoal],
    logs: list[ResearchLog],
    labels: list[GoalLogLabel],
) -> list[dict]:
    """(goal, log, label) 전체에 대해 컴포넌트 점수 사전 계산.

    semantic score: DenseRetriever로 실제 Gemini 임베딩 cosine 계산.
    로그 임베딩은 캐시에서 로드, query 임베딩은 미캐시 시 API 호출.
    """
    goal_map  = {g.goal_id: g for g in goals}
    log_map   = {l.log_id:  l for l in logs}
    exp_cache: dict[str, dict] = {}

    # 유저별 로그 그룹화 (DenseRetriever 인덱싱 재사용)
    user_logs: dict[str, list[ResearchLog]] = defaultdict(list)
    for l in logs:
        user_logs[l.user_id].append(l)

    # goal별 레이블 그룹화
    goal_labels: dict[str, list[GoalLogLabel]] = defaultdict(list)
    for lbl in labels:
        if lbl.label in ("relevant", "irrelevant"):
            goal_labels[lbl.goal_id].append(lbl)

    # user별 goal 그룹화
    user_goals: dict[str, list[ResearchGoal]] = defaultdict(list)
    for g in goals:
        if g.goal_id in goal_labels:
            user_goals[g.user_id].append(g)

    # DenseRetriever (실제 임베딩 사용)
    dense = DenseRetriever()   # GEMINI_API_KEY 있으면 자동으로 Gemini 사용

    records = []
    total_goals = sum(len(gs) for gs in user_goals.values())
    done = 0

    for user_id, u_goals in user_goals.items():
        u_logs = user_logs[user_id]
        if not u_logs:
            continue

        # 유저 로그 인덱싱 (캐시에서 로드)
        dense.index(u_logs)
        log_id_to_idx = {l.log_id: i for i, l in enumerate(u_logs)}

        for goal in u_goals:
            done += 1
            if done % 50 == 0 or done == total_goals:
                print(f"  [{done}/{total_goals}] {goal.goal_id} ...", flush=True)

            # goal query text로 전체 로그에 대한 semantic score 계산
            query_text = goal.query_text
            sem_pairs = dense.score_all(query_text)   # [(log, norm_cosine), ...]
            sem_map   = {log.log_id: score for log, score in sem_pairs}

            if goal.goal_id not in exp_cache:
                exp_cache[goal.goal_id] = _heuristic_expansion(goal, max_terms=15)
            exp = exp_cache[goal.goal_id]

            pri_terms = exp.get("priority_terms", [])
            ev_terms  = exp.get("evidence_terms", [])
            rel_terms = exp.get("related_terms",  [])
            neg_terms = exp.get("negative_terms", [])

            for lbl in goal_labels[goal.goal_id]:
                log = log_map.get(lbl.log_id)
                if not log:
                    continue

                text  = log.full_text
                title = log.title
                tl    = title.lower()
                tt    = _tok_set(tl)
                txl   = text.lower()
                txt   = _tok_set(txl)

                def _score_pri(terms):
                    if not terms:
                        return 0.0
                    matches = [match_priority_phrase(t, txl, txt, tl, tt) for t in terms]
                    total = sum(1.0 for m in matches if m.score > 0.0)
                    return min(total / len(terms), 1.0)

                def _score_rel(terms):
                    if not terms:
                        return 0.0
                    matches = [match_term(t, txl, txt, tl, tt) for t in terms]
                    total = sum(1.0 for m in matches if m.level != "none")
                    return min(total / len(terms), 1.0)

                goal_toks  = _tok_set(goal.query_text)
                base_score = len(goal_toks & txt) / len(goal_toks) if goal_toks else 0.0

                raw_dm, _ = penalty_score(
                    neg_terms, text, title,
                    phrase_penalty=CFG.negative_penalty_phrase,
                    token_penalty=CFG.negative_penalty_token,
                )
                noise  = CFG.negative_daily_penalty if log.activity_type == "daily" else 0.0
                capped = min(raw_dm + noise, 1.0)

                records.append({
                    "goal_id": lbl.goal_id,
                    "log_id":  lbl.log_id,
                    "label":   lbl.label,
                    "pri":     _score_pri(pri_terms),
                    "ev":      _score_pri(ev_terms),
                    "rel":     _score_rel(rel_terms),
                    "sem":     sem_map.get(lbl.log_id, 0.0),
                    "base":    base_score,
                    "raw_dm":  raw_dm,
                    "capped":  capped,
                })

    return records


# ── 스코어 계산 ───────────────────────────────────────────────────────────────

def compute_final(r: dict, w: dict) -> float:
    """주어진 weight로 final_score 계산."""
    pri_w  = w["pri"]
    ev_w   = w["ev"]
    rel_w  = w["rel"]
    sem_w  = w["sem"]
    base_w = w["base"]

    total_w = pri_w + ev_w + rel_w + sem_w + base_w
    if total_w == 0:
        return 0.0
    scale = 1.0 / total_w

    # Veto: pri_score 재계산 필요 (pri_w=0이어도 veto 판단은 원래 pri_score로)
    veto_pri = r["pri"]  # veto는 실제 pri_score로 판단
    if (CFG.negative_veto_enabled
            and r["raw_dm"] >= CFG.negative_veto_dm_threshold
            and veto_pri < CFG.negative_veto_priority_min):
        return 0.0

    rel_score = scale * (
        pri_w  * r["pri"]
        + ev_w   * r["ev"]
        + rel_w  * r["rel"]
        + sem_w  * r["sem"]
        + base_w * r["base"]
    )
    return max(0.0, round(rel_score - r["capped"], 4))


# ── 메트릭 계산 ───────────────────────────────────────────────────────────────

def evaluate(records: list[dict], w: dict) -> dict:
    """weight 설정 하나에 대한 전체 메트릭 계산."""
    goal_records: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        goal_records[r["goal_id"]].append(r)

    goal_metrics = []
    for goal_id, grecs in goal_records.items():
        scored = [(r["log_id"], r["label"], compute_final(r, w)) for r in grecs]
        admitted = sorted(
            [(lid, lbl, fs) for lid, lbl, fs in scored if fs >= THRESHOLD],
            key=lambda x: x[2], reverse=True,
        )
        selected = admitted[:TOP_K]

        total_rel = sum(1 for _, lbl, _ in scored if lbl == "relevant")
        if total_rel == 0:
            continue

        tp        = sum(1 for _, lbl, _ in selected if lbl == "relevant")
        sel_n     = len(selected)
        precision = tp / sel_n if sel_n > 0 else 0.0
        recall    = tp / total_rel
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        fpr       = 1.0 - precision if sel_n > 0 else 0.0

        goal_metrics.append({
            "precision": precision, "recall": recall,
            "f1": f1, "fpr": fpr, "sel_n": sel_n,
        })

    def mean(k):
        vals = [m[k] for m in goal_metrics]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    return {
        "n_goals":   len(goal_metrics),
        "precision": mean("precision"),
        "recall":    mean("recall"),
        "f1":        mean("f1"),
        "fpr":       mean("fpr"),
        "avg_sel":   mean("sel_n"),
    }


# ── Phase 1: Ablation ─────────────────────────────────────────────────────────

def run_phase1(records: list[dict]) -> list[dict]:
    print("\n" + "=" * 68)
    print("PHASE 1  Ablation — 컴포넌트 기여도 측정")
    print("=" * 68)

    baseline_m = None
    rows = []

    for case in ABLATION_CASES:
        w = {k: case[k] for k in ("pri", "ev", "rel", "sem", "base")}
        m = evaluate(records, w)
        if case["name"] == "baseline":
            baseline_m = m
        rows.append({**case, **m})

    # ── 그룹 A: 제거 실험 ────────────────────────────────────────────────────
    print("\n  [A] 제거 실험 — 하나를 0으로 (나머지 scale 자동 보정)")
    print(f"  {'case':<16} {'precision':>10} {'recall':>8} {'F1':>8}  {'F1 변화':>8}")
    print("  " + "-" * 56)
    remove_cases = [r for r in rows if not r["name"].startswith("only_")]
    for r in remove_cases:
        delta = f"{r['f1'] - baseline_m['f1']:+.4f}" if r["name"] != "baseline" else "  (기준)"
        print(
            f"  {r['name']:<16} {r['precision']:>10.4f}"
            f" {r['recall']:>8.4f} {r['f1']:>8.4f}  {delta:>8}"
        )

    # ── 그룹 B: 단독 실험 ────────────────────────────────────────────────────
    print("\n  [B] 단독 실험 — 하나만 활성화 (동일 조건 순수 비교)")
    print(f"  {'case':<16} {'precision':>10} {'recall':>8} {'F1':>8}")
    print("  " + "-" * 46)
    only_cases = sorted(
        [r for r in rows if r["name"].startswith("only_")],
        key=lambda x: x["f1"], reverse=True,
    )
    for r in only_cases:
        print(
            f"  {r['name']:<16} {r['precision']:>10.4f}"
            f" {r['recall']:>8.4f} {r['f1']:>8.4f}"
        )

    # ── 단독 실험 기반 중요도 순위 ────────────────────────────────────────────
    print("\n  [단독 실험 기반 중요도 순위]")
    for i, r in enumerate(only_cases, 1):
        comp = r["name"].replace("only_", "")
        sign = "중요" if r["f1"] > 0.05 else "미미"
        print(f"  {i}. {comp:<12}  F1={r['f1']:.4f}  ({sign})")

    out = RESULTS_DIR / "phase1_ablation.csv"
    save_csv(rows, out)
    print(f"\n[저장] {out}")
    return rows


# ── Phase 2: Grid Search ──────────────────────────────────────────────────────

def run_phase2(records: list[dict]) -> list[dict]:
    print("\n" + "=" * 68)
    print("PHASE 2  Grid Search — 최적 weight 탐색")
    print("=" * 68)
    print(f"  threshold={THRESHOLD}  top_k={TOP_K}\n")

    rows = []
    for case in GRID_CASES:
        w = {k: case[k] for k in ("pri", "ev", "rel", "sem", "base")}
        m = evaluate(records, w)
        rows.append({**case, **m})
        print(
            f"  [{case['name']:<16}]"
            f"  pri={case['pri']:.2f}  ev={case['ev']:.2f}"
            f"  rel={case['rel']:.2f}  sem={case['sem']:.2f}  base={case['base']:.2f}"
            f"  →  F1={m['f1']:.4f}"
            f"  P={m['precision']:.4f}"
            f"  R={m['recall']:.4f}"
        )

    # 비교 테이블
    print("\n" + "=" * 80)
    print("비교 테이블  (F1 내림차순)")
    print("=" * 80)
    sorted_rows = sorted(rows, key=lambda x: x["f1"], reverse=True)
    hdr = (
        f"{'case':<18} {'pri':>5} {'ev':>5} {'rel':>5} {'sem':>5} {'base':>5}"
        f"  {'precision':>10} {'recall':>8} {'F1':>8} {'FPR':>6}"
    )
    print(hdr)
    print("-" * 80)
    for r in sorted_rows:
        print(
            f"{r['name']:<18} {r['pri']:>5.2f} {r['ev']:>5.2f}"
            f" {r['rel']:>5.2f} {r['sem']:>5.2f} {r['base']:>5.2f}"
            f"  {r['precision']:>10.4f} {r['recall']:>8.4f}"
            f" {r['f1']:>8.4f} {r['fpr']:>6.4f}"
        )

    # 최고 케이스 추천
    best     = sorted_rows[0]
    best_r   = max(rows, key=lambda x: x["recall"])
    best_p   = max(rows, key=lambda x: x["precision"])
    print(f"\n  F1 최고:         {best['name']:<18}  F1={best['f1']:.4f}")
    print(f"  Recall 최고:     {best_r['name']:<18}  R={best_r['recall']:.4f}")
    print(f"  Precision 최고:  {best_p['name']:<18}  P={best_p['precision']:.4f}")
    print(f"\n  [권장 설정]  pri={best['pri']}  ev={best['ev']}"
          f"  rel={best['rel']}  sem={best['sem']}  base={best['base']}")

    out = RESULTS_DIR / "phase2_grid.csv"
    save_csv(rows, out)
    print(f"\n[저장] {out}")
    return rows


# ── Phase 3: Semantic Weight Sweep (recall 기준) ──────────────────────────────

# semantic weight를 0.05에서 1.0까지 sweep
# 나머지 lexical 가중치는 비례해서 줄여나감
# Phase 1 [B] 단독 실험 recall 비율로 weight 설정
# only_semantic=0.757, only_evidence=0.246, only_priority=0.200,
# only_base=0.144, only_related=0.120
# 비교군: all_equal, sem_1.00, 기존 baseline
RECALL_CASES = [
    # 핵심 케이스: sem=0.70 sweep1 최적 + ev만 0.06→0.20으로 상향
    {"name": "target",        "pri": 0.10, "ev": 0.20, "rel": 0.03, "sem": 0.70, "base": 0.05},
    # 비교군
    {"name": "sweep1_best",   "pri": 0.10, "ev": 0.06, "rel": 0.03, "sem": 0.70, "base": 0.05},
    {"name": "sem_1.00_fix",  "pri": 0.10, "ev": 0.20, "rel": 0.10, "sem": 1.00, "base": 0.10},
    {"name": "only_semantic", "pri": 0.00, "ev": 0.00, "rel": 0.00, "sem": 1.00, "base": 0.00},
    {"name": "baseline",      "pri": 0.35, "ev": 0.20, "rel": 0.10, "sem": 0.05, "base": 0.05},
]


def run_phase3(records: list[dict]) -> list[dict]:
    print("\n" + "=" * 68)
    print("PHASE 3  Semantic Weight Sweep — recall 기준")
    print("=" * 68)
    print("  semantic weight를 0.05→1.0으로 증가, 나머지 lexical 비례 감소\n")

    rows = []
    for case in RECALL_CASES:
        w = {k: case[k] for k in ("pri", "ev", "rel", "sem", "base")}
        m = evaluate(records, w)
        rows.append({**case, **m})
        print(
            f"  [{case['name']:<10}]"
            f"  sem={case['sem']:.2f}  pri={case['pri']:.2f}  ev={case['ev']:.2f}"
            f"  rel={case['rel']:.2f}  base={case['base']:.2f}"
            f"  →  recall={m['recall']:.4f}"
            f"  precision={m['precision']:.4f}"
            f"  F1={m['f1']:.4f}"
        )

    # recall 기준 정렬
    print("\n" + "=" * 72)
    print("비교 테이블  (recall 내림차순)")
    print("=" * 72)
    sorted_rows = sorted(rows, key=lambda x: x["recall"], reverse=True)
    hdr = (
        f"{'case':<12} {'sem':>5} {'pri':>5} {'ev':>5} {'rel':>5}"
        f"  {'recall':>8} {'precision':>10} {'F1':>8} {'FPR':>6}"
    )
    print(hdr)
    print("-" * 72)
    for r in sorted_rows:
        print(
            f"{r['name']:<12} {r['sem']:>5.2f} {r['pri']:>5.2f}"
            f" {r['ev']:>5.2f} {r['rel']:>5.2f}"
            f"  {r['recall']:>8.4f} {r['precision']:>10.4f}"
            f" {r['f1']:>8.4f} {r['fpr']:>6.4f}"
        )

    best_r = sorted_rows[0]
    best_f = max(rows, key=lambda x: x["f1"])
    print(f"\n  Recall 최고:  {best_r['name']}  R={best_r['recall']:.4f}"
          f"  P={best_r['precision']:.4f}  F1={best_r['f1']:.4f}")
    print(f"  F1 최고:      {best_f['name']}  R={best_f['recall']:.4f}"
          f"  P={best_f['precision']:.4f}  F1={best_f['f1']:.4f}")

    # recall 우선 권장 설정
    print(f"\n  [recall 우선 권장]  sem={best_r['sem']}  pri={best_r['pri']}"
          f"  ev={best_r['ev']}  rel={best_r['rel']}  base={best_r['base']}")

    out = RESULTS_DIR / "phase3_recall_sweep.csv"
    save_csv(rows, out)
    print(f"\n[저장] {out}")
    return rows


# ── CSV 저장 ──────────────────────────────────────────────────────────────────

def save_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Step 3: Relevance Score 가중치 실험")
    parser.add_argument(
        "--phase", choices=["1", "2", "3", "all"], default="all",
        help="실행할 phase (default: all)",
    )
    args = parser.parse_args()

    print("[데이터 로드]")
    _, goals, logs, labels = load_dataset_from_json(str(DATA_DIR))
    print(f"  goals={len(goals)}  logs={len(logs)}  labels={len(labels)}")

    print("\n[사전 계산 — 컴포넌트 점수 수집]")
    records = precompute(goals, logs, labels)
    print(f"  처리 완료: {len(records)}개 (goal, log) 쌍")

    if args.phase in ("1", "all"):
        run_phase1(records)

    if args.phase in ("2", "all"):
        run_phase2(records)

    if args.phase in ("3", "all"):
        run_phase3(records)


if __name__ == "__main__":
    main()
