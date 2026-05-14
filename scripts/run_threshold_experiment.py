#!/usr/bin/env python3
"""Dense threshold experiment script.

6개 threshold case로 전체 데이터셋을 실행하고
precision@k, recall@k, FPR 등을 집계하여 CSV로 저장한다.

Usage:
    .venv/bin/python scripts/run_threshold_experiment.py
    .venv/bin/python scripts/run_threshold_experiment.py --top_k 5 --data_dir data/synthetic
    .venv/bin/python scripts/run_threshold_experiment.py --dry_run   # 실행 계획만 출력
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

# thresholds: case name → normalized dense_score cutoff
THRESHOLDS: dict[str, float] = {
    "case1_0.96": 0.96,
    "case2_0.94": 0.94,
    "case3_0.92": 0.92,
    "case4_0.90": 0.90,
    "case5_0.88": 0.88,
    "case6_0.85": 0.85,
}

RESULTS_DIR = Path("results/threshold_experiment")
VENV_PYTHON = ".venv/bin/python"


def load_goal_ids(data_dir: str) -> list[str]:
    """data_dir/goals.json에서 전체 goal_id 목록 로드."""
    goals_path = Path(data_dir) / "goals.json"
    if not goals_path.exists():
        print(f"[ERROR] goals.json not found at {goals_path}")
        sys.exit(1)
    with open(goals_path, encoding="utf-8") as f:
        goals = json.load(f)
    return [g["goal_id"] for g in goals]


def run_single(
    goal_id: str,
    threshold: float,
    case_name: str,
    top_k: int,
    data_dir: str,
) -> dict | None:
    """단일 goal + threshold 실행 후 결과 dict 반환."""
    result_path = RESULTS_DIR / f"{case_name}_{goal_id}.json"

    if result_path.exists():
        # 캐시 활용: 이미 실행된 결과 재사용
        with open(result_path, encoding="utf-8") as f:
            return json.load(f)

    cmd = [
        VENV_PYTHON, "scripts/run_stage1.py",
        "--goal_id", goal_id,
        "--top_k", str(top_k),
        "--real_embeddings",
        "--baseline", "ours",
        "--dense_threshold", str(threshold),
        "--save_result",
        "--result_path", str(result_path),
        "--data_dir", data_dir,
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.returncode != 0:
        # stderr 마지막 300자만 출력 (긴 traceback 방지)
        err_tail = proc.stderr.strip()[-300:] if proc.stderr else "(no stderr)"
        print(f"\n  [ERROR] {goal_id} / {case_name}:\n  {err_tail}")
        return None

    if result_path.exists():
        with open(result_path, encoding="utf-8") as f:
            return json.load(f)
    return None


def extract_metrics(result: dict | None, k: int) -> dict:
    """결과 JSON에서 지표 추출.

    result_writer.py 저장 구조:
        {"metrics": {"recall@k": ..., "precision@k": ..., ...}}
    """
    _empty = {
        f"precision@{k}": None,
        f"recall@{k}": None,
        "fpr": None,
        f"f1@{k}": None,
        f"ndcg@{k}": None,
        "candidate_count": None,
        "admitted_count": None,
    }
    if result is None:
        return _empty

    m = result.get("metrics", {})
    return {
        f"precision@{k}": m.get(f"precision@{k}"),
        f"recall@{k}":    m.get(f"recall@{k}"),
        "fpr":            m.get("false_positive_rate"),
        f"f1@{k}":        m.get(f"f1@{k}"),
        f"ndcg@{k}":      m.get(f"ndcg@{k}"),
        "candidate_count": m.get("candidate_count"),
        "admitted_count":  m.get("admitted_count"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Dense threshold experiment")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--data_dir", default="data/synthetic")
    parser.add_argument(
        "--dry_run", action="store_true",
        help="실행 계획만 출력, 실제 실행 안 함",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    k = args.top_k

    goal_ids = load_goal_ids(args.data_dir)
    total_runs = len(goal_ids) * len(THRESHOLDS)

    print(f"전체 goal 수    : {len(goal_ids)}")
    print(f"Threshold cases : {len(THRESHOLDS)}")
    print(f"총 실행 횟수    : {total_runs}")
    print(f"결과 저장 위치  : {RESULTS_DIR}/\n")

    if args.dry_run:
        print("[DRY RUN] 실행 계획:")
        for case_name, threshold in THRESHOLDS.items():
            print(f"  {case_name}  threshold={threshold}  → {len(goal_ids)} goals")
        return

    all_rows: list[dict] = []

    for case_name, threshold in THRESHOLDS.items():
        print(f"\n{'='*60}")
        print(f"[{case_name}]  threshold={threshold}")
        print(f"{'='*60}")

        case_metrics: list[dict] = []

        for i, goal_id in enumerate(goal_ids):
            cached = (RESULTS_DIR / f"{case_name}_{goal_id}.json").exists()
            tag = "(cached)" if cached else ""
            print(f"  [{i+1:2d}/{len(goal_ids)}] {goal_id} {tag} ...", end=" ", flush=True)

            result = run_single(goal_id, threshold, case_name, k, args.data_dir)
            m = extract_metrics(result, k)

            p = m[f"precision@{k}"]
            r_val = m[f"recall@{k}"]
            if p is not None and r_val is not None:
                print(f"P={p:.3f}  R={r_val:.3f}  FPR={m['fpr']:.3f}  "
                      f"cand={m['candidate_count']}  admit={m['admitted_count']}")
            else:
                print("FAIL")

            row = {
                "case": case_name,
                "threshold": threshold,
                "goal_id": goal_id,
                **m,
            }
            all_rows.append(row)
            case_metrics.append(m)

        # case별 평균 출력
        valid = [m for m in case_metrics if m[f"precision@{k}"] is not None]
        if valid:
            avg_p   = sum(m[f"precision@{k}"] for m in valid) / len(valid)
            avg_r   = sum(m[f"recall@{k}"]    for m in valid) / len(valid)
            avg_fpr = sum(m["fpr"]             for m in valid) / len(valid)
            avg_f1  = sum(m[f"f1@{k}"]         for m in valid) / len(valid)
            avg_c   = sum(m["candidate_count"] or 0 for m in valid) / len(valid)
            avg_a   = sum(m["admitted_count"]  or 0 for m in valid) / len(valid)
            print(
                f"\n  [{case_name}] 평균 — "
                f"P={avg_p:.3f}  R={avg_r:.3f}  FPR={avg_fpr:.3f}  "
                f"F1={avg_f1:.3f}  cand={avg_c:.1f}  admit={avg_a:.1f}  "
                f"(유효 {len(valid)}/{len(goal_ids)})"
            )

    if not all_rows:
        print("\n[ERROR] 실행된 결과 없음. 위 오류 메시지를 확인하세요.")
        sys.exit(1)

    # ── 전체 결과 CSV ──────────────────────────────────────────────────────────
    csv_path = RESULTS_DIR / "threshold_experiment_all.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n전체 결과 저장: {csv_path}")

    # ── 요약 CSV ──────────────────────────────────────────────────────────────
    summary_rows: list[dict] = []
    for case_name, threshold in THRESHOLDS.items():
        case_rows = [r for r in all_rows if r["case"] == case_name]
        valid = [r for r in case_rows if r[f"precision@{k}"] is not None]
        if not valid:
            continue
        summary_rows.append({
            "case":             case_name,
            "threshold":        threshold,
            "goal_count":       len(valid),
            "avg_precision":    round(sum(r[f"precision@{k}"] for r in valid) / len(valid), 4),
            "avg_recall":       round(sum(r[f"recall@{k}"]    for r in valid) / len(valid), 4),
            "avg_fpr":          round(sum(r["fpr"]             for r in valid) / len(valid), 4),
            "avg_f1":           round(sum(r[f"f1@{k}"]         for r in valid) / len(valid), 4),
            "avg_ndcg":         round(sum(r[f"ndcg@{k}"]       for r in valid) / len(valid), 4),
            "avg_candidates":   round(sum(r["candidate_count"] or 0 for r in valid) / len(valid), 1),
            "avg_admitted":     round(sum(r["admitted_count"]  or 0 for r in valid) / len(valid), 1),
            "fail_count":       len(case_rows) - len(valid),
        })

    summary_path = RESULTS_DIR / "threshold_experiment_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"요약 결과 저장: {summary_path}")

    # ── 최종 콘솔 요약 ────────────────────────────────────────────────────────
    col = f"{'case':<20} {'thr':<6} {'P@'+str(k):<8} {'R@'+str(k):<8} {'FPR':<8} {'F1':<8} {'cand':<7} {'admit':<7} {'fail':<5}"
    print("\n" + "=" * len(col))
    print(col)
    print("-" * len(col))
    for row in summary_rows:
        print(
            f"{row['case']:<20} {row['threshold']:<6} "
            f"{row['avg_precision']:<8} {row['avg_recall']:<8} "
            f"{row['avg_fpr']:<8} {row['avg_f1']:<8} "
            f"{row['avg_candidates']:<7} {row['avg_admitted']:<7} "
            f"{row['fail_count']:<5}"
        )
    print("=" * len(col))

    # ── 최적 threshold 판단 ───────────────────────────────────────────────────
    if summary_rows:
        best_f1  = max(summary_rows, key=lambda r: r["avg_f1"])
        best_rec = max(summary_rows, key=lambda r: r["avg_recall"])
        best_prec = max(summary_rows, key=lambda r: r["avg_precision"])
        # FPR ≤ 0.20 중 Recall 최대
        low_fpr = [r for r in summary_rows if r["avg_fpr"] <= 0.20]
        best_balanced = max(low_fpr, key=lambda r: r["avg_recall"]) if low_fpr else best_f1

        print("\n[최적 Threshold 판단]")
        print(f"  F1 최고       : {best_f1['case']}  threshold={best_f1['threshold']}  F1={best_f1['avg_f1']}")
        print(f"  Recall 최고   : {best_rec['case']}  threshold={best_rec['threshold']}  R={best_rec['avg_recall']}")
        print(f"  Precision 최고: {best_prec['case']}  threshold={best_prec['threshold']}  P={best_prec['avg_precision']}")
        print(f"  권장 (FPR≤0.20 & Recall↑): {best_balanced['case']}  threshold={best_balanced['threshold']}")


if __name__ == "__main__":
    main()
