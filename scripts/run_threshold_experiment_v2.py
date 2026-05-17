#!/usr/bin/env python3
"""Dense threshold 실험 스크립트 v2.

설계:
  goal 1개당 DenseRetriever 직접 호출 (Gate/Reranker 없음)
  threshold 6개를 메모리에서 동시 평가
  개별 JSON 저장 없음 → CSV 2개만 저장
  resume 지원 (중단 후 재실행 시 완료된 goal 자동 skip)

Usage:
    .venv/bin/python scripts/run_threshold_experiment_v2.py
    .venv/bin/python scripts/run_threshold_experiment_v2.py --top_k 5
    .venv/bin/python scripts/run_threshold_experiment_v2.py --dry_run
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

from app.utils.logging_utils import setup_logging
setup_logging(level=logging.WARNING)   # 실험 중 INFO 로그 억제

THRESHOLDS: dict[str, float] = {
    "case1": 0.96,
    "case2": 0.94,
    "case3": 0.92,
    "case4": 0.90,
    "case5": 0.88,
    "case6": 0.85,
}

TOP_K = 5
RESULTS_DIR = Path("results/threshold_experiment")
ALL_CSV     = RESULTS_DIR / "threshold_experiment_all.csv"
SUMMARY_CSV = RESULTS_DIR / "threshold_experiment_summary.csv"

ALL_FIELDNAMES = [
    "case", "threshold", "goal_id",
    "precision_at_k", "recall_at_k", "f1_at_k", "fpr",
    "candidate_count", "admitted_count",
    "relevant_count", "tp", "fp",
]


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data(data_dir: str):
    from app.data_generation.export_utils import load_dataset_from_json
    _, goals, logs, labels = load_dataset_from_json(data_dir)

    # goal_id → {log_id: label}
    label_index: dict[str, dict[str, str]] = {}
    for lb in labels:
        label_index.setdefault(lb.goal_id, {})[lb.log_id] = lb.label

    # user_id → logs
    logs_by_user: dict[str, list] = {}
    for log in logs:
        logs_by_user.setdefault(log.user_id, []).append(log)

    return goals, label_index, logs_by_user


# ── Dense-only candidate retrieval (Method B) ─────────────────────────────────

def get_all_candidates(goal, logs):
    """DenseRetriever만 직접 호출 — Gate/Reranker 없음.

    1. query expansion으로 dense_query 생성 (Gemini → heuristic fallback)
    2. DenseRetriever로 전체 로그 score 계산 (score_all)
    3. CandidateLog 리스트 반환 (dense_score 기준 내림차순)

    반환값: list[CandidateLog]  (.log_id, .dense_score)
    """
    from app.retrieval.query_understanding import build_query
    from app.retrieval.query_expansion import expand_goal_query
    from app.retrieval.dense_retriever import DenseRetriever
    from app.retrieval.embedding_provider import get_embedding_provider
    from app.schemas import CandidateLog

    # 1. Dense query 생성 (expansion 실패 시 canonical_text fallback)
    query_obj = build_query(goal)
    try:
        expanded = expand_goal_query(
            goal, query_obj, use_mock_fallback=True
        )
        dense_text = expanded.dense_query
    except Exception:
        dense_text = query_obj.canonical_text

    # 2. DenseRetriever — doc provider: Gemini embedding-001
    provider = get_embedding_provider(real=True)
    dense = DenseRetriever(doc_provider=provider)
    dense.index(logs)

    # 3. 전체 로그 score (정규화된 cosine similarity)
    pairs = dense.score_all(dense_text)            # (ResearchLog, float) list

    candidates = [
        CandidateLog(log=log, dense_score=round(score, 6))
        for log, score in sorted(pairs, key=lambda x: x[1], reverse=True)
    ]
    return candidates


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(
    selected_ids: list[str],
    relevant_ids: set[str],
    candidate_count: int,
    total_logs: int,
    k: int,
) -> dict:
    tp = sum(1 for sid in selected_ids if sid in relevant_ids)
    fp = len(selected_ids) - tp
    total_irrel = total_logs - len(relevant_ids)
    fpr       = fp / total_irrel if total_irrel > 0 else 0.0
    precision = tp / len(selected_ids) if selected_ids else 0.0
    recall    = tp / len(relevant_ids) if relevant_ids else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return {
        "precision_at_k":  round(precision, 4),
        "recall_at_k":     round(recall,    4),
        "f1_at_k":         round(f1,        4),
        "fpr":             round(fpr,        4),
        "candidate_count": candidate_count,
        "admitted_count":  len(selected_ids),
        "relevant_count":  len(relevant_ids),
        "tp": tp,
        "fp": fp,
    }


# ── Summary ────────────────────────────────────────────────────────────────────

def save_and_print_summary(case_accum: dict[str, list[dict]]) -> None:
    summary_rows: list[dict] = []

    for case_name, threshold in THRESHOLDS.items():
        rows = case_accum.get(case_name, [])
        if not rows:
            continue

        def avg(key: str) -> float:
            vals = [r[key] for r in rows if r.get(key) is not None]
            return round(sum(vals) / len(vals), 4) if vals else 0.0

        summary_rows.append({
            "case":           case_name,
            "threshold":      threshold,
            "goal_count":     len(rows),
            "avg_precision":  avg("precision_at_k"),
            "avg_recall":     avg("recall_at_k"),
            "avg_fpr":        avg("fpr"),
            "avg_f1":         avg("f1_at_k"),
            "avg_candidates": avg("candidate_count"),
            "avg_admitted":   avg("admitted_count"),
        })

    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    col = (f"{'case':<8} {'thresh':<8} {'goals':<7} "
           f"{'P@5':<8} {'R@5':<8} {'FPR':<8} {'F1':<8} "
           f"{'cand':<8} {'admit':<6}")
    sep = "=" * len(col)
    print(f"\n{sep}\n{col}\n{'-'*len(col)}")
    for row in summary_rows:
        print(
            f"{row['case']:<8} {row['threshold']:<8} {row['goal_count']:<7} "
            f"{row['avg_precision']:<8} {row['avg_recall']:<8} "
            f"{row['avg_fpr']:<8} {row['avg_f1']:<8} "
            f"{row['avg_candidates']:<8} {row['avg_admitted']:<6}"
        )
    print(sep)
    print(f"\n전체 결과: {ALL_CSV}")
    print(f"요약 결과: {SUMMARY_CSV}")

    # 최적 threshold 판단
    if summary_rows:
        best_f1      = max(summary_rows, key=lambda r: r["avg_f1"])
        best_rec     = max(summary_rows, key=lambda r: r["avg_recall"])
        best_prec    = max(summary_rows, key=lambda r: r["avg_precision"])
        low_fpr      = [r for r in summary_rows if r["avg_fpr"] <= 0.20]
        best_balanced = (max(low_fpr, key=lambda r: r["avg_recall"])
                         if low_fpr else best_f1)
        print("\n[최적 Threshold 판단]")
        print(f"  F1 최고              : {best_f1['case']}  "
              f"(thr={best_f1['threshold']})  F1={best_f1['avg_f1']}")
        print(f"  Recall 최고          : {best_rec['case']}  "
              f"(thr={best_rec['threshold']})  R={best_rec['avg_recall']}")
        print(f"  Precision 최고       : {best_prec['case']}  "
              f"(thr={best_prec['threshold']})  P={best_prec['avg_precision']}")
        print(f"  권장 (FPR≤0.20 & ↑R): {best_balanced['case']}  "
              f"(thr={best_balanced['threshold']})")


# ── Resume helper ──────────────────────────────────────────────────────────────

def load_processed_goals() -> set[str]:
    """ALL_CSV에 이미 기록된 goal_id 집합 반환 (resume용)."""
    if not ALL_CSV.exists():
        return set()
    done: set[str] = set()
    with open(ALL_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            done.add(row["goal_id"])
    return done


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Dense threshold experiment v2")
    parser.add_argument("--top_k",    type=int, default=TOP_K)
    parser.add_argument("--data_dir", default="data/synthetic")
    parser.add_argument("--dry_run",  action="store_true",
                        help="실행 계획만 출력, 실제 실행 안 함")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    k = args.top_k

    goals, label_index, logs_by_user = load_data(args.data_dir)
    processed = load_processed_goals()

    remaining = [g for g in goals if g.goal_id not in processed]

    print(f"총 goal 수    : {len(goals)}")
    print(f"완료(skip)    : {len(processed)}")
    print(f"실행 예정     : {len(remaining)}")
    print(f"threshold 수  : {len(THRESHOLDS)}  (메모리 처리)")
    print(f"결과 저장     : {ALL_CSV.name}, {SUMMARY_CSV.name}\n")

    if args.dry_run:
        print("[DRY RUN] threshold cases:")
        for c, t in THRESHOLDS.items():
            print(f"  {c}: {t}")
        return

    if not remaining:
        print("모든 goal 처리 완료. summary만 재생성합니다.")
        # all.csv에서 accum 재구성 후 summary 생성
        case_accum: dict[str, list[dict]] = {c: [] for c in THRESHOLDS}
        with open(ALL_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                case_accum[row["case"]].append({
                    "precision_at_k": float(row["precision_at_k"]),
                    "recall_at_k":    float(row["recall_at_k"]),
                    "f1_at_k":        float(row["f1_at_k"]),
                    "fpr":            float(row["fpr"]),
                    "candidate_count": int(row["candidate_count"]),
                    "admitted_count": int(row["admitted_count"]),
                })
        save_and_print_summary(case_accum)
        return

    # resume: CSV를 append 모드로 열기 (새 파일이면 header 추가)
    csv_mode = "a" if ALL_CSV.exists() else "w"
    case_accum = {c: [] for c in THRESHOLDS}

    # 기존 CSV 데이터를 accum에 로드 (summary 계산에 필요)
    if ALL_CSV.exists():
        with open(ALL_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                c = row["case"]
                if c in case_accum:
                    case_accum[c].append({
                        "precision_at_k": float(row["precision_at_k"]),
                        "recall_at_k":    float(row["recall_at_k"]),
                        "f1_at_k":        float(row["f1_at_k"]),
                        "fpr":            float(row["fpr"]),
                        "candidate_count": int(row["candidate_count"]),
                        "admitted_count": int(row["admitted_count"]),
                    })

    fail_goals: list[str] = []

    with open(ALL_CSV, csv_mode, newline="", encoding="utf-8") as all_f:
        writer = csv.DictWriter(all_f, fieldnames=ALL_FIELDNAMES)
        if csv_mode == "w":
            writer.writeheader()

        for i, goal in enumerate(remaining):
            user_logs = logs_by_user.get(goal.user_id, [])
            relevant_ids = {
                lid for lid, lbl in label_index.get(goal.goal_id, {}).items()
                if lbl == "relevant"
            }

            tag = f"[{i+1:3d}/{len(remaining)}]"
            print(f"{tag} {goal.goal_id} ({goal.title[:22]:<22}) "
                  f"logs={len(user_logs)}  rel={len(relevant_ids)} ",
                  end="", flush=True)

            if not user_logs or not relevant_ids:
                reason = "로그 없음" if not user_logs else "relevant 없음"
                print(f"⚠ {reason} — skip")
                fail_goals.append(goal.goal_id)
                continue

            try:
                candidates = get_all_candidates(goal, user_logs)
            except Exception as exc:
                print(f"❌ {exc}")
                fail_goals.append(goal.goal_id)
                continue

            line_parts: list[str] = []
            for case_name, threshold in THRESHOLDS.items():
                passed = [c for c in candidates if c.dense_score >= threshold]
                selected = sorted(
                    passed, key=lambda c: c.dense_score, reverse=True
                )[:k]

                m = compute_metrics(
                    selected_ids=[c.log_id for c in selected],
                    relevant_ids=relevant_ids,
                    candidate_count=len(passed),
                    total_logs=len(user_logs),
                    k=k,
                )
                row = {
                    "case": case_name, "threshold": threshold,
                    "goal_id": goal.goal_id, **m,
                }
                writer.writerow(row)
                case_accum[case_name].append(m)
                line_parts.append(
                    f"{case_name}:c{m['candidate_count']}"
                    f"/P{m['precision_at_k']:.2f}"
                    f"/R{m['recall_at_k']:.2f}"
                )

            all_f.flush()
            print("  " + "  ".join(line_parts))

    print(f"\n전체 결과 저장: {ALL_CSV}")
    if fail_goals:
        print(f"실패/skip: {len(fail_goals)}개 — {fail_goals[:10]}")

    save_and_print_summary(case_accum)


if __name__ == "__main__":
    main()
