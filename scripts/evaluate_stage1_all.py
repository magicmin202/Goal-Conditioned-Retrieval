#!/usr/bin/env python3
"""Stage 1 전체 데이터 배치 평가.

모든 goal에 대해 현재 시스템(ours)으로 Stage 1을 실행하고
per-goal 메트릭을 계산한 뒤 전체 평균을 출력하고
results/stage1_eval_all.csv 에 저장한다.

Usage:
    .venv/bin/python scripts/evaluate_stage1_all.py --real_embeddings
    .venv/bin/python scripts/evaluate_stage1_all.py --real_embeddings --top_k 5
    .venv/bin/python scripts/evaluate_stage1_all.py --real_embeddings --user_id U0001
    .venv/bin/python scripts/evaluate_stage1_all.py --real_embeddings --output results/my_eval.csv
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
setup_logging(level="WARNING")   # 배치 실행 중 per-log 노이즈 억제

from app.config import DEFAULT_CONFIG
from app.data_generation.dataset_builder import build_dataset
from app.data_generation.export_utils import load_dataset_from_json
from app.evaluation.ranking_metrics import compute_all_metrics, compute_candidate_metrics
from app.pipeline.stage1_ranking_pipeline import Stage1Pipeline
from app.schemas import GoalLogLabel, ResearchGoal, ResearchLog

logger = logging.getLogger(__name__)

_DEFAULT_DATA_DIR = "data/synthetic"
_DEFAULT_OUTPUT   = "results/stage1_eval_all.csv"


# ── 수치 메트릭 컬럼 (평균 계산 대상) ──────────────────────────────────────
_FLOAT_KEYS = [
    "candidate_recall",
    "candidate_precision",
    "candidate_f1",
    "recall@k",
    "precision@k",
    "f1@k",
    "selected_precision",
    "false_positive_rate",
    "mrr",
    "ndcg@k",
    "diversity_coverage",
]


def _dynamic_candidate_size(corpus_size: int, top_k: int) -> int:
    if corpus_size <= 20:
        ratio = 0.80
    elif corpus_size <= 50:
        ratio = 0.60
    elif corpus_size <= 100:
        ratio = 0.50
    else:
        ratio = 0.40
    return max(top_k, int(corpus_size * ratio))


def load_data(data_dir: str):
    p = Path(data_dir) / "goals.json"
    if p.exists():
        _, goals, logs, labels = load_dataset_from_json(data_dir)
    else:
        print(f"[INFO] {data_dir} 없음 → 소형 데이터셋 자동 생성")
        ds = build_dataset(small_mode=True, seed=42)
        goals, logs, labels = ds.goals, ds.logs, ds.labels
    return goals, logs, labels


def run_one(
    goal: ResearchGoal,
    user_logs: list[ResearchLog],
    user_labels: list[GoalLogLabel],
    top_k: int,
    use_real_embeddings: bool,
) -> dict:
    """한 goal에 대해 Stage 1을 실행하고 메트릭 dict를 반환한다."""
    cfg = DEFAULT_CONFIG.stage1
    cfg.retrieval.top_k = top_k
    cfg.retrieval.candidate_size = _dynamic_candidate_size(len(user_logs), top_k)

    pipeline = Stage1Pipeline(
        config=cfg,
        use_real_embeddings=use_real_embeddings,
        disable_lexical_gate=False,  # 현재 시스템 그대로
    )
    pipeline.index(user_logs)
    result = pipeline.run(goal, use_expansion=True)

    all_types = {l.activity_type for l in user_logs}

    cand_m = compute_candidate_metrics(result.candidates, user_labels)
    sel_m  = compute_all_metrics(
        result.ranked_logs,
        user_labels,
        k=top_k,
        all_activity_types=all_types,
        selected_logs=result.selected_logs,
    )

    # label 분포 (relevant/partial/irrelevant/unknown)
    label_map = {lb.log_id: lb.label for lb in user_labels}
    dist: dict[str, int] = {}
    for r in result.selected_logs:
        lbl = label_map.get(r.log_id, "unknown")
        dist[lbl] = dist.get(lbl, 0) + 1

    return {
        "goal_id":             goal.goal_id,
        "user_id":             goal.user_id,
        "goal_title":          goal.title,
        "corpus_size":         len(user_logs),
        "relevant_total":      cand_m["relevant_total"],
        # Layer 1: candidate pool
        "candidate_size":      cand_m["candidate_size"],
        "candidate_recall":    cand_m["candidate_recall"],
        "candidate_precision": cand_m["candidate_precision"],
        "candidate_f1":        cand_m["candidate_f1"],
        # Layer 2: final selected
        "selected_count":      sel_m["selected_count"],
        "recall@k":            sel_m[f"recall@{top_k}"],
        "precision@k":         sel_m[f"precision@{top_k}"],
        "f1@k":                sel_m[f"f1@{top_k}"],
        "selected_precision":  sel_m["selected_precision"],
        "false_positive_rate": sel_m["false_positive_rate"],
        "mrr":                 sel_m["mrr"],
        "ndcg@k":              sel_m[f"ndcg@{top_k}"],
        "diversity_coverage":  sel_m["diversity_coverage"],
        # label 분포
        "label_dist":          str(dist),
    }


def print_table(rows: list[dict]) -> None:
    hdr = (
        f"{'goal_id':<18} {'title':<28} "
        f"{'c_R':>5} {'c_P':>5} "
        f"{'R@k':>5} {'P@k':>5} {'F1@k':>5} "
        f"{'sel_P':>5} {'FPR':>5} {'MRR':>5} {'nDCG':>5} {'labels'}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['goal_id']:<18} {r['goal_title'][:26]:<28} "
            f"{r['candidate_recall']:>5.3f} {r['candidate_precision']:>5.3f} "
            f"{r['recall@k']:>5.3f} {r['precision@k']:>5.3f} {r['f1@k']:>5.3f} "
            f"{r['selected_precision']:>5.3f} {r['false_positive_rate']:>5.3f} "
            f"{r['mrr']:>5.3f} {r['ndcg@k']:>5.3f} {r['label_dist']}"
        )


def _mean(rows: list[dict], key: str) -> float:
    if not rows:
        return 0.0
    return round(sum(r[key] for r in rows) / len(rows), 4)


def print_aggregate(rows: list[dict], top_k: int) -> None:
    # ── 세 그룹으로 분리 ────────────────────────────────────────────────
    # A: 선택했고 관련 로그도 있음  → precision/recall 의미 있음
    # B: 선택 안 했고 관련 로그도 없음 → 올바르게 abstain한 케이스
    # C: 선택 안 했지만 관련 로그는 있음 → recall 실패
    selected_rows   = [r for r in rows if r["selected_count"] > 0]
    abstain_correct = [r for r in rows if r["selected_count"] == 0 and r["relevant_total"] == 0]
    abstain_miss    = [r for r in rows if r["selected_count"] == 0 and r["relevant_total"] > 0]

    n_all  = len(rows)
    n_sel  = len(selected_rows)
    n_abs  = len(abstain_correct)
    n_miss = len(abstain_miss)

    print(f"\n{'='*66}")
    print(f"  AGGREGATE  |  n_total={n_all}  top_k={top_k}  system=ours")
    print(f"{'='*66}")

    # ── 전체 평균 ────────────────────────────────────────────────────────
    print(f"\n  [전체 평균  n={n_all}]")
    _print_metric_block(rows, top_k, label="all")

    # ── 실제 선택한 goal만 ────────────────────────────────────────────────
    print(f"\n  [선택 있는 goal만  n={n_sel}]  (selected_count > 0)")
    if selected_rows:
        _print_metric_block(selected_rows, top_k, label="selected_only")
    else:
        print("  (해당 없음)")

    # ── Abstain 분석 ──────────────────────────────────────────────────────
    print(f"\n  [Abstain 분석]")
    print(f"  ✓ 올바르게 선택 안 함 (관련 로그 없음): {n_abs}개")
    print(f"  ⚠ 관련 로그 있는데 선택 0개 (recall 실패): {n_miss}개")
    if abstain_miss:
        for r in abstain_miss:
            print(f"     - {r['goal_id']}  {r['goal_title'][:30]}  relevant={r['relevant_total']}")

    # ── Bottleneck 진단 ────────────────────────────────────────────────────
    base = selected_rows if selected_rows else rows
    cr  = _mean(rows, "candidate_recall")
    sp  = _mean(base, "selected_precision")
    fpr = _mean(base, "false_positive_rate")
    print(f"\n  [Bottleneck 진단]  (선택 있는 goal 기준)")
    if cr < 0.70:
        print(f"  ⚠ candidate_recall={cr:.3f} — retrieval recall 부족")
    else:
        print(f"  ✓ candidate_recall={cr:.3f} — retrieval recall 충분")
    if sp < 0.60:
        print(f"  ⚠ selected_precision={sp:.3f} — reranker/filter 노이즈 과다")
    else:
        print(f"  ✓ selected_precision={sp:.3f} — precision 양호")
    if fpr > 0.30:
        print(f"  ⚠ false_positive_rate={fpr:.3f} — 무관 로그 혼입")
    else:
        print(f"  ✓ false_positive_rate={fpr:.3f}")


def _print_metric_block(rows: list[dict], top_k: int, label: str = "") -> None:
    labels_map = {
        "candidate_recall":    f"후보 recall  (retrieval이 관련 로그를 담았나)",
        "candidate_precision": "후보 precision",
        "candidate_f1":        "후보 F1",
        "recall@k":            f"recall@{top_k}",
        "precision@k":         f"precision@{top_k}",
        "f1@k":                f"F1@{top_k}",
        "selected_precision":  "selected_precision  ← 최종 선택의 정확도",
        "false_positive_rate": "false_positive_rate",
        "mrr":                 "MRR",
        "ndcg@k":              f"nDCG@{top_k}",
        "diversity_coverage":  "diversity_coverage",
    }
    for k in _FLOAT_KEYS:
        v = _mean(rows, k)
        note = labels_map.get(k, "")
        print(f"  {k:<24} {v:>7.4f}   {note}")


def save_csv(rows: list[dict], output: str, top_k: int) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # 마지막 행: 평균
    n = len(rows)
    mean_row: dict = {k: "" for k in fieldnames}
    mean_row["goal_id"]    = "MEAN"
    mean_row["goal_title"] = f"n={n}, top_k={top_k}"
    for k in _FLOAT_KEYS:
        mean_row[k] = round(sum(r[k] for r in rows) / n, 4)

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(mean_row)

    print(f"\n[CSV 저장] {path}  ({n} goals + MEAN row)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1 전체 배치 평가 (현재 시스템)")
    parser.add_argument("--data_dir",        default=_DEFAULT_DATA_DIR)
    parser.add_argument("--output",          default=_DEFAULT_OUTPUT)
    parser.add_argument("--top_k",           type=int, default=10)
    parser.add_argument("--user_id",         default=None, help="특정 user만 평가")
    parser.add_argument("--sample",          type=int, default=None, help="무작위 N개 goal만 평가")
    parser.add_argument("--seed",            type=int, default=42)
    parser.add_argument(
        "--real_embeddings", action="store_true",
        help="Gemini Embedding API 사용 (권장, GEMINI_API_KEY 필요)",
    )
    args = parser.parse_args()

    if not args.real_embeddings:
        print("[WARNING] --real_embeddings 없이 실행 중 → mock embedding 사용 (메트릭 신뢰도 낮음)")

    goals, logs, labels = load_data(args.data_dir)
    print(f"[데이터] goals={len(goals)}  logs={len(logs)}  labels={len(labels)}")

    if args.user_id:
        goals = [g for g in goals if g.user_id == args.user_id]
        if not goals:
            print(f"[ERROR] user_id={args.user_id} 없음")
            sys.exit(1)

    if args.sample and args.sample < len(goals):
        import random
        random.seed(args.seed)
        goals = random.sample(goals, args.sample)
        print(f"[샘플링] {args.sample}개 goal 무작위 선택 (seed={args.seed})")

    rows: list[dict] = []
    skipped = 0

    print(f"\n[평가 시작]  n_goals={len(goals)}  top_k={args.top_k}  real_emb={args.real_embeddings}\n")

    for i, goal in enumerate(goals):
        user_logs   = [l  for l  in logs   if l.user_id  == goal.user_id]
        user_labels = [lb for lb in labels
                       if lb.user_id == goal.user_id and lb.goal_id == goal.goal_id]

        if not user_labels:
            skipped += 1
            continue

        try:
            row = run_one(goal, user_logs, user_labels, args.top_k, args.real_embeddings)
            rows.append(row)
            print(
                f"[{i+1:>3}/{len(goals)}] {goal.goal_id}  {goal.title[:28]:<28}  "
                f"R@k={row['recall@k']:.3f}  P@k={row['precision@k']:.3f}  "
                f"F1={row['f1@k']:.3f}  sel_P={row['selected_precision']:.3f}  "
                f"FPR={row['false_positive_rate']:.3f}"
            )
        except Exception as e:
            print(f"[{i+1:>3}/{len(goals)}] {goal.goal_id}  ERROR: {e}")
            skipped += 1

    if not rows:
        print("[ERROR] 평가 가능한 goal이 없습니다.")
        sys.exit(1)

    print(f"\n{'─'*90}")
    print("[Per-goal 상세 테이블]")
    print_table(rows)

    print_aggregate(rows, args.top_k)

    save_csv(rows, args.output, args.top_k)

    if skipped:
        print(f"  (skipped {skipped} goals: label 없음 또는 오류)")


if __name__ == "__main__":
    main()
