#!/usr/bin/env python3
"""Dense vs Hybrid 대화형 비교 도구.

사용법:
    .venv/bin/python scripts/compare_dense_hybrid.py
    .venv/bin/python scripts/compare_dense_hybrid.py --goal_id G-U0001-01
    .venv/bin/python scripts/compare_dense_hybrid.py --goal_id G-U0001-01 --top_k 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

from app.utils.logging_utils import setup_logging
setup_logging(level="WARNING")  # 실행 중 로그는 경고 이상만 표시

import logging
# reranker 내부 semantic tie-breaker(5% 가중치)는 mock 사용 — 경고 억제
logging.getLogger("app.retrieval.embedding_provider").setLevel(logging.ERROR)

import logging
from app.config import DEFAULT_CONFIG
from app.data_generation.export_utils import load_dataset_from_json
from app.evaluation.ranking_metrics import compute_all_metrics
from app.pipeline.stage1_ranking_pipeline import Stage1Pipeline
from app.retrieval.candidate_retrieval import RetrievalMode
from app.schemas import ResearchGoal, ResearchLog

logger = logging.getLogger(__name__)

_DATA_DIR = "data/synthetic"
_BASELINES = {
    "dense":  {"retrieval_mode": RetrievalMode.DENSE,  "use_expansion": False, "disable_lexical_gate": True},
    "hybrid": {"retrieval_mode": RetrievalMode.HYBRID, "use_expansion": False, "disable_lexical_gate": True},
}


def load_data():
    _, goals, logs, labels = load_dataset_from_json(_DATA_DIR)
    return goals, logs, labels


def list_goals(goals: list[ResearchGoal]) -> None:
    print("\n사용 가능한 goal_id 목록:")
    print(f"  {'goal_id':<16} {'유저':<8} {'목표'}")
    print("  " + "─" * 50)
    for g in goals:
        print(f"  {g.goal_id:<16} {g.user_id:<8} {g.title}")


def run_baseline(
    goal: ResearchGoal,
    user_logs: list[ResearchLog],
    baseline: str,
    bcfg: dict,
    top_k: int,
) -> dict:
    cfg = DEFAULT_CONFIG.stage1
    cfg.retrieval.top_k = top_k
    cfg.retrieval.candidate_size = max(top_k * 3, min(len(user_logs) * 6 // 10, 30))

    pipeline = Stage1Pipeline(
        config=cfg,
        use_real_embeddings=True,
        retrieval_mode=bcfg["retrieval_mode"],
        disable_lexical_gate=bcfg["disable_lexical_gate"],
    )
    pipeline.index(user_logs)
    result = pipeline.run(goal, use_expansion=bcfg["use_expansion"])
    return result


def print_comparison(
    goal: ResearchGoal,
    results: dict,
    labels,
    user_logs: list[ResearchLog],
    top_k: int,
) -> None:
    print("\n" + "=" * 64)
    print(f"  목표: {goal.title}  ({goal.goal_id})")
    print(f"  corpus: {len(user_logs)} logs  |  top_k={top_k}")
    print("=" * 64)

    label_map = {lb.log_id: lb.label for lb in labels}
    all_types = {l.activity_type for l in user_logs}

    metrics_per_bl: dict[str, dict] = {}
    selected_per_bl: dict[str, list] = {}

    for bl, result in results.items():
        m = compute_all_metrics(
            result.ranked_logs, labels,
            k=top_k,
            all_activity_types=all_types,
            selected_logs=result.selected_logs,
        )
        metrics_per_bl[bl] = m
        selected_per_bl[bl] = result.selected_logs

    # ── 선택된 로그 비교 ──────────────────────────────────────────────────────
    print(f"\n{'─'*64}")
    for bl in _BASELINES:
        result = results[bl]
        print(f"\n[{bl.upper()} — Top-{top_k} 선택 로그]")
        for r in result.selected_logs:
            lbl = label_map.get(r.log_id, "?")
            mark = "✓" if lbl == "relevant" else ("△" if lbl == "partial" else "✗")
            print(f"  {mark} [{r.rank:2d}] {r.final_score:.4f}  {r.log.date}  {r.log.title}")

    # ── 지표 비교 테이블 ───────────────────────────────────────────────────────
    metric_keys = ["recall@5", "precision@5", "selected_precision",
                   "f1@5", "false_positive_rate", "mrr", "ndcg@5"]
    metric_keys = [m.replace("5", str(top_k)) for m in metric_keys]
    # ndcg key fix
    metric_keys = [
        f"recall@{top_k}", f"precision@{top_k}", "selected_precision",
        f"f1@{top_k}", "false_positive_rate", "mrr", f"ndcg@{top_k}",
    ]

    print(f"\n{'─'*64}")
    print(f"  {'지표':<24} {'dense':>10} {'hybrid':>10}  {'winner':>8}")
    print(f"  {'─'*56}")

    lower_is_better = {"false_positive_rate"}

    for mk in metric_keys:
        d_val = metrics_per_bl["dense"].get(mk, 0.0)
        h_val = metrics_per_bl["hybrid"].get(mk, 0.0)

        if mk in lower_is_better:
            winner = "dense" if d_val < h_val else ("hybrid" if h_val < d_val else "tie")
        else:
            winner = "dense" if d_val > h_val else ("hybrid" if h_val > d_val else "tie")

        mark = "◀" if winner != "tie" else "="
        print(f"  {mk:<24} {d_val:>10.4f} {h_val:>10.4f}  {winner:>6} {mark}")

    print(f"\n  ✓=relevant  △=partial  ✗=irrelevant")
    print("=" * 64 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dense vs Hybrid 비교")
    parser.add_argument("--goal_id", default=None, help="비교할 goal_id (생략 시 대화형 입력)")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    print("데이터 로딩 중...", end=" ", flush=True)
    goals, logs, labels = load_data()
    print("완료")

    goal_map = {g.goal_id: g for g in goals}

    while True:
        # ── goal 선택 ──────────────────────────────────────────────────────────
        if args.goal_id:
            goal_id = args.goal_id.strip()
        else:
            list_goals(goals)
            print()
            goal_id = input("goal_id 입력 (종료: q): ").strip()

        if goal_id.lower() in ("q", "quit", "exit", ""):
            print("종료합니다.")
            break

        if goal_id not in goal_map:
            print(f"[오류] '{goal_id}' 를 찾을 수 없습니다.")
            if args.goal_id:
                break
            continue

        goal = goal_map[goal_id]
        user_logs = [l for l in logs if l.user_id == goal.user_id]
        user_labels = [lb for lb in labels if lb.user_id == goal.user_id and lb.goal_id == goal.goal_id]

        print(f"\n'{goal.title}' ({goal.user_id}, {len(user_logs)} logs) — 임베딩 중...")

        results = {}
        for bl, bcfg in _BASELINES.items():
            print(f"  [{bl}] 실행 중...", end=" ", flush=True)
            try:
                results[bl] = run_baseline(goal, user_logs, bl, bcfg, args.top_k)
                print("완료")
            except Exception as e:
                print(f"오류: {e}")
                results[bl] = None

        results = {bl: r for bl, r in results.items() if r is not None}

        if results:
            print_comparison(goal, results, user_labels, user_logs, args.top_k)

        # 단발 실행 시 종료
        if args.goal_id:
            break

        again = input("다른 goal을 테스트하시겠습니까? (y/n): ").strip().lower()
        if again != "y":
            print("종료합니다.")
            break


if __name__ == "__main__":
    main()
