#!/usr/bin/env python3
"""Weight sweep: compare BM25 / Dense / VocabBoost weight combinations.

Experiments (vocab_boost contribution fixed at 0.15):
  A: BM25=0.55  Dense=0.30
  B: BM25=0.50  Dense=0.35
  C: BM25=0.45  Dense=0.40  (current)
  D: BM25=0.40  Dense=0.45
  E: BM25=0.35  Dense=0.50
  F: BM25=0.25  Dense=0.60

--mode candidate (default):
  Evaluates at the candidate level (before reranking).
  This is the correct mode to compare BM25/Dense weight effects.
  Reranker re-scores all candidates identically regardless of weights,
  so end-to-end comparison only reflects candidate-level differences.

--mode rerank:
  Full pipeline evaluation after reranking.
  Only meaningful when candidate_size < corpus size (tight recall setting).

Usage:
    python scripts/compare_retrieval_weights.py
    python scripts/compare_retrieval_weights.py --mode rerank --candidate_ratio 0.4
    python scripts/compare_retrieval_weights.py --goal_id G-U0001-01
    python scripts/compare_retrieval_weights.py --real_embeddings
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

from app.utils.logging_utils import setup_logging
setup_logging(level="WARNING")   # suppress INFO noise during sweep

import logging
from app.config import DEFAULT_CONFIG
from app.data_generation.dataset_builder import build_dataset
from app.data_generation.export_utils import load_dataset_from_json
from app.evaluation.ranking_metrics import compute_all_metrics
from app.pipeline.stage1_ranking_pipeline import Stage1Pipeline
from app.retrieval.candidate_retrieval import RetrievalMode
from app.schemas import ResearchGoal, ResearchLog, GoalLogLabel

logger = logging.getLogger(__name__)

_DEFAULT_DATA_DIR = "data/synthetic"

# ── Experiment definitions ─────────────────────────────────────────────────────
EXPERIMENTS: list[dict] = [
    {"name": "A", "bm25": 0.55, "dense": 0.30, "note": "BM25 우세"},
    {"name": "B", "bm25": 0.50, "dense": 0.35, "note": "BM25 중간 우세"},
    {"name": "C", "bm25": 0.45, "dense": 0.40, "note": "균형 (현재)"},
    {"name": "D", "bm25": 0.40, "dense": 0.45, "note": "Dense 중간 우세"},
    {"name": "E", "bm25": 0.35, "dense": 0.50, "note": "Dense 우세"},
    {"name": "F", "bm25": 0.25, "dense": 0.60, "note": "Dense 강세"},
]


# ── Data helpers ───────────────────────────────────────────────────────────────
def load_data(data_dir: str):
    goals_path = Path(data_dir) / "goals.json"
    if goals_path.exists():
        _, goals, logs, labels = load_dataset_from_json(data_dir)
    else:
        ds = build_dataset(small_mode=True, seed=42)
        goals, logs, labels = ds.goals, ds.logs, ds.labels
    return goals, logs, labels


def _labels_for(goal: ResearchGoal, labels: list[GoalLogLabel]) -> list[GoalLogLabel]:
    return [lb for lb in labels if lb.goal_id == goal.goal_id and lb.user_id == goal.user_id]


def _logs_for(goal: ResearchGoal, logs: list[ResearchLog]) -> list[ResearchLog]:
    return [l for l in logs if l.user_id == goal.user_id]


# ── Single-experiment runner ───────────────────────────────────────────────────
def run_experiment(
    exp: dict,
    goals: list[ResearchGoal],
    logs: list[ResearchLog],
    labels: list[GoalLogLabel],
    top_k: int,
    use_real_embeddings: bool,
    goal_filter: str | None,
    mode: str = "candidate",
    candidate_ratio: float | None = None,
) -> dict:
    """Run one weight configuration across all (or one) goal(s).

    mode='candidate': evaluate on raw candidates (before reranking).
                      This is the correct mode to compare BM25/Dense weights
                      since reranker re-scores identically regardless of weights.
    mode='rerank':    full pipeline evaluation after reranking.
    """
    from app.retrieval.hybrid_retriever import HybridRetriever
    from app.retrieval.query_understanding import build_query
    from app.retrieval.query_expansion import _heuristic_expansion
    from app.schemas import RankedLog

    cfg = copy.deepcopy(DEFAULT_CONFIG.stage1)
    cfg.retrieval.top_k = top_k
    cfg.retrieval.sparse_weight = exp["bm25"]
    cfg.retrieval.dense_weight = exp["dense"]

    target_goals = [g for g in goals if _labels_for(g, labels)]
    if goal_filter:
        target_goals = [g for g in target_goals if g.goal_id == goal_filter]

    if not target_goals:
        return {"metrics_avg": {}, "per_goal": []}

    per_goal: list[dict] = []

    for goal in target_goals:
        user_logs = _logs_for(goal, logs)
        user_labels = _labels_for(goal, labels)
        n_corpus = len(user_logs)

        if candidate_ratio is not None:
            cand_size = max(top_k, int(n_corpus * candidate_ratio))
        elif mode == "candidate":
            # Tight: top_k only — forces the retriever to make real choices
            cand_size = top_k
        else:
            cand_size = max(top_k * 3, min(n_corpus * 6 // 10, 30))

        cfg.retrieval.candidate_size = cand_size

        if mode == "candidate":
            # Direct hybrid retrieval — no reranker, no gate
            # This shows the raw effect of BM25/Dense weight balance
            from app.retrieval.embedding_provider import get_embedding_provider
            provider = get_embedding_provider(real=use_real_embeddings)
            retriever = HybridRetriever(cfg.retrieval, embedding_provider=provider)
            retriever.index(user_logs)

            query_obj = build_query(goal)
            candidates = retriever.retrieve(query_obj.canonical_text, top_n=cand_size)

            # Convert CandidateLog → RankedLog-like for metrics
            label_map = {lb.log_id: lb for lb in user_labels}
            ranked_as_ranked = [
                RankedLog(
                    log=c.log,
                    semantic_relevance=c.dense_score,
                    goal_focus=c.sparse_score,
                    evidence_value=0.0,
                    final_score=c.hybrid_score,
                    rank=i + 1,
                )
                for i, c in enumerate(candidates)
            ]
            selected = ranked_as_ranked  # candidate mode: all retrieved = selected
        else:
            # Full pipeline (rerank mode)
            pipeline = Stage1Pipeline(
                config=cfg,
                use_real_embeddings=use_real_embeddings,
                retrieval_mode=RetrievalMode.HYBRID,
                disable_lexical_gate=False,
            )
            pipeline.index(user_logs)
            result = pipeline.run(goal, use_expansion=True)
            ranked_as_ranked = result.ranked_logs
            selected = result.selected_logs

        all_types = {l.activity_type for l in user_logs}
        m = compute_all_metrics(
            ranked_as_ranked, user_labels,
            k=top_k,
            all_activity_types=all_types,
            selected_logs=selected,
        )
        per_goal.append({
            "goal_id": goal.goal_id,
            "goal_title": goal.title,
            "metrics": m,
            "cand_size": cand_size,
            "corpus_size": n_corpus,
            "selected_logs": [(r.log.title, r.final_score) for r in selected],
            "neg_in_selected": [
                r.log.title for r in selected
                if next(
                    (lb.label for lb in user_labels if lb.log_id == r.log_id),
                    "unknown",
                ) == "irrelevant"
            ],
        })

    # Average metrics across goals
    metric_keys = list(per_goal[0]["metrics"].keys()) if per_goal else []
    avg: dict[str, float] = {}
    for key in metric_keys:
        if key == "selected_count":
            avg[key] = sum(g["metrics"][key] for g in per_goal) / len(per_goal)
        else:
            avg[key] = round(sum(g["metrics"][key] for g in per_goal) / len(per_goal), 4)

    return {"metrics_avg": avg, "per_goal": per_goal}


# ── Display helpers ────────────────────────────────────────────────────────────
def print_comparison_table(results: list[dict], top_k: int) -> None:
    k = top_k
    metrics_to_show = [
        f"recall@{k}",
        f"precision@{k}",
        f"f1@{k}",
        "selected_precision",
        "false_positive_rate",
        "mrr",
        f"ndcg@{k}",
    ]

    col_w = 14

    print("\n" + "=" * 90)
    print("  Retrieval Weight Sweep — Stage 1 Averaged Metrics")
    print("=" * 90)

    # Header
    header = f"  {'Exp':<4} {'BM25':>5} {'Dense':>5} {'VocabB':>6} | "
    header += " ".join(f"{m[:col_w]:>{col_w}}" for m in metrics_to_show)
    print(header)
    print("-" * 90)

    best: dict[str, tuple[str, float]] = {}  # metric → (exp_name, value)

    for exp, res in results:
        avg = res["metrics_avg"]
        if not avg:
            continue

        row = f"  {exp['name']:<4} {exp['bm25']:>5.2f} {exp['dense']:>5.2f} {'0.15':>6} | "
        vals = []
        for m in metrics_to_show:
            v = avg.get(m, 0.0)
            vals.append(v)
            # track best (for fpr: lower is better)
            if m == "false_positive_rate":
                if m not in best or v < best[m][1]:
                    best[m] = (exp["name"], v)
            else:
                if m not in best or v > best[m][1]:
                    best[m] = (exp["name"], v)
        row += " ".join(f"{v:>{col_w}.4f}" for v in vals)
        row += f"  ({exp['note']})"
        print(row)

    print("-" * 90)

    # Best row
    best_row = f"  {'Best':<4} {'':>5} {'':>5} {'':>6} | "
    best_row += " ".join(f"{best.get(m, ('-',''))[0]:>{col_w}}" for m in metrics_to_show)
    print(best_row)
    print("=" * 90)


def print_per_goal_breakdown(results: list[dict], top_k: int) -> None:
    print("\n" + "=" * 90)
    print("  Per-Goal Breakdown (negative logs admitted → False Positives)")
    print("=" * 90)

    all_goal_ids: list[str] = []
    for _, res in results:
        for g in res["per_goal"]:
            if g["goal_id"] not in all_goal_ids:
                all_goal_ids.append(g["goal_id"])

    for goal_id in all_goal_ids:
        print(f"\n  Goal: {goal_id}")
        print(f"  {'Exp':<4} {'f1':>6} {'prec':>6} {'recall':>6} {'fpr':>6}  neg_admitted")
        print("  " + "-" * 70)
        for exp, res in results:
            goal_data = next((g for g in res["per_goal"] if g["goal_id"] == goal_id), None)
            if goal_data is None:
                continue
            m = goal_data["metrics"]
            neg = goal_data["neg_in_selected"]
            k = top_k
            print(
                f"  {exp['name']:<4} "
                f"{m.get(f'f1@{k}', 0):>6.3f} "
                f"{m.get('selected_precision', 0):>6.3f} "
                f"{m.get(f'recall@{k}', 0):>6.3f} "
                f"{m.get('false_positive_rate', 0):>6.3f}  "
                f"{neg if neg else '(none)'}"
            )


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="BM25/Dense weight sweep experiment")
    parser.add_argument("--goal_id", default=None, help="Restrict to a single goal")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--data_dir", default=_DEFAULT_DATA_DIR)
    parser.add_argument(
        "--real_embeddings", action="store_true",
        help="Use Gemini Embedding API (requires GEMINI_API_KEY)",
    )
    parser.add_argument(
        "--exps", default=None,
        help="Comma-separated experiment names to run (e.g. A,C,F). Default: all.",
    )
    parser.add_argument(
        "--mode", choices=["candidate", "rerank"], default="candidate",
        help=(
            "candidate (default): evaluate at candidate level before reranking. "
            "rerank: full pipeline. Use candidate to see actual weight effects."
        ),
    )
    parser.add_argument(
        "--candidate_ratio", type=float, default=None,
        help=(
            "Fraction of corpus to use as candidate_size (e.g. 0.4). "
            "Lower = tighter recall, shows weight differences more clearly. "
            "Default: top_k (tight) for candidate mode, top_k*3 for rerank mode."
        ),
    )
    args = parser.parse_args()

    print(f"Loading data from {args.data_dir} ...")
    goals, logs, labels = load_data(args.data_dir)

    exps = EXPERIMENTS
    if args.exps:
        allowed = set(args.exps.upper().split(","))
        exps = [e for e in EXPERIMENTS if e["name"] in allowed]

    print(f"Mode     : {args.mode}")
    print(f"Running {len(exps)} experiments × goals ...")
    if args.real_embeddings:
        print("[real_embeddings=True] Using Gemini Embedding API")
    else:
        print("[real_embeddings=False] Using mock hash embeddings")

    if args.mode == "candidate":
        print("[candidate mode] Evaluating at retriever output level (before reranking)")
        print("  → candidate_size = top_k (tight) — weight differences are visible here")
    else:
        print("[rerank mode] Full pipeline evaluation")
        if args.candidate_ratio is None:
            print("  ⚠  candidate_size may cover full corpus → all exps will look identical")
            print("  → use --candidate_ratio 0.3 to tighten recall and expose weight differences")

    results: list[tuple[dict, dict]] = []
    for exp in exps:
        print(f"  Exp {exp['name']} (BM25={exp['bm25']}, Dense={exp['dense']}) ...", end="", flush=True)
        res = run_experiment(
            exp, goals, logs, labels,
            top_k=args.top_k,
            use_real_embeddings=args.real_embeddings,
            goal_filter=args.goal_id,
            mode=args.mode,
            candidate_ratio=args.candidate_ratio,
        )
        n_goals = len(res["per_goal"])
        print(f" done ({n_goals} goals)")
        results.append((exp, res))

    print_comparison_table(results, top_k=args.top_k)
    print_per_goal_breakdown(results, top_k=args.top_k)

    print("\n[참고]")
    if args.mode == "candidate":
        print("  현재 candidate 모드: retriever 출력 단계에서 평가합니다.")
        print("  BM25/Dense 가중치 효과를 직접 볼 수 있습니다.")
    else:
        print("  rerank 모드: reranker 이후 최종 결과를 평가합니다.")
        print("  candidate_size가 corpus를 전부 커버하면 weight 차이가 사라집니다.")
    print()
    print("  mock embedding: Dense는 hash 기반이라 의미론적 유사도 없음")
    print("  → BM25 우세 조합(A,B)이 유리하게 나오는 것이 정상")
    print()
    print("  실제 semantic recall을 보려면:")
    print("  PYTHONHASHSEED=0 .venv/bin/python scripts/compare_retrieval_weights.py --real_embeddings")


if __name__ == "__main__":
    main()
