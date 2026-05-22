#!/usr/bin/env python3
"""Baseline comparison: Dense retrieval variants.

Compares the effect of query expansion and lexical gate on retrieval quality.
BM25 and VocabBoost have been removed; Dense (Gemini embedding-001) is the
sole retrieval mechanism.

Baselines:
  dense              Dense only, no gate, no expansion
  dense_expand       Dense + query expansion, no gate
  ours               Dense + expansion + lexical gate (full pipeline)
  ours_wo_lexical_gate  Dense + expansion, no gate

Usage:
    python scripts/compare_retrieval_weights.py
    python scripts/compare_retrieval_weights.py --real_embeddings
    python scripts/compare_retrieval_weights.py --mode rerank
    python scripts/compare_retrieval_weights.py --goal_id G-U0001-01
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
from app.evaluation.ranking_metrics import compute_all_metrics, compute_candidate_metrics
from app.pipeline.stage1_ranking_pipeline import Stage1Pipeline
from app.schemas import ResearchGoal, ResearchLog, GoalLogLabel


def dynamic_candidate_size(corpus_size: int, top_k: int) -> int:
    if corpus_size <= 20:
        ratio = 0.80
    elif corpus_size <= 50:
        ratio = 0.60
    elif corpus_size <= 100:
        ratio = 0.50
    else:
        ratio = 0.40
    return max(top_k, int(corpus_size * ratio))


logger = logging.getLogger(__name__)

_DEFAULT_DATA_DIR = "data/synthetic"

# ── Baseline definitions ───────────────────────────────────────────────────────
EXPERIMENTS: list[dict] = [
    {"name": "dense",              "use_expansion": False, "disable_lexical_gate": True,  "note": "Dense only"},
    {"name": "dense_expand",       "use_expansion": True,  "disable_lexical_gate": True,  "note": "Dense + expansion"},
    {"name": "ours",               "use_expansion": True,  "disable_lexical_gate": False, "note": "Full pipeline"},
    {"name": "ours_wo_gate",       "use_expansion": True,  "disable_lexical_gate": True,  "note": "Full pipeline, no gate"},
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
    shared_provider=None,
) -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG.stage1)
    cfg.retrieval.top_k = top_k

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
        else:
            cand_size = dynamic_candidate_size(n_corpus, top_k)

        cfg.retrieval.candidate_size = cand_size

        if mode == "candidate":
            from app.retrieval.embedding_provider import get_embedding_provider
            from app.retrieval.candidate_retrieval import CandidateRetriever
            from app.retrieval.query_understanding import build_query
            from app.retrieval.query_expansion import ExpandedQuery, _heuristic_expansion
            from app.schemas import RankedLog

            provider = shared_provider or get_embedding_provider(real=use_real_embeddings)
            retriever = CandidateRetriever(config=cfg.retrieval, embedding_provider=provider)
            retriever.index(user_logs)

            query_obj = build_query(goal)

            if exp["use_expansion"]:
                heuristic = _heuristic_expansion(goal, max_terms=15)
                active_query = ExpandedQuery(
                    base_query=query_obj,
                    expanded_terms=heuristic.get("evidence_terms", []),
                    priority_terms=heuristic.get("priority_terms", []),
                    related_terms=heuristic.get("related_terms", []),
                    negative_terms=heuristic.get("negative_terms", []),
                )
            else:
                active_query = query_obj

            candidates = retriever.retrieve(active_query, top_n=cand_size)

            cand_m = compute_candidate_metrics(candidates, user_labels)

            ranked_as_ranked = [
                RankedLog(
                    log=c.log,
                    semantic_relevance=c.dense_score,
                    goal_focus=0.0,
                    evidence_value=0.0,
                    final_score=c.dense_score,
                    rank=i + 1,
                )
                for i, c in enumerate(candidates)
            ]
            selected = ranked_as_ranked
        else:
            pipeline = Stage1Pipeline(
                config=cfg,
                use_real_embeddings=use_real_embeddings,
                disable_lexical_gate=exp["disable_lexical_gate"],
            )
            pipeline.index(user_logs)
            result = pipeline.run(goal, use_expansion=exp["use_expansion"])
            ranked_as_ranked = result.ranked_logs
            selected = result.selected_logs
            candidates = result.candidates
            cand_m = compute_candidate_metrics(candidates, user_labels)

        all_types = {l.activity_type for l in user_logs}
        m = compute_all_metrics(
            ranked_as_ranked, user_labels,
            k=top_k,
            all_activity_types=all_types,
            selected_logs=selected,
        )
        m["candidate_recall"]    = cand_m["candidate_recall"]
        m["candidate_precision"] = cand_m["candidate_precision"]
        m["relevant_in_pool"]    = cand_m["relevant_in_pool"]

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
        "candidate_recall",
        "candidate_precision",
        f"recall@{k}",
        f"precision@{k}",
        f"f1@{k}",
        "selected_precision",
        "false_positive_rate",
        "mrr",
    ]

    col_w = 14

    print("\n" + "=" * 110)
    print("  Dense Retrieval Baseline Comparison — Stage 1 Averaged Metrics")
    print("  [candidate_recall] = 관련 로그가 pool에 들어왔나?  (retrieval 품질 핵심 지표)")
    print("=" * 110)

    header = f"  {'Exp':<18} {'cand_n':>6} | "
    header += " ".join(f"{m[:col_w]:>{col_w}}" for m in metrics_to_show)
    print(header)
    print("-" * 110)

    best: dict[str, tuple[str, float]] = {}

    exp_cand_sizes: dict[str, float] = {}
    for exp, res in results:
        sizes = [g["cand_size"] for g in res["per_goal"]]
        exp_cand_sizes[exp["name"]] = round(sum(sizes) / len(sizes), 1) if sizes else 0

    for exp, res in results:
        avg = res["metrics_avg"]
        if not avg:
            continue

        avg_n = exp_cand_sizes.get(exp["name"], 0)
        row = f"  {exp['name']:<18} {avg_n:>6.0f} | "
        vals = []
        for m in metrics_to_show:
            v = avg.get(m, 0.0)
            vals.append(v)
            if m == "false_positive_rate":
                if m not in best or v < best[m][1]:
                    best[m] = (exp["name"], v)
            else:
                if m not in best or v > best[m][1]:
                    best[m] = (exp["name"], v)
        row += " ".join(f"{v:>{col_w}.4f}" for v in vals)
        row += f"  ({exp['note']})"
        print(row)

    print("-" * 110)
    best_row = f"  {'Best':<18} {'':>6} | "
    best_row += " ".join(f"{best.get(m, ('-',''))[0]:>{col_w}}" for m in metrics_to_show)
    print(best_row)
    print("=" * 110)


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
        print(f"  {'Exp':<20} {'f1':>6} {'prec':>6} {'recall':>6} {'fpr':>6}  neg_admitted")
        print("  " + "-" * 70)
        for exp, res in results:
            goal_data = next((g for g in res["per_goal"] if g["goal_id"] == goal_id), None)
            if goal_data is None:
                continue
            m = goal_data["metrics"]
            neg = goal_data["neg_in_selected"]
            k = top_k
            print(
                f"  {exp['name']:<20} "
                f"{m.get(f'f1@{k}', 0):>6.3f} "
                f"{m.get('selected_precision', 0):>6.3f} "
                f"{m.get(f'recall@{k}', 0):>6.3f} "
                f"{m.get('false_positive_rate', 0):>6.3f}  "
                f"{neg if neg else '(none)'}"
            )


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Dense retrieval baseline comparison")
    parser.add_argument("--goal_id", default=None, help="Restrict to a single goal")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--data_dir", default=_DEFAULT_DATA_DIR)
    parser.add_argument(
        "--no_real_embeddings", dest="real_embeddings", action="store_false",
        help="Disable Gemini Embedding API and use mock embeddings instead",
    )
    parser.set_defaults(real_embeddings=True)
    parser.add_argument(
        "--exps", default=None,
        help="Comma-separated experiment names to run (e.g. dense,ours). Default: all.",
    )
    parser.add_argument(
        "--mode", choices=["candidate", "rerank"], default="candidate",
        help=(
            "candidate (default): evaluate at candidate level before reranking. "
            "rerank: full pipeline."
        ),
    )
    parser.add_argument(
        "--candidate_ratio", type=float, default=None,
        help="Fraction of corpus to use as candidate_size (e.g. 0.4).",
    )
    args = parser.parse_args()

    print(f"Loading data from {args.data_dir} ...")
    goals, logs, labels = load_data(args.data_dir)

    exps = EXPERIMENTS
    if args.exps:
        allowed = set(args.exps.lower().split(","))
        exps = [e for e in EXPERIMENTS if e["name"] in allowed]

    print(f"Mode     : {args.mode}")
    print(f"Running {len(exps)} experiments × goals ...")
    if args.real_embeddings:
        print("[real_embeddings=True] Using Gemini Embedding API")
    else:
        print("[real_embeddings=False] Using mock hash embeddings")

    shared_provider = None
    if args.real_embeddings:
        from app.retrieval.embedding_provider import get_embedding_provider
        shared_provider = get_embedding_provider(real=True)
        all_texts = list({log.full_text for log in logs} | {g.query_text for g in goals})
        print(f"Pre-warming embeddings: {len(all_texts)} unique texts ...", flush=True)
        shared_provider.encode_batch(all_texts)
        print(f"  → done (cache has {len(shared_provider._cache)} entries)")

    results: list[tuple[dict, dict]] = []
    for exp in exps:
        print(f"  Exp {exp['name']} ...", end="", flush=True)
        res = run_experiment(
            exp, goals, logs, labels,
            top_k=args.top_k,
            use_real_embeddings=args.real_embeddings,
            goal_filter=args.goal_id,
            mode=args.mode,
            candidate_ratio=args.candidate_ratio,
            shared_provider=shared_provider,
        )
        n_goals = len(res["per_goal"])
        print(f" done ({n_goals} goals)")
        results.append((exp, res))

    print_comparison_table(results, top_k=args.top_k)
    print_per_goal_breakdown(results, top_k=args.top_k)


if __name__ == "__main__":
    main()
