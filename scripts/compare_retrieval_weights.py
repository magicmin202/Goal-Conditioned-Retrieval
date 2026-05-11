#!/usr/bin/env python3
"""Weight sweep: compare BM25 / Dense / VocabBoost weight combinations.

Experiments (vocab_boost contribution fixed at 0.15):
  A: BM25=0.55  Dense=0.30
  B: BM25=0.50  Dense=0.35
  C: BM25=0.45  Dense=0.40  (current)
  D: BM25=0.40  Dense=0.45
  E: BM25=0.35  Dense=0.50
  F: BM25=0.25  Dense=0.60

Candidate size is dynamic (scales with corpus size):
  ≤ 20 logs  → 80 %  of corpus
  ≤ 50 logs  → 60 %
  ≤ 100 logs → 50 %
  > 100 logs → 40 %
  (always ≥ top_k)

This ensures the "top-N" pool is proportional to corpus, not hardcoded at 30.

--mode candidate (default):
  Evaluates at the candidate level (before reranking).
  Reports both candidate_recall (how many relevant logs entered the pool)
  and selected_precision (precision of the pool itself).

--mode rerank:
  Full pipeline evaluation after reranking + diversity selection.

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
from app.retrieval.candidate_retrieval import RetrievalMode
from app.schemas import ResearchGoal, ResearchLog, GoalLogLabel


def dynamic_candidate_size(corpus_size: int, top_k: int) -> int:
    """Scale candidate pool proportionally to corpus size.

    Corpus size   Ratio   Example (top_k=10)
    ──────────────────────────────────────────
    ≤  20         80 %    16
    ≤  50         60 %    30
    ≤ 100         50 %    50
    > 100         40 %    40+ (corpus-dependent)

    Always returns at least top_k so the pipeline never starves.
    """
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

# ── Experiment definitions ─────────────────────────────────────────────────────
EXPERIMENTS: list[dict] = [
    {"name": "A",    "bm25": 0.55, "dense": 0.30, "note": "BM25 우세"},
    {"name": "B",    "bm25": 0.50, "dense": 0.35, "note": "BM25 중간 우세"},
    {"name": "C",    "bm25": 0.45, "dense": 0.40, "note": "균형 (현재)"},
    {"name": "D",    "bm25": 0.40, "dense": 0.45, "note": "Dense 중간 우세"},
    {"name": "E",    "bm25": 0.35, "dense": 0.50, "note": "Dense 우세"},
    {"name": "F",    "bm25": 0.25, "dense": 0.60, "note": "Dense 강세"},
    {"name": "DENSE","bm25": 0.00, "dense": 1.00, "note": "Dense only (no BM25)"},
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
    shared_provider=None,   # pre-warmed provider shared across all experiments
    vocab_boost: bool = False,  # apply vocab boost on top of hybrid score
) -> dict:
    """Run one weight configuration across all (or one) goal(s).

    mode='candidate': evaluate on raw candidates (before reranking).
                      This is the correct mode to compare BM25/Dense weights
                      since reranker re-scores identically regardless of weights.
    mode='rerank':    full pipeline evaluation after reranking.
    vocab_boost:      when True (candidate mode only), apply CandidateRetriever
                      vocab boost using heuristic-expanded query terms.
    """
    from app.retrieval.hybrid_retriever import HybridRetriever
    from app.retrieval.candidate_retrieval import CandidateRetriever, RetrievalMode, _weak_vocab_boost as _apply_vocab_boost
    from app.retrieval.query_understanding import build_query
    from app.retrieval.query_expansion import _heuristic_expansion, ExpandedQuery
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
        else:
            cand_size = dynamic_candidate_size(n_corpus, top_k)

        cfg.retrieval.candidate_size = cand_size

        if mode == "candidate":
            # Direct hybrid retrieval — no reranker, no gate
            # Shows the raw effect of BM25/Dense weight balance at recall level
            from app.retrieval.embedding_provider import get_embedding_provider
            provider = shared_provider or get_embedding_provider(real=use_real_embeddings)
            retriever = HybridRetriever(cfg.retrieval, embedding_provider=provider)
            retriever.index(user_logs)

            query_obj = build_query(goal)
            candidates = retriever.retrieve(query_obj.canonical_text, top_n=cand_size)

            # Optional vocab boost: apply CandidateRetriever-style lexicon boost
            if vocab_boost:
                from app.config import CandidateConfig
                heuristic = _heuristic_expansion(goal, max_terms=15)
                expanded = ExpandedQuery(
                    base_query=query_obj,
                    expanded_terms=heuristic.get("evidence_terms", []),
                    priority_terms=heuristic.get("priority_terms", []),
                    related_terms=heuristic.get("related_terms", []),
                    negative_terms=heuristic.get("negative_terms", []),
                )
                candidates = _apply_vocab_boost(
                    candidates, expanded, cfg.vocab_boost, CandidateConfig()
                )

            # Candidate-level metrics (the primary signal for this mode)
            cand_m = compute_candidate_metrics(candidates, user_labels)

            # Also build RankedLog list so compute_all_metrics can run
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
            selected = ranked_as_ranked
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
            candidates = result.candidates
            cand_m = compute_candidate_metrics(candidates, user_labels)

        all_types = {l.activity_type for l in user_logs}
        m = compute_all_metrics(
            ranked_as_ranked, user_labels,
            k=top_k,
            all_activity_types=all_types,
            selected_logs=selected,
        )
        # Merge candidate metrics into result dict
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
    # candidate_recall first — this is what we're tuning retrieval weights for
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
    print("  Retrieval Weight Sweep — Stage 1 Averaged Metrics")
    print("  [candidate_recall] = 관련 로그가 pool에 들어왔나?  (retrieval 품질 핵심 지표)")
    print("=" * 110)

    # Header
    header = f"  {'Exp':<4} {'BM25':>5} {'Dense':>5} {'cand_n':>6} | "
    header += " ".join(f"{m[:col_w]:>{col_w}}" for m in metrics_to_show)
    print(header)
    print("-" * 110)

    best: dict[str, tuple[str, float]] = {}  # metric → (exp_name, value)

    # Compute avg candidate_size for display
    exp_cand_sizes: dict[str, float] = {}
    for exp, res in results:
        sizes = [g["cand_size"] for g in res["per_goal"]]
        exp_cand_sizes[exp["name"]] = round(sum(sizes) / len(sizes), 1) if sizes else 0

    for exp, res in results:
        avg = res["metrics_avg"]
        if not avg:
            continue

        avg_n = exp_cand_sizes.get(exp["name"], 0)
        row = f"  {exp['name']:<4} {exp['bm25']:>5.2f} {exp['dense']:>5.2f} {avg_n:>6.0f} | "
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

    print("-" * 110)

    # Best row
    best_row = f"  {'Best':<4} {'':>5} {'':>5} {'':>6} | "
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
    parser.add_argument(
        "--vocab_boost", action="store_true",
        help=(
            "candidate mode only: apply vocab boost (priority/evidence/negative lexicon) "
            "on top of the hybrid score using heuristic expansion terms. "
            "Lets you compare F vs F+VocabBoost without running the full reranker."
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
        if args.vocab_boost:
            print("  [vocab_boost=True] Heuristic lexicon boost applied on top of hybrid score")
    else:
        print("[rerank mode] Full pipeline evaluation")
        if args.candidate_ratio is None:
            print("  ⚠  candidate_size may cover full corpus → all exps will look identical")
            print("  → use --candidate_ratio 0.3 to tighten recall and expose weight differences")

    # ── Pre-warm embedding cache (one shared provider for all experiments) ────
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
        print(f"  Exp {exp['name']} (BM25={exp['bm25']}, Dense={exp['dense']}) ...", end="", flush=True)
        res = run_experiment(
            exp, goals, logs, labels,
            top_k=args.top_k,
            use_real_embeddings=args.real_embeddings,
            goal_filter=args.goal_id,
            mode=args.mode,
            candidate_ratio=args.candidate_ratio,
            shared_provider=shared_provider,
            vocab_boost=args.vocab_boost,
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
