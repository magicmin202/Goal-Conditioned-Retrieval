#!/usr/bin/env python3
"""Stage 1 full-pipeline A/B comparison: gemini-2.5-flash vs gemini-2.5-flash-preview.

각 모델에 대해 전체 Stage 1 파이프라인을 실행하고 결과를 비교합니다:
  1. Query Expansion 품질 비교
  2. Candidate Retrieval 결과 비교
  3. Reranking 점수 비교
  4. Relevance Filtering 결과 비교
  5. 최종 Selected Logs 비교
  6. 메트릭 비교 (labels 있을 때)

Usage:
    .venv/bin/python scripts/compare_stage1_models.py --auto
    .venv/bin/python scripts/compare_stage1_models.py --goal_id G-U0001-01
    .venv/bin/python scripts/compare_stage1_models.py --all_goals --limit 10
    .venv/bin/python scripts/compare_stage1_models.py --all_goals --limit 10 --no_checkpoint
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from app.utils.logging_utils import setup_logging
setup_logging(level="WARNING")

import logging
logger = logging.getLogger(__name__)

from app.config import DEFAULT_CONFIG, GeminiConfig
from app.data_generation.dataset_builder import build_dataset
from app.data_generation.export_utils import load_dataset_from_json
from app.evaluation.ranking_metrics import (
    compute_all_metrics,
    compute_candidate_metrics,
)
from app.retrieval.query_understanding import build_query
from app.retrieval.query_expansion import expand_goal_query, ExpandedQuery
from app.retrieval.candidate_retrieval import CandidateRetriever
from app.retrieval.reranker import GoalConditionedReranker
from app.retrieval.diversity_selector import DiversitySelector
from app.schemas import ResearchGoal, ResearchLog, GoalLogLabel

_DEFAULT_DATA_DIR = "data/synthetic"

MODEL_A = "gemini-2.5-flash"
MODEL_B = "gemini-3-flash-preview"

_CHECKPOINT_PATH = Path("data/compare_stage1_checkpoint.json")


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_data(data_dir: str):
    goals_path = Path(data_dir) / "goals.json"
    if goals_path.exists():
        _, goals, logs, labels = load_dataset_from_json(data_dir)
    else:
        ds = build_dataset(small_mode=True, seed=42)
        goals, logs, labels = ds.goals, ds.logs, ds.labels
    return goals, logs, labels


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


# ── Per-model run ─────────────────────────────────────────────────────────────

def run_stage1_with_model(
    goal: ResearchGoal,
    user_logs: list[ResearchLog],
    model_name: str,
    top_k: int,
    dense_threshold: float,
) -> tuple[dict, float]:
    """Run full Stage 1 pipeline with the given expansion model.

    Returns (result_dict, elapsed_sec).
    result_dict keys:
        expansion, candidates, ranked, filtered, selected
    """
    cfg = DEFAULT_CONFIG.stage1
    cfg.retrieval.top_k = top_k
    cfg.retrieval.candidate_size = _dynamic_candidate_size(len(user_logs), top_k)

    gemini_cfg = GeminiConfig(
        model_name=model_name,
        max_output_tokens=2048,
        temperature=0.2,
    )

    t0 = time.time()

    # 1. Query Understanding
    query_obj = build_query(goal)

    # 2. LLM Query Expansion
    expanded = expand_goal_query(
        goal,
        query_obj,
        max_terms=cfg.query_expansion.max_terms,
        mode="structured",
        use_mock_fallback=False,
        use_cache=False,
        gemini_config=gemini_cfg,
    )

    # 3. Candidate Retrieval + vocab boost
    retriever = CandidateRetriever(config=cfg.retrieval)
    retriever.index(user_logs)
    pool_size = max(cfg.retrieval.candidate_size, len(user_logs))
    all_cands = retriever.retrieve(expanded, top_n=pool_size)

    # dense_threshold filter
    above_thresh = [c for c in all_cands if (c.dense_score or 0.0) >= dense_threshold]
    candidates = above_thresh if above_thresh else all_cands[:cfg.retrieval.candidate_size]

    # 4. Reranking
    reranker = GoalConditionedReranker(config=cfg.ranker)
    ranked = reranker.rank(
        goal, candidates,
        expanded_terms=expanded.expanded_terms,
        negative_terms=expanded.negative_terms,
        priority_terms=expanded.priority_terms,
        related_terms=getattr(expanded, "related_terms", []),
    )

    # 5. Relevance Filtering
    threshold = cfg.diversity.relevance_threshold
    above = [r for r in ranked if r.final_score >= threshold]
    filtered = above if len(above) >= top_k else ranked[:top_k * 2]

    # 6. Diversity-aware Top-K
    selector = DiversitySelector(config=cfg.diversity)
    selected = selector.select(goal, filtered, top_k=top_k)

    elapsed = time.time() - t0

    return {
        "expansion": expanded,
        "candidates": candidates,
        "ranked": ranked,
        "filtered": filtered,
        "selected": selected,
    }, elapsed


# ── Print helpers ─────────────────────────────────────────────────────────────

def _score(c) -> float:
    return getattr(c, "final_score", None) or getattr(c, "hybrid_score", None) or getattr(c, "dense_score", None) or 0.0


def print_expansion(label: str, exp: ExpandedQuery, elapsed: float) -> None:
    print(f"\n{'─'*60}")
    print(f"  {label}  [{elapsed:.2f}s]")
    print(f"{'─'*60}")
    print(f"  goal_summary : {exp.goal_summary}")
    print(f"  core_intents : {exp.core_intents}")
    print(f"  priority  ({len(exp.priority_terms):2d}) : {exp.priority_terms}")
    print(f"  evidence  ({len(exp.expanded_terms):2d}) : {exp.expanded_terms}")
    print(f"  related   ({len(getattr(exp,'related_terms',[])):2d}) : {getattr(exp,'related_terms',[])}")
    print(f"  negative  ({len(exp.negative_terms):2d}) : {exp.negative_terms}")


def print_ranked(label: str, ranked: list, top_n: int = 10) -> None:
    print(f"\n  [Ranked — {label}  (top {top_n})]")
    for r in ranked[:top_n]:
        print(f"    [{r.rank:2d}] score={r.final_score:.4f}  gf={r.goal_focus:.3f}  "
              f"sr={r.semantic_relevance:.3f}  {r.log.date}  {r.log.title}")


def print_selected(label: str, selected: list) -> None:
    print(f"\n  [Selected — {label}  ({len(selected)} logs)]")
    for i, r in enumerate(selected, 1):
        print(f"    [{i:2d}] score={r.final_score:.4f}  {r.log.date}  {r.log.title}")


def print_expansion_diff(exp_a: ExpandedQuery, exp_b: ExpandedQuery) -> None:
    def diff(name: str, a: list[str], b: list[str]) -> None:
        sa, sb = set(a), set(b)
        common = sa & sb
        only_a = sa - sb
        only_b = sb - sa
        print(f"\n  [{name}]")
        if common: print(f"    공통  : {sorted(common)}")
        if only_a: print(f"    A만   : {sorted(only_a)}")
        if only_b: print(f"    B만   : {sorted(only_b)}")

    print(f"\n{'═'*60}")
    print("  [Expansion Term Diff]")
    diff("priority_terms", exp_a.priority_terms, exp_b.priority_terms)
    diff("evidence_terms", exp_a.expanded_terms, exp_b.expanded_terms)
    diff("related_terms",  getattr(exp_a, "related_terms", []), getattr(exp_b, "related_terms", []))
    diff("negative_terms", exp_a.negative_terms, exp_b.negative_terms)


def print_selected_diff(res_a: dict, res_b: dict) -> None:
    ids_a = {r.log_id: r for r in res_a["selected"]}
    ids_b = {r.log_id: r for r in res_b["selected"]}
    common = set(ids_a) & set(ids_b)
    only_a = set(ids_a) - set(ids_b)
    only_b = set(ids_b) - set(ids_a)

    print(f"\n{'═'*60}")
    print(f"  [Selected Logs Diff]")
    print(f"  공통 ({len(common)}):")
    for lid in sorted(common):
        r = ids_a[lid]
        print(f"    ○ score_A={_score(r):.4f}  score_B={_score(ids_b[lid]):.4f}  {r.log.title}  [{r.log.date}]")

    print(f"\n  {MODEL_A}만 선택 ({len(only_a)}):")
    for lid in sorted(only_a):
        r = ids_a[lid]
        print(f"    A  score={_score(r):.4f}  {r.log.title}  [{r.log.date}]")

    print(f"\n  {MODEL_B}만 선택 ({len(only_b)}):")
    for lid in sorted(only_b):
        r = ids_b[lid]
        print(f"    B  score={_score(r):.4f}  {r.log.title}  [{r.log.date}]")


def print_metrics(label: str, res: dict, user_labels: list[GoalLogLabel], top_k: int) -> None:
    if not user_labels:
        return
    all_types = set()
    cand_metrics = compute_candidate_metrics(res["candidates"], user_labels)
    metrics = compute_all_metrics(
        res["ranked"], user_labels,
        k=top_k,
        all_activity_types=all_types,
        selected_logs=res["selected"],
    )
    print(f"\n  [Metrics — {label}]")
    print(f"    candidate_recall    : {cand_metrics['candidate_recall']:.4f}  "
          f"({cand_metrics['relevant_in_pool']}/{cand_metrics['relevant_total']})")
    print(f"    candidate_precision : {cand_metrics['candidate_precision']:.4f}")
    print(f"    recall@{top_k:<3}         : {metrics[f'recall@{top_k}']:.4f}")
    print(f"    precision@{top_k:<3}      : {metrics[f'precision@{top_k}']:.4f}")
    print(f"    selected_precision  : {metrics['selected_precision']:.4f}")
    print(f"    f1@{top_k:<3}             : {metrics[f'f1@{top_k}']:.4f}")
    print(f"    false_positive_rate : {metrics['false_positive_rate']:.4f}")
    print(f"    mrr                 : {metrics['mrr']:.4f}")
    print(f"    ndcg@{top_k:<3}           : {metrics[f'ndcg@{top_k}']:.4f}")
    return metrics


# ── Single goal ───────────────────────────────────────────────────────────────

def run_single_goal(
    goal: ResearchGoal,
    user_logs: list[ResearchLog],
    user_labels: list[GoalLogLabel],
    top_k: int,
    dense_threshold: float,
    verbose: bool = True,
) -> dict:
    if verbose:
        print(f"\n{'═'*60}")
        print(f"  Goal : {goal.goal_id}  {goal.title}")
        print(f"  User : {goal.user_id}   Logs: {len(user_logs)}")
        print(f"{'═'*60}")

    try:
        res_a, t_a = run_stage1_with_model(goal, user_logs, MODEL_A, top_k, dense_threshold)
    except Exception as e:
        print(f"  [SKIP] {goal.goal_id}  ERROR Model A: {e}")
        return {}
    try:
        res_b, t_b = run_stage1_with_model(goal, user_logs, MODEL_B, top_k, dense_threshold)
    except Exception as e:
        print(f"  [SKIP] {goal.goal_id}  ERROR Model B: {e}")
        return {}

    exp_a: ExpandedQuery = res_a["expansion"]
    exp_b: ExpandedQuery = res_b["expansion"]

    if verbose:
        print_expansion(f"Model A — {MODEL_A}", exp_a, t_a)
        print_expansion(f"Model B — {MODEL_B}", exp_b, t_b)
        print_expansion_diff(exp_a, exp_b)
        print_ranked(f"Model A — {MODEL_A}", res_a["ranked"])
        print_ranked(f"Model B — {MODEL_B}", res_b["ranked"])
        print_selected(f"Model A — {MODEL_A}", res_a["selected"])
        print_selected(f"Model B — {MODEL_B}", res_b["selected"])
        print_selected_diff(res_a, res_b)

        if user_labels:
            m_a = print_metrics(f"Model A — {MODEL_A}", res_a, user_labels, top_k)
            m_b = print_metrics(f"Model B — {MODEL_B}", res_b, user_labels, top_k)
        else:
            m_a = m_b = {}

    # ── Summary stats ─────────────────────────────────────────────────────────
    ids_a = {r.log_id for r in res_a["selected"]}
    ids_b = {r.log_id for r in res_b["selected"]}
    common = ids_a & ids_b
    only_a = ids_a - ids_b
    only_b = ids_b - ids_a

    avg_score_a = sum(r.final_score for r in res_a["selected"]) / len(res_a["selected"]) if res_a["selected"] else 0.0
    avg_score_b = sum(r.final_score for r in res_b["selected"]) / len(res_b["selected"]) if res_b["selected"] else 0.0

    # Shared logs: A score vs B score diff
    ranked_map_a = {r.log_id: r.final_score for r in res_a["ranked"]}
    ranked_map_b = {r.log_id: r.final_score for r in res_b["ranked"]}
    score_diffs = [ranked_map_a.get(lid, 0) - ranked_map_b.get(lid, 0) for lid in common]
    avg_score_diff = sum(score_diffs) / len(score_diffs) if score_diffs else 0.0

    if verbose:
        print(f"\n{'═'*60}")
        print(f"  A elapsed={t_a:.2f}s  selected={len(ids_a)}  avg_score={avg_score_a:.4f}")
        print(f"  B elapsed={t_b:.2f}s  selected={len(ids_b)}  avg_score={avg_score_b:.4f}")
        print(f"  common={len(common)}  A_only={len(only_a)}  B_only={len(only_b)}")
        print(f"  avg_score_diff(A-B, shared)={avg_score_diff:+.4f}")
        print(f"{'═'*60}\n")

    return {
        "goal_id":          goal.goal_id,
        "goal_title":       goal.title,
        "user_id":          goal.user_id,
        "num_logs":         len(user_logs),
        "t_a":              round(t_a, 2),
        "t_b":              round(t_b, 2),
        "cands_a":          len(res_a["candidates"]),
        "cands_b":          len(res_b["candidates"]),
        "filtered_a":       len(res_a["filtered"]),
        "filtered_b":       len(res_b["filtered"]),
        "selected_a":       len(ids_a),
        "selected_b":       len(ids_b),
        "common":           len(common),
        "only_a":           len(only_a),
        "only_b":           len(only_b),
        "avg_score_a":      round(avg_score_a, 4),
        "avg_score_b":      round(avg_score_b, 4),
        "avg_score_diff":   round(avg_score_diff, 4),
        "a_only_titles":    [next(r for r in res_a["selected"] if r.log_id == lid).log.title for lid in only_a],
        "b_only_titles":    [next(r for r in res_b["selected"] if r.log_id == lid).log.title for lid in only_b],
    }


# ── All goals ─────────────────────────────────────────────────────────────────

def run_all_goals(
    goals: list[ResearchGoal],
    logs: list[ResearchLog],
    labels: list[GoalLogLabel],
    top_k: int,
    dense_threshold: float,
    use_checkpoint: bool = True,
) -> None:
    rows: list[dict] = []
    total = len(goals)

    # Load checkpoint if exists
    done_ids: set[str] = set()
    if use_checkpoint and _CHECKPOINT_PATH.exists():
        try:
            saved = json.loads(_CHECKPOINT_PATH.read_text())
            rows = saved
            done_ids = {r["goal_id"] for r in rows}
            print(f"[Checkpoint] {len(done_ids)} goals already done — resuming.")
        except Exception:
            pass

    for i, goal in enumerate(goals, 1):
        if goal.goal_id in done_ids:
            print(f"\r[{i:3d}/{total}] {goal.goal_id} (skipped — checkpoint)         ", end="", flush=True)
            continue

        user_logs = [l for l in logs if l.user_id == goal.user_id]
        user_labels = [lb for lb in labels if lb.user_id == goal.user_id and lb.goal_id == goal.goal_id]
        print(f"\r[{i:3d}/{total}] {goal.goal_id} {goal.title[:30]:<30}", end="", flush=True)

        row = run_single_goal(goal, user_logs, user_labels, top_k, dense_threshold, verbose=False)
        if row:
            rows.append(row)
            done_ids.add(goal.goal_id)

        # Save checkpoint after each goal
        if use_checkpoint:
            _CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
            _CHECKPOINT_PATH.write_text(json.dumps(rows, ensure_ascii=False, indent=2))

    print()  # newline after progress

    if not rows:
        print("No results.")
        return

    # ── Summary table ─────────────────────────────────────────────────────────
    W = 80
    print(f"\n{'═'*W}")
    print(f"  Stage 1 A/B Comparison — {len(rows)} goals")
    print(f"  Model A: {MODEL_A}   Model B: {MODEL_B}")
    print(f"  top_k={top_k}  dense_threshold={dense_threshold}")
    print(f"{'═'*W}")
    hdr = (f"  {'goal_id':<14} {'title':<22} "
           f"{'selA':>4} {'selB':>4} {'avgA':>6} {'avgB':>6} {'diff':>8}  "
           f"{'com':>4} {'A+':>4} {'B+':>4}  {'tA':>5} {'tB':>5}")
    print(hdr)
    print(f"  {'─'*14} {'─'*22} {'─'*4} {'─'*4} {'─'*6} {'─'*6} {'─'*8}  {'─'*4} {'─'*4} {'─'*4}  {'─'*5} {'─'*5}")

    sum_diff = 0.0
    a_wins = b_wins = ties = 0
    for r in rows:
        d = r["avg_score_diff"]
        sum_diff += d
        if d > 0.001:    a_wins += 1
        elif d < -0.001: b_wins += 1
        else:            ties   += 1
        winner = "A>" if d > 0.001 else ("B>" if d < -0.001 else " =")
        print(
            f"  {r['goal_id']:<14} {r['goal_title'][:22]:<22}"
            f" {r['selected_a']:>4} {r['selected_b']:>4}"
            f" {r['avg_score_a']:>6.4f} {r['avg_score_b']:>6.4f}"
            f" {d:>+8.4f}{winner}"
            f" {r['common']:>4} {r['only_a']:>4} {r['only_b']:>4}"
            f"  {r['t_a']:>5.1f} {r['t_b']:>5.1f}"
        )

    print(f"{'─'*W}")
    macro_diff = sum_diff / len(rows)
    print(f"  macro avg diff (A-B): {macro_diff:+.4f}")
    print(f"  A wins: {a_wins}  B wins: {b_wins}  ties: {ties}  total: {len(rows)}")

    avg_ta = sum(r["t_a"] for r in rows) / len(rows)
    avg_tb = sum(r["t_b"] for r in rows) / len(rows)
    print(f"  avg expansion time  A: {avg_ta:.2f}s  B: {avg_tb:.2f}s")
    print(f"{'═'*W}\n")

    # ── B-only / A-only selected logs ─────────────────────────────────────────
    b_adds = [(r["goal_id"], r["goal_title"], r["b_only_titles"]) for r in rows if r["b_only_titles"]]
    if b_adds:
        print("  [B-only selected (Model B가 추가로 선택한 로그)]")
        for gid, gtitle, titles in b_adds[:20]:
            print(f"    {gid}  {gtitle[:28]:<28}  → {titles}")
        print()

    a_adds = [(r["goal_id"], r["goal_title"], r["a_only_titles"]) for r in rows if r["a_only_titles"]]
    if a_adds:
        print("  [A-only selected (Model A만 선택한 로그)]")
        for gid, gtitle, titles in a_adds[:20]:
            print(f"    {gid}  {gtitle[:28]:<28}  → {titles}")
        print()

    if use_checkpoint and _CHECKPOINT_PATH.exists():
        _CHECKPOINT_PATH.unlink()
        print(f"  [Checkpoint cleared: {_CHECKPOINT_PATH}]")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1 A/B comparison: Model A vs Model B")
    parser.add_argument("--user_id",          default=None)
    parser.add_argument("--goal_id",          default=None)
    parser.add_argument("--auto",             action="store_true", help="First goal only")
    parser.add_argument("--all_goals",        action="store_true", help="Run over every goal")
    parser.add_argument("--limit",            type=int, default=None)
    parser.add_argument("--top_k",            type=int, default=10)
    parser.add_argument("--dense_threshold",  type=float, default=0.92)
    parser.add_argument("--data_dir",         default=_DEFAULT_DATA_DIR)
    parser.add_argument("--no_checkpoint",    action="store_true", help="Disable checkpoint (don't resume)")
    args = parser.parse_args()

    goals, logs, labels = load_data(args.data_dir)

    # ── All-goals mode ─────────────────────────────────────────────────────────
    if args.all_goals:
        target_goals = goals[:args.limit] if args.limit else goals
        print(f"\nStage 1 A/B  —  {len(target_goals)} goals")
        print(f"Model A: {MODEL_A}   Model B: {MODEL_B}")
        print(f"top_k={args.top_k}  dense_threshold={args.dense_threshold}\n")
        run_all_goals(
            target_goals, logs, labels,
            top_k=args.top_k,
            dense_threshold=args.dense_threshold,
            use_checkpoint=not args.no_checkpoint,
        )
        return

    # ── Single-goal mode ───────────────────────────────────────────────────────
    if args.auto or (args.user_id is None and args.goal_id is None):
        goal = goals[0]
    elif args.goal_id:
        goal = next((g for g in goals if g.goal_id == args.goal_id), None)
        if not goal:
            print(f"goal_id={args.goal_id} not found."); sys.exit(1)
    else:
        goal = next((g for g in goals if g.user_id == args.user_id), None)
        if not goal:
            print(f"No goals for user_id={args.user_id}."); sys.exit(1)

    user_logs   = [l  for l  in logs   if l.user_id  == goal.user_id]
    user_labels = [lb for lb in labels if lb.user_id == goal.user_id and lb.goal_id == goal.goal_id]
    run_single_goal(goal, user_logs, user_labels, top_k=args.top_k, dense_threshold=args.dense_threshold, verbose=True)


if __name__ == "__main__":
    main()
