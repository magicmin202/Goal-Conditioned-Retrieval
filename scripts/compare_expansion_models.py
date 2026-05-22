#!/usr/bin/env python3
"""Query Expansion A/B comparison: gemini-2.0-flash vs gemini-3-flash-preview.

비교 항목:
  1. 생성된 expansion 품질 (evidence / priority / related / negative terms)
  2. Candidate Retrieval 결과 (top-N 로그, hybrid_score)
  3. 두 모델의 결과 차이 (공통 / 모델A만 / 모델B만)

Usage:
    .venv/bin/python scripts/compare_expansion_models.py --auto
    .venv/bin/python scripts/compare_expansion_models.py --goal_id G-U0001-01
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from app.utils.logging_utils import setup_logging
setup_logging(level="WARNING")   # suppress INFO noise during comparison

import logging
logger = logging.getLogger(__name__)

from app.config import DEFAULT_CONFIG, GeminiConfig
from app.data_generation.dataset_builder import build_dataset
from app.data_generation.export_utils import load_dataset_from_json
from app.retrieval.query_understanding import build_query
from app.retrieval.query_expansion import expand_goal_query, ExpandedQuery
from app.retrieval.candidate_retrieval import CandidateRetriever
from app.schemas import ResearchGoal, ResearchLog

_DEFAULT_DATA_DIR = "data/synthetic"

MODEL_A = "gemini-2.5-flash"
MODEL_B = "gemini-3-flash-preview"


def load_data(data_dir: str):
    goals_path = Path(data_dir) / "goals.json"
    if goals_path.exists():
        _, goals, logs, labels = load_dataset_from_json(data_dir)
    else:
        ds = build_dataset(small_mode=True, seed=42)
        goals, logs, labels = ds.goals, ds.logs, ds.labels
    return goals, logs, labels


def run_expansion(
    goal: ResearchGoal,
    model_name: str,
) -> tuple[ExpandedQuery, float]:
    """Run query expansion with the given model. Returns (expanded, elapsed_sec)."""
    cfg = GeminiConfig(
        model_name=model_name,
        max_output_tokens=2048,
        temperature=0.2,
    )
    query_obj = build_query(goal)
    t0 = time.time()
    expanded = expand_goal_query(
        goal,
        query_obj,
        max_terms=15,
        mode="structured",
        use_mock_fallback=False,
        use_cache=False,   # bypass disk cache so each model runs independently
        gemini_config=cfg,
    )
    elapsed = time.time() - t0
    return expanded, elapsed


def run_candidate_retrieval(
    logs: list[ResearchLog],
    expanded: ExpandedQuery,
    top_n: int,
    dense_threshold: float = 0.92,
) -> list:
    """Retrieve candidates and filter by dense_score >= dense_threshold.

    Fetches a wide pool first (top_n * 3) then keeps only candidates
    whose dense_score meets the threshold. Falls back to top_n if
    nothing passes the threshold.
    """
    cfg = DEFAULT_CONFIG.stage1.retrieval
    pool_size = max(top_n * 3, len(logs))
    cfg.candidate_size = pool_size
    retriever = CandidateRetriever(config=cfg)
    retriever.index(logs)
    all_cands = retriever.retrieve(expanded, top_n=pool_size)

    above = [c for c in all_cands if (c.dense_score or 0.0) >= dense_threshold]
    if above:
        return above
    # fallback: nothing passes threshold → return top_n as before
    return all_cands[:top_n]


def print_expansion(label: str, exp: ExpandedQuery, elapsed: float) -> None:
    print(f"\n{'─'*60}")
    print(f"  {label}  ({exp.base_query.goal_id})  [{elapsed:.2f}s]")
    print(f"{'─'*60}")
    print(f"  goal_summary : {exp.goal_summary}")
    print(f"  core_intents : {exp.core_intents}")
    print(f"  priority     ({len(exp.priority_terms):2d}) : {exp.priority_terms}")
    print(f"  evidence     ({len(exp.expanded_terms):2d}) : {exp.expanded_terms}")
    print(f"  related      ({len(exp.related_terms):2d}) : {exp.related_terms}")
    print(f"  negative     ({len(exp.negative_terms):2d}) : {exp.negative_terms}")


def print_candidates(label: str, candidates: list) -> None:
    print(f"\n  [Candidates — {label}]")
    for i, c in enumerate(candidates, 1):
        score = c.dense_score or c.hybrid_score or 0.0
        print(f"    {i:2d}. score={score:.4f}  {c.log.date}  {c.log.title}")


def print_diff(candidates_a: list, candidates_b: list) -> None:
    ids_a = {c.log_id: c for c in candidates_a}
    ids_b = {c.log_id: c for c in candidates_b}
    common = set(ids_a) & set(ids_b)
    only_a = set(ids_a) - set(ids_b)
    only_b = set(ids_b) - set(ids_a)

    def _score(c) -> float:
        return c.dense_score or c.hybrid_score or 0.0

    print(f"\n{'═'*60}")
    print(f"  [Candidate Diff]")
    print(f"  공통 ({len(common)}):")
    for lid in sorted(common):
        c = ids_a[lid]
        print(f"    ○ {c.log.title}  [{c.log.date}]")

    print(f"\n  {MODEL_A}만 포함 ({len(only_a)}):")
    for lid in sorted(only_a):
        c = ids_a[lid]
        print(f"    A  score={_score(c):.4f}  {c.log.title}  [{c.log.date}]")

    print(f"\n  {MODEL_B}만 포함 ({len(only_b)}):")
    for lid in sorted(only_b):
        c = ids_b[lid]
        print(f"    B  score={_score(c):.4f}  {c.log.title}  [{c.log.date}]")


def print_expansion_diff(exp_a: ExpandedQuery, exp_b: ExpandedQuery) -> None:
    """Show term-level diff between two expansions."""
    def diff(name: str, a: list[str], b: list[str]) -> None:
        sa, sb = set(a), set(b)
        common = sa & sb
        only_a = sa - sb
        only_b = sb - sa
        print(f"\n  [{name}]")
        if common:
            print(f"    공통  : {sorted(common)}")
        if only_a:
            print(f"    A만   : {sorted(only_a)}")
        if only_b:
            print(f"    B만   : {sorted(only_b)}")

    print(f"\n{'═'*60}")
    print("  [Expansion Term Diff]")
    diff("priority_terms", exp_a.priority_terms, exp_b.priority_terms)
    diff("evidence_terms", exp_a.expanded_terms, exp_b.expanded_terms)
    diff("related_terms",  exp_a.related_terms,  exp_b.related_terms)
    diff("negative_terms", exp_a.negative_terms, exp_b.negative_terms)


def _score(c) -> float:
    return c.dense_score or c.hybrid_score or 0.0


def run_single_goal(
    target_goal: ResearchGoal,
    user_logs: list[ResearchLog],
    top_n: int,
    dense_threshold: float = 0.92,
    verbose: bool = True,
) -> dict:
    """Run A/B comparison for one goal. Returns summary dict."""
    if verbose:
        print(f"\n{'═'*60}")
        print(f"  Goal  : {target_goal.goal_id}  {target_goal.title}")
        print(f"  User  : {target_goal.user_id}   Logs: {len(user_logs)}")
        print(f"{'═'*60}")

    # expansions
    try:
        exp_a, t_a = run_expansion(target_goal, MODEL_A)
    except Exception as e:
        print(f"  [SKIP] {target_goal.goal_id}  ERROR Model A: {e}")
        return {}
    try:
        exp_b, t_b = run_expansion(target_goal, MODEL_B)
    except Exception as e:
        print(f"  [SKIP] {target_goal.goal_id}  ERROR Model B: {e}")
        return {}

    if verbose:
        print_expansion(f"Model A — {MODEL_A}", exp_a, t_a)
        print_expansion(f"Model B — {MODEL_B}", exp_b, t_b)
        print_expansion_diff(exp_a, exp_b)

    # candidates
    cands_a = run_candidate_retrieval(user_logs, exp_a, top_n=top_n, dense_threshold=dense_threshold)
    cands_b = run_candidate_retrieval(user_logs, exp_b, top_n=top_n, dense_threshold=dense_threshold)

    ids_a = {c.log_id: c for c in cands_a}
    ids_b = {c.log_id: c for c in cands_b}
    common  = set(ids_a) & set(ids_b)
    only_a  = set(ids_a) - set(ids_b)
    only_b  = set(ids_b) - set(ids_a)

    # avg score of shared candidates: A score vs B score
    score_diffs = []
    for lid in common:
        diff = _score(ids_a[lid]) - _score(ids_b[lid])
        score_diffs.append(diff)
    avg_score_diff = sum(score_diffs) / len(score_diffs) if score_diffs else 0.0

    # avg score of each model's top candidates
    avg_a = sum(_score(c) for c in cands_a) / len(cands_a) if cands_a else 0.0
    avg_b = sum(_score(c) for c in cands_b) / len(cands_b) if cands_b else 0.0

    if verbose:
        print_candidates(f"Model A — {MODEL_A}", cands_a)
        print_candidates(f"Model B — {MODEL_B}", cands_b)
        print_diff(cands_a, cands_b)
        print(f"\n{'═'*60}")
        print(f"  A elapsed={t_a:.2f}s  avg_score={avg_a:.4f}")
        print(f"  B elapsed={t_b:.2f}s  avg_score={avg_b:.4f}")
        print(f"  common={len(common)}  A_only={len(only_a)}  B_only={len(only_b)}")
        print(f"  avg_score_diff(A-B, shared)={avg_score_diff:+.4f}")
        print(f"{'═'*60}\n")

    return {
        "goal_id":        target_goal.goal_id,
        "goal_title":     target_goal.title,
        "user_id":        target_goal.user_id,
        "num_logs":       len(user_logs),
        "t_a":            round(t_a, 2),
        "t_b":            round(t_b, 2),
        "cands_a":        len(cands_a),
        "cands_b":        len(cands_b),
        "common":         len(common),
        "only_a":         len(only_a),
        "only_b":         len(only_b),
        "avg_score_a":    round(avg_a, 4),
        "avg_score_b":    round(avg_b, 4),
        "avg_score_diff": round(avg_score_diff, 4),  # positive = A scored higher on shared
        "a_only_titles":  [ids_a[lid].log.title for lid in only_a],
        "b_only_titles":  [ids_b[lid].log.title for lid in only_b],
    }


def run_all_goals(goals, logs, top_n: int, dense_threshold: float = 0.92) -> None:
    """Iterate over every goal and print a summary table."""
    rows = []
    total = len(goals)
    for i, goal in enumerate(goals, 1):
        user_logs = [l for l in logs if l.user_id == goal.user_id]
        print(f"\r[{i:3d}/{total}] {goal.goal_id} {goal.title[:30]:<30}", end="", flush=True)
        row = run_single_goal(goal, user_logs, top_n=top_n, dense_threshold=dense_threshold, verbose=False)
        if row:
            rows.append(row)

    print()  # newline after progress

    if not rows:
        print("No results."); return

    # ── Summary table ──────────────────────────────────────────────────────────
    W = 72
    print(f"\n{'═'*W}")
    print(f"  A/B Candidate Score Comparison — all {len(rows)} goals")
    print(f"  Model A: {MODEL_A}   Model B: {MODEL_B}   top_n={top_n}  dense_threshold={dense_threshold}")
    print(f"{'═'*W}")
    hdr = f"  {'goal_id':<14} {'title':<24} {'avgA':>6} {'avgB':>6} {'diff(A-B)':>10} {'com':>4} {'A+':>4} {'B+':>4}"
    print(hdr)
    print(f"  {'─'*14} {'─'*24} {'─'*6} {'─'*6} {'─'*10} {'─'*4} {'─'*4} {'─'*4}")

    sum_diff = 0.0
    a_wins = b_wins = ties = 0
    for r in rows:
        d = r["avg_score_diff"]
        sum_diff += d
        if d > 0.001:   a_wins += 1
        elif d < -0.001: b_wins += 1
        else:            ties   += 1
        winner = "A>" if d > 0.001 else ("B>" if d < -0.001 else " =")
        print(
            f"  {r['goal_id']:<14} {r['goal_title'][:24]:<24}"
            f" {r['avg_score_a']:>6.4f} {r['avg_score_b']:>6.4f}"
            f" {d:>+10.4f} {winner}"
            f" {r['common']:>4} {r['only_a']:>4} {r['only_b']:>4}"
        )

    print(f"{'─'*W}")
    macro_diff = sum_diff / len(rows)
    print(f"  macro avg diff (A-B): {macro_diff:+.4f}")
    print(f"  A wins: {a_wins}  B wins: {b_wins}  ties: {ties}  total: {len(rows)}")

    # speed
    avg_ta = sum(r["t_a"] for r in rows) / len(rows)
    avg_tb = sum(r["t_b"] for r in rows) / len(rows)
    print(f"  avg expansion time  A: {avg_ta:.2f}s  B: {avg_tb:.2f}s")
    print(f"{'═'*W}\n")

    # ── Goals where B adds unique relevant-looking logs ────────────────────────
    b_adds = [(r["goal_id"], r["goal_title"], r["b_only_titles"]) for r in rows if r["b_only_titles"]]
    if b_adds:
        print(f"  [B-only candidates (Model B가 추가로 찾은 로그)]")
        for gid, gtitle, titles in b_adds[:20]:
            print(f"    {gid}  {gtitle[:30]:<30}  → {titles}")
        print()

    a_adds = [(r["goal_id"], r["goal_title"], r["a_only_titles"]) for r in rows if r["a_only_titles"]]
    if a_adds:
        print(f"  [A-only candidates (Model A만 찾은 로그)]")
        for gid, gtitle, titles in a_adds[:20]:
            print(f"    {gid}  {gtitle[:30]:<30}  → {titles}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare query expansion: Model A vs B")
    parser.add_argument("--user_id",   default=None)
    parser.add_argument("--goal_id",   default=None)
    parser.add_argument("--auto",      action="store_true", help="First goal only")
    parser.add_argument("--all_goals", action="store_true", help="Run over every goal → summary table")
    parser.add_argument("--limit",     type=int, default=None, help="Limit number of goals in --all_goals mode")
    parser.add_argument("--data_dir",  default=_DEFAULT_DATA_DIR)
    parser.add_argument("--top_n",           type=int,   default=15,   help="Fallback pool size when no candidate passes threshold")
    parser.add_argument("--dense_threshold", type=float, default=0.92, help="Keep candidates with dense_score >= threshold (default: 0.92)")
    args = parser.parse_args()

    goals, logs, _ = load_data(args.data_dir)

    # ── All-goals mode ─────────────────────────────────────────────────────────
    if args.all_goals:
        target_goals = goals[:args.limit] if args.limit else goals
        print(f"\nRunning A/B comparison over {len(target_goals)} goals (dense_threshold={args.dense_threshold}) ...")
        print(f"Model A: {MODEL_A}   Model B: {MODEL_B}\n")
        run_all_goals(target_goals, logs, top_n=args.top_n, dense_threshold=args.dense_threshold)
        return

    # ── Single-goal mode ───────────────────────────────────────────────────────
    if args.auto or (args.user_id is None and args.goal_id is None):
        target_goal = goals[0]
    elif args.goal_id:
        target_goal = next((g for g in goals if g.goal_id == args.goal_id), None)
        if not target_goal:
            print(f"goal_id={args.goal_id} not found."); sys.exit(1)
    else:
        target_goal = next((g for g in goals if g.user_id == args.user_id), None)
        if not target_goal:
            print(f"No goals for user_id={args.user_id}."); sys.exit(1)

    user_logs = [l for l in logs if l.user_id == target_goal.user_id]
    run_single_goal(target_goal, user_logs, top_n=args.top_n, dense_threshold=args.dense_threshold, verbose=True)


if __name__ == "__main__":
    main()
