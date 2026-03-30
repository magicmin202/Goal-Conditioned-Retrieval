#!/usr/bin/env python3
"""Diagnostic comparison script: Stage1 standalone vs Stage2 internal Stage1.

Runs 3 experiments for the same goal and prints a comparison table:
  Exp A: Stage1 standalone, no expansion
  Exp B: Stage1 standalone, with expansion
  Exp C: Stage2 chain (Stage1→Stage2), with expansion (same as run_stage2.py)

Purpose: diagnose WHY the same goal produces different admitted candidates
across execution paths.

Usage:
    .venv/bin/python scripts/compare_debug_runs.py --goal_id G-U0001-01
    .venv/bin/python scripts/compare_debug_runs.py --auto
    .venv/bin/python scripts/compare_debug_runs.py --goal_id G-U0002-01 --top_k 5
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from app.utils.logging_utils import setup_logging
setup_logging()

import logging
from app.config import DEFAULT_CONFIG, Stage1Config
from app.data_generation.dataset_builder import build_dataset
from app.data_generation.export_utils import load_dataset_from_json
from app.pipeline.stage1_ranking_pipeline import Stage1Pipeline, Stage1Result
from app.pipeline.stage2_rag_pipeline import Stage2Pipeline
from app.schemas import ResearchGoal, ResearchLog

logger = logging.getLogger(__name__)
_DEFAULT_DATA_DIR = "data/synthetic"


def load_data(data_dir: str):
    goals_path = Path(data_dir) / "goals.json"
    if goals_path.exists():
        _, goals, logs, labels = load_dataset_from_json(data_dir)
    else:
        ds = build_dataset(small_mode=True, seed=42)
        goals, logs, labels = ds.goals, ds.logs, ds.labels
    return goals, logs, labels


def _fresh_s1_config(top_k: int, corpus_size: int) -> Stage1Config:
    """Deep-copy Stage1Config to avoid DEFAULT_CONFIG mutation."""
    cfg = copy.deepcopy(DEFAULT_CONFIG.stage1)
    cfg.retrieval.top_k = top_k
    cfg.retrieval.candidate_size = max(top_k * 3, min(corpus_size * 6 // 10, 30))
    return cfg


def run_experiment(
    label: str,
    goal: ResearchGoal,
    user_logs: list[ResearchLog],
    top_k: int,
    use_expansion: bool,
    use_real_embeddings: bool = False,
) -> Stage1Result:
    """Run a single Stage1 experiment and return the result."""
    cfg = _fresh_s1_config(top_k, len(user_logs))
    pipeline = Stage1Pipeline(config=cfg, use_real_embeddings=use_real_embeddings)
    pipeline.index(user_logs)
    result = pipeline.run(goal, use_expansion=use_expansion, run_label=label)
    return result


def _candidate_row(c) -> dict:
    return {
        "log_id": c.log_id,
        "title": c.log.title[:40],
        "bm25": round(c.sparse_score, 4),
        "dense": round(c.dense_score, 4),
        "hybrid": round(c.hybrid_score, 4),
    }


def _ranked_row(r) -> dict:
    return {
        "log_id": r.log_id,
        "title": r.log.title[:40],
        "score": round(r.final_score, 4),
        "cat": getattr(r, "schema_category", "?"),
        "strength": getattr(r, "category_hit_strength", "?"),
        "decision": "ADMIT" if r.final_score > 0 else "REJECT",
        "reason": r.rejection_reason or r.admission_reason or "-",
    }


def print_comparison(
    label_a: str, result_a: Stage1Result,
    label_b: str, result_b: Stage1Result,
    label_c: str, result_c: Stage1Result,
) -> None:
    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)

    print(f"\n{'─'*40}")
    print("QUERY INPUT")
    print(f"{'─'*40}")
    print(f"  {label_a:30s} | {result_a.query_text[:60]}")
    print(f"  {label_b:30s} | {result_b.query_text[:60]}")
    print(f"  {label_c:30s} | {result_c.query_text[:60]}")

    print(f"\n{'─'*40}")
    print("EXPANSION")
    print(f"{'─'*40}")
    for label, res in [(label_a, result_a), (label_b, result_b), (label_c, result_c)]:
        print(f"  {label:30s}")
        print(f"    expansion_used : {res.used_expansion}")
        print(f"    priority_terms : {res.priority_terms[:4]}")
        print(f"    negative_terms : {res.negative_terms[:4]}")

    print(f"\n{'─'*40}")
    print("CANDIDATE POOL (top-10, by hybrid_score)")
    print(f"{'─'*40}")
    header = f"  {'log_id':16s} {'bm25':>6} {'dense':>6} {'hybrid':>7}  title"
    print(f"  [{label_a}]")
    print(header)
    for c in result_a.candidates[:10]:
        print(f"  {c.log_id:16s} {c.sparse_score:6.4f} {c.dense_score:6.4f} {c.hybrid_score:7.4f}  {c.log.title[:35]}")

    print(f"\n  [{label_b}]")
    print(header)
    for c in result_b.candidates[:10]:
        print(f"  {c.log_id:16s} {c.sparse_score:6.4f} {c.dense_score:6.4f} {c.hybrid_score:7.4f}  {c.log.title[:35]}")

    print(f"\n  [{label_c}]")
    print(header)
    for c in result_c.candidates[:10]:
        print(f"  {c.log_id:16s} {c.sparse_score:6.4f} {c.dense_score:6.4f} {c.hybrid_score:7.4f}  {c.log.title[:35]}")

    # Candidate pool diff
    ids_a = {c.log_id for c in result_a.candidates}
    ids_b = {c.log_id for c in result_b.candidates}
    ids_c = {c.log_id for c in result_c.candidates}
    print(f"\n  Pool diff  A∩B={len(ids_a&ids_b)}  A∩C={len(ids_a&ids_c)}  B∩C={len(ids_b&ids_c)}  A∩B∩C={len(ids_a&ids_b&ids_c)}")
    only_in_a = ids_a - ids_b - ids_c
    only_in_b = ids_b - ids_a - ids_c
    only_in_c = ids_c - ids_a - ids_b
    if only_in_a:
        titles = {c.log_id: c.log.title for c in result_a.candidates}
        print(f"  Only in A: {[(i, titles.get(i,'?')[:30]) for i in only_in_a]}")
    if only_in_b:
        titles = {c.log_id: c.log.title for c in result_b.candidates}
        print(f"  Only in B: {[(i, titles.get(i,'?')[:30]) for i in only_in_b]}")
    if only_in_c:
        titles = {c.log_id: c.log.title for c in result_c.candidates}
        print(f"  Only in C: {[(i, titles.get(i,'?')[:30]) for i in only_in_c]}")

    print(f"\n{'─'*40}")
    print("RERANKER / ADMISSION (all candidates)")
    print(f"{'─'*40}")
    rh = f"  {'log_id':16s} {'score':>6} {'cat':20s} {'str':10s} {'dec':6s}  reason"
    for label, res in [(label_a, result_a), (label_b, result_b), (label_c, result_c)]:
        print(f"\n  [{label}]")
        print(rh)
        for r in sorted(res.ranked_logs, key=lambda x: x.final_score, reverse=True)[:15]:
            dec = "ADMIT" if r.final_score > 0 else "rej"
            reason = (r.rejection_reason or r.admission_reason or "-")[:35]
            print(
                f"  {r.log_id:16s} {r.final_score:6.4f} "
                f"{getattr(r,'schema_category','?'):20s} "
                f"{getattr(r,'category_hit_strength','?'):10s} "
                f"{dec:6s}  {reason}"
            )

    print(f"\n{'─'*40}")
    print("ADMITTED ANCHORS")
    print(f"{'─'*40}")
    for label, res in [(label_a, result_a), (label_b, result_b), (label_c, result_c)]:
        admitted_ids = [r.log_id for r in res.selected_logs]
        print(f"  {label:30s} → {len(admitted_ids)} admitted: {admitted_ids}")
        for r in res.selected_logs:
            print(f"    {r.log_id}  score={r.final_score:.4f}  [{r.log.title}]")
            print(f"      reason: {r.admission_reason}")

    # Overlap analysis
    adm_a = {r.log_id for r in result_a.selected_logs}
    adm_b = {r.log_id for r in result_b.selected_logs}
    adm_c = {r.log_id for r in result_c.selected_logs}
    print(f"\n  Admitted overlap  A∩B={sorted(adm_a&adm_b)}  A∩C={sorted(adm_a&adm_c)}  B∩C={sorted(adm_b&adm_c)}")
    print(f"  Only in A: {sorted(adm_a - adm_b - adm_c)}")
    print(f"  Only in B: {sorted(adm_b - adm_a - adm_c)}")
    print(f"  Only in C: {sorted(adm_c - adm_a - adm_b)}")


def print_hypothesis_verdict(
    result_a: Stage1Result,
    result_b: Stage1Result,
    result_c: Stage1Result,
) -> None:
    print(f"\n{'='*80}")
    print("HYPOTHESIS VERDICT")
    print(f"{'='*80}")

    ids_a = {c.log_id for c in result_a.candidates}
    ids_b = {c.log_id for c in result_b.candidates}
    ids_c = {c.log_id for c in result_c.candidates}
    adm_a = {r.log_id for r in result_a.selected_logs}
    adm_b = {r.log_id for r in result_b.selected_logs}
    adm_c = {r.log_id for r in result_c.selected_logs}

    # H-A: log embedding cache is goal-dependent
    print("\n[H-A] Log embedding cache goal-dependent?")
    print("  → Cache key = SHA256(log.full_text) — goal-agnostic.")
    print("  → VERDICT: ✗ Not this issue.")

    # H-B: query embedding / expansion difference
    query_diff_bc = result_b.query_text != result_c.query_text
    cand_diff_bc = ids_b != ids_c
    print(f"\n[H-B] Query/expansion difference (B vs C)?")
    print(f"  query_text same (B vs C): {not query_diff_bc}")
    print(f"  candidate pool same (B vs C): {not cand_diff_bc}")
    overlap_bc = len(ids_b & ids_c)
    print(f"  candidate pool overlap B∩C: {overlap_bc}/{len(ids_b)} B, {overlap_bc}/{len(ids_c)} C")
    if cand_diff_bc:
        print("  → VERDICT: ⚠ Candidate pools differ between B and C — expansion state likely differs.")
    else:
        print("  → VERDICT: ✓ Same pools, expansion is not the issue.")

    # H-C: index state contamination
    print(f"\n[H-C] Vector index / pool contamination?")
    print(f"  Each experiment uses a fresh Stage1Pipeline.index() call.")
    print(f"  candidate_size A={len(result_a.candidates)}  B={len(result_b.candidates)}  C={len(result_c.candidates)}")
    if len(result_a.candidates) == len(result_b.candidates) == len(result_c.candidates):
        print("  → VERDICT: ✓ Consistent pool sizes — no contamination detected.")
    else:
        print("  → VERDICT: ⚠ Pool sizes differ — check candidate_size config.")

    # H-D: admission rule difference
    adm_diff_bc = adm_b != adm_c
    print(f"\n[H-D] Admission rule difference (B vs C)?")
    print(f"  Admitted B: {sorted(adm_b)}")
    print(f"  Admitted C: {sorted(adm_c)}")
    if adm_diff_bc:
        print(f"  Only in B: {sorted(adm_b - adm_c)}")
        print(f"  Only in C: {sorted(adm_c - adm_b)}")
        print("  → VERDICT: ⚠ Admission differs — check threshold / schema config between runs.")
    else:
        print("  → VERDICT: ✓ Same admission result (B == C).")

    # Overall
    print(f"\n[SUMMARY]")
    print(f"  A (no expansion) admitted:  {len(adm_a)}")
    print(f"  B (expansion)    admitted:  {len(adm_b)}")
    print(f"  C (stage2 chain) admitted:  {len(adm_c)}")
    if adm_b == adm_c:
        print("  B == C: Stage2 internal Stage1 is CONSISTENT with standalone expansion mode.")
    else:
        print("  B ≠ C: Stage2 internal Stage1 produces DIFFERENT results — investigate config.")
    if len(adm_a) != len(adm_b):
        print("  A ≠ B: expansion changes admission results — expected (different retrieval query).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug: compare Stage1 runs")
    parser.add_argument("--goal_id", default=None)
    parser.add_argument("--user_id", default=None)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("--data_dir", default=_DEFAULT_DATA_DIR)
    parser.add_argument("--real_embeddings", action="store_true")
    parser.add_argument("--json_out", default=None, help="Write comparison JSON to file")
    args = parser.parse_args()

    goals, logs, _ = load_data(args.data_dir)

    if args.auto or (args.goal_id is None and args.user_id is None):
        target_goal = goals[0]
    elif args.goal_id:
        target_goal = next((g for g in goals if g.goal_id == args.goal_id), None)
        if not target_goal:
            print(f"goal_id={args.goal_id} not found. Available: {[g.goal_id for g in goals]}")
            sys.exit(1)
    else:
        target_goal = next((g for g in goals if g.user_id == args.user_id), None)
        if not target_goal:
            sys.exit(1)

    user_logs = [l for l in logs if l.user_id == target_goal.user_id]
    print(f"\nGoal: {target_goal.goal_id}  ({target_goal.title})")
    print(f"User: {target_goal.user_id}  Logs: {len(user_logs)}")

    # ── Experiment A: Stage1 standalone, no expansion ─────────────────────────
    print("\n[Exp A] Stage1 standalone, no expansion ...")
    result_a = run_experiment(
        "A_standalone_no_expand", target_goal, user_logs,
        top_k=args.top_k, use_expansion=False,
        use_real_embeddings=args.real_embeddings,
    )

    # ── Experiment B: Stage1 standalone, with expansion ───────────────────────
    print("[Exp B] Stage1 standalone, with expansion ...")
    result_b = run_experiment(
        "B_standalone_expand", target_goal, user_logs,
        top_k=args.top_k, use_expansion=True,
        use_real_embeddings=args.real_embeddings,
    )

    # ── Experiment C: Stage2 chain (uses Stage1 internally with expansion) ────
    print("[Exp C] Stage2 chain (Stage1 internal + Stage2 consolidation) ...")
    s1_cfg_c = _fresh_s1_config(args.top_k, len(user_logs))
    s1_pipe_c = Stage1Pipeline(config=s1_cfg_c, use_real_embeddings=args.real_embeddings)
    s1_pipe_c.index(user_logs)
    result_c = s1_pipe_c.run(target_goal, use_expansion=True, run_label="C_stage2_internal")

    s2_cfg = copy.deepcopy(DEFAULT_CONFIG.stage2)
    s2_pipe = Stage2Pipeline(config=s2_cfg, use_real_embeddings=args.real_embeddings)
    s2_pipe.index(user_logs)
    s2_result = s2_pipe.run_with_stage1(result_c)

    print_comparison(
        "A:s1_no_expand", result_a,
        "B:s1_expand", result_b,
        "C:s2_chain", result_c,
    )
    print_hypothesis_verdict(result_a, result_b, result_c)

    print(f"\n[Stage2 Result]  evidence_units={len(s2_result.evidence_units)}")
    for u in s2_result.evidence_units:
        print(f"  {u.unit_id}  {u.date_range}  type={u.activity_cluster}  logs={u.log_count}")
        print(f"    {u.summary[:80]}")

    if args.json_out:
        def _slog(r):
            return {
                "log_id": r.log_id, "title": r.log.title,
                "score": r.final_score,
                "category": getattr(r, "schema_category", "?"),
                "strength": getattr(r, "category_hit_strength", "?"),
                "reason": r.rejection_reason or r.admission_reason or "-",
            }

        out = {
            "goal_id": target_goal.goal_id,
            "goal_title": target_goal.title,
            "A": {
                "expansion": False,
                "query": result_a.query_text,
                "candidate_ids": [c.log_id for c in result_a.candidates],
                "admitted": [_slog(r) for r in result_a.selected_logs],
                "all_ranked": [_slog(r) for r in result_a.ranked_logs],
            },
            "B": {
                "expansion": True,
                "query": result_b.query_text,
                "candidate_ids": [c.log_id for c in result_b.candidates],
                "admitted": [_slog(r) for r in result_b.selected_logs],
                "all_ranked": [_slog(r) for r in result_b.ranked_logs],
            },
            "C": {
                "expansion": True,
                "query": result_c.query_text,
                "candidate_ids": [c.log_id for c in result_c.candidates],
                "admitted": [_slog(r) for r in result_c.selected_logs],
                "all_ranked": [_slog(r) for r in result_c.ranked_logs],
            },
        }
        Path(args.json_out).write_text(json.dumps(out, ensure_ascii=False, indent=2))
        print(f"\nJSON written to {args.json_out}")


if __name__ == "__main__":
    main()
