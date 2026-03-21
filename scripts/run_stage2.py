#!/usr/bin/env python3
"""Stage 2 RAG experiment runner.

Usage:
    python scripts/run_stage2.py --auto
    python scripts/run_stage2.py --user_id U0001 --goal_id G-U0001-01
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.utils.logging_utils import setup_logging
setup_logging()

import logging
from app.config import DEFAULT_CONFIG
from app.data_generation.dataset_builder import build_dataset
from app.data_generation.export_utils import load_dataset_from_json
from app.evaluation.rag_metrics import compute_rag_metrics
from app.pipeline.stage2_rag_pipeline import Stage2Pipeline
from app.schemas import GoalLogLabel, ResearchGoal, ResearchLog

logger = logging.getLogger(__name__)

_DEFAULT_DATA_DIR = "data/synthetic"


def load_data(data_dir: str) -> tuple[list[ResearchGoal], list[ResearchLog], list[GoalLogLabel]]:
    goals_path = Path(data_dir) / "goals.json"
    if goals_path.exists():
        logger.info("Loading dataset from %s", data_dir)
        _, goals, logs, labels = load_dataset_from_json(data_dir)
    else:
        logger.info("No dataset at %s. Generating small dataset...", data_dir)
        ds = build_dataset(small_mode=True, seed=42)
        goals, logs, labels = ds.goals, ds.logs, ds.labels
    return goals, logs, labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 2 RAG Experiment")
    parser.add_argument("--user_id", default=None)
    parser.add_argument("--goal_id", default=None)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("--data_dir", default=_DEFAULT_DATA_DIR)
    parser.add_argument("--mock", action="store_true", help="Use mock LLM instead of Gemini")
    args = parser.parse_args()

    goals, logs, labels = load_data(args.data_dir)

    if args.auto or (args.user_id is None and args.goal_id is None):
        target_goal = goals[0]
    elif args.goal_id:
        target_goal = next((g for g in goals if g.goal_id == args.goal_id), None)
        if target_goal is None:
            logger.error("goal_id=%s not found.", args.goal_id)
            sys.exit(1)
    else:
        target_goal = next((g for g in goals if g.user_id == args.user_id), None)
        if target_goal is None:
            logger.error("No goals for user_id=%s.", args.user_id)
            sys.exit(1)

    user_id = target_goal.user_id
    user_logs = [l for l in logs if l.user_id == user_id]

    logger.info("User: %s | Goal: %s (%s) | Logs: %d", user_id, target_goal.goal_id, target_goal.title, len(user_logs))

    cfg = DEFAULT_CONFIG.stage2
    cfg.retrieval.top_k = args.top_k
    # candidate_size: top-N pruning — use ~60% of corpus, minimum top_k * 3
    cfg.retrieval.candidate_size = max(args.top_k * 3, min(len(user_logs) * 6 // 10, 30))

    pipeline = Stage2Pipeline(config=cfg, use_mock_llm=args.mock)
    pipeline.index(user_logs)

    start = time.time()
    result = pipeline.run(target_goal)
    elapsed = time.time() - start

    print("\n" + "=" * 60)
    print(f"Stage 2  |  Goal: {result.goal.title}")
    print(f"Mode    : {result.metadata.get('adaptive_mode', '-')}")
    print(f"Query   : {result.query_text}")
    print(f"Priority: {result.priority_terms}")
    print(f"Expanded: {result.expanded_terms}")
    print(f"Negative: {result.negative_terms}")
    print(f"Corpus  : {len(user_logs)} logs | Candidates: {result.metadata.get('candidate_size')} → Filter: {result.metadata.get('after_filter')}")
    print("=" * 60)

    print(f"\n[Selected Logs ({len(result.selected_logs)})]")
    for r in result.selected_logs:
        strength = r.log.metadata.get("evidence_strength", "-")
        print(f"  [{r.rank:2d}] score={r.final_score:.4f}  {r.log.date}  [{strength}]  {r.log.title}")

    print(f"\n[Evidence Units ({len(result.evidence_units)})]")
    for u in result.evidence_units:
        print(f"  {u.unit_id}  {u.date_range}  type={u.activity_cluster}  logs={u.log_count}")
        print(f"    {u.summary}")
        if u.temporal_progression:
            print(f"    진행: {u.temporal_progression}")

    print("\n[LLM Analysis]")
    print(result.llm_analysis)

    metrics = compute_rag_metrics(target_goal, user_logs, result.evidence_units)
    print(f"[RAG Metrics]  elapsed={elapsed:.3f}s")
    for name, val in metrics.items():
        print(f"  {name}: {val:.4f}")


if __name__ == "__main__":
    main()
