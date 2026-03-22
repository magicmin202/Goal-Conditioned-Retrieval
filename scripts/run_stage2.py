#!/usr/bin/env python3
"""Stage 2 RAG experiment runner.

Architecture:
  Stage 1 (retrieval + reranking + admission)  →  Stage 2 (consolidation only)

Stage 2 does NOT do independent global retrieval.
It receives Stage 1 admitted anchors and consolidates them.

Usage:
    .venv/bin/python scripts/run_stage2.py --auto
    .venv/bin/python scripts/run_stage2.py --goal_id G-U0001-01 --top_k 5
    .venv/bin/python scripts/run_stage2.py --goal_id G-U0001-01 --real_embeddings
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
setup_logging()

import logging
from app.config import DEFAULT_CONFIG
from app.data_generation.dataset_builder import build_dataset
from app.data_generation.export_utils import load_dataset_from_json
from app.evaluation.rag_metrics import compute_rag_metrics
from app.pipeline.stage1_ranking_pipeline import Stage1Pipeline
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
    parser = argparse.ArgumentParser(description="Run Stage 1 → Stage 2 Pipeline")
    parser.add_argument("--user_id", default=None)
    parser.add_argument("--goal_id", default=None)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("--data_dir", default=_DEFAULT_DATA_DIR)
    parser.add_argument("--mock", action="store_true", help="Use mock LLM instead of Gemini")
    parser.add_argument("--real_embeddings", action="store_true",
                        help="Use Gemini Embedding API (requires GEMINI_API_KEY)")
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
    logger.info(
        "User: %s | Goal: %s (%s) | Logs: %d",
        user_id, target_goal.goal_id, target_goal.title, len(user_logs),
    )

    candidate_size = max(args.top_k * 3, min(len(user_logs) * 6 // 10, 30))

    # ── Stage 1: Retrieval + Reranking + Admission ────────────────────────────
    s1_cfg = DEFAULT_CONFIG.stage1
    s1_cfg.retrieval.top_k = args.top_k
    s1_cfg.retrieval.candidate_size = candidate_size

    s1_pipeline = Stage1Pipeline(config=s1_cfg, use_real_embeddings=args.real_embeddings)
    s1_pipeline.index(user_logs)

    # Always use expansion for Stage1→Stage2 chain (vocabulary needed for neighbor admission)
    s1_result = s1_pipeline.run(target_goal, use_expansion=True)

    logger.info(
        "Stage1 complete  admitted_anchors=%d  goal=%s",
        len(s1_result.selected_logs), target_goal.goal_id,
    )

    # ── Stage 2: Anchor-centered Consolidation ────────────────────────────────
    # Stage 2 does NOT retrieve from corpus again.
    # It receives Stage 1 admitted anchors as fixed input.
    s2_cfg = DEFAULT_CONFIG.stage2

    s2_pipeline = Stage2Pipeline(
        config=s2_cfg,
        use_mock_llm=args.mock,
        use_real_embeddings=args.real_embeddings,
    )
    s2_pipeline.index(user_logs)   # stores corpus for temporal expansion only

    start = time.time()
    result = s2_pipeline.run_with_stage1(s1_result)
    elapsed = time.time() - start

    # ── Output ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Stage 1→2  |  Goal: {result.goal.title}")
    print(f"Mode      : {result.metadata.get('adaptive_mode', '-')}")
    print(f"Query     : {result.query_text}")
    print(f"Priority  : {result.priority_terms}")
    print(f"Expanded  : {result.expanded_terms}")
    print(f"Negative  : {result.negative_terms}")
    print(f"Corpus    : {len(user_logs)} logs  →  Stage1 anchors: {len(s1_result.selected_logs)}")
    print("=" * 60)

    print(f"\n[Stage1 Admitted Anchors ({len(s1_result.selected_logs)})]")
    for r in s1_result.selected_logs:
        strength = r.log.metadata.get("evidence_strength", "-")
        reason = r.admission_reason or "-"
        print(
            f"  [{r.rank:2d}] score={r.final_score:.4f}  {r.log.date}"
            f"  [{strength}]  {r.log.title}"
        )
        print(f"       admission_reason: {reason}")

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
