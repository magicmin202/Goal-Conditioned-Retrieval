#!/usr/bin/env python3
"""Stage 1 → Stage 2 experiment runner.

Architecture:
  Stage 1 (retrieval + reranking + admission)  →  Stage 2 (consolidation only)

Stage 2 does NOT do independent global retrieval.
It receives Stage 1 admitted anchors and consolidates them.

Usage:
    .venv/bin/python scripts/run_stage2.py --auto --baseline ours
    .venv/bin/python scripts/run_stage2.py --goal_id G-U0001-01 --baseline raw_llm
    .venv/bin/python scripts/run_stage2.py --goal_id G-U0001-01 --baseline ours --save_result

Stage2 baselines:
  ours               Full pipeline (default)
  ours_wo_compression  Stage1 ours → skip compression → LLM
  ours_wo_lexical_gate Stage1 wo lexical gate → Stage2 full
  raw_llm            No retrieval, raw logs → LLM directly
  simple_summary     No retrieval, LLM summarization of all logs
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
from app.evaluation.rag_metrics import compute_rag_metrics, coverage_at_k
from app.pipeline.stage1_ranking_pipeline import Stage1Pipeline, Stage1Result
from app.pipeline.stage2_rag_pipeline import Stage2Pipeline, Stage2Result
from app.retrieval.candidate_retrieval import RetrievalMode
from app.schemas import GoalLogLabel, ResearchGoal, ResearchLog, CompressedEvidenceUnit

logger = logging.getLogger(__name__)

_DEFAULT_DATA_DIR = "data/synthetic"


def load_data(data_dir: str):
    goals_path = Path(data_dir) / "goals.json"
    if goals_path.exists():
        logger.info("Loading dataset from %s", data_dir)
        _, goals, logs, labels = load_dataset_from_json(data_dir)
    else:
        logger.info("No dataset at %s. Generating small dataset...", data_dir)
        ds = build_dataset(small_mode=True, seed=42)
        goals, logs, labels = ds.goals, ds.logs, ds.labels
    return goals, logs, labels


def _run_stage1(
    target_goal: ResearchGoal,
    user_logs: list[ResearchLog],
    args,
    disable_lexical_gate: bool = False,
) -> Stage1Result:
    """Run Stage 1 and return the result."""
    s1_cfg = DEFAULT_CONFIG.stage1
    s1_cfg.retrieval.top_k = args.top_k
    s1_cfg.retrieval.candidate_size = max(
        args.top_k * 3, min(len(user_logs) * 6 // 10, 30)
    )
    s1_pipeline = Stage1Pipeline(
        config=s1_cfg,
        use_real_embeddings=args.real_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        disable_lexical_gate=disable_lexical_gate,
    )
    s1_pipeline.index(user_logs)
    return s1_pipeline.run(target_goal, use_expansion=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 1 → Stage 2 Pipeline")
    parser.add_argument("--user_id", default=None)
    parser.add_argument("--goal_id", default=None)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("--data_dir", default=_DEFAULT_DATA_DIR)
    parser.add_argument("--mock", action="store_true", help="Use mock LLM instead of Gemini")
    parser.add_argument(
        "--real_embeddings", action="store_true",
        help="Use Gemini Embedding API (requires GEMINI_API_KEY)",
    )
    parser.add_argument(
        "--baseline",
        choices=["ours", "ours_wo_compression", "ours_wo_lexical_gate",
                 "raw_llm", "simple_summary"],
        default="ours",
        help="Stage 2 baseline mode (default: ours)",
    )
    parser.add_argument(
        "--save_result", action="store_true",
        help="Save result to results/stage2/{goal_id}_{baseline}.json",
    )
    args = parser.parse_args()

    goals, logs, labels = load_data(args.data_dir)

    # ── Resolve target goal ────────────────────────────────────────────────────
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
    user_labels = [
        lb for lb in labels
        if lb.user_id == user_id and lb.goal_id == target_goal.goal_id
    ]
    labels_dict = {lb.log_id: lb.relevance_score for lb in user_labels}

    logger.info(
        "User: %s | Goal: %s (%s) | Logs: %d | Baseline: %s",
        user_id, target_goal.goal_id, target_goal.title, len(user_logs), args.baseline,
    )

    # ── Stage 2 pipeline (shared across baselines that use it) ────────────────
    s2_cfg = DEFAULT_CONFIG.stage2
    s2_pipeline = Stage2Pipeline(
        config=s2_cfg,
        use_mock_llm=args.mock,
        use_real_embeddings=args.real_embeddings,
    )
    s2_pipeline.index(user_logs)

    baseline = args.baseline
    start = time.time()

    # ── Baseline dispatch ──────────────────────────────────────────────────────
    result: Stage2Result | None = None
    llm_analysis: str = ""
    evidence_units: list[CompressedEvidenceUnit] = []

    if baseline == "raw_llm":
        # No retrieval, no compression — all raw logs → LLM
        llm_analysis = s2_pipeline._run_raw_llm(target_goal, user_logs)
        evidence_units = []
        evidence_unit_count = 0

    elif baseline == "simple_summary":
        # No retrieval — LLM summarizes all logs
        llm_analysis = s2_pipeline._run_simple_summary(target_goal, user_logs)
        evidence_units = []
        evidence_unit_count = 0

    elif baseline == "ours_wo_compression":
        # Stage1 ours → compression skipped → LLM
        s1_result = _run_stage1(target_goal, user_logs, args, disable_lexical_gate=False)
        logger.info(
            "Stage1 complete  admitted_anchors=%d  goal=%s",
            len(s1_result.selected_logs), target_goal.goal_id,
        )
        result = s2_pipeline.run_with_stage1(s1_result, skip_compression=True)
        llm_analysis = result.llm_analysis
        evidence_units = result.evidence_units
        evidence_unit_count = len(evidence_units)

    elif baseline == "ours_wo_lexical_gate":
        # Stage1 without lexical gate → Stage2 full
        s1_result = _run_stage1(target_goal, user_logs, args, disable_lexical_gate=True)
        logger.info(
            "Stage1 (no lexical gate)  admitted_anchors=%d  goal=%s",
            len(s1_result.selected_logs), target_goal.goal_id,
        )
        result = s2_pipeline.run_with_stage1(s1_result, skip_compression=False)
        llm_analysis = result.llm_analysis
        evidence_units = result.evidence_units
        evidence_unit_count = len(evidence_units)

    else:  # ours
        s1_result = _run_stage1(target_goal, user_logs, args, disable_lexical_gate=False)
        logger.info(
            "Stage1 complete  admitted_anchors=%d  goal=%s",
            len(s1_result.selected_logs), target_goal.goal_id,
        )
        result = s2_pipeline.run_with_stage1(s1_result, skip_compression=False)
        llm_analysis = result.llm_analysis
        evidence_units = result.evidence_units
        evidence_unit_count = len(evidence_units)

    elapsed = time.time() - start

    # ── Output ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Stage 1→2  |  Goal: {target_goal.title}")
    print(f"Baseline  : {baseline}")
    print(f"Mode      : {result.metadata.get('adaptive_mode', '-') if result else 'N/A'}")
    if result:
        print(f"Query     : {result.query_text}")
        print(f"Priority  : {result.priority_terms}")
        print(f"Expanded  : {result.expanded_terms}")
        print(f"Negative  : {result.negative_terms}")
        anchors = result.anchors if hasattr(result, "anchors") else result.selected_logs
        print(f"Corpus    : {len(user_logs)} logs  →  Stage1 anchors: {len(anchors)}")
    print("=" * 60)

    if result and hasattr(result, "anchors"):
        print(f"\n[Stage1 Admitted Anchors ({len(result.anchors)})]")
        for r in result.anchors:
            strength = r.log.metadata.get("evidence_strength", "-")
            reason = r.admission_reason or "-"
            print(
                f"  [{r.rank:2d}] score={r.final_score:.4f}  {r.log.date}"
                f"  [{strength}]  {r.log.title}"
            )
            print(f"       admission_reason: {reason}")

    print(f"\n[Evidence Units ({evidence_unit_count})]")
    for u in evidence_units:
        print(f"  {u.unit_id}  {u.date_range}  type={u.activity_cluster}  logs={u.log_count}")
        print(f"    {u.summary}")
        if u.temporal_progression:
            print(f"    진행: {u.temporal_progression}")

    print("\n[LLM Analysis]")
    print(llm_analysis)

    # ── Metrics ────────────────────────────────────────────────────────────────
    rag_metrics = compute_rag_metrics(
        target_goal, user_logs, evidence_units,
        labels=labels_dict, k=None,
    )
    cov = rag_metrics.get("coverage@k", 0.0)
    token_red = rag_metrics.get("token_reduction_rate", 0.0)
    red_red = rag_metrics.get("redundancy_reduction", 0.0)
    goal_align = rag_metrics.get("goal_alignment_score", 0.0)
    action = rag_metrics.get("actionability_score", 0.0)

    print(f"\n[RAG Metrics]  elapsed={elapsed:.3f}s  baseline={baseline}")
    print(f"  {'coverage@k':<28} {cov:.4f}")
    print(f"  {'token_reduction_rate':<28} {token_red:.4f}")
    print(f"  {'redundancy_reduction':<28} {red_red:.4f}")
    print(f"  {'evidence_unit_count':<28} {evidence_unit_count}")
    print(f"  {'goal_alignment_score':<28} {goal_align:.4f}")
    print(f"  {'actionability_score':<28} {action:.4f}")
    print(f"  {'llm_judge_score':<28} N/A  (placeholder)")

    if args.save_result:
        from app.evaluation.result_writer import save_stage2_result
        path = save_stage2_result(
            goal_id=target_goal.goal_id,
            baseline=baseline,
            metrics={
                "coverage@k": cov,
                "token_reduction_rate": token_red,
                "redundancy_reduction": red_red,
                "goal_alignment_score": goal_align,
                "actionability_score": action,
                "llm_judge_score": 0.0,
            },
            evidence_unit_count=evidence_unit_count,
            extra={"elapsed_sec": round(elapsed, 3)},
        )
        print(f"\n[Result saved] {path}")


if __name__ == "__main__":
    main()
