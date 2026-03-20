#!/usr/bin/env python3
"""Stage 2 RAG experiment runner.

Usage:
    python scripts/run_stage2.py
    python scripts/run_stage2.py --top_k 7
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
from app.evaluation.rag_metrics import compute_rag_metrics
from app.pipeline.stage2_rag_pipeline import Stage2Pipeline
from scripts.run_stage1 import get_mock_data

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 2 RAG Experiment")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    goal, logs, _ = get_mock_data()
    cfg = DEFAULT_CONFIG.stage2
    cfg.retrieval.top_k = args.top_k
    cfg.retrieval.candidate_size = len(logs)

    pipeline = Stage2Pipeline(config=cfg, use_mock_llm=True)
    pipeline.index(logs)

    start = time.time()
    result = pipeline.run(goal)
    elapsed = time.time() - start

    print("\n" + "=" * 60)
    print(f"Stage 2  |  Goal: {result.goal.title}")
    print(f"Query   : {result.query_text}")
    print(f"Expanded: {result.metadata.get('expanded_terms', [])}")
    print("=" * 60)

    print(f"\n[Selected Logs ({len(result.selected_logs)})]")
    for r in result.selected_logs:
        print(f"  [{r.rank:2d}] score={r.final_score:.4f}  {r.log.date}  {r.log.title}")

    print(f"\n[Evidence Units ({len(result.evidence_units)})]")
    for u in result.evidence_units:
        print(f"  {u.unit_id}  {u.date_range}  logs={u.log_count}")
        print(f"    {u.summary}")
        if u.temporal_progression:
            print(f"    진행: {u.temporal_progression}")

    print("\n[LLM Analysis]")
    print(result.llm_analysis)

    metrics = compute_rag_metrics(goal, logs, result.evidence_units)
    print(f"[RAG Metrics]  elapsed={elapsed:.3f}s")
    for name, val in metrics.items():
        print(f"  {name}: {val:.4f}")


if __name__ == "__main__":
    main()
