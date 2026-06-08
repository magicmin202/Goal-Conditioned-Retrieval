#!/usr/bin/env python3
"""스케일링 메서드별 threshold sweep → precision/recall 평가.

Usage:
    LD_PRELOAD=... .venv/bin/python scripts/run_eval.py
    LD_PRELOAD=... .venv/bin/python scripts/run_eval.py --thresholds 100 --save results/eval.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

from app.pipeline.similarity import compute_similarities
from app.evaluation.threshold_eval import evaluate, print_summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--thresholds", type=int, default=100)
    parser.add_argument("--save", type=str, default="")
    args = parser.parse_args()

    goal_sims = compute_similarities()
    summary   = evaluate(goal_sims, n_thresholds=args.thresholds)
    print_summary(summary)

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        # per_goal은 너무 크므로 summary만 저장
        compact = {
            m: {k: v for k, v in s.items() if k != "per_goal"}
            for m, s in summary.items()
        }
        out.write_text(json.dumps(compact, ensure_ascii=False, indent=2))
        full_out = out.with_stem(out.stem + "_full")
        full_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"\n요약 저장: {out}")
        print(f"전체 저장: {full_out}")
