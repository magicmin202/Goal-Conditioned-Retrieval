#!/usr/bin/env python3
"""Aggregate baseline result JSON files to a CSV summary.

Usage:
    .venv/bin/python scripts/aggregate_results.py --stage stage1
    .venv/bin/python scripts/aggregate_results.py --stage stage2
    .venv/bin/python scripts/aggregate_results.py --stage all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.evaluation.result_writer import aggregate_to_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate baseline results to CSV"
    )
    parser.add_argument(
        "--stage",
        choices=["stage1", "stage2", "all"],
        default="all",
        help="Stage to aggregate (default: all)",
    )
    args = parser.parse_args()

    if args.stage in ("stage1", "all"):
        path = aggregate_to_csv("stage1")
        print(f"Stage1 summary → {path}")

    if args.stage in ("stage2", "all"):
        path = aggregate_to_csv("stage2")
        print(f"Stage2 summary → {path}")


if __name__ == "__main__":
    main()
