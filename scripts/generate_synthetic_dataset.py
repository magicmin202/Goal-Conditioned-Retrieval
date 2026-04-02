#!/usr/bin/env python3
"""Generate and export synthetic dataset.

Usage:
    # Small mode (3 users, quick test)
    python scripts/generate_synthetic_dataset.py --small

    # Full mode (100 users)
    python scripts/generate_synthetic_dataset.py --num_users 100

    # Upload to Firestore after export
    python scripts/generate_synthetic_dataset.py --small --upload
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.utils.logging_utils import setup_logging
setup_logging()

import logging
from app.data_generation.dataset_builder import build_dataset
from app.data_generation.export_utils import export_dataset_to_json, upload_dataset_to_firestore

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic research dataset")
    parser.add_argument("--small",  action="store_true", help="Small  mode: 3 users, 25-40 logs, 1 month")
    parser.add_argument("--medium", action="store_true", help="Medium mode: 10 users, 50-70 logs, 2 months")
    parser.add_argument("--num_users", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="data/synthetic")
    parser.add_argument("--upload", action="store_true", help="Upload to Firestore after export")
    args = parser.parse_args()

    if args.medium:
        num_users = 10
        start_date = date(2026, 2, 1)
        end_date   = date(2026, 3, 31)
        small_mode = False
    elif args.small:
        num_users  = 3
        start_date = date(2026, 3, 1)
        end_date   = date(2026, 3, 31)
        small_mode = True
    else:
        num_users  = args.num_users
        start_date = date(2026, 3, 1)
        end_date   = date(2026, 3, 31)
        small_mode = False

    logger.info("Building dataset (mode=%s, num_users=%d, seed=%d)",
                "medium" if args.medium else ("small" if args.small else "full"),
                num_users, args.seed)

    dataset = build_dataset(
        num_users=num_users,
        start_date=start_date,
        end_date=end_date,
        seed=args.seed,
        small_mode=small_mode,
    )

    stats = dataset.stats()
    print("\n[Dataset Stats]")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    export_dataset_to_json(
        users=dataset.users,
        goals=dataset.goals,
        logs=dataset.logs,
        labels=dataset.labels,
        output_dir=args.output_dir,
    )
    print(f"\nExported to: {args.output_dir}/")

    if args.upload:
        logger.info("Uploading to Firestore...")
        upload_dataset_to_firestore(dataset.users, dataset.goals, dataset.logs, dataset.labels)
        print("Uploaded to Firestore.")


if __name__ == "__main__":
    main()
