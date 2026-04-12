"""Experiment result persistence utilities.

save_stage1_result / save_stage2_result  — write per-run JSON
aggregate_to_csv                          — collapse JSON files to CSV summary
"""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path("results")


def save_stage1_result(
    goal_id: str,
    baseline: str,
    metrics: dict,
    selected_log_ids: list[str],
    selected_titles: list[str],
    extra: dict | None = None,
) -> Path:
    """Write Stage 1 result to results/stage1/{goal_id}_{baseline}.json."""
    out_dir = RESULTS_DIR / "stage1"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{goal_id}_{baseline}.json"

    payload: dict = {
        "goal_id": goal_id,
        "baseline": baseline,
        "stage": "stage1",
        "timestamp": datetime.utcnow().isoformat(),
        "metrics": metrics,
        "selected_log_ids": selected_log_ids,
        "selected_titles": selected_titles,
    }
    if extra:
        payload.update(extra)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_path


def save_stage2_result(
    goal_id: str,
    baseline: str,
    metrics: dict,
    evidence_unit_count: int,
    extra: dict | None = None,
) -> Path:
    """Write Stage 2 result to results/stage2/{goal_id}_{baseline}.json."""
    out_dir = RESULTS_DIR / "stage2"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{goal_id}_{baseline}.json"

    payload: dict = {
        "goal_id": goal_id,
        "baseline": baseline,
        "stage": "stage2",
        "timestamp": datetime.utcnow().isoformat(),
        "metrics": {**metrics, "evidence_unit_count": evidence_unit_count},
    }
    if extra:
        payload.update(extra)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_path


def aggregate_to_csv(stage: str = "stage1") -> Path:
    """Read all JSON files under results/{stage}/ and write a CSV summary.

    Rows with goal_id starting with 'G-TEST' are excluded from the output
    so that verification dummy runs don't pollute the aggregation.
    """
    in_dir = RESULTS_DIR / stage
    out_path = RESULTS_DIR / f"{stage}_summary.csv"

    rows: list[dict] = []
    for json_file in sorted(in_dir.glob("*.json")):
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("goal_id", "").startswith("G-TEST"):
            continue
        row: dict = {
            "goal_id": data.get("goal_id"),
            "baseline": data.get("baseline"),
        }
        row.update(data.get("metrics", {}))
        rows.append(row)

    if not rows:
        print(f"No results found in {in_dir}")
        return out_path

    # Union of all keys — missing values become empty strings
    all_keys: list[str] = ["goal_id", "baseline"]
    for row in rows:
        for k in row:
            if k not in all_keys:
                all_keys.append(k)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in all_keys})

    return out_path
