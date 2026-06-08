"""goal별 threshold sweep → precision/recall/F1 평가.

평가 방식:
  - 각 goal에 대해 자기 user의 logs와의 유사도 점수 사용
  - threshold sweep → 각 goal별 best threshold 탐색
  - 전체 goal macro-avg P/R/F1 + 글로벌 best threshold 보고
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from app.pipeline.similarity import GoalSim, SCALING_METHODS

logger = logging.getLogger(__name__)

_DATA = Path(__file__).resolve().parent.parent.parent / "data" / "synthetic_v2"


def _load_labels() -> dict[tuple[str, str], int]:
    """(goal_id, log_id) → 1(relevant) / 0(irrelevant)."""
    labels = json.loads((_DATA / "labels.json").read_text(encoding="utf-8"))
    return {
        (lbl["goal_id"], lbl["log_id"]): (1 if lbl["label"] == "relevant" else 0)
        for lbl in labels
    }


def _pr_curve(
    scores: np.ndarray,
    labels: np.ndarray,
    n_thresholds: int = 100,
) -> list[dict]:
    lo = float(np.percentile(scores, 1))
    hi = float(np.percentile(scores, 99))
    thresholds = np.linspace(lo, hi, n_thresholds)

    rows = []
    for t in thresholds:
        pred = scores >= t
        tp = int((pred & (labels == 1)).sum())
        fp = int((pred & (labels == 0)).sum())
        fn = int((~pred & (labels == 1)).sum())
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        rows.append({"threshold": round(float(t), 6),
                     "precision": round(p, 4), "recall": round(r, 4),
                     "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn})
    return rows


def evaluate(
    goal_sims: list[GoalSim],
    n_thresholds: int = 100,
) -> dict:
    """goal별 threshold sweep 후 macro-avg 집계."""
    label_map = _load_labels()

    # method → 각 goal의 best P/R/F1 목록
    per_method: dict[str, list[dict]] = {m: [] for m in SCALING_METHODS}

    for gs in goal_sims:
        y = np.array([label_map.get((gs.goal_id, lid), 0) for lid in gs.log_ids])
        n_pos = int(y.sum())
        if n_pos == 0:
            continue  # relevant 없는 goal은 집계 제외

        for method in SCALING_METHODS:
            curve = _pr_curve(gs.scores[method], y, n_thresholds)
            best  = max(curve, key=lambda x: x["f1"])
            per_method[method].append({
                "goal_id":   gs.goal_id,
                "user_id":   gs.user_id,
                "n_logs":    len(gs.log_ids),
                "n_relevant": n_pos,
                **best,
            })

    # macro-avg 집계
    summary: dict[str, dict] = {}
    for method, rows in per_method.items():
        if not rows:
            continue
        avg_p  = float(np.mean([r["precision"] for r in rows]))
        avg_r  = float(np.mean([r["recall"]    for r in rows]))
        avg_f1 = float(np.mean([r["f1"]        for r in rows]))
        summary[method] = {
            "macro_precision": round(avg_p,  4),
            "macro_recall":    round(avg_r,  4),
            "macro_f1":        round(avg_f1, 4),
            "n_goals_evaluated": len(rows),
            "per_goal": rows,
        }
        logger.info(
            "%-14s  macro P=%.3f  R=%.3f  F1=%.3f  (goals=%d)",
            method, avg_p, avg_r, avg_f1, len(rows),
        )

    return summary


def print_summary(summary: dict) -> None:
    n = next(iter(summary.values()))["n_goals_evaluated"]
    print(f"\n평가 goals: {n}개  (macro-avg across goals)\n")
    print(f"{'method':<16} {'precision':>10} {'recall':>10} {'F1':>8}")
    print("─" * 50)
    for method, s in summary.items():
        print(
            f"{method:<16} {s['macro_precision']:>10.4f} "
            f"{s['macro_recall']:>10.4f} {s['macro_f1']:>8.4f}"
        )
