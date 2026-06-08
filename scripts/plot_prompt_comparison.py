#!/usr/bin/env python3
"""프롬프트 개선 전/후 지표 비교 bar 차트.

Usage:
    LD_PRELOAD=... .venv/bin/python scripts/plot_prompt_comparison.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "figure.dpi":        150,
})

LABEL_BEFORE = "Previous Prompt"
LABEL_AFTER  = "Improved Prompt"
COLOR_BEFORE = "#4C72B0"
COLOR_AFTER  = "#DD8452"

RATIO_METRICS = [
    ("Recall",    0.247, 0.257),
    ("Precision", 0.783, 0.800),
    ("F1",        0.372, 0.387),
]
COUNT_METRICS = [
    ("Avg Retrieved", 3.2, 3.3),
]
TOKEN_METRICS = [
    ("Input Tokens", 2711, 3038),
]


def _grouped_bar(ax, items, fmt="{:.3f}", ylabel=""):
    names   = [n for n, _, _ in items]
    befores = [b for _, b, _ in items]
    afters  = [a for _, _, a in items]

    x = np.arange(len(names))
    w = 0.32
    b1 = ax.bar(x - w/2, befores, w, label=LABEL_BEFORE, color=COLOR_BEFORE, alpha=0.85)
    b2 = ax.bar(x + w/2, afters,  w, label=LABEL_AFTER,  color=COLOR_AFTER,  alpha=0.85)

    for bars, vals in ((b1, befores), (b2, afters)):
        for rect, v in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width()/2, rect.get_height(),
                    fmt.format(v), ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    ax.legend(fontsize=9)
    return befores, afters


def main(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # 패널 1: Recall / Precision / F1 (0~1 비율)
    ax = axes[0]
    _grouped_bar(ax, RATIO_METRICS, fmt="{:.3f}")
    ax.set_ylim(0, 1.0)
    ax.set_title("Retrieval Quality (Recall / Precision / F1)", fontsize=11)

    # 패널 2: Avg Retrieved (개수)
    ax = axes[1]
    _grouped_bar(ax, COUNT_METRICS, fmt="{:.1f}")
    ax.set_ylim(0, 5)
    ax.set_title("Avg # Retrieved Logs", fontsize=11)

    # 패널 3: Input Tokens (토큰 수)
    ax = axes[2]
    _grouped_bar(ax, TOKEN_METRICS, fmt="{:,.0f}")
    ax.set_ylim(0, 3500)
    ax.set_title("Avg Input Tokens", fontsize=11)

    fig.suptitle("Prompt Comparison: Before vs. After Improvement", fontsize=13, y=1.04)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"저장: {out_path}")


if __name__ == "__main__":
    main(ROOT / "results" / "plots" / "prompt_comparison.png")
