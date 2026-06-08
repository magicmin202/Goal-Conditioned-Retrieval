#!/usr/bin/env python3
"""평가 결과 시각화.

Usage:
    LD_PRELOAD=... .venv/bin/python scripts/plot_eval.py
    LD_PRELOAD=... .venv/bin/python scripts/plot_eval.py --out results/plots
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── 스타일 ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "figure.dpi":        150,
})

METHOD_LABELS = {
    "max_sim":      "Max(sim_L,M,S)",
    "avg_sim":      "Avg(sim_L,M,S)",
    "max_scale":    "Max-Scale",
    "minmax_scale": "MinMax-Scale",
    "zscore_scale": "Z-Score",
    "raw_long":     "Raw Long-term",
}
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]
METHOD_ORDER = list(METHOD_LABELS.keys())


def load_full(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ── 플롯 1: Macro P/R/F1 bar chart ────────────────────────────────────────────
def plot_bar_summary(summary: dict, ax: plt.Axes) -> None:
    methods = METHOD_ORDER
    labels  = [METHOD_LABELS[m] for m in methods]
    ps  = [summary[m]["macro_precision"] for m in methods]
    rs  = [summary[m]["macro_recall"]    for m in methods]
    f1s = [summary[m]["macro_f1"]        for m in methods]

    x   = np.arange(len(methods))
    w   = 0.25
    ax.bar(x - w, ps,  w, label="Precision", color="#4C72B0", alpha=0.85)
    ax.bar(x,     rs,  w, label="Recall",    color="#55A868", alpha=0.85)
    ax.bar(x + w, f1s, w, label="F1",        color="#DD8452", alpha=0.85)

    for i, (p, r, f) in enumerate(zip(ps, rs, f1s)):
        ax.text(i - w, p + 0.005, f"{p:.3f}", ha="center", va="bottom", fontsize=7)
        ax.text(i,     r + 0.005, f"{r:.3f}", ha="center", va="bottom", fontsize=7)
        ax.text(i + w, f + 0.005, f"{f:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title("Macro-avg P / R / F1  (per-goal best threshold)", fontsize=11)
    ax.legend(fontsize=9)


# ── 플롯 2: PR curve (threshold sweep, avg across goals) ─────────────────────
def plot_pr_curve(summary: dict, ax: plt.Axes) -> None:
    for color, method in zip(COLORS, METHOD_ORDER):
        per_goal = summary[method]["per_goal"]
        # 각 goal의 best P/R 점들을 scatter
        ps = [g["precision"] for g in per_goal]
        rs = [g["recall"]    for g in per_goal]
        ax.scatter(rs, ps, alpha=0.15, s=10, color=color)

    # macro 점 (굵게)
    for color, method in zip(COLORS, METHOD_ORDER):
        s = summary[method]
        ax.scatter(
            s["macro_recall"], s["macro_precision"],
            s=120, color=color, zorder=5,
            label=f"{METHOD_LABELS[method]}  F1={s['macro_f1']:.3f}",
            edgecolors="white", linewidths=0.8,
        )

    ax.set_xlabel("Recall", fontsize=10)
    ax.set_ylabel("Precision", fontsize=10)
    ax.set_title("Per-Goal Best (P, R)  dot=single goal  filled=macro avg", fontsize=11)
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=7.5, loc="lower left")


# ── 플롯 3: per-goal F1 분포 violin ───────────────────────────────────────────
def plot_f1_violin(summary: dict, ax: plt.Axes) -> None:
    data   = [summary[m]["per_goal"] for m in METHOD_ORDER]
    f1_all = [[g["f1"] for g in d] for d in data]
    labels = [METHOD_LABELS[m] for m in METHOD_ORDER]

    parts = ax.violinplot(f1_all, showmedians=True, showextrema=True)
    for pc, color in zip(parts["bodies"], COLORS):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(1.5)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("F1 (best threshold per goal)", fontsize=10)
    ax.set_title("Per-Goal F1 Distribution", fontsize=11)
    ax.set_ylim(-0.05, 1.1)


# ── 플롯 4: F1 vs noise_ratio scatter ────────────────────────────────────────
def plot_f1_vs_noise(summary: dict, ax: plt.Axes) -> None:
    import re

    goals_json = ROOT / "data" / "synthetic_v2" / "goals.json"
    goals      = json.loads(goals_json.read_text(encoding="utf-8"))
    noise_map  = {g["goal_id"]: g["noise_ratio"] for g in goals}

    best_method = max(METHOD_ORDER, key=lambda m: summary[m]["macro_f1"])
    per_goal    = summary[best_method]["per_goal"]

    xs = [noise_map.get(g["goal_id"], 0.0) for g in per_goal]
    ys = [g["f1"] for g in per_goal]

    # 노이즈별 평균
    bins = sorted(set(xs))
    means = [np.mean([y for x, y in zip(xs, ys) if x == b]) for b in bins]

    ax.scatter(xs, ys, alpha=0.25, s=15, color="#4C72B0")
    ax.plot(bins, means, "o-", color="#DD8452", lw=2, ms=7, label="Mean F1 per noise level")

    ax.set_xlabel("noise_ratio (goal)", fontsize=10)
    ax.set_ylabel("F1 (best threshold)", fontsize=10)
    ax.set_title(f"F1 vs noise_ratio   [{METHOD_LABELS[best_method]}]", fontsize=11)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=9)


# ── 플롯 5: n_relevant vs best F1 ─────────────────────────────────────────────
def plot_f1_vs_nrel(summary: dict, ax: plt.Axes) -> None:
    best_method = max(METHOD_ORDER, key=lambda m: summary[m]["macro_f1"])
    per_goal    = summary[best_method]["per_goal"]

    xs = [g["n_relevant"] for g in per_goal]
    ys = [g["f1"]         for g in per_goal]

    ax.scatter(xs, ys, alpha=0.3, s=15, color="#55A868")
    # 추세선
    if len(xs) > 2:
        z = np.polyfit(xs, ys, 1)
        p = np.poly1d(z)
        xr = np.linspace(min(xs), max(xs), 100)
        ax.plot(xr, p(xr), "--", color="#C44E52", lw=1.5, label=f"trend")

    ax.set_xlabel("# relevant logs per goal", fontsize=10)
    ax.set_ylabel("F1", fontsize=10)
    ax.set_title(f"F1 vs #Relevant  [{METHOD_LABELS[best_method]}]", fontsize=11)
    ax.legend(fontsize=9)


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    full_path = ROOT / "results" / "eval_full.json"
    if not full_path.exists():
        print(f"ERROR: {full_path} not found. run scripts/run_eval.py first.")
        sys.exit(1)

    summary = load_full(full_path)

    # ── Figure 1: 3-panel overview ─────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)
    plot_bar_summary(summary, fig.add_subplot(gs[0]))
    plot_pr_curve(summary,    fig.add_subplot(gs[1]))
    plot_f1_violin(summary,   fig.add_subplot(gs[2]))
    fig.suptitle("Embedding-based Retrieval Evaluation  (gemini-embedding-2 / per-goal threshold)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    p1 = out_dir / "eval_overview.png"
    fig.savefig(p1, bbox_inches="tight")
    print(f"저장: {p1}")

    # ── Figure 2: noise & n_relevant 분석 ─────────────────────────────────────
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    plot_f1_vs_noise(summary, ax1)
    plot_f1_vs_nrel(summary,  ax2)
    fig2.suptitle("F1 Analysis: impact of noise_ratio & #relevant logs", fontsize=12)
    fig2.tight_layout()
    p2 = out_dir / "eval_analysis.png"
    fig2.savefig(p2, bbox_inches="tight")
    print(f"저장: {p2}")

    plt.close("all")
    print("\n완료.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/plots")
    args = parser.parse_args()
    main(ROOT / args.out)
