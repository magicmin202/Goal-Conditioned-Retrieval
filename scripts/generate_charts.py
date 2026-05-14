#!/usr/bin/env python3
"""두 개의 연구 차트 생성.

Chart 1 — Threshold Tuning Line Chart
    Dense threshold vs Precision / Recall / F1

Chart 2 — Full-context vs Proposed Token Usage Bar Chart
    [A] Raw (all logs → LLM) vs [B] Pipeline (expansion + compressed analysis)
    여러 goal에 대한 평균 토큰 비교

Usage:
    .venv/bin/python scripts/generate_charts.py
    .venv/bin/python scripts/generate_charts.py --output_dir results/charts
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── 공통 스타일 ────────────────────────────────────────────────────────────────
PALETTE = {
    "precision": "#2563EB",   # blue
    "recall":    "#16A34A",   # green
    "f1":        "#DC2626",   # red
    "raw":       "#94A3B8",   # slate-400
    "pipeline":  "#2563EB",   # blue
    "expansion": "#60A5FA",   # blue-400  (pipeline breakdown)
    "analysis":  "#1D4ED8",   # blue-700
}
FONT_FAMILY = "DejaVu Sans"
plt.rcParams.update({
    "font.family": FONT_FAMILY,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})


# ─────────────────────────────────────────────────────────────────────────────
# Chart 1: Threshold Tuning Line Chart
# ─────────────────────────────────────────────────────────────────────────────

def load_threshold_data(csv_path: str) -> dict:
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "threshold": float(row["threshold"]),
                "precision": float(row["avg_precision"]),
                "recall":    float(row["avg_recall"]),
                "f1":        float(row["avg_f1"]),
            })
    rows.sort(key=lambda r: r["threshold"])
    return {
        "thresholds":  [r["threshold"] for r in rows],
        "precisions":  [r["precision"] for r in rows],
        "recalls":     [r["recall"]    for r in rows],
        "f1s":         [r["f1"]        for r in rows],
    }


def plot_threshold_chart(data: dict, output_path: str, production_threshold: float = 0.92) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    x = data["thresholds"]

    ax.plot(x, data["precisions"], marker="o", linewidth=2.2, markersize=7,
            color=PALETTE["precision"], label="Precision", zorder=3)
    ax.plot(x, data["recalls"],    marker="s", linewidth=2.2, markersize=7,
            color=PALETTE["recall"],    label="Recall",    zorder=3)
    ax.plot(x, data["f1s"],        marker="^", linewidth=2.2, markersize=7,
            color=PALETTE["f1"],        label="F1",        zorder=3)

    # 선택된 production threshold 마킹
    prod_idx = None
    for i, t in enumerate(x):
        if abs(t - production_threshold) < 1e-4:
            prod_idx = i
            break

    if prod_idx is not None:
        ax.axvline(x=production_threshold, color="#6B7280", linestyle="--",
                   linewidth=1.5, alpha=0.7, zorder=2)
        ax.text(production_threshold + 0.002, ax.get_ylim()[0] + 0.02,
                f"Selected\n(t={production_threshold})",
                fontsize=8.5, color="#374151", va="bottom")
        # F1 최고점 dot 강조
        f1_val = data["f1s"][prod_idx]
        ax.scatter([production_threshold], [f1_val], s=120, color=PALETTE["f1"],
                   zorder=5, edgecolors="white", linewidths=1.5)
        ax.annotate(f"F1={f1_val:.3f}",
                    xy=(production_threshold, f1_val),
                    xytext=(production_threshold + 0.006, f1_val + 0.015),
                    fontsize=8.5, color=PALETTE["f1"],
                    arrowprops=dict(arrowstyle="-", color=PALETTE["f1"], lw=1.2))

    ax.set_xlabel("Dense Similarity Threshold", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Threshold Tuning: Precision / Recall / F1\n"
                 "(393 goals, Gemini Embedding, top-k=10)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_xlim(min(x) - 0.005, max(x) + 0.01)
    ax.set_ylim(0, 1.0)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.legend(fontsize=10, loc="center left")

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Chart 1 saved] {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Chart 2: Full-context vs Proposed Token Usage Bar Chart
# ─────────────────────────────────────────────────────────────────────────────

def compute_token_samples(n_goals: int = 8, seed: int = 42) -> list[dict]:
    """compare_token_usage.py 로직을 재사용해 여러 goal의 토큰 수치를 수집."""
    from dotenv import load_dotenv
    load_dotenv()

    from app.utils.logging_utils import setup_logging
    setup_logging(level="WARNING")

    from app.config import DEFAULT_CONFIG
    from app.data_generation.export_utils import load_dataset_from_json
    from app.data_generation.dataset_builder import build_dataset
    from app.llm.analysis import _PROMPT_TEMPLATE, build_evidence_text
    from app.pipeline.stage1_ranking_pipeline import Stage1Pipeline
    from app.pipeline.stage2_rag_pipeline import Stage2Pipeline
    from app.retrieval.query_expansion import _EXPANSION_PROMPT

    _CHARS_PER_TOKEN       = 3.5
    _EXPANSION_OUT_TOKENS  = 300
    _ANALYSIS_OUT_TOKENS   = 400

    def _tok(text: str) -> int:
        return max(1, int(len(text) / _CHARS_PER_TOKEN))

    # 데이터 로드
    try:
        _, goals, logs, _ = load_dataset_from_json("data/synthetic")
    except Exception:
        ds = build_dataset(small_mode=True, seed=seed)
        goals, logs = ds.goals, ds.logs

    import random
    rng = random.Random(seed)
    sampled = rng.sample(goals, min(n_goals, len(goals)))

    results = []
    for goal in sampled:
        user_logs = [l for l in logs if l.user_id == goal.user_id]
        if len(user_logs) < 5:
            continue

        try:
            cfg1 = DEFAULT_CONFIG.stage1
            cfg1.retrieval.top_k = 5
            cfg1.retrieval.candidate_size = max(15, len(user_logs) // 2)
            cfg1.query_expansion.enabled = True

            s1 = Stage1Pipeline(config=cfg1)
            s1.index(user_logs)
            s1_result = s1.run(goal, use_expansion=True)

            cfg2 = DEFAULT_CONFIG.stage2
            s2 = Stage2Pipeline(config=cfg2, use_mock_llm=True)
            s2.index(user_logs)
            s2_result = s2.run_with_stage1(s1_result)
            evidence_units = s2_result.evidence_units

            # [A] Raw
            log_lines = "\n".join(
                f"[{i}] {l.date}  {l.title}\n    {l.content}"
                for i, l in enumerate(user_logs, 1)
            )
            raw_prompt = _PROMPT_TEMPLATE.format(
                goal_title=goal.title,
                goal_description=goal.description,
                evidence_text=log_lines,
            )
            raw_in    = _tok(raw_prompt)
            raw_total = raw_in + _ANALYSIS_OUT_TOKENS

            # [B] Pipeline
            exp_in    = _tok(_EXPANSION_PROMPT.format(
                title=goal.title,
                description=goal.description or goal.title,
            ))
            ana_prompt = _PROMPT_TEMPLATE.format(
                goal_title=goal.title,
                goal_description=goal.description,
                evidence_text=build_evidence_text(evidence_units),
            )
            ana_in   = _tok(ana_prompt)
            pipe_total = exp_in + _EXPANSION_OUT_TOKENS + ana_in + _ANALYSIS_OUT_TOKENS

            # goal title → ASCII-safe label (goal_id 뒤 번호 + 영문 키워드)
            _KO_EN = {
                "개발자": "Dev Job", "AI 엔지니어": "AI Eng", "대학원": "Grad School",
                "토익": "TOEIC", "운동": "Exercise", "식단": "Diet",
                "저축": "Savings", "투자": "Investment", "독서": "Reading",
                "여행": "Travel", "연애": "Romance", "기상": "Morning",
                "사진": "Photo", "요리": "Cooking", "영어": "English",
                "일본어": "Japanese", "공무원": "Civil Exam", "기타": "Guitar",
                "명상": "Meditation", "드로잉": "Drawing",
            }
            eng = goal.goal_id.split("-")[-1]  # 01, 02 등
            for ko, en in _KO_EN.items():
                if ko in goal.title:
                    eng = en
                    break
            label = f"{eng}\n({goal.user_id[-4:]})"

            results.append({
                "goal_title":     label,
                "goal_title_ko":  goal.title[:16],
                "total_logs":     len(user_logs),
                "admitted_logs":  len(s1_result.selected_logs),
                "raw_total":      raw_total,
                "pipe_exp":       exp_in + _EXPANSION_OUT_TOKENS,
                "pipe_ana":       ana_in + _ANALYSIS_OUT_TOKENS,
                "pipe_total":     pipe_total,
                "reduction_pct":  round((1 - pipe_total / raw_total) * 100, 1),
            })
            print(f"  Token sample: {goal.goal_id}  raw={raw_total:,}  pipe={pipe_total:,}  -{results[-1]['reduction_pct']}%")
        except Exception as e:
            print(f"  Skip {goal.goal_id}: {e}")

    return results


def plot_token_chart(samples: list[dict], output_path: str) -> None:
    if not samples:
        print("[Chart 2] No data — skip")
        return

    # ── 평균 집계 ────────────────────────────────────────────────────────────
    avg_raw      = sum(s["raw_total"]   for s in samples) / len(samples)
    avg_pipe     = sum(s["pipe_total"]  for s in samples) / len(samples)
    avg_pipe_exp = sum(s["pipe_exp"]    for s in samples) / len(samples)
    avg_pipe_ana = sum(s["pipe_ana"]    for s in samples) / len(samples)
    avg_red      = sum(s["reduction_pct"] for s in samples) / len(samples)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5),
                              gridspec_kw={"width_ratios": [1, 1.6]})

    # ── 왼쪽: 평균 bar ───────────────────────────────────────────────────────
    ax = axes[0]
    categories = ["[A] Full-Context\n(Raw)", "[B] Proposed\n(Pipeline)"]
    values      = [avg_raw, avg_pipe]
    colors      = [PALETTE["raw"], PALETTE["pipeline"]]

    bars = ax.bar(categories, values, color=colors, width=0.5, zorder=3,
                  edgecolor="white", linewidth=1.2)

    # Proposed bar를 stacked로 표현 (Expansion / Analysis)
    bar_pipe = bars[1]
    x_pos = bar_pipe.get_x()
    w     = bar_pipe.get_width()
    ax.bar([categories[1]], [avg_pipe_exp], color=PALETTE["expansion"],
           width=0.5, zorder=4, edgecolor="white", linewidth=1.2,
           label=f"Expansion ({int(avg_pipe_exp):,} tok)")
    ax.bar([categories[1]], [avg_pipe_ana], bottom=[avg_pipe_exp],
           color=PALETTE["analysis"], width=0.5, zorder=4,
           edgecolor="white", linewidth=1.2,
           label=f"Analysis ({int(avg_pipe_ana):,} tok)")

    # 숫자 레이블
    ax.text(0, avg_raw + avg_raw * 0.02, f"{int(avg_raw):,}", ha="center",
            va="bottom", fontsize=10, fontweight="bold", color="#374151")
    ax.text(1, avg_pipe + avg_pipe * 0.02, f"{int(avg_pipe):,}", ha="center",
            va="bottom", fontsize=10, fontweight="bold", color="#374151")

    # 절감율 화살표
    ax.annotate("",
        xy=(1, avg_pipe * 1.05),
        xytext=(0, avg_raw * 0.95),
        arrowprops=dict(arrowstyle="->", color="#DC2626", lw=2.0),
    )
    mid_x = 0.5
    mid_y = (avg_raw + avg_pipe) / 2
    ax.text(mid_x, mid_y, f"−{avg_red:.0f}%", ha="center", va="center",
            fontsize=13, fontweight="bold", color="#DC2626",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#DC2626", lw=1.5))

    ax.set_ylabel("Total Tokens (input + output)", fontsize=10)
    ax.set_title("Average Token Usage\n(per goal)", fontsize=11, fontweight="bold")
    ax.set_ylim(0, avg_raw * 1.25)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.legend(fontsize=8.5, loc="upper right")

    # ── 오른쪽: goal별 bar ───────────────────────────────────────────────────
    ax2 = axes[1]
    n = len(samples)
    ind = np.arange(n)
    bar_w = 0.35

    raw_vals  = [s["raw_total"]  for s in samples]
    pipe_vals = [s["pipe_total"] for s in samples]
    labels    = [s["goal_title"] for s in samples]

    b1 = ax2.bar(ind - bar_w / 2, raw_vals,  bar_w, label="[A] Full-Context",
                 color=PALETTE["raw"],      zorder=3, edgecolor="white")
    b2 = ax2.bar(ind + bar_w / 2, pipe_vals, bar_w, label="[B] Pipeline",
                 color=PALETTE["pipeline"], zorder=3, edgecolor="white")

    # 절감율 표시
    for i, (rv, pv) in enumerate(zip(raw_vals, pipe_vals)):
        red = (1 - pv / rv) * 100
        ax2.text(i, max(rv, pv) + 50, f"−{red:.0f}%",
                 ha="center", va="bottom", fontsize=7.5, color="#DC2626",
                 fontweight="bold")

    ax2.set_xticks(ind)
    ax2.set_xticklabels(labels, rotation=35, ha="right", fontsize=8.5)
    ax2.set_ylabel("Total Tokens", fontsize=10)
    ax2.set_title("Per-Goal Token Comparison\n(Full-Context vs Pipeline)",
                  fontsize=11, fontweight="bold")
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax2.legend(fontsize=9)

    # ── 전체 제목 ─────────────────────────────────────────────────────────────
    fig.suptitle(
        "Token Efficiency: Full-Context Baseline vs Proposed RAG Pipeline",
        fontsize=13, fontweight="bold", y=1.01,
    )
    n_goals = len(samples)
    avg_logs = sum(s["total_logs"] for s in samples) / n_goals
    avg_adm  = sum(s["admitted_logs"] for s in samples) / n_goals
    # 하단에 한글 원제 범례 텍스트 (영문 label과 매핑)
    ko_labels = [s.get("goal_title_ko", "") for s in samples]
    note_parts = [f"{s['goal_title'].split(chr(10))[0]}={ko}" for s, ko in zip(samples, ko_labels) if ko]
    note = "  |  ".join(note_parts[:4])  # 너무 길면 잘라냄

    fig.text(0.5, -0.02,
             f"n={n_goals} goals  |  avg corpus={avg_logs:.0f} logs  |  "
             f"avg admitted={avg_adm:.1f} logs  |  est: 1 token ≈ 3.5 chars",
             ha="center", fontsize=8.5, color="#6B7280")

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Chart 2 saved] {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="연구 차트 생성")
    parser.add_argument("--output_dir", default="results/charts")
    parser.add_argument("--threshold_csv",
                        default="results/threshold_experiment/threshold_experiment_summary.csv")
    parser.add_argument("--production_threshold", type=float, default=0.92)
    parser.add_argument("--token_goals", type=int, default=8,
                        help="토큰 비교 차트에 사용할 goal 샘플 수")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Chart 1 ──────────────────────────────────────────────────────────────
    print("\n[Chart 1] Threshold Tuning Line Chart ...")
    data = load_threshold_data(args.threshold_csv)
    plot_threshold_chart(data, str(out / "chart1_threshold_tuning.png"),
                         production_threshold=args.production_threshold)

    # ── Chart 2 ──────────────────────────────────────────────────────────────
    print(f"\n[Chart 2] Token Usage Bar Chart ({args.token_goals} goals) ...")
    samples = compute_token_samples(n_goals=args.token_goals)
    if samples:
        plot_token_chart(samples, str(out / "chart2_token_usage.png"))
    else:
        print("[Chart 2] 샘플 데이터 없음 — 스킵")

    print(f"\n완료. 저장 위치: {out}/")


if __name__ == "__main__":
    main()
