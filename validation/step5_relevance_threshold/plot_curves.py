#!/usr/bin/env python3
"""Relevance Filtering Threshold — Recall & Precision 곡선 시각화.

Usage:
    cd /home/user/retrieval
    .venv/bin/python validation/step5_relevance_threshold/plot_curves.py
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

import logging
logging.basicConfig(level=logging.WARNING)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

from app.config import DEFAULT_CONFIG
from app.data_generation.export_utils import load_dataset_from_json
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.query_expansion import _heuristic_expansion
from app.schemas import GoalLogLabel, ResearchGoal, ResearchLog
from app.utils.text_matching import _tok_set, match_priority_phrase, match_term

DATA_DIR  = ROOT / "data" / "synthetic"
OUT_PATH  = Path(__file__).parent / "results" / "threshold_curve.png"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

TOP_K = 10
CFG   = DEFAULT_CONFIG.stage1.ranker

WEIGHTS = {"pri": 0.10, "ev": 0.06, "rel": 0.03, "sem": 0.70, "base": 0.05}
TOTAL_W = sum(WEIGHTS.values())
SCALE   = 1.0 / TOTAL_W

PHRASE_PEN  = CFG.negative_penalty_phrase
TOKEN_PEN   = CFG.negative_penalty_token
DAILY_NOISE = CFG.negative_daily_penalty

THRESHOLDS = [round(0.60 + i * 0.01, 2) for i in range(11)]   # 0.60~0.70


def precompute(goals, logs, labels):
    goal_map  = {g.goal_id: g for g in goals}
    log_map   = {l.log_id:  l for l in logs}
    exp_cache: dict = {}

    user_logs_map = defaultdict(list)
    for l in logs:
        user_logs_map[l.user_id].append(l)

    goal_labels_map = defaultdict(list)
    for lbl in labels:
        if lbl.label in ("relevant", "irrelevant"):
            goal_labels_map[lbl.goal_id].append(lbl)

    user_goals_map = defaultdict(list)
    for g in goals:
        if g.goal_id in goal_labels_map:
            user_goals_map[g.user_id].append(g)

    dense = DenseRetriever()
    records = []
    total = sum(len(v) for v in user_goals_map.values())
    done = 0

    for user_id, u_goals in user_goals_map.items():
        u_logs = user_logs_map[user_id]
        if not u_logs:
            continue
        dense.index(u_logs)

        for goal in u_goals:
            done += 1
            if done % 100 == 0:
                print(f"  [{done}/{total}] ...", flush=True)

            sem_map = {log.log_id: s for log, s in dense.score_all(goal.query_text)}

            if goal.goal_id not in exp_cache:
                exp_cache[goal.goal_id] = _heuristic_expansion(goal, max_terms=15)
            exp = exp_cache[goal.goal_id]

            pri_terms = exp.get("priority_terms", [])
            ev_terms  = exp.get("evidence_terms", [])
            rel_terms = exp.get("related_terms",  [])
            neg_terms = exp.get("negative_terms", [])

            for lbl in goal_labels_map[goal.goal_id]:
                log = log_map.get(lbl.log_id)
                if not log:
                    continue

                txl, txt = log.full_text.lower(), _tok_set(log.full_text.lower())
                tl,  tt  = log.title.lower(),     _tok_set(log.title.lower())

                def _pri(terms):
                    if not terms: return 0.0
                    m = [match_priority_phrase(t, txl, txt, tl, tt) for t in terms]
                    return min(sum(1.0 for x in m if x.score > 0.0) / len(terms), 1.0)

                def _rel(terms):
                    if not terms: return 0.0
                    m = [match_term(t, txl, txt, tl, tt) for t in terms]
                    return min(sum(1.0 for x in m if x.level != "none") / len(terms), 1.0)

                pri_score = _pri(pri_terms)
                goal_toks = _tok_set(goal.query_text)
                base      = len(goal_toks & txt) / len(goal_toks) if goal_toks else 0.0

                n_multi = sum(
                    1 for t in neg_terms
                    if len(set(re.findall(r"[\w가-힣]+", t.lower()))) > 1
                    and t.lower() in txl
                )
                n_single = sum(
                    1 for t in neg_terms
                    if len(set(re.findall(r"[\w가-힣]+", t.lower()))) == 1
                    and set(re.findall(r"[\w가-힣]+", t.lower())) & txt
                )

                raw_dm = n_multi * PHRASE_PEN + n_single * TOKEN_PEN
                noise  = DAILY_NOISE if log.activity_type == "daily" else 0.0
                capped = min(raw_dm + noise, 1.0)

                veto = (
                    CFG.negative_veto_enabled
                    and raw_dm >= CFG.negative_veto_dm_threshold
                    and pri_score < CFG.negative_veto_priority_min
                )
                if veto:
                    final_score = 0.0
                else:
                    rel = SCALE * (
                        WEIGHTS["pri"]  * pri_score
                        + WEIGHTS["ev"] * _pri(ev_terms)
                        + WEIGHTS["rel"] * _rel(rel_terms)
                        + WEIGHTS["sem"] * sem_map.get(lbl.log_id, 0.0)
                        + WEIGHTS["base"]* base
                    )
                    final_score = max(0.0, round(rel - capped, 4))

                records.append({
                    "goal_id":     lbl.goal_id,
                    "log_id":      lbl.log_id,
                    "label":       lbl.label,
                    "final_score": final_score,
                })

    return records


def evaluate(records, threshold):
    goal_records = defaultdict(list)
    for r in records:
        goal_records[r["goal_id"]].append(r)

    goal_metrics = []
    for goal_id, grecs in goal_records.items():
        total_rel = sum(1 for r in grecs if r["label"] == "relevant")
        if total_rel == 0:
            continue

        admitted = sorted(
            [r for r in grecs if r["final_score"] >= threshold],
            key=lambda x: x["final_score"], reverse=True,
        )
        selected = admitted[:TOP_K]

        tp    = sum(1 for r in selected if r["label"] == "relevant")
        sel_n = len(selected)
        prec  = tp / sel_n if sel_n > 0 else 0.0
        rec   = tp / total_rel

        goal_metrics.append({"rec": rec, "prec": prec})

    mean = lambda k: round(sum(m[k] for m in goal_metrics) / len(goal_metrics), 4) if goal_metrics else 0.0
    return mean("rec"), mean("prec")


def main():
    print("[데이터 로드]")
    _, goals, logs, labels = load_dataset_from_json(str(DATA_DIR))
    print(f"  goals={len(goals)}  logs={len(logs)}  labels={len(labels)}")

    print("\n[사전 계산]")
    records = precompute(goals, logs, labels)
    print(f"  처리 완료: {len(records)}개")

    print("\n[Threshold Sweep 0.60~0.70]")
    recalls, precisions = [], []
    for thr in THRESHOLDS:
        rec, prec = evaluate(records, thr)
        recalls.append(rec)
        precisions.append(prec)
        print(f"  threshold={thr:.2f}  recall={rec:.4f}  precision={prec:.4f}")

    # ── 교차점 찾기 ───────────────────────────────────────────────────────────
    cross_thr, cross_val = None, None
    for i in range(len(THRESHOLDS) - 1):
        r1, r2 = recalls[i], recalls[i + 1]
        p1, p2 = precisions[i], precisions[i + 1]
        # recall이 precision을 넘거나 같아지는 점 (r > p → r <= p)
        if (r1 - p1) * (r2 - p2) <= 0:
            # 선형 보간
            denom = (r1 - p1) - (r2 - p2)
            if abs(denom) > 1e-9:
                t = (r1 - p1) / denom
                cross_thr = THRESHOLDS[i] + t * (THRESHOLDS[i + 1] - THRESHOLDS[i])
                cross_val = r1 + t * (r2 - r1)
            break

    # ── ASCII 그래프 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  Recall & Precision vs Threshold  (R=파랑 P=빨강)")
    print("=" * 62)

    height = 20
    y_min, y_max = 0.55, 0.90

    def to_row(val):
        return height - 1 - int((val - y_min) / (y_max - y_min) * (height - 1))

    cols = len(THRESHOLDS)
    grid = [[" "] * cols for _ in range(height)]

    for j, (r, p) in enumerate(zip(recalls, precisions)):
        rr = max(0, min(height - 1, to_row(r)))
        pr = max(0, min(height - 1, to_row(p)))
        grid[rr][j] = "R"
        grid[pr][j] = "P"
        if rr == pr:
            grid[rr][j] = "X"   # 교차점

    for i, row in enumerate(grid):
        y_val = y_max - i * (y_max - y_min) / (height - 1)
        bar = " ".join(row)
        print(f"  {y_val:.3f} | {bar}")

    print("         " + "  ".join(f"{t:.2f}"[1:] for t in THRESHOLDS))
    print("         threshold →")

    print()
    print("  thr    Recall  Precis")
    print("  " + "-" * 24)
    for thr, r, p in zip(THRESHOLDS, recalls, precisions):
        mark = " ← current" if abs(thr - 0.68) < 0.005 else ""
        cross_mark = " ← CROSS" if cross_thr and abs(thr - round(cross_thr, 2)) < 0.015 else ""
        print(f"  {thr:.2f}  {r:.4f}  {p:.4f}{mark}{cross_mark}")

    if cross_thr:
        print(f"\n  교차점: threshold ≈ {cross_thr:.3f}  (recall=precision≈{cross_val:.3f})")
    print(f"  현재 설정: threshold=0.68")

    # matplotlib 가능하면 PNG도 저장
    if HAS_MPL:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(THRESHOLDS, recalls,    "b-o", linewidth=2, markersize=5, label="Recall")
        ax.plot(THRESHOLDS, precisions, "r-s", linewidth=2, markersize=5, label="Precision")
        if cross_thr:
            ax.axvline(cross_thr, color="gray", linestyle="--", alpha=0.6)
            ax.plot(cross_thr, cross_val, "g*", markersize=14, zorder=5,
                    label=f"Intersection ≈ {cross_thr:.3f}")
        ax.axvline(0.68, color="orange", linestyle=":", alpha=0.8, label="Current (0.68)")
        ax.set_xlabel("Threshold"); ax.set_ylabel("Score")
        ax.set_title("Recall & Precision vs Threshold")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT_PATH, dpi=150)
        print(f"\n[PNG 저장] {OUT_PATH}")


if __name__ == "__main__":
    main()
