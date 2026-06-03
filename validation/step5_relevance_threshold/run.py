#!/usr/bin/env python3
"""Step 5: Relevance Filtering Threshold 검증
==============================================
final_score >= threshold 조건으로 로그를 걸러내는 Relevance Filtering의
최적 threshold를 recall 기준으로 탐색한다.

고정 조건:
  relevance 가중치: sem=0.70, pri=0.10, ev=0.06, rel=0.03, base=0.05
  negative penalty: phrase=0.70, token=0.40, daily=0.20
  top_k=10

비교: threshold 0.00 ~ 0.30 sweep

평가 지표 (주: recall, 부: precision / F1 / avg_admitted)

Usage:
    cd /home/user/retrieval
    .venv/bin/python validation/step5_relevance_threshold/run.py
"""
from __future__ import annotations

import csv
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

from app.config import DEFAULT_CONFIG
from app.data_generation.export_utils import load_dataset_from_json
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.query_expansion import _heuristic_expansion
from app.schemas import GoalLogLabel, ResearchGoal, ResearchLog
from app.utils.text_matching import _tok_set, match_priority_phrase, match_term

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = ROOT / "data" / "synthetic"

TOP_K = 10
CFG   = DEFAULT_CONFIG.stage1.ranker

# ── 고정 파라미터 ────────────────────────────────────────────────────────────
WEIGHTS      = {"pri": 0.10, "ev": 0.06, "rel": 0.03, "sem": 0.70, "base": 0.05}
TOTAL_W      = sum(WEIGHTS.values())
SCALE        = 1.0 / TOTAL_W
PHRASE_PEN   = CFG.negative_penalty_phrase   # 0.70
TOKEN_PEN    = CFG.negative_penalty_token    # 0.40
DAILY_NOISE  = CFG.negative_daily_penalty    # 0.20

# ── threshold 후보 ───────────────────────────────────────────────────────────
THRESHOLDS = [0.57, 0.58, 0.59, 0.60, 0.61, 0.62, 0.63]


# ── 사전 계산 ─────────────────────────────────────────────────────────────────

def precompute(
    goals: list[ResearchGoal],
    logs: list[ResearchLog],
    labels: list[GoalLogLabel],
) -> list[dict]:
    goal_map = {g.goal_id: g for g in goals}
    log_map  = {l.log_id:  l for l in logs}
    exp_cache: dict[str, dict] = {}

    user_logs_map: dict[str, list[ResearchLog]] = defaultdict(list)
    for l in logs:
        user_logs_map[l.user_id].append(l)

    goal_labels_map: dict[str, list[GoalLogLabel]] = defaultdict(list)
    for lbl in labels:
        if lbl.label in ("relevant", "irrelevant"):
            goal_labels_map[lbl.goal_id].append(lbl)

    user_goals_map: dict[str, list[ResearchGoal]] = defaultdict(list)
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
            if done % 50 == 0 or done == total:
                print(f"  [{done}/{total}] {goal.goal_id} ...", flush=True)

            sem_map = {log.log_id: score for log, score in dense.score_all(goal.query_text)}

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

                pri_score = _pri(pri_terms)
                raw_dm    = n_multi * PHRASE_PEN + n_single * TOKEN_PEN
                noise     = DAILY_NOISE if log.activity_type == "daily" else 0.0
                capped    = min(raw_dm + noise, 1.0)

                # veto
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


# ── 메트릭 계산 ───────────────────────────────────────────────────────────────

def evaluate(records: list[dict], threshold: float) -> dict:
    goal_records: dict[str, list[dict]] = defaultdict(list)
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
        f1    = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        goal_metrics.append({
            "rec": rec, "prec": prec, "f1": f1, "sel_n": sel_n, "admitted": len(admitted),
        })

    def mean(k):
        v = [m[k] for m in goal_metrics]
        return round(sum(v) / len(v), 4) if v else 0.0

    return {
        "recall":        mean("rec"),
        "precision":     mean("prec"),
        "f1":            mean("f1"),
        "avg_selected":  mean("sel_n"),
        "avg_admitted":  mean("admitted"),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("[데이터 로드]")
    _, goals, logs, labels = load_dataset_from_json(str(DATA_DIR))
    print(f"  goals={len(goals)}  logs={len(logs)}  labels={len(labels)}")

    print("\n[사전 계산]")
    records = precompute(goals, logs, labels)
    print(f"  처리 완료: {len(records)}개 (goal, log) 쌍")

    print("\n" + "=" * 64)
    print("Relevance Filtering Threshold Sweep  (recall 기준)")
    print(f"  weights={WEIGHTS}  top_k={TOP_K}")
    print("=" * 64)

    rows = []
    for thr in THRESHOLDS:
        m = evaluate(records, thr)
        marker = " ← 현재" if abs(thr - 0.08) < 1e-9 else ""
        rows.append({"threshold": thr, **m})
        print(
            f"  threshold={thr:.2f}{marker:<8}"
            f"  recall={m['recall']:.4f}"
            f"  prec={m['precision']:.4f}"
            f"  F1={m['f1']:.4f}"
            f"  avg_admitted={m['avg_admitted']:.1f}"
            f"  avg_selected={m['avg_selected']:.1f}"
        )

    # ── 결과 테이블 ───────────────────────────────────────────────────────────
    print("\n" + "=" * 66)
    print("비교 테이블  (threshold 오름차순)")
    print("=" * 66)
    hdr = (
        f"{'threshold':>10} {'recall':>8} {'prec':>8} {'F1':>8}"
        f" {'avg_adm':>8} {'avg_sel':>8}"
    )
    print(hdr)
    print("-" * 66)
    for r in rows:
        mark = " *" if abs(r["threshold"] - 0.08) < 1e-9 else "  "
        print(
            f"{r['threshold']:>10.2f}{mark}"
            f" {r['recall']:>8.4f} {r['precision']:>8.4f} {r['f1']:>8.4f}"
            f" {r['avg_admitted']:>8.1f} {r['avg_selected']:>8.1f}"
        )

    best_r = max(rows, key=lambda x: x["recall"])
    best_f = max(rows, key=lambda x: x["f1"])
    print(f"\n  (* = 현재 설정)")
    print(f"  Recall 최고:  threshold={best_r['threshold']:.2f}  R={best_r['recall']:.4f}")
    print(f"  F1 최고:      threshold={best_f['threshold']:.2f}  F1={best_f['f1']:.4f}")

    # recall이 처음으로 최고치에 도달하는 threshold 찾기
    max_recall = best_r["recall"]
    first_max = next(r for r in rows if r["recall"] >= max_recall)
    print(f"  최고 recall 최초 달성: threshold={first_max['threshold']:.2f}")

    # CSV 저장
    out = RESULTS_DIR / "threshold_sweep.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[저장] {out}")


if __name__ == "__main__":
    main()
