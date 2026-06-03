#!/usr/bin/env python3
"""Step 4: Negative Penalty 파라미터 검증 (새 relevance 가중치 기준)
======================================================================
Step 3에서 확정한 relevance 가중치(sem=0.70, pri=0.10, ev=0.06, rel=0.03, base=0.05)를
고정한 상태에서 negative penalty 파라미터를 비교한다.

비교 축 3가지:
  Group A — 전체 스케일 변화 (phrase:token 비율 고정)
  Group B — phrase:token 비율 변화 (스케일 고정)
  Group C — daily noise 변화

평가 지표:
  주: recall@selected (관련 로그를 penalty가 잘못 걸러내는지)
  부: precision@selected, f1@selected, n_vetoed, n_zero_by_penalty, FPR

Usage:
    cd /home/user/retrieval
    .venv/bin/python validation/step4_negative_penalty/run.py
"""
from __future__ import annotations

import csv
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

TOP_K     = 10
THRESHOLD = 0.08
CFG       = DEFAULT_CONFIG.stage1.ranker

# ── 확정된 relevance 가중치 ───────────────────────────────────────────────────
WEIGHTS = {"pri": 0.10, "ev": 0.06, "rel": 0.03, "sem": 0.70, "base": 0.05}
TOTAL_W = sum(WEIGHTS.values())   # 0.94
SCALE   = 1.0 / TOTAL_W

# ── 10가지 비교 케이스 ────────────────────────────────────────────────────────
CASES = [
    # Group A: 전체 스케일 변화
    {"name": "current",     "group": "A", "phrase": 0.70, "token": 0.40, "daily": 0.20},
    {"name": "scale_down",  "group": "A", "phrase": 0.50, "token": 0.29, "daily": 0.15},
    {"name": "scale_up",    "group": "A", "phrase": 0.90, "token": 0.51, "daily": 0.26},
    # Group B: phrase:token 비율 변화
    {"name": "ratio_equal", "group": "B", "phrase": 0.50, "token": 0.50, "daily": 0.20},
    {"name": "ratio_2to1",  "group": "B", "phrase": 0.60, "token": 0.30, "daily": 0.20},
    {"name": "ratio_3to1",  "group": "B", "phrase": 0.75, "token": 0.25, "daily": 0.20},
    # Group C: daily noise 변화
    {"name": "no_daily",    "group": "C", "phrase": 0.70, "token": 0.40, "daily": 0.00},
    {"name": "low_daily",   "group": "C", "phrase": 0.70, "token": 0.40, "daily": 0.10},
    {"name": "high_daily",  "group": "C", "phrase": 0.70, "token": 0.40, "daily": 0.30},
]


# ── 사전 계산 ─────────────────────────────────────────────────────────────────

def precompute(
    goals: list[ResearchGoal],
    logs: list[ResearchLog],
    labels: list[GoalLogLabel],
) -> list[dict]:
    """(goal, log, label) 전체에 대해 필요한 데이터 사전 계산.

    저장 항목:
      - relevance 컴포넌트 점수 (pri, ev, rel, sem, base)  ← 고정
      - negative 매칭 카운트 (n_multi_phrase, n_single_token) ← 케이스별 재계산용
      - is_daily                                             ← daily noise 재계산용
      - pri_score                                            ← veto 판단용
    """
    goal_map  = {g.goal_id: g for g in goals}
    log_map   = {l.log_id:  l for l in logs}
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
    total_goals = sum(len(v) for v in user_goals_map.values())
    done = 0

    for user_id, u_goals in user_goals_map.items():
        u_logs = user_logs_map[user_id]
        if not u_logs:
            continue

        dense.index(u_logs)

        for goal in u_goals:
            done += 1
            if done % 50 == 0 or done == total_goals:
                print(f"  [{done}/{total_goals}] {goal.goal_id} ...", flush=True)

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

                text  = log.full_text
                title = log.title
                tl, tt = title.lower(), _tok_set(title.lower())
                txl, txt = text.lower(), _tok_set(text.lower())

                def _score_pri(terms):
                    if not terms:
                        return 0.0
                    matches = [match_priority_phrase(t, txl, txt, tl, tt) for t in terms]
                    return min(sum(1.0 for m in matches if m.score > 0.0) / len(terms), 1.0)

                def _score_rel(terms):
                    if not terms:
                        return 0.0
                    matches = [match_term(t, txl, txt, tl, tt) for t in terms]
                    return min(sum(1.0 for m in matches if m.level != "none") / len(terms), 1.0)

                goal_toks  = _tok_set(goal.query_text)
                base_score = len(goal_toks & txt) / len(goal_toks) if goal_toks else 0.0

                pri_score = _score_pri(pri_terms)
                ev_score  = _score_pri(ev_terms)

                # negative 매칭 카운트 (penalty 값과 독립적으로 저장)
                n_multi_phrase = 0
                n_single_token = 0
                import re
                for term in neg_terms:
                    term_l  = term.lower()
                    term_toks = set(re.findall(r"[\w가-힣]+", term_l))
                    is_multi  = len(term_toks) > 1
                    if is_multi:
                        if term_l in txl:
                            n_multi_phrase += 1
                    else:
                        if term_toks & txt:
                            n_single_token += 1

                records.append({
                    "goal_id":       lbl.goal_id,
                    "log_id":        lbl.log_id,
                    "label":         lbl.label,
                    "pri":           pri_score,
                    "ev":            ev_score,
                    "rel":           _score_rel(rel_terms),
                    "sem":           sem_map.get(lbl.log_id, 0.0),
                    "base":          base_score,
                    "n_multi_phrase": n_multi_phrase,
                    "n_single_token": n_single_token,
                    "is_daily":       log.activity_type == "daily",
                })

    return records


# ── 스코어 계산 ───────────────────────────────────────────────────────────────

def compute_final(r: dict, phrase_w: float, token_w: float, daily_w: float) -> tuple[float, bool]:
    """(final_score, is_vetoed) 반환."""
    raw_dm = r["n_multi_phrase"] * phrase_w + r["n_single_token"] * token_w
    noise  = daily_w if r["is_daily"] else 0.0
    capped = min(raw_dm + noise, 1.0)

    # Veto: raw_dm(penalty값 기반) >= 0.70 AND pri_score < 0.05
    veto = (
        CFG.negative_veto_enabled
        and raw_dm >= CFG.negative_veto_dm_threshold
        and r["pri"] < CFG.negative_veto_priority_min
    )
    if veto:
        return 0.0, True

    rel = SCALE * (
        WEIGHTS["pri"]  * r["pri"]
        + WEIGHTS["ev"] * r["ev"]
        + WEIGHTS["rel"] * r["rel"]
        + WEIGHTS["sem"] * r["sem"]
        + WEIGHTS["base"]* r["base"]
    )
    return max(0.0, round(rel - capped, 4)), False


# ── 메트릭 계산 ───────────────────────────────────────────────────────────────

def evaluate(records: list[dict], phrase_w: float, token_w: float, daily_w: float) -> dict:
    goal_records: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        goal_records[r["goal_id"]].append(r)

    goal_metrics = []
    total_vetoed = 0
    total_zero_by_penalty = 0

    for goal_id, grecs in goal_records.items():
        scored = []
        n_vetoed = 0
        n_zero_pen = 0
        for r in grecs:
            fs, vetoed = compute_final(r, phrase_w, token_w, daily_w)
            scored.append((r["log_id"], r["label"], fs))
            if vetoed:
                n_vetoed += 1
            elif fs == 0.0 and (r["n_multi_phrase"] > 0 or r["n_single_token"] > 0 or r["is_daily"]):
                n_zero_pen += 1

        total_vetoed       += n_vetoed
        total_zero_by_penalty += n_zero_pen

        admitted = sorted(
            [(lid, lbl, fs) for lid, lbl, fs in scored if fs >= THRESHOLD],
            key=lambda x: x[2], reverse=True,
        )
        selected = admitted[:TOP_K]

        total_rel = sum(1 for _, lbl, _ in scored if lbl == "relevant")
        if total_rel == 0:
            continue

        tp    = sum(1 for _, lbl, _ in selected if lbl == "relevant")
        sel_n = len(selected)
        prec  = tp / sel_n if sel_n > 0 else 0.0
        rec   = tp / total_rel
        f1    = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        fpr   = 1.0 - prec if sel_n > 0 else 0.0

        goal_metrics.append({"prec": prec, "rec": rec, "f1": f1, "fpr": fpr, "sel_n": sel_n})

    def mean(k):
        v = [m[k] for m in goal_metrics]
        return round(sum(v) / len(v), 4) if v else 0.0

    return {
        "n_goals":             len(goal_metrics),
        "recall":              mean("rec"),
        "precision":           mean("prec"),
        "f1":                  mean("f1"),
        "fpr":                 mean("fpr"),
        "avg_selected":        mean("sel_n"),
        "total_vetoed":        total_vetoed,
        "total_zero_by_pen":   total_zero_by_penalty,
    }


# ── 실행 ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("[데이터 로드]")
    _, goals, logs, labels = load_dataset_from_json(str(DATA_DIR))
    print(f"  goals={len(goals)}  logs={len(logs)}  labels={len(labels)}")

    print("\n[사전 계산]")
    records = precompute(goals, logs, labels)
    print(f"  처리 완료: {len(records)}개 (goal, log) 쌍")

    print("\n" + "=" * 70)
    print("Negative Penalty 파라미터 비교  (relevance 가중치 고정)")
    print(f"  weights: {WEIGHTS}  threshold={THRESHOLD}  top_k={TOP_K}")
    print("=" * 70)

    rows = []
    for case in CASES:
        m = evaluate(records, case["phrase"], case["token"], case["daily"])
        row = {**case, **m}
        rows.append(row)
        print(
            f"  [{case['group']}] {case['name']:<14}"
            f"  phrase={case['phrase']:.2f}  token={case['token']:.2f}  daily={case['daily']:.2f}"
            f"  →  recall={m['recall']:.4f}"
            f"  prec={m['precision']:.4f}"
            f"  F1={m['f1']:.4f}"
            f"  vetoed={m['total_vetoed']}"
        )

    # ── 결과 테이블 (recall 내림차순) ────────────────────────────────────────
    print("\n" + "=" * 78)
    print("비교 테이블  (recall 내림차순)")
    print("=" * 78)
    hdr = (
        f"{'name':<14} {'grp':>3} {'phrase':>7} {'token':>7} {'daily':>7}"
        f"  {'recall':>8} {'prec':>8} {'F1':>8} {'FPR':>6}"
        f"  {'vetoed':>7} {'zero_pen':>8}"
    )
    print(hdr)
    print("-" * 78)
    for r in sorted(rows, key=lambda x: x["recall"], reverse=True):
        print(
            f"{r['name']:<14} {r['group']:>3} {r['phrase']:>7.2f} {r['token']:>7.2f} {r['daily']:>7.2f}"
            f"  {r['recall']:>8.4f} {r['precision']:>8.4f} {r['f1']:>8.4f} {r['fpr']:>6.4f}"
            f"  {r['total_vetoed']:>7} {r['total_zero_by_pen']:>8}"
        )

    best_r = max(rows, key=lambda x: x["recall"])
    best_f = max(rows, key=lambda x: x["f1"])
    print(f"\n  Recall 최고:  {best_r['name']}  R={best_r['recall']:.4f}")
    print(f"  F1 최고:      {best_f['name']}  F1={best_f['f1']:.4f}")

    # ── 그룹별 요약 ───────────────────────────────────────────────────────────
    for grp in ("A", "B", "C"):
        grp_rows = [r for r in rows if r["group"] == grp]
        if not grp_rows:
            continue
        grp_names = {"A": "스케일", "B": "phrase:token 비율", "C": "daily noise"}
        print(f"\n  [Group {grp} — {grp_names[grp]}]")
        for r in sorted(grp_rows, key=lambda x: x["recall"], reverse=True):
            print(f"    {r['name']:<14}  recall={r['recall']:.4f}  prec={r['precision']:.4f}")

    # ── CSV 저장 ──────────────────────────────────────────────────────────────
    out = RESULTS_DIR / "negative_penalty_cases.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[저장] {out}")


if __name__ == "__main__":
    main()
