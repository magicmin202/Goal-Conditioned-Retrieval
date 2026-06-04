#!/usr/bin/env python3
"""threshold=0.60 통과 but 0.68 미달인 relevant 로그 분석.

recall 손실 원인: dense_score 낮고 lexical 매칭 없는 관련 로그.

Usage:
    cd /home/user/retrieval
    .venv/bin/python validation/step5_relevance_threshold/analyze_recall_loss.py
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

from app.config import DEFAULT_CONFIG
from app.data_generation.export_utils import load_dataset_from_json
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.query_expansion import _heuristic_expansion
from app.schemas import GoalLogLabel, ResearchGoal, ResearchLog
from app.utils.text_matching import _tok_set, match_priority_phrase, match_term

DATA_DIR = ROOT / "data" / "synthetic"
CFG      = DEFAULT_CONFIG.stage1.ranker

WEIGHTS = {"pri": 0.10, "ev": 0.06, "rel": 0.03, "sem": 0.70, "base": 0.05}
TOTAL_W = sum(WEIGHTS.values())
SCALE   = 1.0 / TOTAL_W

PHRASE_PEN  = CFG.negative_penalty_phrase
TOKEN_PEN   = CFG.negative_penalty_token
DAILY_NOISE = CFG.negative_daily_penalty

THR_LOW  = 0.60   # 현재 통과 기준
THR_HIGH = 0.68   # 새 기준


def main() -> None:
    print("[데이터 로드]")
    _, goals, logs, labels = load_dataset_from_json(str(DATA_DIR))

    goal_map  = {g.goal_id: g for g in goals}
    log_map   = {l.log_id:  l for l in logs}
    exp_cache: dict[str, dict] = {}

    user_logs_map: dict[str, list[ResearchLog]] = defaultdict(list)
    for l in logs:
        user_logs_map[l.user_id].append(l)

    goal_labels_map: dict[str, list[GoalLogLabel]] = defaultdict(list)
    for lbl in labels:
        if lbl.label == "relevant":
            goal_labels_map[lbl.goal_id].append(lbl)

    user_goals_map: dict[str, list[ResearchGoal]] = defaultdict(list)
    for g in goals:
        if g.goal_id in goal_labels_map:
            user_goals_map[g.user_id].append(g)

    dense = DenseRetriever()

    lost_logs: list[dict] = []
    total_relevant = 0
    total_done = 0
    n_goals = sum(len(v) for v in user_goals_map.values())

    for user_id, u_goals in user_goals_map.items():
        u_logs = user_logs_map[user_id]
        if not u_logs:
            continue
        dense.index(u_logs)

        for goal in u_goals:
            total_done += 1
            if total_done % 100 == 0:
                print(f"  [{total_done}/{n_goals}] ...", flush=True)

            sem_map = {log.log_id: s for log, s in dense.score_all(goal.query_text)}

            if goal.goal_id not in exp_cache:
                exp_cache[goal.goal_id] = _heuristic_expansion(goal, max_terms=15)
            exp = exp_cache[goal.goal_id]

            pri_terms = exp.get("priority_terms", [])
            ev_terms  = exp.get("evidence_terms", [])
            rel_terms = exp.get("related_terms",  [])
            neg_terms = exp.get("negative_terms", [])

            for lbl in goal_labels_map[goal.goal_id]:
                total_relevant += 1
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

                sem   = sem_map.get(lbl.log_id, 0.0)
                pri   = _pri(pri_terms)
                ev    = _pri(ev_terms)
                rel   = _rel(rel_terms)
                goal_toks = _tok_set(goal.query_text)
                base  = len(goal_toks & txt) / len(goal_toks) if goal_toks else 0.0

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
                    and pri < CFG.negative_veto_priority_min
                )
                if veto:
                    final_score = 0.0
                else:
                    relevance = SCALE * (
                        WEIGHTS["pri"]  * pri
                        + WEIGHTS["ev"] * ev
                        + WEIGHTS["rel"] * rel
                        + WEIGHTS["sem"] * sem
                        + WEIGHTS["base"]* base
                    )
                    final_score = max(0.0, round(relevance - capped, 4))

                # 손실 구간: 0.60 통과 but 0.68 미달
                if THR_LOW <= final_score < THR_HIGH:
                    lost_logs.append({
                        "goal_id":    goal.goal_id,
                        "goal_title": goal.title,
                        "log_id":     lbl.log_id,
                        "log_title":  log.title,
                        "activity":   log.activity_type,
                        "final_score":final_score,
                        "sem":        round(sem, 4),
                        "pri":        round(pri, 4),
                        "ev":         round(ev, 4),
                        "base":       round(base, 4),
                        "penalty":    round(capped, 4),
                    })

    # ── 결과 출력 ─────────────────────────────────────────────────────────────
    print(f"\n총 relevant 로그: {total_relevant}개")
    print(f"recall 손실 로그 (0.60 ≤ score < 0.68): {len(lost_logs)}개")
    print(f"손실률: {len(lost_logs)/total_relevant*100:.1f}%\n")

    # sem 구간별 분포
    sem_buckets = defaultdict(int)
    for r in lost_logs:
        b = round(r["sem"] * 20) / 20   # 0.05 단위
        sem_buckets[b] += 1

    print("[sem 구간별 분포]")
    for s in sorted(sem_buckets):
        bar = "█" * sem_buckets[s]
        print(f"  sem={s:.2f}  {sem_buckets[s]:>4}개  {bar}")

    # lexical 기여 없는 것 (pri=ev=rel=base=0) 비율
    no_lexical = sum(1 for r in lost_logs if r["pri"] == 0 and r["ev"] == 0 and r["base"] == 0)
    print(f"\nlexical 매칭 전혀 없음: {no_lexical}/{len(lost_logs)} ({no_lexical/len(lost_logs)*100:.1f}%)")

    # activity_type 분포
    act_dist: dict[str, int] = defaultdict(int)
    for r in lost_logs:
        act_dist[r["activity"]] += 1
    print("\n[activity_type 분포]")
    for act, cnt in sorted(act_dist.items(), key=lambda x: -x[1]):
        print(f"  {act:<16}  {cnt}개")

    # 샘플 출력 (상위 20개)
    print(f"\n[샘플 — final_score 오름차순 (낮을수록 가장자리)]")
    hdr = (
        f"{'goal_title':<20} {'log_title':<24} {'score':>7}"
        f" {'sem':>6} {'pri':>5} {'ev':>5} {'base':>5} {'pen':>5} {'act'}"
    )
    print(hdr)
    print("-" * 90)
    for r in sorted(lost_logs, key=lambda x: x["final_score"])[:30]:
        print(
            f"{r['goal_title'][:18]:<20} {r['log_title'][:22]:<24}"
            f" {r['final_score']:>7.4f}"
            f" {r['sem']:>6.4f} {r['pri']:>5.3f} {r['ev']:>5.3f}"
            f" {r['base']:>5.3f} {r['penalty']:>5.3f} {r['activity']}"
        )


if __name__ == "__main__":
    main()
