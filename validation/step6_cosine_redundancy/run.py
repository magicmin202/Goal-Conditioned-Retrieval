#!/usr/bin/env python3
"""Step 6: Cosine Similarity 기반 Redundancy 실험
===================================================
Relevance Filtering 이후, 이미 캐싱된 log embedding_text 벡터를 재사용하여
greedy cosine similarity로 중복 로그를 압축한다.

현재 title Jaccard 방식과 비교:
  current:  title token Jaccard overlap >= 0.60 → -0.15 penalty
  proposed: embedding cosine similarity >= threshold → -0.15 penalty

비교 케이스: cosine threshold 0.70 ~ 0.98

평가 지표: recall@selected, precision@selected, F1, n_removed

Usage:
    cd /home/user/retrieval
    .venv/bin/python validation/step6_cosine_redundancy/run.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
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
DATA_DIR    = ROOT / "data" / "synthetic"
CACHE_PATH  = ROOT / ".cache" / "embeddings" / "models_gemini-embedding-001_retrieval_document.json"

TOP_K      = 10
THRESHOLD  = 0.60   # relevance_threshold
CFG        = DEFAULT_CONFIG.stage1.ranker

WEIGHTS = {"pri": 0.10, "ev": 0.06, "rel": 0.03, "sem": 0.70, "base": 0.05}
TOTAL_W = sum(WEIGHTS.values())
SCALE   = 1.0 / TOTAL_W

PHRASE_PEN  = CFG.negative_penalty_phrase
TOKEN_PEN   = CFG.negative_penalty_token
DAILY_NOISE = CFG.negative_daily_penalty

# ── 비교 케이스: cosine threshold 탐색 ──────────────────────────────────────
COSINE_THRESHOLDS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98]

REDUNDANCY_PENALTY = 0.15   # 현재 similar_penalty와 동일


# ── 임베딩 캐시 로드 ──────────────────────────────────────────────────────────

def _text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_embedding_cache() -> dict[str, list[float]]:
    if not CACHE_PATH.exists():
        raise FileNotFoundError(f"Embedding cache not found: {CACHE_PATH}")
    cache = json.loads(CACHE_PATH.read_text())
    print(f"  임베딩 캐시 로드: {len(cache)}개 항목")
    return cache


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x**2 for x in a)) or 1.0
    nb  = math.sqrt(sum(x**2 for x in b)) or 1.0
    return dot / (na * nb)


# ── 사전 계산 ─────────────────────────────────────────────────────────────────

def precompute(
    goals: list[ResearchGoal],
    logs: list[ResearchLog],
    labels: list[GoalLogLabel],
    emb_cache: dict[str, list[float]],
) -> list[dict]:
    """(goal, log, label) 전체에 대해 final_score + embedding 벡터 수집."""
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
    total = sum(len(v) for v in user_goals_map.values())
    done = 0
    missing_emb = 0

    for user_id, u_goals in user_goals_map.items():
        u_logs = user_logs_map[user_id]
        if not u_logs:
            continue
        dense.index(u_logs)

        for goal in u_goals:
            done += 1
            if done % 50 == 0 or done == total:
                print(f"  [{done}/{total}] {goal.goal_id} ...", flush=True)

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

                # 캐싱된 embedding 벡터 로드
                emb_key = _text_key(log.embedding_text)
                emb_vec = emb_cache.get(emb_key)
                if emb_vec is None:
                    missing_emb += 1

                records.append({
                    "goal_id":     lbl.goal_id,
                    "log_id":      lbl.log_id,
                    "label":       lbl.label,
                    "final_score": final_score,
                    "emb":         emb_vec,  # None이면 코사인 계산 불가
                })

    if missing_emb:
        print(f"  ⚠ 임베딩 캐시 미스: {missing_emb}개 (cosine 계산 제외)")
    return records


# ── Redundancy 적용 ───────────────────────────────────────────────────────────

def apply_cosine_redundancy(
    scored: list[tuple[str, str, float, list | None]],
    cosine_threshold: float,
) -> list[tuple[str, str, float]]:
    """greedy cosine redundancy 적용.

    scored: [(log_id, label, final_score, emb_vec), ...]  score 내림차순
    반환:   [(log_id, label, adjusted_score), ...]
    """
    admitted_embs: list[list[float]] = []
    result = []

    for log_id, label, score, emb in scored:
        if emb is None or score < THRESHOLD:
            result.append((log_id, label, score))
            continue

        # 이미 admitted된 로그들과 cosine 비교
        max_sim = max(
            (cosine(emb, prev) for prev in admitted_embs),
            default=0.0,
        )

        if max_sim >= cosine_threshold:
            adjusted = max(0.0, round(score - REDUNDANCY_PENALTY, 4))
        else:
            adjusted = score

        admitted_embs.append(emb)
        result.append((log_id, label, adjusted))

    return result


def apply_title_redundancy(
    scored: list[tuple[str, str, float, list | None]],
    log_map: dict[str, ResearchLog],
    jaccard_threshold: float = 0.60,
) -> list[tuple[str, str, float]]:
    """현재 title Jaccard 방식 (비교 baseline)."""
    admitted_toks: list[frozenset] = []
    admitted_norms: list[str] = []
    result = []

    for log_id, label, score, _ in scored:
        log = log_map.get(log_id)
        if log is None or score < THRESHOLD:
            result.append((log_id, label, score))
            continue

        title_norm = re.sub(r"\s+", "", log.title.lower())
        log_toks   = frozenset(re.findall(r"[\w가-힣]{2,}", log.title.lower()))

        adjusted = score
        for prev_norm, prev_toks in zip(admitted_norms, admitted_toks):
            if title_norm == prev_norm:
                adjusted = max(0.0, round(score - 0.30, 4))
                break
            if log_toks and prev_toks:
                union = log_toks | prev_toks
                sim   = len(log_toks & prev_toks) / len(union) if union else 0.0
                if sim >= jaccard_threshold:
                    adjusted = max(0.0, round(score - 0.15, 4))
                    break

        admitted_norms.append(title_norm)
        admitted_toks.append(log_toks)
        result.append((log_id, label, adjusted))

    return result


# ── 메트릭 계산 ───────────────────────────────────────────────────────────────

def evaluate_with_redundancy(
    records: list[dict],
    log_map: dict[str, ResearchLog],
    mode: str,
    cosine_threshold: float = 0.85,
) -> dict:
    """mode: 'cosine_thr' | 'jaccard' | 'none'"""
    goal_records: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        goal_records[r["goal_id"]].append(r)

    goal_metrics = []
    total_removed = 0

    for goal_id, grecs in goal_records.items():
        total_rel = sum(1 for r in grecs if r["label"] == "relevant")
        if total_rel == 0:
            continue

        # threshold 통과 후 score 내림차순
        base_scored = sorted(
            [(r["log_id"], r["label"], r["final_score"], r["emb"])
             for r in grecs if r["final_score"] >= THRESHOLD],
            key=lambda x: x[2], reverse=True,
        )

        # redundancy 적용
        if mode == "cosine":
            adjusted = apply_cosine_redundancy(base_scored, cosine_threshold)
        elif mode == "jaccard":
            adjusted = apply_title_redundancy(base_scored, log_map)
        else:  # none
            adjusted = [(lid, lbl, s) for lid, lbl, s, _ in base_scored]

        # score > 0인 것만 pool에 올림
        pool = sorted(
            [(lid, lbl, s) for lid, lbl, s in adjusted if s > 0],
            key=lambda x: x[2], reverse=True,
        )
        n_removed = len(base_scored) - len(pool)
        total_removed += n_removed

        selected = pool[:TOP_K]

        tp    = sum(1 for _, lbl, _ in selected if lbl == "relevant")
        sel_n = len(selected)
        prec  = tp / sel_n if sel_n > 0 else 0.0
        rec   = tp / total_rel
        f1    = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        goal_metrics.append({"rec": rec, "prec": prec, "f1": f1, "sel_n": sel_n})

    def mean(k):
        v = [m[k] for m in goal_metrics]
        return round(sum(v) / len(v), 4) if v else 0.0

    return {
        "recall":        mean("rec"),
        "precision":     mean("prec"),
        "f1":            mean("f1"),
        "avg_selected":  mean("sel_n"),
        "total_removed": total_removed,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("[데이터 로드]")
    _, goals, logs, labels = load_dataset_from_json(str(DATA_DIR))
    print(f"  goals={len(goals)}  logs={len(logs)}  labels={len(labels)}")

    print("\n[임베딩 캐시 로드]")
    emb_cache = load_embedding_cache()

    log_map = {l.log_id: l for l in logs}

    print("\n[사전 계산]")
    records = precompute(goals, logs, labels, emb_cache)
    print(f"  처리 완료: {len(records)}개 (goal, log) 쌍")

    print("\n" + "=" * 66)
    print("Cosine Similarity 기반 Redundancy 실험")
    print(f"  relevance_threshold={THRESHOLD}  redundancy_penalty={REDUNDANCY_PENALTY}")
    print("=" * 66)

    rows = []

    # baseline: redundancy 없음
    m = evaluate_with_redundancy(records, log_map, "none")
    rows.append({"mode": "no_redundancy", "threshold": "-", **m})
    print(f"\n  [no_redundancy    ]  recall={m['recall']:.4f}  prec={m['precision']:.4f}"
          f"  F1={m['f1']:.4f}  removed={m['total_removed']}")

    # 현재 방식: title Jaccard
    m = evaluate_with_redundancy(records, log_map, "jaccard")
    rows.append({"mode": "jaccard_current", "threshold": "0.60", **m})
    print(f"  [jaccard (현재)   ]  recall={m['recall']:.4f}  prec={m['precision']:.4f}"
          f"  F1={m['f1']:.4f}  removed={m['total_removed']}")

    # 제안 방식: cosine similarity
    print()
    for thr in COSINE_THRESHOLDS:
        m = evaluate_with_redundancy(records, log_map, "cosine", thr)
        rows.append({"mode": "cosine", "threshold": str(thr), **m})
        print(f"  [cosine thr={thr:.2f}  ]  recall={m['recall']:.4f}  prec={m['precision']:.4f}"
              f"  F1={m['f1']:.4f}  removed={m['total_removed']}")

    # ── 비교 테이블 ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("비교 테이블  (recall 내림차순)")
    print("=" * 70)
    hdr = (
        f"{'mode':<20} {'threshold':>10}"
        f"  {'recall':>8} {'prec':>8} {'F1':>8}"
        f"  {'avg_sel':>8} {'removed':>8}"
    )
    print(hdr)
    print("-" * 70)
    for r in sorted(rows, key=lambda x: x["recall"], reverse=True):
        print(
            f"{r['mode']:<20} {str(r['threshold']):>10}"
            f"  {r['recall']:>8.4f} {r['precision']:>8.4f} {r['f1']:>8.4f}"
            f"  {r['avg_selected']:>8.2f} {r['total_removed']:>8}"
        )

    best_r = max(rows, key=lambda x: x["recall"])
    best_f = max(rows, key=lambda x: x["f1"])
    print(f"\n  Recall 최고:  {best_r['mode']} (thr={best_r['threshold']})  R={best_r['recall']:.4f}")
    print(f"  F1 최고:      {best_f['mode']} (thr={best_f['threshold']})  F1={best_f['f1']:.4f}")

    out = RESULTS_DIR / "cosine_redundancy.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[저장] {out}")


if __name__ == "__main__":
    main()
