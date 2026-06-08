#!/usr/bin/env python3
"""노이즈 영향 검증용 미니 LLM 실험 (Full vs RAG).

irrelevant 로그 비중이 높은 goal을 선택하여
  - Full : 사용자 전체 로그(노이즈 다수 포함) → LLM
  - RAG  : embedding 필터(avg_sim >= threshold) → LLM
두 분석 결과를 비교해 "노이즈가 섞이면 분석 품질이 저하되는지" 검증합니다.
일일 API 호출 한도(20회)를 고려해 goal 수를 최소(기본 2개)로 제한합니다.

Usage:
    LD_PRELOAD=... .venv/bin/python scripts/run_noise_test.py
    LD_PRELOAD=... .venv/bin/python scripts/run_noise_test.py --goal-ids G-U0014-02 G-U0009-02
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

import numpy as np

from app.pipeline.similarity import compute_similarities
from app.pipeline.llm_analysis import analyze, DEFAULT_MODEL

_DATA = ROOT / "data" / "synthetic_v2"

# noise=1.0은 relevant 로그가 0개라 RAG 비교 자체가 불가능하므로 제외
DEFAULT_GOAL_IDS = ["G-U0014-02", "G-U0009-02"]   # noise=0.8, 0.9 (각 user 로그 수가 적어 토큰 절약)
DEFAULT_THRESHOLD = 0.5439


def load_data() -> tuple[dict, dict, dict]:
    goals = json.loads((_DATA / "goals.json").read_text())
    logs  = json.loads((_DATA / "logs.json").read_text())

    goal_map  = {g["goal_id"]: g for g in goals}
    log_map   = {l["log_id"]:  l for l in logs}
    user_logs: dict[str, list[dict]] = {}
    for l in logs:
        user_logs.setdefault(l["user_id"], []).append(l)

    return goal_map, log_map, user_logs


def run(args) -> None:
    out_dir = ROOT / "results" / "noise_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "noise_test_results.json"

    goal_map, log_map, user_logs = load_data()
    goal_sims = compute_similarities()
    sim_map   = {gs.goal_id: gs for gs in goal_sims}

    selected = []
    for gid in args.goal_ids:
        g = goal_map.get(gid)
        if g is None:
            log.warning("goal_id %s 를 찾을 수 없음 — 건너뜀", gid)
            continue
        selected.append(g)

    log.info("총 %d개 goal × 2모드(Full/RAG) = %d회 LLM 호출 예정 (모델: %s)",
             len(selected), len(selected) * 2, args.model)

    if ckpt_path.exists():
        results = json.loads(ckpt_path.read_text())
        done_ids = {r["goal_id"] for r in results}
        log.info("체크포인트 로드: %d건 복원, 완료 goal=%d개", len(results), len(done_ids))
    else:
        results = []
        done_ids = set()

    for i, goal in enumerate(selected):
        gid = goal["goal_id"]
        uid = goal["user_id"]
        nr  = round(goal.get("noise_ratio", 0.0), 1)
        all_logs = user_logs.get(uid, [])

        if gid in done_ids:
            log.info("[%d/%d] %s noise=%.1f → 이미 완료, 건너뜀", i+1, len(selected), gid, nr)
            continue

        log.info("[%d/%d] %s  user=%s  noise=%.1f  전체로그=%d",
                 i+1, len(selected), gid, uid, nr, len(all_logs))

        # ── Full: 노이즈 포함 전체 로그 ──────────────────────────────────────
        log.info("  → Full 방식 실행 (노이즈 포함 전체 %d개 로그 전달)", len(all_logs))
        full_result = analyze(
            goal=goal, logs=all_logs,
            mode="full", n_logs_total=len(all_logs),
            model=args.model,
        )
        log.info("  Full: logs=%d  tokens_in=%d  tokens_out=%d",
                 full_result.n_logs_sent, full_result.input_tokens, full_result.output_tokens)

        # ── RAG: 임베딩 필터링된 relevant 로그만 ─────────────────────────────
        gs = sim_map.get(gid)
        if gs is not None:
            rag_logs = [
                log_map[lid]
                for j, lid in enumerate(gs.log_ids)
                if float(gs.scores["avg_sim"][j]) >= args.threshold
                and lid in log_map
            ]
        else:
            rag_logs = []

        if not rag_logs:
            log.warning("  RAG: 필터 후 로그 0개 → 상위 10개 사용")
            if gs is not None:
                top_idx = np.argsort(gs.scores["avg_sim"])[::-1][:10]
                rag_logs = [log_map[gs.log_ids[j]] for j in top_idx if gs.log_ids[j] in log_map]

        log.info("  → RAG 방식 실행 (필터된 로그: %d개 / 전체 %d개)", len(rag_logs), len(all_logs))
        rag_result = analyze(
            goal=goal, logs=rag_logs,
            mode="rag", n_logs_total=len(all_logs),
            model=args.model,
        )
        log.info("  RAG:  logs=%d  tokens_in=%d  tokens_out=%d",
                 rag_result.n_logs_sent, rag_result.input_tokens, rag_result.output_tokens)

        results.extend([asdict(full_result), asdict(rag_result)])
        ckpt_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
        log.info("  체크포인트 저장 (%d건) → %s", len(results), ckpt_path)

    log.info("전체 완료: %s", ckpt_path)
    _print_comparison(results)


def _print_comparison(results: list[dict]) -> None:
    print("\n" + "="*70)
    print("  노이즈 영향 검증 — Full vs RAG 분석 비교")
    print("="*70)

    by_goal: dict[str, dict] = {}
    for r in results:
        by_goal.setdefault(r["goal_id"], {})[r["mode"]] = r

    for gid, modes in by_goal.items():
        full = modes.get("full")
        rag  = modes.get("rag")
        if not full or not rag:
            continue
        print(f"\n[{gid}]  user={full['user_id']}  noise_ratio={full['noise_ratio']:.1f}")
        print(f"  전체 로그: {full['n_logs_total']}개")
        print(f"  Full → 전달 로그: {full['n_logs_sent']}개  "
              f"input_tokens={full['input_tokens']}  output_tokens={full['output_tokens']}")
        print(f"  RAG  → 전달 로그: {rag['n_logs_sent']}개  "
              f"input_tokens={rag['input_tokens']}  output_tokens={rag['output_tokens']}")
        print(f"\n  ── Full 분석 결과 ──")
        print(f"  {full['analysis']}")
        print(f"\n  ── RAG 분석 결과 ──")
        print(f"  {rag['analysis']}")
        print("-"*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--goal-ids", nargs="+", default=DEFAULT_GOAL_IDS)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    run(parser.parse_args())
