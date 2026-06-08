#!/usr/bin/env python3
"""Full vs RAG LLM 분석 벤치마크.

noise_ratio별로 대표 goal을 선택하여
  - Full : 사용자 전체 로그 → LLM
  - RAG  : embedding 필터(avg_sim >= threshold) → LLM
두 방식을 비교합니다.

Usage:
    LD_PRELOAD=... .venv/bin/python scripts/run_benchmark.py
    LD_PRELOAD=... .venv/bin/python scripts/run_benchmark.py --threshold 0.544 --n-per-noise 2
"""
from __future__ import annotations

import argparse
import json
import logging
import random
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

# noise=1.0은 relevant 없으므로 제외
EVAL_NOISE_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
DEFAULT_THRESHOLD = 0.5439


def load_data() -> tuple[dict, dict, dict]:
    """goal_id→goal, log_id→log, user_id→[logs] 반환."""
    goals = json.loads((_DATA / "goals.json").read_text())
    logs  = json.loads((_DATA / "logs.json").read_text())

    goal_map  = {g["goal_id"]: g for g in goals}
    log_map   = {l["log_id"]:  l for l in logs}
    user_logs: dict[str, list[dict]] = {}
    for l in logs:
        user_logs.setdefault(l["user_id"], []).append(l)

    return goal_map, log_map, user_logs


def select_goals(goal_map: dict, n_per_noise: int, seed: int = 42) -> list[dict]:
    """noise_ratio별로 n_per_noise개 goal을 랜덤 선택."""
    rng = random.Random(seed)
    selected = []
    by_noise: dict[float, list] = {}
    for g in goal_map.values():
        nr = round(g.get("noise_ratio", 0.0), 1)
        by_noise.setdefault(nr, []).append(g)

    for nr in EVAL_NOISE_LEVELS:
        pool = by_noise.get(nr, [])
        chosen = rng.sample(pool, min(n_per_noise, len(pool)))
        selected.extend(chosen)
        log.info("noise=%.1f: %d goals 선택 (pool=%d)", nr, len(chosen), len(pool))

    return selected


def run(args) -> None:
    out_dir = ROOT / "results" / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)

    goal_map, log_map, user_logs = load_data()
    goal_sims = compute_similarities()

    # goal_id → GoalSim 매핑
    sim_map = {gs.goal_id: gs for gs in goal_sims}

    selected = select_goals(goal_map, n_per_noise=args.n_per_noise)
    log.info("총 %d개 goal 벤치마크 시작 (모델: %s)", len(selected), args.model)

    # 체크포인트: 이미 완료된 goal_id 로드
    ckpt_path = out_dir / "benchmark_results.json"
    if ckpt_path.exists():
        results = json.loads(ckpt_path.read_text())
        done_ids = {r["goal_id"] for r in results}
        log.info("체크포인트 로드: %d개 결과 복원, 완료 goal=%d개", len(results), len(done_ids))
    else:
        results = []
        done_ids = set()

    for i, goal in enumerate(selected):
        gid  = goal["goal_id"]
        uid  = goal["user_id"]
        nr   = round(goal.get("noise_ratio", 0.0), 1)
        all_logs = user_logs.get(uid, [])

        if gid in done_ids:
            log.info("[%d/%d] %s  noise=%.1f  → 이미 완료, 건너뜀",
                     i+1, len(selected), gid, nr)
            continue

        log.info("[%d/%d] %s  noise=%.1f  전체로그=%d",
                 i+1, len(selected), gid, nr, len(all_logs))

        # ── Full 방식 ─────────────────────────────────────────────────────
        log.info("  → Full 방식 실행")
        full_result = analyze(
            goal=goal, logs=all_logs,
            mode="full", n_logs_total=len(all_logs),
            model=args.model,
        )
        log.info("  Full: logs=%d  tokens_in=%d  tokens_out=%d",
                 full_result.n_logs_sent, full_result.input_tokens, full_result.output_tokens)

        # ── RAG 방식 ─────────────────────────────────────────────────────
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

        log.info("  → RAG 방식 실행 (필터된 로그: %d개)", len(rag_logs))
        rag_result = analyze(
            goal=goal, logs=rag_logs,
            mode="rag", n_logs_total=len(all_logs),
            model=args.model,
        )
        log.info("  RAG:  logs=%d  tokens_in=%d  tokens_out=%d",
                 rag_result.n_logs_sent, rag_result.input_tokens, rag_result.output_tokens)

        results.extend([asdict(full_result), asdict(rag_result)])

        # 매 goal 완료마다 저장 (체크포인트)
        ckpt_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
        log.info("  체크포인트 저장 (%d건)", len(results))

    log.info("전체 완료: %s", ckpt_path)

    # 요약 출력
    _print_summary(results)


def _print_summary(results: list[dict]) -> None:
    full_res = [r for r in results if r["mode"] == "full"]
    rag_res  = [r for r in results if r["mode"] == "rag"]

    print("\n" + "="*65)
    print("  벤치마크 요약")
    print("="*65)
    print(f"  평가 goal 수: {len(full_res)}")
    print()

    by_noise: dict[float, dict] = {}
    for f, r in zip(full_res, rag_res):
        nr = f["noise_ratio"]
        by_noise.setdefault(nr, {"full_tokens": [], "rag_tokens": [],
                                  "full_logs": [],   "rag_logs": []})
        by_noise[nr]["full_tokens"].append(f["input_tokens"])
        by_noise[nr]["rag_tokens"].append(r["input_tokens"])
        by_noise[nr]["full_logs"].append(f["n_logs_sent"])
        by_noise[nr]["rag_logs"].append(r["n_logs_sent"])

    print(f"  {'noise':>6}  {'full_logs':>10} {'rag_logs':>9} {'full_tok':>10} {'rag_tok':>9} {'tok_save%':>10}")
    print("  " + "─"*58)
    for nr in sorted(by_noise):
        d = by_noise[nr]
        fl = sum(d["full_logs"])   / len(d["full_logs"])
        rl = sum(d["rag_logs"])    / len(d["rag_logs"])
        ft = sum(d["full_tokens"]) / len(d["full_tokens"])
        rt = sum(d["rag_tokens"])  / len(d["rag_tokens"])
        save = (ft - rt) / ft * 100 if ft > 0 else 0
        print(f"  {nr:>6.1f}  {fl:>10.1f} {rl:>9.1f} {ft:>10.0f} {rt:>9.0f} {save:>9.1f}%")

    total_full = sum(r["input_tokens"] for r in full_res)
    total_rag  = sum(r["input_tokens"] for r in rag_res)
    print("  " + "─"*58)
    print(f"  {'전체':>6}  "
          f"{'─':>10} {'─':>9} "
          f"{total_full:>10,} {total_rag:>9,} "
          f"{(total_full-total_rag)/total_full*100:>9.1f}%")
    print()
    print(f"  토큰 절감: {total_full - total_rag:,} tokens  ({(total_full-total_rag)/total_full*100:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold",   type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--n-per-noise", type=int,   default=2)
    parser.add_argument("--model",       type=str,   default=DEFAULT_MODEL)
    run(parser.parse_args())
