#!/usr/bin/env python3
"""Token usage comparison: Pipeline vs Raw (all logs to LLM).

Compares two approaches for a given goal:

  [A] Raw Dump
      LLM receives ALL user logs directly in a single call.
      LLM calls : 1
      Token cost : 1 × (prompt template + all log text)

  [B] Pipeline (Stage1 + Stage2 + LLM Analysis)
      LLM call 1 — Query Expansion  : expansion prompt → JSON vocab
      LLM call 2 — Goal Analysis    : compressed evidence → analysis
      LLM calls : 2
      Token cost : expansion_prompt + expansion_response
                 + analysis_prompt (compressed) + analysis_response

Both sides include output token estimates (responses).

Metrics:
  - LLM calls
  - input tokens per call
  - output tokens per call
  - total tokens (in + out)
  - reduction ratio

Usage:
    python scripts/compare_token_usage.py --goal_id G-U0001-01
    python scripts/compare_token_usage.py --auto
    python scripts/compare_token_usage.py --auto --all_goals   # all goals for user
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

from app.utils.logging_utils import setup_logging
setup_logging(level="WARNING")   # suppress pipeline logs for clean output

from app.config import DEFAULT_CONFIG
from app.data_generation.dataset_builder import build_dataset
from app.data_generation.export_utils import load_dataset_from_json
from app.llm.analysis import _PROMPT_TEMPLATE, build_evidence_text
from app.pipeline.stage1_ranking_pipeline import Stage1Pipeline, Stage1Result
from app.pipeline.stage2_rag_pipeline import Stage2Pipeline
from app.retrieval.query_expansion import _EXPANSION_PROMPT
from app.schemas import GoalLogLabel, ResearchGoal, ResearchLog

_DEFAULT_DATA_DIR = "data/synthetic"
_CHARS_PER_TOKEN  = 3.5   # Korean ~2.5-3 / English ~4 / mixed ~3.5

# Typical output token estimates (actual Gemini responses — conservative)
_EXPANSION_OUTPUT_TOKENS  = 300   # JSON with ~40 terms across 6 fields
_ANALYSIS_OUTPUT_TOKENS   = 400   # 3-part analysis: progress / patterns / next steps


def _tok(text: str) -> int:
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


# ── Prompt builders ───────────────────────────────────────────────────────────

def _raw_prompt(goal: ResearchGoal, logs: list[ResearchLog]) -> str:
    log_lines = [
        f"[{i}] {log.date}  {log.activity_type}  {log.title}\n    {log.content}"
        for i, log in enumerate(logs, 1)
    ]
    return _PROMPT_TEMPLATE.format(
        goal_title=goal.title,
        goal_description=goal.description,
        evidence_text="\n".join(log_lines),
    )


def _expansion_prompt(goal: ResearchGoal) -> str:
    return _EXPANSION_PROMPT.format(
        title=goal.title,
        description=goal.description or goal.title,
    )


def _analysis_prompt(goal: ResearchGoal, evidence_units) -> str:
    return _PROMPT_TEMPLATE.format(
        goal_title=goal.title,
        goal_description=goal.description,
        evidence_text=build_evidence_text(evidence_units),
    )


# ── Main comparison ───────────────────────────────────────────────────────────

def run_comparison(goal: ResearchGoal, user_logs: list[ResearchLog], top_k: int = 5) -> dict:
    # ── Pipeline ──────────────────────────────────────────────────────────────
    cfg1 = DEFAULT_CONFIG.stage1
    cfg1.retrieval.top_k = top_k
    cfg1.retrieval.candidate_size = max(top_k * 3, min(len(user_logs) * 6 // 10, 30))
    cfg1.query_expansion.enabled = True

    s1 = Stage1Pipeline(config=cfg1)
    s1.index(user_logs)
    s1_result: Stage1Result = s1.run(goal, use_expansion=True)

    cfg2 = DEFAULT_CONFIG.stage2
    s2 = Stage2Pipeline(config=cfg2, use_mock_llm=True)
    s2.index(user_logs)
    s2_result = s2.run_with_stage1(s1_result)

    evidence_units = s2_result.evidence_units
    admitted_logs  = s1_result.selected_logs

    # ── [A] Raw: single LLM call with all logs ────────────────────────────────
    raw_in   = _tok(_raw_prompt(goal, user_logs))
    raw_out  = _ANALYSIS_OUTPUT_TOKENS
    raw_total = raw_in + raw_out

    # ── [B] Pipeline: 2 LLM calls ─────────────────────────────────────────────
    # Call 1 — Query Expansion
    exp_prompt   = _expansion_prompt(goal)
    exp_in       = _tok(exp_prompt)
    exp_out      = _EXPANSION_OUTPUT_TOKENS

    # Call 2 — Goal Analysis (compressed evidence)
    analysis_prompt = _analysis_prompt(goal, evidence_units)
    ana_in          = _tok(analysis_prompt)
    ana_out         = _ANALYSIS_OUTPUT_TOKENS

    pipe_total = exp_in + exp_out + ana_in + ana_out

    reduction_pct = (1 - pipe_total / raw_total) * 100 if raw_total else 0

    return {
        "goal_id":      goal.goal_id,
        "goal_title":   goal.title,
        "total_logs":   len(user_logs),
        "admitted_logs": len(admitted_logs),
        "evidence_units": len(evidence_units),
        "admission_rate": round(len(admitted_logs) / len(user_logs) * 100, 1),

        # [A] Raw
        "raw_in":    raw_in,
        "raw_out":   raw_out,
        "raw_total": raw_total,
        "raw_calls": 1,

        # [B] Pipeline breakdown
        "exp_in":    exp_in,
        "exp_out":   exp_out,
        "ana_in":    ana_in,
        "ana_out":   ana_out,
        "pipe_total": pipe_total,
        "pipe_calls": 2,

        # Summary
        "token_savings":       raw_total - pipe_total,
        "reduction_pct":       round(reduction_pct, 1),
    }


def print_report(results: list[dict]) -> None:
    SEP = "─" * 74

    print("\n" + "=" * 74)
    print("  TOKEN USAGE COMPARISON  (입력 + 출력 포함 전체 토큰)")
    print("=" * 74)
    print(f"  추정 기준: 1 token ≈ {_CHARS_PER_TOKEN} chars (한/영 혼용)")
    print(f"  출력 토큰 가정: expansion≈{_EXPANSION_OUTPUT_TOKENS} / analysis≈{_ANALYSIS_OUTPUT_TOKENS} (보수적 추정)")
    print()

    sum_raw = sum_pipe = 0

    for r in results:
        print(SEP)
        print(f"  Goal : {r['goal_title']}  [{r['goal_id']}]")
        print(f"  코퍼스: {r['total_logs']}개 로그  →  admitted {r['admitted_logs']}개 ({r['admission_rate']}%)  →  {r['evidence_units']}개 evidence unit")
        print()
        print(f"  {'':45}  {'[A] Raw':>8}  {'[B] Pipeline':>12}")
        print(f"  {'':─<45}  {'':─>8}  {'':─>12}")
        print(f"  {'LLM 호출 횟수':45}  {'1':>8}  {'2':>12}")
        print()
        print(f"  {'[Call 1]':45}")

        # Raw call 1
        print(f"  {'  분석 프롬프트 (입력)':45}  {r['raw_in']:>8,}  ", end="")
        # Pipeline call 1 = expansion
        print(f"{'Query Expansion (입력)':>12}")
        print(f"  {'':45}  {'':>8}  {r['exp_in']:>10,}  ← expansion prompt")
        print(f"  {'  분석 응답 (출력)':45}  {r['raw_out']:>8,}  {r['exp_out']:>12,}  ← expansion JSON")
        print()
        print(f"  {'[Call 2]':45}  {'—':>8}  {'Goal Analysis':>12}")
        print(f"  {'':45}  {'':>8}  {r['ana_in']:>10,}  ← compressed prompt")
        print(f"  {'':45}  {'':>8}  {r['ana_out']:>12,}  ← analysis 응답")
        print()
        print(f"  {'합계 (입력 + 출력)':45}  {r['raw_total']:>8,}  {r['pipe_total']:>12,}")
        print()
        print(f"  → 토큰 절감  : {r['token_savings']:,} tokens  ({r['reduction_pct']:.1f}% 감소)")

        sum_raw  += r['raw_total']
        sum_pipe += r['pipe_total']

    if len(results) > 1:
        total_red = (1 - sum_pipe / sum_raw) * 100 if sum_raw else 0
        print(SEP)
        print(f"\n  [전체 합산  —  목표 {len(results)}개]")
        hdr = f"  {'':45}  {'[A] Raw':>8}  {'[B] Pipeline':>12}"
        print(hdr)
        print(f"  {'총 토큰 (입력+출력)':45}  {sum_raw:>8,}  {sum_pipe:>12,}")
        print(f"  {'총 절감':45}  {sum_raw - sum_pipe:>8,}  {total_red:>11.1f}%")

    print("=" * 74)
    print()
    print("  [해석]")
    print("  [A] Raw  : 로그 전체 → LLM 1회 호출 (입력 大, 노이즈 多)")
    print("  [B] Pipeline :")
    print("      Call 1 — Query Expansion  : goal → 검색 vocabulary 생성")
    print("      Call 2 — Goal Analysis    : compressed evidence → 목표 분석")
    print()
    print("  Pipeline은 Call 1에서 expansion 비용이 추가되지만,")
    print("  Call 2의 입력이 대폭 줄어 전체 토큰은 절감됩니다.")
    print("  또한 LLM이 받는 컨텍스트 품질(노이즈 제거)도 높아집니다.")
    print()


def load_data(data_dir: str):
    goals_path = Path(data_dir) / "goals.json"
    if goals_path.exists():
        _, goals, logs, labels = load_dataset_from_json(data_dir)
    else:
        ds = build_dataset(small_mode=True, seed=42)
        goals, logs, labels = ds.goals, ds.logs, ds.labels
    return goals, logs, labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Token usage: Pipeline vs Raw Dump")
    parser.add_argument("--goal_id",   default=None)
    parser.add_argument("--user_id",   default=None)
    parser.add_argument("--top_k",     type=int, default=5)
    parser.add_argument("--all_goals", action="store_true",
                        help="Run for ALL goals of the resolved user")
    parser.add_argument("--auto",      action="store_true",
                        help="Auto-pick first user/goal")
    parser.add_argument("--data_dir",  default=_DEFAULT_DATA_DIR)
    args = parser.parse_args()

    goals, logs, labels = load_data(args.data_dir)

    # Resolve target goal(s)
    if args.goal_id:
        target_goal = next((g for g in goals if g.goal_id == args.goal_id), None)
        if not target_goal:
            print(f"goal_id={args.goal_id} not found."); sys.exit(1)
        user_id = target_goal.user_id
        target_goals = [g for g in goals if g.user_id == user_id] if args.all_goals else [target_goal]
    elif args.user_id:
        user_id = args.user_id
        target_goals = [g for g in goals if g.user_id == user_id]
    else:  # --auto or default
        target_goal = goals[0]
        user_id = target_goal.user_id
        target_goals = [g for g in goals if g.user_id == user_id] if args.all_goals else [target_goal]

    user_logs = [l for l in logs if l.user_id == user_id]
    print(f"\n  User: {user_id}  |  Logs: {len(user_logs)}  |  Goals to compare: {len(target_goals)}")

    results = []
    for goal in target_goals:
        print(f"  Running pipeline for: {goal.goal_id} ({goal.title}) ...", end=" ", flush=True)
        try:
            r = run_comparison(goal, user_logs, top_k=args.top_k)
            results.append(r)
            print("done")
        except Exception as e:
            print(f"ERROR: {e}")

    if results:
        print_report(results)


if __name__ == "__main__":
    main()
