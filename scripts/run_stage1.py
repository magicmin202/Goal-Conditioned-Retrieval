#!/usr/bin/env python3
"""Stage 1 experiment runner.

Usage:
    python scripts/run_stage1.py
    python scripts/run_stage1.py --expand
    python scripts/run_stage1.py --top_k 7
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.utils.logging_utils import setup_logging
setup_logging()

import logging
from app.config import DEFAULT_CONFIG
from app.evaluation.ranking_metrics import compute_all_metrics
from app.pipeline.stage1_ranking_pipeline import Stage1Pipeline
from app.schemas import GoalLogLabel, ResearchGoal, ResearchLog

logger = logging.getLogger(__name__)


def get_mock_data() -> tuple[ResearchGoal, list[ResearchLog], list[GoalLogLabel]]:
    goal = ResearchGoal(
        goal_id="G0001", user_id="U0001",
        title="AI 개발 역량 강화",
        description="수학, 머신러닝, 논문 읽기, 구현 역량을 강화한다.",
        goal_embedding_text="AI 개발 역량 강화 수학 머신러닝 딥러닝 논문 구현",
    )
    logs = [
        ResearchLog("L0001", "U0001", "2026-03-01", "선형대수 개념 복습", "행렬 연산과 고유값 분해를 정리했다.", "study"),
        ResearchLog("L0002", "U0001", "2026-03-02", "확률론 공부", "베이즈 정리와 확률분포를 학습했다.", "study"),
        ResearchLog("L0003", "U0001", "2026-03-03", "PyTorch 튜토리얼", "CNN 구현 실습을 완료했다.", "implementation"),
        ResearchLog("L0004", "U0001", "2026-03-04", "Attention is All You Need 논문", "트랜스포머 구조를 이해하고 노트를 작성했다.", "reading"),
        ResearchLog("L0005", "U0001", "2026-03-05", "ML 모델 실험", "sklearn으로 분류 모델을 비교했다.", "implementation"),
        ResearchLog("L0006", "U0001", "2026-03-06", "점심 식사", "된장찌개를 먹었다.", "daily"),
        ResearchLog("L0007", "U0001", "2026-03-07", "딥러닝 강의 수강", "역전파 알고리즘을 배웠다.", "study"),
        ResearchLog("L0008", "U0001", "2026-03-08", "선형대수 문제 풀이", "고유값 문제 20개를 풀고 오답을 정리했다.", "study"),
        ResearchLog("L0009", "U0001", "2026-03-09", "GAN 논문 리뷰", "생성 모델의 학습 과정을 정리했다.", "reading"),
        ResearchLog("L0010", "U0001", "2026-03-10", "코딩 테스트 준비", "그래프 탐색 문제를 풀었다.", "coding"),
    ]
    labels = [
        GoalLogLabel("GL001", "U0001", "G0001", "L0001", "relevant", 1.0),
        GoalLogLabel("GL002", "U0001", "G0001", "L0002", "relevant", 0.9),
        GoalLogLabel("GL003", "U0001", "G0001", "L0003", "relevant", 1.0),
        GoalLogLabel("GL004", "U0001", "G0001", "L0004", "relevant", 1.0),
        GoalLogLabel("GL005", "U0001", "G0001", "L0005", "relevant", 0.8),
        GoalLogLabel("GL006", "U0001", "G0001", "L0006", "irrelevant", 0.0),
        GoalLogLabel("GL007", "U0001", "G0001", "L0007", "relevant", 1.0),
        GoalLogLabel("GL008", "U0001", "G0001", "L0008", "relevant", 1.0),
        GoalLogLabel("GL009", "U0001", "G0001", "L0009", "relevant", 0.9),
        GoalLogLabel("GL010", "U0001", "G0001", "L0010", "relevant", 0.5),
    ]
    return goal, logs, labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 1 Retrieval Experiment")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--expand", action="store_true", help="Enable query expansion variant")
    args = parser.parse_args()

    goal, logs, labels = get_mock_data()
    cfg = DEFAULT_CONFIG.stage1
    cfg.retrieval.top_k = args.top_k
    cfg.retrieval.candidate_size = len(logs)

    pipeline = Stage1Pipeline(config=cfg)
    pipeline.index(logs)
    result = pipeline.run(goal, use_expansion=args.expand)

    print("\n" + "=" * 60)
    print(f"Stage 1  |  Goal: {result.goal.title}")
    print(f"Query   : {result.query_text}")
    print(f"Expand  : {result.used_expansion}")
    print("=" * 60)

    print(f"\n[Top-{args.top_k} Selected Logs]")
    for r in result.selected_logs:
        print(f"  [{r.rank:2d}] score={r.final_score:.4f}  {r.log.date}  {r.log.title}")

    all_types = {log.activity_type for log in logs}
    metrics = compute_all_metrics(result.selected_logs, labels, k=args.top_k, all_activity_types=all_types)
    print(f"\n[Metrics @ k={args.top_k}]")
    for name, val in metrics.items():
        print(f"  {name}: {val:.4f}")


if __name__ == "__main__":
    main()
