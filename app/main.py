"""Research app entry point — quick smoke test of both pipelines."""
from __future__ import annotations
from app.utils.logging_utils import setup_logging

setup_logging()

from app.config import DEFAULT_CONFIG
from app.pipeline.stage1_ranking_pipeline import Stage1Pipeline
from app.pipeline.stage2_rag_pipeline import Stage2Pipeline
from app.schemas import GoalLogLabel, ResearchGoal, ResearchLog


def _mock_data() -> tuple[ResearchGoal, list[ResearchLog], list[GoalLogLabel]]:
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


if __name__ == "__main__":
    goal, logs, _ = _mock_data()
    cfg = DEFAULT_CONFIG

    print("=== Stage 1 smoke test ===")
    p1 = Stage1Pipeline(config=cfg.stage1)
    p1.index(logs)
    r1 = p1.run(goal)
    print(f"Selected {len(r1.selected_logs)} logs")
    for r in r1.selected_logs:
        print(f"  [{r.rank}] {r.log.title}  score={r.final_score:.4f}")

    print("\n=== Stage 2 smoke test ===")
    p2 = Stage2Pipeline(config=cfg.stage2, use_mock_llm=True)
    p2.index(logs)
    r2 = p2.run(goal)
    print(f"Evidence units: {len(r2.evidence_units)}")
    print(r2.llm_analysis)
