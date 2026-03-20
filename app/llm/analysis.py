"""Goal progress analysis using LLM + compressed evidence."""
from __future__ import annotations
import logging
from app.llm.llm_client import BaseLLMClient, get_llm_client
from app.schemas import CompressedEvidenceUnit, ResearchGoal

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = """\
[Goal]
{goal_title}: {goal_description}

[Evidence Summary]
{evidence_text}

[Task]
1. 이 사용자가 목표를 향해 얼마나 진행했는지 분석하세요.
2. 반복되는 패턴이나 주요 성과를 식별하세요.
3. 다음 단계로 추천할 행동을 제안하세요."""


def build_evidence_text(units: list[CompressedEvidenceUnit]) -> str:
    lines = []
    for unit in units:
        lines.append(
            f"[{unit.unit_id}] {unit.date_range} | {unit.activity_cluster} | logs={unit.log_count}\n"
            f"  {unit.summary}"
        )
        if unit.temporal_progression:
            lines.append(f"  진행: {unit.temporal_progression}")
    return "\n".join(lines)


class GoalAnalyzer:
    def __init__(self, llm: BaseLLMClient | None = None) -> None:
        self.llm = llm or get_llm_client(mock=True)

    def analyze(
        self, goal: ResearchGoal, evidence_units: list[CompressedEvidenceUnit]
    ) -> str:
        prompt = _PROMPT_TEMPLATE.format(
            goal_title=goal.title,
            goal_description=goal.description,
            evidence_text=build_evidence_text(evidence_units),
        )
        logger.info("Running LLM analysis for goal=%s", goal.goal_id)
        return self.llm.generate(prompt)
