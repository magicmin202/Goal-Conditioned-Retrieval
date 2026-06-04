"""HarnessAnalyzer — Stage 1 selected_logs를 LLM에 직접 전달 + 캐시 관리.

흐름:
  1. HarnessStore에서 (user_id, goal_id) 캐시 로드
  2. new_logs = selected_logs - already_analyzed_log_ids
  3. 첫 실행: 전체 로그로 초기 분석 프롬프트
     이후 실행: prev_summary + new_logs로 업데이트 프롬프트
  4. LLM 호출
  5. 캐시 업데이트 저장
  6. 분석 텍스트 반환
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from app.harness.prompts import build_initial_prompt, build_update_prompt
from app.harness.store import HarnessEntry, HarnessStore
from app.llm.llm_client import get_llm_client
from app.schemas import ResearchGoal, ResearchLog

logger = logging.getLogger(__name__)


@dataclass
class HarnessResult:
    goal_id:        str
    user_id:        str
    analysis:       str
    new_log_count:  int
    total_log_count: int
    is_first_run:   bool
    cached_log_ids: set[str] = field(default_factory=set)


class HarnessAnalyzer:
    """Stage 1 결과를 LLM에 직접 전달하고 하네스를 관리합니다."""

    def __init__(
        self,
        use_mock_llm: bool = False,
        harness_dir: str | Path = ".harness",
    ) -> None:
        self._llm   = get_llm_client(mock=use_mock_llm)
        self._store = HarnessStore(harness_dir=harness_dir)

    def analyze(
        self,
        goal: ResearchGoal,
        selected_logs: list[ResearchLog],
    ) -> HarnessResult:
        """selected_logs를 분석하고 하네스를 업데이트합니다.

        Parameters
        ----------
        goal:
            분석 대상 목표.
        selected_logs:
            Stage 1 relevance filtering 통과 로그 (title/content/time 포함).
        """
        entry = self._store.load(goal.user_id, goal.goal_id)
        is_first = entry.is_empty()

        # 신규 로그 = 전체 - 이미 분석된 것
        current_ids = {log.log_id for log in selected_logs}
        new_ids     = current_ids - entry.analyzed_log_ids
        new_logs    = [l for l in selected_logs if l.log_id in new_ids]

        logger.info(
            "HarnessAnalyzer  goal=%s  total=%d  cached=%d  new=%d  first=%s",
            goal.goal_id, len(selected_logs),
            len(entry.analyzed_log_ids), len(new_logs), is_first,
        )

        # 신규 로그가 없으면 이전 분석 그대로 반환
        if not new_logs and not is_first:
            logger.info(
                "HarnessAnalyzer: no new logs — returning cached analysis  goal=%s",
                goal.goal_id,
            )
            return HarnessResult(
                goal_id         = goal.goal_id,
                user_id         = goal.user_id,
                analysis        = entry.analysis_summary,
                new_log_count   = 0,
                total_log_count = entry.log_count,
                is_first_run    = False,
                cached_log_ids  = entry.analyzed_log_ids,
            )

        # 프롬프트 구성 및 LLM 호출
        if is_first:
            prompt = build_initial_prompt(goal, selected_logs)
            logs_for_cache = selected_logs
        else:
            prompt = build_update_prompt(goal, entry.analysis_summary, new_logs)
            logs_for_cache = new_logs

        logger.info(
            "HarnessAnalyzer: LLM call  goal=%s  prompt_len=%d  logs=%d",
            goal.goal_id, len(prompt), len(logs_for_cache),
        )
        analysis = self._llm.generate(prompt)

        # 캐시 업데이트
        self._store.update(
            entry,
            new_log_ids      = [l.log_id for l in logs_for_cache],
            analysis_summary = analysis,
        )

        return HarnessResult(
            goal_id         = goal.goal_id,
            user_id         = goal.user_id,
            analysis        = analysis,
            new_log_count   = len(new_logs),
            total_log_count = len(entry.analyzed_log_ids),
            is_first_run    = is_first,
            cached_log_ids  = entry.analyzed_log_ids,
        )

    def clear_cache(self, user_id: str, goal_id: str) -> None:
        """특정 goal의 하네스 초기화."""
        self._store.clear(user_id, goal_id)
        logger.info("HarnessAnalyzer: cache cleared  goal=%s", goal_id)
