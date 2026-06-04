"""HarnessStore — (user_id, goal_id) 단위 분석 캐시 관리.

저장 위치: .harness/{user_id}/{goal_id}.json
저장 항목:
  analyzed_log_ids  이미 LLM에 전달된 로그 ID 집합
  analysis_summary  마지막 LLM 분석 결과
  last_updated      마지막 업데이트 시각 (ISO 8601)
  log_count         누적 분석 로그 수
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_HARNESS_DIR = Path(".harness")


@dataclass
class HarnessEntry:
    goal_id:           str
    user_id:           str
    analyzed_log_ids:  set[str]        = field(default_factory=set)
    analysis_summary:  str             = ""
    last_updated:      str             = ""
    log_count:         int             = 0

    def is_empty(self) -> bool:
        return len(self.analyzed_log_ids) == 0


class HarnessStore:
    """캐시 I/O 담당."""

    def __init__(self, harness_dir: str | Path = _DEFAULT_HARNESS_DIR) -> None:
        self._dir = Path(harness_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, user_id: str, goal_id: str) -> Path:
        user_dir = self._dir / user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir / f"{goal_id}.json"

    def load(self, user_id: str, goal_id: str) -> HarnessEntry:
        """캐시에서 로드. 없으면 빈 entry 반환."""
        path = self._path(user_id, goal_id)
        if not path.exists():
            return HarnessEntry(goal_id=goal_id, user_id=user_id)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return HarnessEntry(
                goal_id          = data.get("goal_id", goal_id),
                user_id          = data.get("user_id", user_id),
                analyzed_log_ids = set(data.get("analyzed_log_ids", [])),
                analysis_summary = data.get("analysis_summary", ""),
                last_updated     = data.get("last_updated", ""),
                log_count        = data.get("log_count", 0),
            )
        except Exception as exc:
            logger.warning("HarnessStore: failed to load %s: %s", path, exc)
            return HarnessEntry(goal_id=goal_id, user_id=user_id)

    def save(self, entry: HarnessEntry) -> None:
        path = self._path(entry.user_id, entry.goal_id)
        data = {
            "goal_id":          entry.goal_id,
            "user_id":          entry.user_id,
            "analyzed_log_ids": sorted(entry.analyzed_log_ids),
            "analysis_summary": entry.analysis_summary,
            "last_updated":     entry.last_updated,
            "log_count":        entry.log_count,
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.debug("HarnessStore: saved %s  logs=%d", path, entry.log_count)

    def update(
        self,
        entry: HarnessEntry,
        new_log_ids: list[str],
        analysis_summary: str,
    ) -> HarnessEntry:
        """새 로그 ID와 분석 결과로 entry를 업데이트하고 저장."""
        entry.analyzed_log_ids.update(new_log_ids)
        entry.analysis_summary = analysis_summary
        entry.last_updated     = datetime.now().isoformat()
        entry.log_count        = len(entry.analyzed_log_ids)
        self.save(entry)
        return entry

    def clear(self, user_id: str, goal_id: str) -> None:
        """특정 goal의 캐시 초기화."""
        path = self._path(user_id, goal_id)
        if path.exists():
            path.unlink()
            logger.info("HarnessStore: cleared %s", path)
