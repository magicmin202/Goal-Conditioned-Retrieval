"""사용자별 Goal-Log 코사인 유사도 계산 및 6가지 스케일링."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_CACHE = Path(__file__).resolve().parent.parent.parent / ".cache" / "embeddings"

SCALING_METHODS = [
    "max_sim",
    "avg_sim",
    "max_scale",
    "minmax_scale",
    "zscore_scale",
    "raw_long",
]


@dataclass
class GoalSim:
    """단일 goal에 대한 유사도 결과."""
    goal_id: str
    user_id: str
    log_ids: list[str]
    sim_long:  np.ndarray   # (n_logs,)
    sim_mid:   np.ndarray
    sim_short: np.ndarray
    scores: dict[str, np.ndarray] = field(default_factory=dict)  # method → (n_logs,)


def _cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (dim,)  b: (n, dim) → (n,) cosine similarity."""
    a_norm = a / (np.linalg.norm(a) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return (b_norm @ a_norm).astype(np.float32)


def _max_scale(v: np.ndarray) -> np.ndarray:
    m = v.max()
    return v / m if m != 0 else v


def _minmax_scale(v: np.ndarray) -> np.ndarray:
    lo, hi = v.min(), v.max()
    return (v - lo) / (hi - lo) if hi != lo else np.zeros_like(v)


def _zscore_scale(v: np.ndarray) -> np.ndarray:
    mu, std = v.mean(), v.std()
    return (v - mu) / std if std != 0 else np.zeros_like(v)


def _user_id_from(id_str: str) -> str:
    m = re.match(r"[GL]-?(U\d+)", id_str)
    return m.group(1) if m else id_str.split("-")[1]


def compute_similarities() -> list[GoalSim]:
    """캐시에서 임베딩을 로드하고 사용자별로 goal ↔ 자기 logs 유사도를 계산."""
    index = json.loads((_CACHE / "index.json").read_text(encoding="utf-8"))
    goal_ids = index["goal_ids"]
    log_ids  = index["log_ids"]

    long_embs  = np.load(_CACHE / "goals_long.npy")   # (n_goals, dim)
    mid_embs   = np.load(_CACHE / "goals_mid.npy")
    short_embs = np.load(_CACHE / "goals_short.npy")
    log_embs   = np.load(_CACHE / "logs.npy")          # (n_logs, dim)

    # 사용자별 log 인덱스 그룹화
    user_log_idx: dict[str, list[int]] = {}
    for j, lid in enumerate(log_ids):
        uid = _user_id_from(lid)
        user_log_idx.setdefault(uid, []).append(j)

    results: list[GoalSim] = []

    for i, gid in enumerate(goal_ids):
        uid = _user_id_from(gid)
        log_indices = user_log_idx.get(uid, [])
        if not log_indices:
            continue

        user_log_embs = log_embs[log_indices]   # (n_user_logs, dim)
        user_log_ids  = [log_ids[j] for j in log_indices]

        sl = _cosine_rows(long_embs[i],  user_log_embs)
        sm = _cosine_rows(mid_embs[i],   user_log_embs)
        ss = _cosine_rows(short_embs[i], user_log_embs)

        stack = np.stack([sl, sm, ss], axis=0)  # (3, n_user_logs)
        max_sim  = stack.max(axis=0)
        avg_sim  = stack.mean(axis=0)

        scores = {
            "max_sim":      max_sim,
            "avg_sim":      avg_sim,
            "max_scale":    _max_scale(max_sim.copy()),
            "minmax_scale": _minmax_scale(max_sim.copy()),
            "zscore_scale": _zscore_scale(max_sim.copy()),
            "raw_long":     sl,
        }

        results.append(GoalSim(
            goal_id=gid,
            user_id=uid,
            log_ids=user_log_ids,
            sim_long=sl,
            sim_mid=sm,
            sim_short=ss,
            scores=scores,
        ))

    logger.info("유사도 계산 완료: %d goals", len(results))
    return results
