"""Goal(long/mid/short) 및 Log 텍스트를 임베딩하고 .npy 캐시에 저장."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from app.retrieval.embedding_provider import get_embedding_provider

logger = logging.getLogger(__name__)

_CACHE = Path(__file__).resolve().parent.parent.parent / ".cache" / "embeddings"
_DATA  = Path(__file__).resolve().parent.parent.parent / "data" / "synthetic_v2"


def _load_json(name: str) -> list[dict]:
    return json.loads((_DATA / f"{name}.json").read_text(encoding="utf-8"))


def _embed_batch(texts: list[str], provider, batch_size: int = 200) -> np.ndarray:
    vecs = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        vecs.extend(provider.encode_batch(chunk))
        logger.info("  임베딩 %d / %d", min(i + batch_size, len(texts)), len(texts))
    return np.array(vecs, dtype=np.float32)


def run_embed(real: bool = False) -> None:
    """goal(long/mid/short)과 log를 임베딩하여 .cache/embeddings/에 저장."""
    _CACHE.mkdir(parents=True, exist_ok=True)

    provider = get_embedding_provider(real=real)
    logger.info("임베딩 프로바이더: %s", type(provider).__name__)

    goals = _load_json("goals")
    logs  = _load_json("logs")

    goal_ids = [g["goal_id"] for g in goals]
    log_ids  = [l["log_id"]  for l in logs]

    logger.info("goal %d개 임베딩 시작…", len(goals))
    long_embs  = _embed_batch([g["long_term"]  for g in goals], provider)
    mid_embs   = _embed_batch([g["mid_term"]   for g in goals], provider)
    short_embs = _embed_batch([g["short_term"] for g in goals], provider)

    logger.info("log %d개 임베딩 시작…", len(logs))
    log_texts = []
    for l in logs:
        topic = (l.get("metadata") or {}).get("topic", "")
        parts = [f"title: {l['title']}"]
        if l.get("activity_type") and l["activity_type"] != "unknown":
            parts.append(f"activity_type: {l['activity_type']}")
        if topic:
            parts.append(f"topic: {topic}")
        if l.get("content"):
            parts.append(f"content: {l['content']}")
        log_texts.append("\n".join(parts))
    log_embs = _embed_batch(log_texts, provider)

    np.save(_CACHE / "goals_long.npy",  long_embs)
    np.save(_CACHE / "goals_mid.npy",   mid_embs)
    np.save(_CACHE / "goals_short.npy", short_embs)
    np.save(_CACHE / "logs.npy",        log_embs)

    index = {"goal_ids": goal_ids, "log_ids": log_ids}
    (_CACHE / "index.json").write_text(
        json.dumps(index, ensure_ascii=False), encoding="utf-8"
    )

    logger.info("캐시 저장 완료: %s", _CACHE)
    logger.info("  goals_long:  %s", long_embs.shape)
    logger.info("  goals_mid:   %s", mid_embs.shape)
    logger.info("  goals_short: %s", short_embs.shape)
    logger.info("  logs:        %s", log_embs.shape)
