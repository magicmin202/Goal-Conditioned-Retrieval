"""Candidate Retrieval entry point.

Dispatches to Dense / Hybrid based on mode and stage configuration.
"""
from __future__ import annotations
import logging
from enum import Enum
from app.config import RetrievalConfig
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.query_understanding import QueryObject
from app.retrieval.query_expansion import ExpandedQuery
from app.schemas import CandidateLog, ResearchLog

logger = logging.getLogger(__name__)


class RetrievalMode(str, Enum):
    DENSE = "dense"
    HYBRID = "hybrid"
    HYBRID_EXPANDED = "hybrid_expanded"


class CandidateRetriever:
    """Unified retrieval entry point."""

    def __init__(
        self,
        mode: RetrievalMode = RetrievalMode.HYBRID,
        config: RetrievalConfig | None = None,
    ) -> None:
        self.mode = mode
        self.config = config or RetrievalConfig()
        self._hybrid = HybridRetriever(self.config)
        self._dense = DenseRetriever()
        self._indexed = False

    def index(self, logs: list[ResearchLog]) -> None:
        self._hybrid.index(logs)
        self._dense.index(logs)
        self._indexed = True
        logger.info("CandidateRetriever indexed %d logs [mode=%s]", len(logs), self.mode)

    def retrieve(
        self,
        query: QueryObject | ExpandedQuery,
        top_n: int | None = None,
    ) -> list[CandidateLog]:
        if not self._indexed:
            raise RuntimeError("Call index() before retrieve().")
        n = top_n or self.config.candidate_size
        query_text = (
            query.full_text if isinstance(query, ExpandedQuery) else query.canonical_text
        )
        if self.mode == RetrievalMode.DENSE:
            return self._dense.retrieve(query_text, top_n=n)
        return self._hybrid.retrieve(query_text, top_n=n)
