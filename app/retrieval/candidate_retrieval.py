"""Candidate Retrieval entry point.

Dispatches to Dense / Hybrid based on mode and stage configuration.

When an ExpandedQuery is provided, applies vocabulary-based score adjustment
post-hoc on the RRF results before returning candidates:
  - priority_terms  → strong positive boost
  - evidence_terms  → normal positive boost
  - related_terms   → weak positive boost
  - negative_terms  → penalty

This pre-reranker signal ensures high-vocabulary-match candidates surface
to the top before the full Goal-Conditioned Reranker runs.
"""
from __future__ import annotations

import logging
import re
from enum import Enum

from app.config import RetrievalConfig, VocabularyBoostConfig
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


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[\w가-힣]+", text.lower())


def _token_set(text: str) -> set[str]:
    return set(_tokenize(text))


def _apply_vocab_boost(
    candidates: list[CandidateLog],
    expanded: ExpandedQuery,
    boost_cfg: VocabularyBoostConfig,
) -> list[CandidateLog]:
    """Adjust hybrid_score using vocabulary match signals.

    Phrase match (multi-token term found as substring) counts heavier
    than single-token match (phrase_match_multiplier).
    """
    result = []
    pm = boost_cfg.phrase_match_multiplier

    for cand in candidates:
        text = cand.log.full_text
        text_lower = text.lower()
        tokens = _token_set(text_lower)
        title_lower = cand.log.title.lower()
        title_tokens = _token_set(title_lower)

        boost = 0.0
        reasons: list[str] = []

        # Priority terms — strong boost; extra bonus if matched in title
        for term in expanded.priority_terms:
            t_lower = term.lower()
            t_tokens = _tokenize(t_lower)
            if len(t_tokens) > 1 and t_lower in text_lower:
                b = boost_cfg.priority_term_boost * pm
                boost += b
                reasons.append(f"+pri_phrase:{term}(+{b:.3f})")
            elif t_tokens and any(t in tokens for t in t_tokens):
                b = boost_cfg.priority_term_boost
                boost += b
                # Extra bonus if match is in title (title is high-precision)
                if any(t in title_tokens for t in t_tokens):
                    boost += b * 0.5
                    reasons.append(f"+pri_title:{term}(+{b*1.5:.3f})")
                else:
                    reasons.append(f"+pri:{term}(+{b:.3f})")

        # Evidence terms — normal boost
        for term in expanded.expanded_terms:
            t_lower = term.lower()
            t_tokens = _tokenize(t_lower)
            if len(t_tokens) > 1 and t_lower in text_lower:
                b = boost_cfg.evidence_term_boost * pm
                boost += b
                reasons.append(f"+ev_phrase:{term}(+{b:.3f})")
            elif t_tokens and any(t in tokens for t in t_tokens):
                b = boost_cfg.evidence_term_boost
                boost += b

        # Related terms — weak boost
        for term in expanded.related_terms:
            t_lower = term.lower()
            t_tokens = _tokenize(t_lower)
            if t_tokens and any(t in tokens for t in t_tokens):
                boost += boost_cfg.related_term_boost

        # Negative terms — penalty; phrase match is stronger signal
        for term in expanded.negative_terms:
            t_lower = term.lower()
            t_tokens = _tokenize(t_lower)
            if len(t_tokens) > 1 and t_lower in text_lower:
                p = boost_cfg.negative_term_penalty * pm
                boost -= p
                reasons.append(f"-neg_phrase:{term}(-{p:.3f})")
            elif t_tokens and any(t in tokens for t in t_tokens):
                p = boost_cfg.negative_term_penalty
                boost -= p
                reasons.append(f"-neg:{term}(-{p:.3f})")

        new_score = max(0.0, round(cand.hybrid_score + boost, 6))

        if reasons:
            logger.debug(
                "VocabBoost  log=%s  boost=%.4f  score: %.6f→%.6f  reasons=%s",
                cand.log_id, boost, cand.hybrid_score, new_score, reasons,
            )

        result.append(CandidateLog(
            log=cand.log,
            sparse_score=cand.sparse_score,
            dense_score=cand.dense_score,
            hybrid_score=new_score,
        ))

    result.sort(key=lambda x: x.hybrid_score, reverse=True)
    return result


class CandidateRetriever:
    """Unified retrieval entry point."""

    def __init__(
        self,
        mode: RetrievalMode = RetrievalMode.HYBRID,
        config: RetrievalConfig | None = None,
        vocab_boost_config: VocabularyBoostConfig | None = None,
    ) -> None:
        self.mode = mode
        self.config = config or RetrievalConfig()
        self.vocab_boost_config = vocab_boost_config or VocabularyBoostConfig()
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
            candidates = self._dense.retrieve(query_text, top_n=n)
        else:
            candidates = self._hybrid.retrieve(query_text, top_n=n)

        # Apply vocabulary boost when ExpandedQuery with vocabulary is provided
        if (
            isinstance(query, ExpandedQuery)
            and self.vocab_boost_config
            and (query.priority_terms or query.expanded_terms or query.negative_terms)
        ):
            logger.debug(
                "Applying vocab boost  priority=%d  evidence=%d  related=%d  negative=%d",
                len(query.priority_terms), len(query.expanded_terms),
                len(query.related_terms), len(query.negative_terms),
            )
            candidates = _apply_vocab_boost(candidates, query, self.vocab_boost_config)

        return candidates
