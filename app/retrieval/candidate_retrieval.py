"""Candidate Retrieval entry point.

Dispatches to Dense / Hybrid based on mode and stage configuration.

When an ExpandedQuery is provided, applies vocabulary-based score adjustment
post-hoc on the RRF results before returning candidates:

  Candidate stage = moderate boosts (recall-focused, keeps all relevant logs)
  Reranker stage  = strong scoring (precision-focused, see reranker.py)

Vocabulary tiers:
  priority_terms  → strongest positive boost (phrase > token, title bonus)
  evidence_terms  → normal positive boost
  related_terms   → weak positive boost
  negative_terms  → penalty (phrase > token)
"""
from __future__ import annotations

import logging
from enum import Enum

from app.config import RetrievalConfig, VocabularyBoostConfig
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.query_understanding import QueryObject
from app.retrieval.query_expansion import ExpandedQuery
from app.schemas import CandidateLog, ResearchLog
from app.utils.text_matching import TermMatch, match_term, _tok_set

logger = logging.getLogger(__name__)


class RetrievalMode(str, Enum):
    DENSE = "dense"
    HYBRID = "hybrid"
    HYBRID_EXPANDED = "hybrid_expanded"


def _apply_vocab_boost(
    candidates: list[CandidateLog],
    expanded: ExpandedQuery,
    cfg: VocabularyBoostConfig,
) -> list[CandidateLog]:
    """Adjust hybrid_score using vocabulary match signals.

    Boost/penalty values are intentionally moderate here (candidate stage).
    The reranker applies the heavy-handed precision scoring later.
    """
    result = []

    for cand in candidates:
        text = cand.log.full_text
        text_lower = text.lower()
        text_tokens = _tok_set(text_lower)
        title = cand.log.title
        title_lower = title.lower()
        title_tokens = _tok_set(title_lower)

        boost = 0.0
        reasons: list[str] = []

        # ── Priority terms (strong positive) ─────────────────────────────────
        for term in expanded.priority_terms:
            m = match_term(term, text_lower, text_tokens, title_lower, title_tokens)
            if m.level == "phrase":
                b = cfg.priority_phrase_boost
                boost += b
                reasons.append(f"+pri_phrase:{term}(+{b:.3f})")
            elif m.level == "token":
                b = cfg.priority_token_boost
                boost += b
                if m.in_title:
                    boost += cfg.priority_title_bonus
                    reasons.append(f"+pri_title:{term}(+{b + cfg.priority_title_bonus:.3f})")
                else:
                    reasons.append(f"+pri_token:{term}(+{b:.3f})")

        # ── Evidence terms (normal positive) ──────────────────────────────────
        for term in expanded.expanded_terms:
            m = match_term(term, text_lower, text_tokens, title_lower, title_tokens)
            if m.level == "phrase":
                b = cfg.evidence_phrase_boost
                boost += b
                reasons.append(f"+ev_phrase:{term}(+{b:.3f})")
            elif m.level == "token":
                boost += cfg.evidence_token_boost

        # ── Related terms (weak positive) ────────────────────────────────────
        for term in expanded.related_terms:
            m = match_term(term, text_lower, text_tokens, title_lower, title_tokens)
            if m.level != "none":
                boost += cfg.related_token_boost

        # ── Negative terms (penalty) ──────────────────────────────────────────
        for term in expanded.negative_terms:
            m = match_term(term, text_lower, text_tokens, title_lower, title_tokens)
            if m.level == "phrase":
                p = cfg.negative_phrase_penalty
                boost -= p
                reasons.append(f"-neg_phrase:{term}(-{p:.3f})")
            elif m.level == "token":
                p = cfg.negative_token_penalty
                boost -= p
                reasons.append(f"-neg_token:{term}(-{p:.3f})")

        new_score = max(0.0, round(cand.hybrid_score + boost, 6))

        if reasons:
            logger.debug(
                "VocabBoost  log=%s  Δ=%.4f  score: %.4f→%.4f  %s  [%s]",
                cand.log_id, boost, cand.hybrid_score, new_score,
                reasons, cand.log.title,
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

        # Apply vocabulary boost when ExpandedQuery carries vocabulary
        if (
            isinstance(query, ExpandedQuery)
            and self.vocab_boost_config
            and (query.priority_terms or query.expanded_terms or query.negative_terms)
        ):
            logger.debug(
                "VocabBoost  priority=%d  evidence=%d  related=%d  negative=%d",
                len(query.priority_terms), len(query.expanded_terms),
                len(query.related_terms), len(query.negative_terms),
            )
            candidates = _apply_vocab_boost(candidates, query, self.vocab_boost_config)

        return candidates
