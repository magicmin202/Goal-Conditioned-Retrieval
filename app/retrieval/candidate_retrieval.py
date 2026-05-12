"""Candidate Retrieval — dual-space recall-focused retrieval.

Architecture (Stage 1):
  candidate_score = bm25_weight * bm25_score    (0.40)
                  + dense_weight * dense_score   (0.45)
                  + vocab_weight * vocab_boost   (0.15)

Vocabulary boost is intentionally weak (recall-focused):
  - Nudges direction toward lexicon signals
  - Does NOT override semantic recall
  - Strong precision control is the reranker's responsibility

Boost values are normalized [0, 1] internally, then scaled by vocab_weight.
"""
from __future__ import annotations

import logging
import re
from enum import Enum

from app.config import CandidateConfig, RetrievalConfig, VocabularyBoostConfig
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.embedding_provider import EmbeddingProvider
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.query_expansion import ExpandedQuery
from app.retrieval.query_understanding import QueryObject
from app.schemas import CandidateLog, ResearchLog
from app.utils.text_matching import match_term, _tok_set

logger = logging.getLogger(__name__)

_TOKENIZE_RE = re.compile(r"[\w가-힣]+")


def _build_expanded_bm25_query(query: ExpandedQuery) -> str:
    """BM25 recall 확대용 쿼리 구성.

    순서: canonical → priority (전체) → evidence (전체) → related (전체)
    negative_terms 제외 — BM25는 감점을 지원하지 않으므로 포함 시
    irrelevant 로그에 점수를 부여하게 됨.

    중복 토큰을 set으로 제거하여 특정 토큰의
    과도한 가중치(BM25 IDF 왜곡) 방지.
    """
    seen: set[str] = set()
    parts: list[str] = []

    def add_terms(terms: list[str]) -> None:
        for term in terms:
            tokens = _TOKENIZE_RE.findall(term.lower())
            new_tokens = [t for t in tokens if t not in seen]
            if new_tokens:
                parts.append(" ".join(new_tokens))
                seen.update(new_tokens)

    add_terms([query.canonical_text])
    add_terms(query.priority_terms)       # 핵심 positive signal
    add_terms(query.expanded_terms)       # evidence_terms: 직접 관련 활동
    add_terms(query.related_terms)        # 간접 표현: recall 확대
    # negative_terms는 포함하지 않음

    result = " ".join(parts).strip()
    logger.debug("expanded_bm25_q (%d tokens): %s", len(seen), result[:120])
    return result


def _dynamic_candidate_size(corpus_size: int) -> int:
    """Corpus 크기에 비례한 candidate_size 동적 계산.

    corpus_size   candidate_size
    ────────────────────────────
    ≤  60         30  (최소 보장)
       70         35
      100         50
      200        100  (상한 적용)
     1000        100  (상한 적용)
    """
    return min(max(30, int(corpus_size * 0.50)), 100)


class RetrievalMode(str, Enum):
    SPARSE = "sparse"           # BM25 only (baseline)
    DENSE = "dense"             # Dense only (baseline)
    HYBRID = "hybrid"           # BM25 + Dense (default)
    HYBRID_EXPANDED = "hybrid_expanded"


def _weak_vocab_boost(
    candidates: list[CandidateLog],
    expanded: ExpandedQuery,
    boost_cfg: VocabularyBoostConfig,
    candidate_cfg: CandidateConfig,
) -> list[CandidateLog]:
    """Compute and apply weak vocabulary boost to hybrid scores.

    The raw boost is normalized [0, 1], then scaled by vocab_boost_weight
    before being added to hybrid_score. This keeps the candidate stage
    recall-focused rather than precision-focused.
    """
    results = []
    vw = candidate_cfg.vocab_boost_weight  # e.g., 0.15

    for cand in candidates:
        text = cand.log.full_text
        text_lower = text.lower()
        text_tokens = _tok_set(text_lower)
        title_lower = cand.log.title.lower()
        title_tokens = _tok_set(title_lower)

        # Raw boost accumulator (can be > 1 before normalization)
        raw_boost = 0.0
        reasons: list[str] = []

        # ── Priority terms ────────────────────────────────────────────────────
        for term in expanded.priority_terms:
            m = match_term(term, text_lower, text_tokens, title_lower, title_tokens)
            if m.level == "phrase":
                raw_boost += boost_cfg.priority_phrase_boost
                reasons.append(f"+pri_phrase:{term}")
            elif m.level == "token":
                b = boost_cfg.priority_token_boost
                if m.in_title:
                    b += boost_cfg.priority_title_bonus
                    reasons.append(f"+pri_title:{term}")
                else:
                    reasons.append(f"+pri_tok:{term}")
                raw_boost += b

        # ── Evidence terms ────────────────────────────────────────────────────
        for term in expanded.expanded_terms:
            m = match_term(term, text_lower, text_tokens, title_lower, title_tokens)
            if m.level == "phrase":
                raw_boost += boost_cfg.evidence_phrase_boost
            elif m.level == "token":
                raw_boost += boost_cfg.evidence_token_boost

        # ── Related terms (very weak) ─────────────────────────────────────────
        for term in expanded.related_terms:
            m = match_term(term, text_lower, text_tokens, title_lower, title_tokens)
            if m.level != "none":
                raw_boost += boost_cfg.related_token_boost

        # ── Negative terms (mild penalty at candidate stage) ──────────────────
        for term in expanded.negative_terms:
            m = match_term(term, text_lower, text_tokens, title_lower, title_tokens)
            if m.level == "phrase":
                raw_boost -= boost_cfg.negative_phrase_penalty
                reasons.append(f"-neg_phrase:{term}")
            elif m.level == "token":
                raw_boost -= boost_cfg.negative_token_penalty
                reasons.append(f"-neg_tok:{term}")

        # Normalize to [-1, 1] and scale by vocab_weight
        # The denominator is the max possible raw boost (all priority phrase hits)
        max_possible = len(expanded.priority_terms) * boost_cfg.priority_phrase_boost or 1.0
        normalized = max(-1.0, min(1.0, raw_boost / max_possible))
        delta = vw * normalized

        new_score = max(0.0, round(cand.hybrid_score + delta, 6))

        if reasons:
            logger.debug(
                "WeakVocab  log=%s  raw=%.3f  norm=%.3f  Δ=%.4f  score:%.4f→%.4f  %s  [%s]",
                cand.log_id, raw_boost, normalized, delta,
                cand.hybrid_score, new_score, reasons, cand.log.title,
            )

        results.append(CandidateLog(
            log=cand.log,
            sparse_score=cand.sparse_score,
            dense_score=cand.dense_score,
            hybrid_score=new_score,
        ))

    results.sort(key=lambda x: x.hybrid_score, reverse=True)
    return results


class CandidateRetriever:
    """Dual-space recall-focused candidate retriever."""

    def __init__(
        self,
        mode: RetrievalMode = RetrievalMode.HYBRID,
        config: RetrievalConfig | None = None,
        candidate_config: CandidateConfig | None = None,
        vocab_boost_config: VocabularyBoostConfig | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self.mode = mode
        self.config = config or RetrievalConfig()
        self.candidate_config = candidate_config or CandidateConfig()
        self.vocab_boost_config = vocab_boost_config or VocabularyBoostConfig()
        self._hybrid = HybridRetriever(
            config=self.config,
            embedding_provider=embedding_provider,
        )
        self._dense = DenseRetriever(doc_provider=embedding_provider)
        self._indexed = False

    def index(self, logs: list[ResearchLog]) -> None:
        self._hybrid.index(logs)
        # Dense is indexed inside hybrid; also expose for reranker semantic scoring
        self._dense = self._hybrid.dense
        self._corpus_size = len(logs)
        self._indexed = True
        logger.info(
            "CandidateRetriever indexed %d logs [mode=%s  bm25=%.2f  dense=%.2f  vocab=%.2f]",
            len(logs), self.mode,
            self.config.sparse_weight, self.config.dense_weight,
            self.candidate_config.vocab_boost_weight,
        )

    def retrieve(
        self,
        query: QueryObject | ExpandedQuery,
        top_n: int | None = None,
    ) -> list[CandidateLog]:
        if not self._indexed:
            raise RuntimeError("Call index() before retrieve().")

        # top_n 우선 사용; 없으면 corpus 크기 기반 동적 산정
        if top_n is not None:
            n = top_n
        else:
            corpus_size = getattr(self, "_corpus_size", None) or self.config.candidate_size
            n = _dynamic_candidate_size(corpus_size)

        if isinstance(query, ExpandedQuery):
            # BM25: canonical + priority(전체) + evidence(전체) + related(전체)
            # Dense가 못 잡는 keyword-sparse 로그를 BM25로 보완 (recall 확대)
            bm25_text = _build_expanded_bm25_query(query)
            # Dense: 최소 semantic core만 사용 (semantic drift 방지)
            dense_text = query.dense_query
            logger.debug(
                "CandidateRetriever  bm25_q=%s  dense_q=%s",
                bm25_text[:80], dense_text[:60],
            )
        else:
            bm25_text = query.canonical_text
            dense_text = query.canonical_text

        if self.mode == RetrievalMode.SPARSE:
            # BM25-only baseline — no dense scoring
            candidates = self._hybrid.sparse.retrieve(bm25_text, top_n=n)
            # Normalise to hybrid_score field so downstream code is uniform
            candidates = [
                CandidateLog(
                    log=c.log,
                    sparse_score=c.sparse_score,
                    dense_score=0.0,
                    hybrid_score=c.sparse_score,
                )
                for c in candidates
            ]
        elif self.mode == RetrievalMode.DENSE:
            candidates = self._dense.retrieve(dense_text, top_n=n)
        else:
            candidates = self._hybrid.retrieve(bm25_text, top_n=n, dense_query=dense_text)

        logger.debug(
            "Stage1 candidates: %d  bm25_q=%s",
            len(candidates), bm25_text[:60],
        )

        # Apply weak vocabulary boost when ExpandedQuery vocabulary is present
        if (
            isinstance(query, ExpandedQuery)
            and (query.priority_terms or query.expanded_terms or query.negative_terms)
        ):
            candidates = _weak_vocab_boost(
                candidates, query,
                self.vocab_boost_config,
                self.candidate_config,
            )

        return candidates
