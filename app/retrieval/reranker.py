"""Lexical-Control Reranker (Stage 2 — precision-focused).

Dual-space architecture:
  Stage 1 (candidate) = recall  → semantic 45%, BM25 40%, vocab 15%
  Stage 2 (reranker)  = precision → lexical 90%, semantic 5–10%

Final score formula:
  final_score =
      priority_weight   * priority_phrase_score   (0.35)
    + evidence_weight   * evidence_phrase_score    (0.20)
    + related_weight    * related_score            (0.10)
    + action_weight     * action_signal            (0.15)
    + domain_weight     * domain_consistency       (0.10)
    + semantic_weight   * semantic_similarity      (0.05)
    + base_weight       * base_goal_overlap        (0.05)
    − negative_penalty

Negative veto:
  If domain_mismatch ≥ veto_threshold AND priority_score < veto_priority_min
  → score = 0.0 (hard veto)

All positive components use phrase-aware scoring from text_matching:
  exact phrase > substring (multi-token) > token

Title matches receive title_weight_multiplier bonus (1.5×).
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from app.config import RankerConfig
from app.retrieval.dense_retriever import DenseRetriever, cosine
from app.retrieval.embedding_provider import MockEmbeddingProvider
from app.schemas import CandidateLog, RankedLog, ResearchGoal, ResearchLog
from app.utils.text_matching import (
    TermMatch,
    _tok_set,
    penalty_score,
    score_terms,
)

logger = logging.getLogger(__name__)

# ── Action/completion signal keywords ────────────────────────────────────────
_ACTION_KEYWORDS: set[str] = {
    # Completion/achievement
    "완료", "완성", "달성", "성공", "해결", "제출", "발표", "합격",
    "finished", "completed", "achieved", "solved", "submitted",
    # Active work
    "구현", "작성", "풀었", "풀이", "풀기", "실습", "수행",
    "implemented", "wrote", "built", "practiced",
    # Study/progress
    "공부", "연습", "정리", "분석", "학습완료", "오답",
    "reviewed", "analyzed",
}

_PRODUCTIVE_TYPES: set[str] = {
    "study", "implementation", "reading", "exercise",
    "planning", "execution", "reflection", "coding", "social",
}
_NOISE_TYPES: set[str] = {"daily"}


class GoalConditionedReranker:
    """Lexical-control precision reranker for Stage 2.

    Semantic similarity acts as tie-breaker only (5% weight).
    Lexical phrase/token matching and rule-based signals = 90%+ of score.
    """

    def __init__(
        self,
        config: RankerConfig | None = None,
        dense_retriever: DenseRetriever | None = None,
        # Back-compat parameter (ignored; penalty is in config)
        negative_term_penalty: Optional[float] = None,
    ) -> None:
        self.config = config or RankerConfig()
        self._dense = dense_retriever or DenseRetriever(provider=MockEmbeddingProvider())

    # ── Priority phrase score ─────────────────────────────────────────────────

    def _priority_phrase_score(
        self, log_text: str, log_title: str, priority_terms: list[str]
    ) -> tuple[float, list[TermMatch]]:
        """Strongest positive signal: phrase/token match of priority_terms."""
        score, matches = score_terms(
            priority_terms, log_text, log_title,
            phrase_weight=1.5, token_weight=1.0,
            title_multiplier=self.config.title_weight_multiplier,
        )
        return score, matches

    # ── Evidence phrase score ─────────────────────────────────────────────────

    def _evidence_phrase_score(
        self, log_text: str, log_title: str, evidence_terms: list[str]
    ) -> tuple[float, list[TermMatch]]:
        """Direct evidence vocabulary match."""
        score, matches = score_terms(
            evidence_terms, log_text, log_title,
            phrase_weight=1.5, token_weight=1.0,
            title_multiplier=self.config.title_weight_multiplier,
        )
        return score, matches

    # ── Related score ─────────────────────────────────────────────────────────

    def _related_score(
        self, log_text: str, log_title: str, related_terms: list[str]
    ) -> float:
        """Weak signal: indirect/related vocabulary."""
        score, _ = score_terms(
            related_terms, log_text, log_title,
            phrase_weight=1.2, token_weight=1.0,
            title_multiplier=self.config.title_weight_multiplier,
        )
        return score

    # ── Action signal ─────────────────────────────────────────────────────────

    def _action_signal(self, candidate: CandidateLog) -> float:
        """Does the log contain evidence of real action / progress / completion?

        Components:
          - Action/completion keywords in text
          - activity_type (productive > noise)
          - evidence_strength metadata
        """
        tokens = _tok_set(candidate.log.full_text)
        kw_hits = len(tokens & _ACTION_KEYWORDS)
        kw_score = min(kw_hits / 3.0, 1.0)

        act = candidate.log.activity_type
        if act in _PRODUCTIVE_TYPES:
            act_score = 0.8
        elif act in _NOISE_TYPES:
            act_score = 0.1
        else:
            act_score = 0.4

        strength = candidate.log.metadata.get("evidence_strength", "medium")
        strength_score = {"high": 1.0, "medium": 0.6, "low": 0.2}.get(strength, 0.5)

        return min(0.4 * kw_score + 0.4 * act_score + 0.2 * strength_score, 1.0)

    # ── Domain consistency ────────────────────────────────────────────────────

    def _domain_consistency(
        self, candidate: CandidateLog, goal: ResearchGoal
    ) -> float:
        """Is this log in the right domain for the goal?

        Measured positively (higher = more consistent).
        Separate from negative_penalty which measures harmful mismatch.
        """
        act = candidate.log.activity_type
        if act in _PRODUCTIVE_TYPES:
            type_score = 0.7
        elif act in _NOISE_TYPES:
            type_score = 0.1
        else:
            type_score = 0.4

        # Specificity: longer, denser logs are more informative
        tokens = _tok_set(candidate.log.full_text)
        specificity = min(len(tokens) / 40.0, 0.3)

        return min(type_score + specificity, 1.0)

    # ── Semantic similarity (tie-breaker) ─────────────────────────────────────

    def _semantic_similarity(
        self, goal: ResearchGoal, log_text: str, precomputed: float | None = None
    ) -> float:
        """Return semantic similarity.

        If precomputed is provided (candidate.dense_score from retrieval),
        use it directly — avoids redundant Gemini API calls.
        Dense embedding is already computed during candidate retrieval;
        reusing it saves N embedding calls per reranking run.
        """
        if precomputed is not None:
            return precomputed
        g_emb = self._dense.embed(goal.query_text)
        l_emb = self._dense.embed(log_text)
        return max(0.0, cosine(g_emb, l_emb))

    # ── Base goal overlap ─────────────────────────────────────────────────────

    def _base_goal_overlap(self, goal: ResearchGoal, log_text: str) -> float:
        """Raw token overlap of goal text with log text."""
        goal_tokens = _tok_set(goal.query_text + " " + goal.goal_embedding_text)
        log_tokens = _tok_set(log_text)
        if not goal_tokens:
            return 0.0
        return len(goal_tokens & log_tokens) / len(goal_tokens)

    # ── Negative penalty ──────────────────────────────────────────────────────

    def _negative_penalty(
        self, candidate: CandidateLog, negative_terms: list[str]
    ) -> tuple[float, float, list[str]]:
        """Compute raw domain mismatch score and final penalty.

        Returns (raw_dm, capped_penalty, matched_descriptions).
        raw_dm is used for veto decision; capped_penalty for score deduction.
        """
        cfg = self.config
        log_text = candidate.log.full_text
        log_title = candidate.log.title

        raw_dm, matched = penalty_score(
            negative_terms, log_text, log_title,
            phrase_penalty=cfg.negative_penalty_phrase,
            token_penalty=cfg.negative_penalty_token,
            title_extra_penalty=cfg.negative_penalty_title,
        )

        noise = cfg.negative_daily_penalty if candidate.log.activity_type in _NOISE_TYPES else 0.0
        capped = min(raw_dm + noise, 1.0)

        if capped > 0.2:
            logger.debug(
                "NegPenalty  log=%s  dm=%.2f  matched=%s  [%s]",
                candidate.log.log_id, capped, matched, candidate.log.title,
            )
        return raw_dm, capped, matched

    # ── Scoring entry point ───────────────────────────────────────────────────

    def score(
        self,
        goal: ResearchGoal,
        candidate: CandidateLog,
        expanded_terms: list[str] | None = None,
        negative_terms: list[str] | None = None,
        priority_terms: list[str] | None = None,
        related_terms: list[str] | None = None,
    ) -> RankedLog:
        pri_terms = priority_terms or []
        ev_terms = expanded_terms or []
        rel_terms = related_terms or []
        neg_terms = negative_terms or []

        log_text = candidate.log.full_text
        log_title = candidate.log.title
        cfg = self.config

        # ── Compute all components ────────────────────────────────────────────
        pri_score, pri_matches = self._priority_phrase_score(log_text, log_title, pri_terms)
        ev_score, ev_matches = self._evidence_phrase_score(log_text, log_title, ev_terms)
        rel_score = self._related_score(log_text, log_title, rel_terms)
        action = self._action_signal(candidate)
        domain = self._domain_consistency(candidate, goal)
        # Reuse dense_score from candidate retrieval (avoids redundant API call)
        sem = self._semantic_similarity(
            goal, log_text,
            precomputed=candidate.dense_score if candidate.dense_score > 0 else None,
        )
        base = self._base_goal_overlap(goal, log_text)

        # ── Negative penalty ──────────────────────────────────────────────────
        raw_dm, penalty, neg_matched = self._negative_penalty(candidate, neg_terms)

        # ── Zero-positive hard block ───────────────────────────────────────────
        # Logs with no lexical evidence of goal relevance are rejected regardless
        # of action/domain/semantic signal. This prevents lifestyle/exercise logs
        # from being admitted via the action_signal + domain_consistency path alone.
        #
        # A log MUST have at least ONE of:
        #   - priority phrase/token match     (direct goal vocabulary)
        #   - evidence phrase/token match     (supporting vocabulary)
        #   - base goal token overlap         (raw goal text overlap)
        if pri_score == 0.0 and ev_score == 0.0 and base == 0.0:
            logger.debug(
                "ZERO-POSITIVE BLOCK  log=%s  [%s]  "
                "(no priority/evidence/base match → reject)",
                candidate.log.log_id, log_title,
            )
            return RankedLog(
                log=candidate.log,
                semantic_relevance=round(sem, 4),
                goal_focus=0.0,
                evidence_value=0.0,
                final_score=0.0,
                rejection_reason="zero_positive_evidence",
            )

        # ── Negative veto (domain conflict gate) ──────────────────────────────
        # Veto fires when: significant domain mismatch AND no priority evidence.
        # This is a domain conflict gate, not a keyword blacklist.
        veto = (
            cfg.negative_veto_enabled
            and raw_dm >= cfg.negative_veto_dm_threshold
            and pri_score < cfg.negative_veto_priority_min
        )

        if veto:
            logger.debug(
                "DOMAIN-CONFLICT VETO  log=%s  dm=%.2f  pri=%.3f  matched=%s  [%s]",
                candidate.log.log_id, raw_dm, pri_score, neg_matched, log_title,
            )
            return RankedLog(
                log=candidate.log,
                semantic_relevance=round(sem, 4),
                goal_focus=0.0,
                evidence_value=0.0,
                final_score=0.0,
                matched_negative=list(neg_matched),
                rejection_reason=f"domain_conflict_veto(dm={raw_dm:.2f})",
            )

        # ── Final score ───────────────────────────────────────────────────────
        final = max(0.0, round(
            cfg.priority_weight * pri_score
            + cfg.evidence_weight * ev_score
            + cfg.related_weight * rel_score
            + cfg.action_weight * action
            + cfg.domain_weight * domain
            + cfg.semantic_weight * sem
            + cfg.base_weight * base
            - penalty,
            4,
        ))

        # Explanation trace
        matched_pri = [m.term for m in pri_matches if m.level != "none"]
        matched_ev = [m.term for m in ev_matches if m.level != "none"]

        if final > 0:
            reason_parts = []
            if matched_pri:
                reason_parts.append(f"priority={matched_pri}")
            if matched_ev:
                reason_parts.append(f"evidence={matched_ev}")
            if base > 0:
                reason_parts.append(f"base_overlap={base:.3f}")
            admission_reason = " | ".join(reason_parts) if reason_parts else "weak_match"
        else:
            admission_reason = ""

        # ── Debug breakdown ───────────────────────────────────────────────────
        logger.debug(
            "[Reranker Score]  log=%s  [%s]\n"
            "  priority_phrase: %.3f  matched=%s\n"
            "  evidence_phrase: %.3f  matched=%s\n"
            "  related:         %.3f\n"
            "  action_signal:   %.3f\n"
            "  domain_consist:  %.3f\n"
            "  semantic:        %.3f\n"
            "  base_overlap:    %.3f\n"
            "  neg_penalty:     %.3f  (matched=%s)\n"
            "  → final:         %.4f  reason=%s",
            candidate.log.log_id, log_title,
            pri_score, matched_pri,
            ev_score, matched_ev,
            rel_score, action, domain, sem, base,
            penalty, neg_matched,
            final, admission_reason,
        )

        return RankedLog(
            log=candidate.log,
            semantic_relevance=round(sem, 4),
            goal_focus=round(
                cfg.priority_weight * pri_score + cfg.evidence_weight * ev_score, 4
            ),
            evidence_value=round(action, 4),
            final_score=final,
            matched_priority=matched_pri,
            matched_evidence=matched_ev,
            matched_negative=list(neg_matched),
            admission_reason=admission_reason,
        )

    def score_log(
        self,
        goal: ResearchGoal,
        log: ResearchLog,
        expanded_terms: list[str] | None = None,
        negative_terms: list[str] | None = None,
        priority_terms: list[str] | None = None,
        related_terms: list[str] | None = None,
    ) -> float:
        """Score a single ResearchLog for neighbor admission check.

        Used by LocalExpander to re-admit temporal neighbors through the same
        admission gate as anchors — prevents non-admitted logs from entering
        the compressor via the expansion path.
        """
        candidate = CandidateLog(log=log, sparse_score=0.0, dense_score=0.0, hybrid_score=0.0)
        ranked = self.rank(
            goal, [candidate],
            expanded_terms=expanded_terms,
            negative_terms=negative_terms,
            priority_terms=priority_terms,
            related_terms=related_terms,
        )
        return ranked[0].final_score if ranked else 0.0

    def rank(
        self,
        goal: ResearchGoal,
        candidates: list[CandidateLog],
        expanded_terms: list[str] | None = None,
        negative_terms: list[str] | None = None,
        priority_terms: list[str] | None = None,
        related_terms: list[str] | None = None,
    ) -> list[RankedLog]:
        ranked = [
            self.score(
                goal, c,
                expanded_terms=expanded_terms,
                negative_terms=negative_terms,
                priority_terms=priority_terms,
                related_terms=related_terms,
            )
            for c in candidates
        ]
        ranked.sort(key=lambda x: x.final_score, reverse=True)
        for i, r in enumerate(ranked):
            r.rank = i + 1

        vetoed = sum(1 for r in ranked if r.final_score == 0.0)
        logger.info(
            "Reranked %d candidates  vetoed=%d  goal=%s",
            len(ranked), vetoed, goal.goal_id,
        )
        return ranked
