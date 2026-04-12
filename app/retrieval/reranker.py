"""Lexical-Control Reranker with Category-First Admission.

Dual-space architecture:
  Stage 1 (candidate) = recall  → semantic 45%, BM25 40%, vocab 15%
  Stage 2 (reranker)  = precision → lexical 90%, semantic 5–10%

Admission gates (applied IN ORDER before scoring):
  1. Schema category gate  — log must belong to a category relevant to goal domain
       "none" → immediate reject (category_mismatch)
  2. Goal lexical gate     — log must have at least ONE direct goal signal
       pri=0 AND ev=0 AND base<threshold → reject (no_goal_signal)
  3. Negative veto         — domain conflict + no positive evidence
       dm≥veto_threshold AND pri<veto_min → reject (domain_conflict_veto)

Scoring formula (only reached after all gates pass):
  final_score =
      priority_weight   * priority_phrase_score   (0.35)
    + evidence_weight   * evidence_phrase_score    (0.20)
    + related_weight    * related_score            (0.10)
    + action_weight     * action_signal            (0.15)
    + domain_weight     * domain_consistency       (0.10)  ← now goal-domain-aware
    + semantic_weight   * semantic_similarity      (0.05)
    + base_weight       * base_goal_overlap        (0.05)
    − negative_penalty

Schema signals:
  - Used ONLY in category mapper (gate logic)
  - NOT injected into reranker scoring terms
  - Small additive boost stays at candidate stage (VocabularyBoostConfig)
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from app.config import RankerConfig
from app.retrieval.dense_retriever import DenseRetriever, cosine
from app.retrieval.embedding_provider import MockEmbeddingProvider
from app.retrieval.evidence_quality import _quality_scorer
from app.retrieval.schema_category import CategoryScore, SchemaMapper, _schema_mapper
from app.schemas import CandidateLog, RankedLog, ResearchGoal, ResearchLog
from app.utils.text_matching import (
    PriorityTermMatch,
    TermMatch,
    _tok_set,
    penalty_score,
    score_priority_terms,
    score_terms,
)

logger = logging.getLogger(__name__)

# ── Tier2 semantic gate threshold ─────────────────────────────────────────────
# Only active when real embeddings are used (use_real_embeddings=True).
# Skipped entirely for mock embeddings to prevent random hash-based scores
# from incorrectly rejecting relevant logs.
# Initial value: 0.50.  Tuning range: 0.45 – 0.55.
SEMANTIC_GATE_THRESHOLD: float = 0.50

# ── Action/completion signal keywords ────────────────────────────────────────
_ACTION_KEYWORDS: set[str] = {
    "완료", "완성", "달성", "성공", "해결", "제출", "발표", "합격",
    "finished", "completed", "achieved", "solved", "submitted",
    "구현", "작성", "풀었", "풀이", "풀기", "실습", "수행",
    "implemented", "wrote", "built", "practiced",
    "공부", "연습", "정리", "분석", "학습완료", "오답",
    "reviewed", "analyzed",
}

_NOISE_TYPES: set[str] = {"daily"}


class GoalConditionedReranker:
    """Category-first lexical-control precision reranker.

    Schema category gate fires before scoring — ensures:
      - Lifestyle/unrelated logs are rejected by domain mismatch
      - action_signal + domain_consistency cannot carry zero-evidence logs
      - Scoring is only reached by categorically-relevant logs with goal signal
    """

    def __init__(
        self,
        config: RankerConfig | None = None,
        dense_retriever: DenseRetriever | None = None,
        negative_term_penalty: Optional[float] = None,  # back-compat, ignored
        use_real_embeddings: bool = False,
    ) -> None:
        self.config = config or RankerConfig()
        self._dense = dense_retriever or DenseRetriever(provider=MockEmbeddingProvider())
        self._schema = _schema_mapper   # module-level singleton, stateless
        # When False, Tier2 semantic gate is skipped (mock embeddings produce
        # meaningless cosine scores that would incorrectly reject relevant logs).
        self._use_real_embeddings = use_real_embeddings

    # ── Priority phrase score ─────────────────────────────────────────────────

    def _priority_phrase_score(
        self,
        log_id: str,
        log_text: str,
        log_title: str,
        priority_terms: list[str],
    ) -> tuple[float, list[PriorityTermMatch]]:
        """Weak-token-filtered priority matching.

        Uses score_priority_terms so that generic tokens ("완료", "정리", …)
        cannot trigger a positive match on their own.
        """
        score, matches = score_priority_terms(
            priority_terms, log_text, log_title,
            phrase_weight=1.5, token_weight=0.4,
            title_multiplier=self.config.title_weight_multiplier,
        )
        for m in matches:
            if m.mode in ("weak_token_only", "none") and m.weak_hits_only:
                logger.debug(
                    "[LEXICAL_MATCH]  log_id=%s  term=%r  mode=%s"
                    "  core_hits=%s  weak_hits=%s  score=%.1f",
                    log_id, m.term, m.mode, m.core_hits, m.weak_hits_only, m.score,
                )
        return score, matches

    # ── Evidence phrase score ─────────────────────────────────────────────────

    def _evidence_phrase_score(
        self,
        log_id: str,
        log_text: str,
        log_title: str,
        evidence_terms: list[str],
    ) -> tuple[float, list[PriorityTermMatch]]:
        """Weak-token-filtered evidence matching (same logic as priority)."""
        score, matches = score_priority_terms(
            evidence_terms, log_text, log_title,
            phrase_weight=1.5, token_weight=0.4,
            title_multiplier=self.config.title_weight_multiplier,
        )
        return score, matches

    # ── Related score ─────────────────────────────────────────────────────────

    def _related_score(
        self, log_text: str, log_title: str, related_terms: list[str]
    ) -> float:
        score, _ = score_terms(
            related_terms, log_text, log_title,
            phrase_weight=1.2, token_weight=1.0,
            title_multiplier=self.config.title_weight_multiplier,
        )
        return score

    # ── Action signal ─────────────────────────────────────────────────────────

    def _action_signal(self, candidate: CandidateLog) -> float:
        """Evidence of real action / completion (goal-agnostic signal)."""
        tokens = _tok_set(candidate.log.full_text)
        kw_hits = len(tokens & _ACTION_KEYWORDS)
        kw_score = min(kw_hits / 3.0, 1.0)

        act = candidate.log.activity_type
        # Note: activity_type relevance is handled by category gate;
        # here we only measure "how active" the log is (not goal-domain relevance).
        if act in _NOISE_TYPES:
            act_score = 0.1
        else:
            act_score = 0.7

        strength = candidate.log.metadata.get("evidence_strength", "medium")
        strength_score = {"high": 1.0, "medium": 0.6, "low": 0.2}.get(strength, 0.5)

        return min(0.4 * kw_score + 0.4 * act_score + 0.2 * strength_score, 1.0)

    # ── Domain consistency (now GOAL-DOMAIN-AWARE) ────────────────────────────

    def _domain_consistency(
        self, candidate: CandidateLog, cat_result: CategoryScore
    ) -> float:
        """Goal-domain-aware domain consistency.

        Uses schema category relevance instead of goal-agnostic _PRODUCTIVE_TYPES.
        Logs that reach this point passed the category gate, so relevance ∈ {core, supporting}.
        """
        base_scores = {"core": 0.80, "supporting": 0.50, "none": 0.05}
        base = base_scores.get(cat_result.relevance, 0.05)

        # Specificity bonus: denser logs are more informative
        tokens = _tok_set(candidate.log.full_text)
        specificity = min(len(tokens) / 40.0, 0.20)

        return min(base + specificity, 1.0)

    # ── Semantic similarity (tie-breaker) ─────────────────────────────────────

    def _semantic_similarity(
        self, goal: ResearchGoal, log_text: str, precomputed: float | None = None
    ) -> float:
        if precomputed is not None:
            return precomputed
        g_emb = self._dense.embed(goal.query_text)
        l_emb = self._dense.embed(log_text)
        return max(0.0, cosine(g_emb, l_emb))

    # ── Base goal overlap ─────────────────────────────────────────────────────

    def _base_goal_overlap(self, goal: ResearchGoal, log_text: str) -> float:
        goal_tokens = _tok_set(goal.query_text + " " + goal.goal_embedding_text)
        log_tokens = _tok_set(log_text)
        if not goal_tokens:
            return 0.0
        return len(goal_tokens & log_tokens) / len(goal_tokens)

    # ── Negative penalty ──────────────────────────────────────────────────────

    def _negative_penalty(
        self, candidate: CandidateLog, negative_terms: list[str]
    ) -> tuple[float, float, list[str]]:
        cfg = self.config
        raw_dm, matched = penalty_score(
            negative_terms, candidate.log.full_text, candidate.log.title,
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
        skip_semantic_gate: bool = False,
        disable_lexical_gate: bool = False,
    ) -> RankedLog:
        pri_terms = priority_terms or []
        ev_terms = expanded_terms or []
        rel_terms = related_terms or []
        neg_terms = negative_terms or []

        log_text = candidate.log.full_text
        log_title = candidate.log.title
        cfg = self.config

        # ══════════════════════════════════════════════════════════════════════
        # GATE 1: Schema Category Gate
        # Goal-domain-aware hard filter.
        # Unrelated categories are rejected before any scoring.
        # Schema signals are NOT in the scoring formula — category is gate-only.
        # ══════════════════════════════════════════════════════════════════════
        cat_result = self._schema.evaluate(candidate.log, goal)

        # Tier1 gate logging (activity-type included for diagnosis)
        from app.retrieval.schema_category import classify_log_activity_type
        _log_at = classify_log_activity_type(
            candidate.log.title + " " + (candidate.log.content or "")
        )

        if cat_result.relevance == "none":
            logger.info(
                "[TIER1_FAIL] %s  log_type=%s  cat=%s  domain=%s  reason=%s  [%s]",
                candidate.log.log_id, _log_at,
                cat_result.log_category, cat_result.goal_domain,
                cat_result.reason, log_title,
            )
            return RankedLog(
                log=candidate.log,
                semantic_relevance=0.0,
                goal_focus=0.0,
                evidence_value=0.0,
                final_score=0.0,
                rejection_reason=(
                    f"category_mismatch(cat={cat_result.log_category}"
                    f"  domain={cat_result.goal_domain}"
                    f"  reason={cat_result.reason})"
                ),
                schema_category=cat_result.log_category,
                goal_domain=cat_result.goal_domain,
                category_hit_strength=cat_result.relevance,
            )

        logger.debug(
            "[TIER1_PASS] %s  log_type=%s  relevance=%s  cat=%s  domain=%s  [%s]",
            candidate.log.log_id, _log_at,
            cat_result.relevance, cat_result.log_category,
            cat_result.goal_domain, log_title,
        )

        # ══════════════════════════════════════════════════════════════════════
        # Compute goal lexical scores (needed for GATE 2 and scoring)
        # ══════════════════════════════════════════════════════════════════════
        pri_score, pri_matches = self._priority_phrase_score(
            candidate.log.log_id, log_text, log_title, pri_terms
        )
        ev_score, ev_matches = self._evidence_phrase_score(
            candidate.log.log_id, log_text, log_title, ev_terms
        )
        base = self._base_goal_overlap(goal, log_text)

        # ══════════════════════════════════════════════════════════════════════
        # GATE 2: Goal Lexical Gate + Support Gate
        #
        # Primary path (gate_mode="direct"):
        #   Any of: priority match, evidence match, base_overlap ≥ 0.04
        #   → proceed to full scoring, no score cap
        #
        # Support path (gate_mode="supporting"):
        #   When no primary signal exists BUT:
        #     (a) log text contains a support_context_signal for the goal domain
        #     (b) log category is a supporting_category for the goal domain
        #   → proceed to scoring with score_cap=0.12
        #   score_cap ensures support logs never rank above direct evidence.
        #
        # Reject: neither path satisfied → no_goal_signal
        # ══════════════════════════════════════════════════════════════════════
        primary_signal = (pri_score > 0.0 or ev_score > 0.0 or base >= 0.04)
        score_cap: float | None = None
        support_context_matched: list[str] = []

        if disable_lexical_gate:
            # Bypass Gate 2 entirely — used by non-ours baselines.
            # Activity-type Gate (Gate 1) is still applied above.
            gate_mode = "direct"
            logger.debug(
                "LEXICAL GATE BYPASSED  log=%s  [%s]",
                candidate.log.log_id, log_title,
            )
        elif primary_signal:
            gate_mode = "direct"
        else:
            support_hit, sup_matched = self._schema.support_context_hit(
                candidate.log, cat_result.goal_domain
            )
            subdomain_ok = self._schema.is_subdomain_consistent(
                cat_result.log_category, cat_result.goal_domain
            )

            if support_hit and subdomain_ok:
                gate_mode = "supporting"
                score_cap = 0.12
                support_context_matched = sup_matched
                logger.debug(
                    "SUPPORT GATE ADMIT  log=%s  cat=%s  domain=%s"
                    "  matched=%s  score_cap=%.2f  [%s]",
                    candidate.log.log_id, cat_result.log_category,
                    cat_result.goal_domain, sup_matched, score_cap, log_title,
                )
            else:
                logger.debug(
                    "GOAL LEXICAL GATE REJECT  log=%s  cat=%s  relevance=%s  "
                    "pri=%.3f  ev=%.3f  base=%.3f  "
                    "support_hit=%s  subdomain_ok=%s  [%s]",
                    candidate.log.log_id, cat_result.log_category, cat_result.relevance,
                    pri_score, ev_score, base,
                    support_hit, subdomain_ok, log_title,
                )
                _reason = "no_goal_signal(pri=0,ev=0,base<0.04"
                if support_hit and not subdomain_ok:
                    _reason += ",subdomain_mismatch"
                elif not support_hit and subdomain_ok:
                    _reason += ",no_support_context"
                _reason += ")"
                return RankedLog(
                    log=candidate.log,
                    semantic_relevance=0.0,
                    goal_focus=0.0,
                    evidence_value=0.0,
                    final_score=0.0,
                    gate_mode="reject",
                    rejection_reason=_reason,
                    schema_category=cat_result.log_category,
                    goal_domain=cat_result.goal_domain,
                    category_hit_strength=cat_result.relevance,
                )

        # ══════════════════════════════════════════════════════════════════════
        # Remaining components
        # ══════════════════════════════════════════════════════════════════════
        rel_score = self._related_score(log_text, log_title, rel_terms)
        action = self._action_signal(candidate)
        domain = self._domain_consistency(candidate, cat_result)  # goal-domain-aware
        sem = self._semantic_similarity(
            goal, log_text,
            precomputed=candidate.dense_score if candidate.dense_score > 0 else None,
        )

        # ══════════════════════════════════════════════════════════════════════
        # TIER 2: Semantic Relevance Gate
        # Only fires when real embeddings are active (use_real_embeddings=True)
        # AND skip_semantic_gate=False.
        #
        # skip_semantic_gate=True is set for Stage2 neighbor re-admission:
        #   - Neighbors are already gated by temporal locality + lexical gates.
        #   - dense_score here is raw cosine (not max-normalized like Stage1),
        #     so the Stage1-calibrated threshold (0.50) would be misapplied.
        #
        # Condition: sem > 0 AND sem < threshold
        #   → log is semantically too far from goal even after lexical gates pass
        # supporting-mode logs are exempt (score_cap already limits their impact)
        # ══════════════════════════════════════════════════════════════════════
        logger.debug(
            "[STAGE2_SEMANTIC_DEBUG]  log_id=%s  title=%s  "
            "dense_score_raw=%.4f  threshold=%.2f  skip_semantic_gate=%s",
            candidate.log.log_id, log_title,
            sem, SEMANTIC_GATE_THRESHOLD, skip_semantic_gate,
        )
        if (
            not skip_semantic_gate
            and self._use_real_embeddings
            and gate_mode != "supporting"
            and sem > 0.0
            and sem < SEMANTIC_GATE_THRESHOLD
        ):
            logger.info(
                "[TIER2_FAIL] %s  dense=%.4f < %.2f  reason=semantic_irrelevant  [%s]",
                candidate.log.log_id, sem, SEMANTIC_GATE_THRESHOLD, log_title,
            )
            return RankedLog(
                log=candidate.log,
                semantic_relevance=round(sem, 4),
                goal_focus=0.0,
                evidence_value=0.0,
                final_score=0.0,
                gate_mode="reject",
                rejection_reason=f"semantic_irrelevant(dense={sem:.4f}<{SEMANTIC_GATE_THRESHOLD})",
                schema_category=cat_result.log_category,
                goal_domain=cat_result.goal_domain,
                category_hit_strength=cat_result.relevance,
            )
        else:
            logger.debug(
                "[TIER2_PASS] %s  dense=%.4f  real_emb=%s  [%s]",
                candidate.log.log_id, sem, self._use_real_embeddings, log_title,
            )

        # ── Negative penalty ──────────────────────────────────────────────────
        raw_dm, penalty, neg_matched = self._negative_penalty(candidate, neg_terms)

        # ══════════════════════════════════════════════════════════════════════
        # GATE 3: Negative Veto (domain conflict gate)
        # Fires when: significant domain mismatch AND no priority evidence.
        # NOT a keyword blacklist — requires BOTH conditions.
        # ══════════════════════════════════════════════════════════════════════
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
                schema_category=cat_result.log_category,
                goal_domain=cat_result.goal_domain,
                category_hit_strength=cat_result.relevance,
            )

        # ── Evidence Quality Score ────────────────────────────────────────────
        # Separate from relevance — measures analysis value of the log.
        # Components: specificity, actionability, goal_progress, domain_consist
        qs = _quality_scorer.score(
            candidate.log,
            log_category=cat_result.log_category,
            goal_domain=cat_result.goal_domain,
            cat_relevance=cat_result.relevance,
        )

        # ── Final score ───────────────────────────────────────────────────────
        # Dual-component formula:
        #   relevance_score = priority + evidence + related + semantic + base
        #   quality_score   = EvidenceQualityScorer.total
        #   final = (1 - quality_weight) * relevance_score
        #         + quality_weight       * quality_score
        #         - negative_penalty
        #
        # Effective relevance weights (each × 0.70):
        #   priority=0.30, evidence=0.18, related=0.08, semantic=0.04, base=0.05
        # Effective quality weight: 0.30
        qw = cfg.quality_weight           # 0.30
        rw = 1.0 - qw                     # 0.70

        # Normalise relevance sub-weights so they sum to rw
        total_rel_w = (
            cfg.priority_weight + cfg.evidence_weight + cfg.related_weight
            + cfg.semantic_weight + cfg.base_weight
        ) or 1.0
        scale = rw / total_rel_w

        relevance_score = round(
            scale * (
                cfg.priority_weight * pri_score
                + cfg.evidence_weight * ev_score
                + cfg.related_weight * rel_score
                + cfg.semantic_weight * sem
                + cfg.base_weight * base
            ),
            4,
        )
        quality_score = round(qw * qs.total, 4)

        final = max(0.0, round(relevance_score + quality_score - penalty, 4))

        # Apply score cap for supporting-mode logs
        if score_cap is not None:
            final = min(final, score_cap)

        # ── Explanation trace ─────────────────────────────────────────────────
        matched_pri = [m.term for m in pri_matches if m.mode not in ("none", "weak_token_only")]
        matched_ev  = [m.term for m in ev_matches  if m.mode not in ("none", "weak_token_only")]

        if final > 0:
            reason_parts = []
            if gate_mode == "supporting":
                reason_parts.append(f"support_gate(matched={support_context_matched})")
            if matched_pri:
                reason_parts.append(f"priority={matched_pri}")
            if matched_ev:
                reason_parts.append(f"evidence={matched_ev}")
            if base >= 0.04:
                reason_parts.append(f"base_overlap={base:.3f}")
            reason_parts.append(
                f"cat={cat_result.log_category}({cat_result.relevance})"
            )
            reason_parts.append(
                f"quality(spec={qs.specificity:.2f},act={qs.actionability:.2f}"
                f",prog={qs.goal_progress:.2f})"
            )
            if score_cap is not None:
                reason_parts.append(f"score_cap={score_cap:.2f}")
            admission_reason = " | ".join(reason_parts) if reason_parts else "weak_match"
        else:
            admission_reason = ""

        logger.debug(
            "[Reranker Score]  log=%s  [%s]\n"
            "  category:        %s  (relevance=%s  domain=%s)\n"
            "  priority_phrase: %.3f  matched=%s\n"
            "  evidence_phrase: %.3f  matched=%s\n"
            "  related:         %.3f\n"
            "  semantic:        %.3f\n"
            "  base_overlap:    %.3f\n"
            "  → relevance_score: %.4f\n"
            "  specificity:     %.3f  (metrics=%s  nums=%s)\n"
            "  actionability:   %.3f  (hits=%s  browse=%s)\n"
            "  goal_progress:   %.3f  (cat_prior)\n"
            "  domain_consist:  %.3f\n"
            "  → quality_score: %.4f  (raw=%.4f × %.2f)\n"
            "  neg_penalty:     %.3f  (matched=%s)\n"
            "  → final:         %.4f  [rel=%.4f + qual=%.4f - pen=%.4f]",
            candidate.log.log_id, log_title,
            cat_result.log_category, cat_result.relevance, cat_result.goal_domain,
            pri_score, matched_pri,
            ev_score, matched_ev,
            rel_score, sem, base,
            relevance_score,
            qs.specificity, qs.has_metrics, qs.has_numbers,
            qs.actionability, qs.action_hits, qs.browse_hits,
            qs.goal_progress,
            qs.domain_consist,
            quality_score, qs.total, qw,
            penalty, neg_matched,
            final, relevance_score, quality_score, penalty,
        )

        return RankedLog(
            log=candidate.log,
            semantic_relevance=round(sem, 4),
            goal_focus=round(
                cfg.priority_weight * pri_score + cfg.evidence_weight * ev_score, 4
            ),
            evidence_value=round(qs.actionability, 4),
            final_score=final,
            matched_priority=matched_pri,
            matched_evidence=matched_ev,
            matched_negative=list(neg_matched),
            admission_reason=admission_reason,
            schema_category=cat_result.log_category,
            goal_domain=cat_result.goal_domain,
            category_hit_strength=cat_result.relevance,
            gate_mode=gate_mode,
            support_context_matched=support_context_matched,
            # Quality trace
            relevance_score=relevance_score,
            evidence_quality_score=quality_score,
            specificity_score=round(qs.specificity, 4),
            actionability_score=round(qs.actionability, 4),
            goal_progress_score=round(qs.goal_progress, 4),
        )

    def rank_log(
        self,
        goal: ResearchGoal,
        log: ResearchLog,
        expanded_terms: list[str] | None = None,
        negative_terms: list[str] | None = None,
        priority_terms: list[str] | None = None,
        related_terms: list[str] | None = None,
        skip_semantic_gate: bool = False,
    ) -> RankedLog:
        """Score a single ResearchLog and return full RankedLog (with category trace).

        Preferred over score_log() when caller needs category/admission info.
        Used by LocalExpander for neighbor re-admission.

        Parameters
        ----------
        skip_semantic_gate:
            When True, Tier2 semantic gate is skipped regardless of
            use_real_embeddings.  Set to True for Stage2 neighbor
            re-admission: neighbors are already gated by temporal locality
            and lexical filters, and dense_score=0.0 (raw cosine, not
            max-normalized) would give a misleading scale vs Stage1.
        """
        candidate = CandidateLog(log=log, sparse_score=0.0, dense_score=0.0, hybrid_score=0.0)
        results = self.rank(
            goal, [candidate],
            expanded_terms=expanded_terms,
            negative_terms=negative_terms,
            priority_terms=priority_terms,
            related_terms=related_terms,
            skip_semantic_gate=skip_semantic_gate,
        )
        return results[0]

    def score_log(
        self,
        goal: ResearchGoal,
        log: ResearchLog,
        expanded_terms: list[str] | None = None,
        negative_terms: list[str] | None = None,
        priority_terms: list[str] | None = None,
        related_terms: list[str] | None = None,
    ) -> float:
        """Return final_score only. Back-compat wrapper around rank_log()."""
        return self.rank_log(
            goal, log,
            expanded_terms=expanded_terms,
            negative_terms=negative_terms,
            priority_terms=priority_terms,
            related_terms=related_terms,
        ).final_score

    def rank(
        self,
        goal: ResearchGoal,
        candidates: list[CandidateLog],
        expanded_terms: list[str] | None = None,
        negative_terms: list[str] | None = None,
        priority_terms: list[str] | None = None,
        related_terms: list[str] | None = None,
        skip_semantic_gate: bool = False,
        disable_lexical_gate: bool = False,
    ) -> list[RankedLog]:
        ranked = [
            self.score(
                goal, c,
                expanded_terms=expanded_terms,
                negative_terms=negative_terms,
                priority_terms=priority_terms,
                related_terms=related_terms,
                skip_semantic_gate=skip_semantic_gate,
                disable_lexical_gate=disable_lexical_gate,
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
