"""Goal-Conditioned Evidence Ranker.

Final score:
  0.30 * semantic_relevance
+ 0.50 * goal_focus            ← primary signal
+ 0.10 * evidence_strength
+ priority_boost               ← additive: phrase=+0.30, token=+0.10
- 0.40 * domain_mismatch       ← negative penalty

goal_focus (3-tier, title-aware):
  0.60 * priority_overlap      ← phrase/token × title_multiplier
+ 0.25 * evidence_overlap
+ 0.10 * related_overlap
+ 0.05 * base_goal_overlap

domain_mismatch (phrase-first, title-extra):
  phrase match → +0.70
  token match  → +0.40
  in title     → extra +0.30
  daily type   → +0.20
  cap at 1.0

Phrase match always beats token match. Title match adds a multiplier bonus.
All matching delegated to app.utils.text_matching.
"""
from __future__ import annotations

import logging
from typing import Optional

from app.config import RankerConfig
from app.retrieval.dense_retriever import DenseRetriever, cosine
from app.schemas import CandidateLog, RankedLog, ResearchGoal
from app.utils.text_matching import TermMatch, score_terms, penalty_score, _tok_set

logger = logging.getLogger(__name__)

_PRODUCTIVE_TYPES = {
    "study", "implementation", "reading", "exercise",
    "planning", "execution", "reflection", "coding", "social",
}
_NOISE_TYPES = {"daily"}

_EVIDENCE_KEYWORDS = {
    "완료", "완성", "제출", "발표", "구현", "해결", "달성", "성공",
    "finished", "completed", "implemented", "solved", "achieved",
    "공부", "연습", "작성", "읽기", "풀이", "실습", "수행",
}


class GoalConditionedReranker:
    """Multi-component scorer: semantic + goal_focus + evidence - penalty."""

    def __init__(
        self,
        config: RankerConfig | None = None,
        dense_retriever: DenseRetriever | None = None,
        negative_term_penalty: Optional[float] = None,
    ) -> None:
        self.config = config or RankerConfig()
        self._dense = dense_retriever or DenseRetriever()
        self.negative_term_penalty = (
            negative_term_penalty if negative_term_penalty is not None
            else self.config.negative_term_penalty
        )

    # ── Semantic relevance ────────────────────────────────────────────────────

    def _semantic_relevance(self, goal: ResearchGoal, log_text: str) -> float:
        g_emb = self._dense.embed(goal.query_text)
        l_emb = self._dense.embed(log_text)
        return max(0.0, cosine(g_emb, l_emb))

    # ── Goal focus (3-tier, title-aware) ──────────────────────────────────────

    def _goal_focus(
        self,
        goal: ResearchGoal,
        log_text: str,
        log_title: str,
        priority_terms: list[str],
        evidence_terms: list[str],
        related_terms: list[str],
    ) -> tuple[float, dict]:
        """3-tier goal relevance score with title weighting.

        Returns (score, debug_dict) where debug_dict has per-tier breakdown.
        """
        cfg = self.config
        tm = cfg.title_weight_multiplier

        pri_score, pri_matches = score_terms(
            priority_terms, log_text, log_title,
            phrase_weight=1.5, token_weight=1.0, title_multiplier=tm,
        )
        ev_score, ev_matches = score_terms(
            evidence_terms, log_text, log_title,
            phrase_weight=1.5, token_weight=1.0, title_multiplier=tm,
        )
        rel_score, rel_matches = score_terms(
            related_terms, log_text, log_title,
            phrase_weight=1.5, token_weight=1.0, title_multiplier=tm,
        )

        # Base anchor: raw goal text token overlap (small, non-dominant)
        base_tokens = _tok_set(goal.query_text + " " + goal.goal_embedding_text)
        log_tokens = _tok_set(log_text)
        base_score = len(base_tokens & log_tokens) / len(base_tokens) if base_tokens else 0.0

        final = (
            cfg.priority_focus_weight * pri_score
            + cfg.evidence_focus_weight * ev_score
            + cfg.related_focus_weight * rel_score
            + cfg.base_goal_anchor * base_score
        )

        debug = {
            "priority_overlap": round(pri_score, 3),
            "evidence_overlap": round(ev_score, 3),
            "related_overlap": round(rel_score, 3),
            "base_overlap": round(base_score, 3),
            "pri_matched": [m.term for m in pri_matches if m.level != "none"],
            "ev_matched": [m.term for m in ev_matches if m.level != "none"],
        }
        return min(final, 1.0), debug

    # ── Evidence value ────────────────────────────────────────────────────────

    def _evidence_value(self, candidate: CandidateLog) -> float:
        tokens = _tok_set(candidate.log.full_text)
        keyword_score = min(len(tokens & _EVIDENCE_KEYWORDS) / 3.0, 1.0)
        specificity = min(len(tokens) / 50.0, 0.4)
        act_bonus = 0.2 if candidate.log.activity_type in _PRODUCTIVE_TYPES else 0.0
        strength = candidate.log.metadata.get("evidence_strength", "medium")
        strength_bonus = {"high": 0.2, "medium": 0.1, "low": 0.0}.get(strength, 0.0)
        return min(keyword_score + specificity + act_bonus + strength_bonus, 1.0)

    # ── Domain mismatch (negative penalty) ───────────────────────────────────

    def _domain_mismatch(
        self,
        candidate: CandidateLog,
        negative_terms: list[str],
    ) -> tuple[float, list[str]]:
        """Penalty 0.0→1.0.  Returns (penalty, matched_descriptions)."""
        cfg = self.config
        log_text = candidate.log.full_text
        log_title = candidate.log.title

        raw_penalty, matched = penalty_score(
            negative_terms, log_text, log_title,
            phrase_penalty=cfg.negative_penalty_phrase,
            token_penalty=cfg.negative_penalty_token,
            title_extra_penalty=cfg.negative_penalty_title,
        )

        # Activity-type noise penalty (relaxed: only for non-trivially daily logs)
        noise = cfg.negative_daily_penalty if candidate.log.activity_type in _NOISE_TYPES else 0.0

        total = min(raw_penalty + noise, 1.0)

        if total > 0.15:
            logger.debug(
                "NegMatch  log=%s  dm=%.2f  matched=%s  [%s]",
                candidate.log.log_id, total, matched, candidate.log.title,
            )
        return total, matched

    # ── Priority boost (additive, outside goal_focus) ─────────────────────────

    def _priority_boost(
        self,
        log_text: str,
        log_title: str,
        priority_terms: list[str],
    ) -> tuple[float, str]:
        """Additive boost for logs matching high-signal priority terms.

        phrase match → priority_boost_phrase
        token match  → priority_boost_token
        Max capped at priority_boost_phrase (one strong hit is enough).
        """
        if not priority_terms:
            return 0.0, ""
        cfg = self.config
        text_lower = log_text.lower()
        text_tokens = _tok_set(text_lower)
        title_lower = log_title.lower()
        title_tokens = _tok_set(title_lower)

        best_level = "none"
        best_term = ""
        from app.utils.text_matching import match_term
        for term in priority_terms:
            m = match_term(term, text_lower, text_tokens, title_lower, title_tokens)
            if m.level == "phrase":
                best_level = "phrase"
                best_term = term
                break  # phrase is the best possible, stop early
            elif m.level == "token" and best_level == "none":
                best_level = "token"
                best_term = term

        if best_level == "phrase":
            boost = cfg.priority_boost_phrase
            tag = f"phrase:{best_term}"
        elif best_level == "token":
            boost = cfg.priority_boost_token
            tag = f"token:{best_term}"
        else:
            return 0.0, ""

        logger.debug(
            "PriorityBoost  level=%s  term=%s  boost=%.4f  [%s]",
            best_level, best_term, boost, log_text[:60],
        )
        return boost, tag

    # ── Main scoring entry point ──────────────────────────────────────────────

    def score(
        self,
        goal: ResearchGoal,
        candidate: CandidateLog,
        expanded_terms: list[str] | None = None,    # evidence_terms
        negative_terms: list[str] | None = None,
        priority_terms: list[str] | None = None,
        related_terms: list[str] | None = None,
    ) -> RankedLog:
        ev_terms = expanded_terms or []
        neg_terms = negative_terms or []
        pri_terms = priority_terms or []
        rel_terms = related_terms or []

        log_text = candidate.log.full_text
        log_title = candidate.log.title

        sr = self._semantic_relevance(goal, log_text)
        gf, gf_debug = self._goal_focus(goal, log_text, log_title, pri_terms, ev_terms, rel_terms)
        ev = self._evidence_value(candidate)
        dm, dm_matched = self._domain_mismatch(candidate, neg_terms)
        pb, pb_tag = self._priority_boost(log_text, log_title, pri_terms)

        w = self.config
        final = max(0.0, round(
            w.semantic_weight * sr
            + w.goal_focus_weight * gf
            + w.evidence_value_weight * ev
            + pb
            - self.negative_term_penalty * dm,
            4,
        ))

        logger.debug(
            "[Scoring Breakdown]  log=%s  [%s]\n"
            "  priority_overlap: %.3f  evidence_overlap: %.3f  related_overlap: %.3f\n"
            "  pri_matched=%s  ev_matched=%s\n"
            "  priority_boost: +%.3f (%s)\n"
            "  domain_mismatch: %.3f  neg_matched=%s\n"
            "  sr=%.3f  gf=%.3f  ev=%.3f  → final=%.4f",
            candidate.log.log_id, log_title,
            gf_debug["priority_overlap"], gf_debug["evidence_overlap"], gf_debug["related_overlap"],
            gf_debug["pri_matched"], gf_debug["ev_matched"],
            pb, pb_tag,
            dm, dm_matched,
            sr, gf, ev, final,
        )

        return RankedLog(
            log=candidate.log,
            semantic_relevance=round(sr, 4),
            goal_focus=round(gf, 4),
            evidence_value=round(ev, 4),
            final_score=final,
        )

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
        logger.debug("Reranked %d candidates  goal=%s", len(ranked), goal.goal_id)
        return ranked
