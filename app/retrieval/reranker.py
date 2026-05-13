"""카테고리 우선 심사를 포함하는 어휘 제어(Lexical-Control) 기반 Reranker.

검색 아키텍처:
  Stage 1 (후보군) = 재현율(Recall)  → Dense 임베딩 검색 (Gemini embedding-001)
  Stage 2 (재정렬) = 정밀도(Precision) → 어휘 매칭 90%, 의미 일치 5–10%

입장 관문 (점수 계산 전 순서대로 적용됨):
  1. 행동 카테고리 관문  — 로그가 목표 도메인과 관련된 카테고리에 속해야 함
       "none" → 즉시 불합격 (category_mismatch)
  2. 목표 핵심어 관문    — 로그에 목표를 나타내는 직접적인 신호가 최소 1개 있어야 함
       pri=0 AND ev=0 AND base<커트라인 → 불합격 (no_goal_signal)
  3. 부정어 거부 관문    — 도메인이 충돌하면서 긍정적인 증거마저 없는 경우
       dm≥veto_threshold AND pri<veto_min → 불합격 (domain_conflict_veto)

점수 계산 공식 (모든 관문을 통과한 경우에만 계산):
  최종 점수 =
      우선순위_가중치   * 우선순위_구문_점수       (0.35)
    + 증거_가중치       * 증거_구문_점수           (0.20)
    + 연관_가중치       * 연관어_점수              (0.10)
    + 실행_가중치       * 실행_신호_점수           (0.15)
    + 도메인_가중치     * 도메인_일치도            (0.10)  ← 목표 도메인 반영
    + 의미_가중치       * 의미_유사도(Dense)       (0.05)
    + 기본_가중치       * 기본_목표문_일치도       (0.05)
    − 부정어_감점

스키마 신호:
  - 카테고리 매퍼(관문 로직)에서만 단독으로 사용됨
  - Reranker의 세부 채점 항목에는 주입되지 않음
  - 초기 후보 검색(Candidate) 단계는 Dense 검색만 사용함 (어휘 가중치 보정 없음)
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from app.config import RankerConfig
from app.retrieval.dense_retriever import DenseRetriever, cosine
from app.retrieval.embedding_provider import MockEmbeddingProvider
from app.retrieval.evidence_quality import _quality_scorer
from app.retrieval.schema_category import (
    classify_log_activity_type,
    get_goal_expected_activity_types,
    is_activity_type_compatible,
)
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

# ── Tier2 의미론적(Semantic) 관문 커트라인 ─────────────────────────────────────────────
# 실제 API(use_real_embeddings=True)를 사용할 때만 활성화됩니다.
# Mock 임베딩 환경에서는 무의미한 해시 기반 점수 때문에 정상적인 로그가
# 억울하게 탈락하는 것을 막기 위해 이 관문을 완전히 건너뜁니다.
# 기본값: 0.50.  튜닝 권장 범위: 0.45 – 0.55.
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
    """카테고리를 우선적으로 심사하는 어휘 기반 정밀도 Reranker.

    점수를 매기기 전에 카테고리 관문을 먼저 적용하여 다음을 보장합니다:
      - 일상(Lifestyle) 등 목표와 무관한 로그는 도메인 불일치로 즉시 거름
      - 실행 신호(action_signal)나 도메인 점수만 높은 '증거 없는 깡통 로그'의 꼼수 합격 방지
      - 오직 카테고리가 적합하고 목표 관련 신호가 있는 로그만 채점 단계에 진입
    """

    def __init__(
        self,
        config: RankerConfig | None = None,
        dense_retriever: DenseRetriever | None = None,
        negative_term_penalty: Optional[float] = None,  # back-compat, ignored
        use_real_embeddings: bool = False,
    ) -> None:
        self.config = config or RankerConfig()
        self._dense = dense_retriever or DenseRetriever(doc_provider=MockEmbeddingProvider())
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
        """약한 토큰 필터링이 적용된 우선순위 단어 매칭.

        score_priority_terms 함수를 사용하여 "완료", "정리" 같은 일반적인 단어들이
        단독으로 매칭되어 점수를 얻는 것(가짜 매칭)을 방지합니다.
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

    # ── Domain consistency ────────────────────────────────────────────────────

    def _domain_consistency(
        self, candidate: CandidateLog, log_activity_type: str
    ) -> float:
        """행동 유형(Activity-type) 기반의 도메인 일치도 (스키마 관문 대체).

        creative/execution → 높음 (구체적인 결과물이 있음)
        learning/planning  → 중간
        lifestyle/unknown  → 낮음
        """
        base_scores = {
            "creative":  0.85,
            "execution": 0.80,
            "learning":  0.60,
            "planning":  0.55,
            "lifestyle": 0.25,
            "unknown":   0.40,
        }
        base = base_scores.get(log_activity_type, 0.40)

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
        goal_tokens = _tok_set(goal.query_text)
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
        # [관문 1] Activity-Type Gate (행동 유형 관문)
        # 역할: 목표의 카테고리(예: 학습)와 로그의 카테고리(예: 일상)가 구조적으로
        #       완전히 엇갈리는 경우(예: 공부가 목표인데 식사 로그가 들어옴),
        #       내용을 볼 필요도 없이 즉시 불합격(Reject)시킵니다.
        # 특징: 알 수 없는 유형(unknown)이거나 오픈 도메인 목표는 항상 통과시킵니다.
        # ══════════════════════════════════════════════════════════════════════
        log_activity_type = classify_log_activity_type(
            candidate.log.title + " " + (candidate.log.content or "")
        )
        goal_activity_types = get_goal_expected_activity_types(
            goal.title,
            getattr(goal, "description", "") or "",
        )

        if goal_activity_types and not is_activity_type_compatible(
            log_activity_type, goal_activity_types
        ):
            logger.info(
                "[ACTIVITY_GATE_FAIL] %s  log_type=%s  goal_types=%s  [%s]",
                candidate.log.log_id, log_activity_type, goal_activity_types, log_title,
            )
            return RankedLog(
                log=candidate.log,
                semantic_relevance=0.0,
                goal_focus=0.0,
                evidence_value=0.0,
                final_score=0.0,
                rejection_reason=f"activity_type_mismatch(log={log_activity_type})",
                schema_category=log_activity_type,
                goal_domain="open",
                category_hit_strength="unknown",
            )

        logger.debug(
            "[ACTIVITY_GATE_PASS] %s  log_type=%s  goal_types=%s  [%s]",
            candidate.log.log_id, log_activity_type, goal_activity_types, log_title,
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
        # [관문 2] Goal Lexical Gate (핵심 어휘 관문)
        # 역할: 목표와 관련된 '결정적 단어(키워드)'가 로그 안에 실제로 존재하는지 검사합니다.
        # 
        # - Direct (정상 통과): 우선순위 단어, 증거 단어, 또는 기본 목표문과 4% 이상 일치하면
        #   제한 없이 점수 채점 단계로 넘어갑니다.
        # - Reject (불합격): 목표를 유추할 수 있는 단서 단어가 단 하나도 없다면
        #   가짜 합격(False Positive)으로 간주하고 가차 없이 불합격(no_goal_signal)시킵니다.
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
            # Support path: domain schema removed — always go to reject.
            # The lexical gate is the only admission path when primary_signal=False.
            logger.debug(
                "GOAL LEXICAL GATE REJECT  log=%s  log_type=%s  "
                "pri=%.3f  ev=%.3f  base=%.3f  [%s]",
                candidate.log.log_id, log_activity_type,
                pri_score, ev_score, base, log_title,
            )
            return RankedLog(
                log=candidate.log,
                semantic_relevance=0.0,
                goal_focus=0.0,
                evidence_value=0.0,
                final_score=0.0,
                gate_mode="reject",
                rejection_reason="no_goal_signal(pri=0,ev=0,base<0.04)",
                schema_category=log_activity_type,
                goal_domain="open",
                category_hit_strength="unknown",
            )

        # ══════════════════════════════════════════════════════════════════════
        # Remaining components
        # ══════════════════════════════════════════════════════════════════════
        rel_score = self._related_score(log_text, log_title, rel_terms)
        action = self._action_signal(candidate)
        domain = self._domain_consistency(candidate, log_activity_type)
        sem = self._semantic_similarity(
            goal, log_text,
            precomputed=candidate.dense_score if candidate.dense_score > 0 else None,
        )

        # ══════════════════════════════════════════════════════════════════════
        # [관문 3] Tier 2 Semantic Relevance Gate (AI 의미/문맥 관문)
        # 역할: 단어 관문을 통과했더라도, 전체적인 문맥(의미)이 정말로 비슷한지 봅니다.
        # 
        # - 실제 API(real embeddings)를 사용할 때만 작동합니다.
        # - AI가 판단한 임베딩(Dense) 유사도 점수가 커트라인(SEMANTIC_GATE_THRESHOLD)
        #   미만이라면, 단어만 겹쳤을 뿐 문맥이 다르다고 판단하여 불합격 처리합니다.
        # - 절대 평가 방식이므로, 초기 검색기가 억지로 가져온 쓰레기 로그들을 차단합니다.
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
                schema_category=log_activity_type,
                goal_domain="open",
                category_hit_strength="unknown",
            )
        else:
            logger.debug(
                "[TIER2_PASS] %s  dense=%.4f  real_emb=%s  [%s]",
                candidate.log.log_id, sem, self._use_real_embeddings, log_title,
            )

        # ── Negative penalty ──────────────────────────────────────────────────
        raw_dm, penalty, neg_matched = self._negative_penalty(candidate, neg_terms)

        # ══════════════════════════════════════════════════════════════════════
        # [관문 4] Negative Veto (부정어 거부 / 도메인 충돌 관문)
        # 역할: 목표와 완전히 반대되거나 무관한 도메인의 이야기인지 검사합니다.
        # 
        # - 무조건 거부하는 것이 아니라, 부정어 점수가 커트라인을 넘을 정도로 높고(심각한 충돌)
        #   동시에 이 로그를 살려줄 만한 강력한 '우선순위 단어'조차 없을 때만
        #   최종적으로 불합격(domain_conflict_veto)시킵니다.
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
                schema_category=log_activity_type,
                goal_domain="open",
                category_hit_strength="unknown",
            )

        # ── Evidence Quality Score ────────────────────────────────────────────
        # Separate from relevance — measures analysis value of the log.
        # Components: specificity, actionability, goal_progress, domain_consist
        qs = _quality_scorer.score(
            candidate.log,
            activity_type=log_activity_type,
        )

        # ── Final score ───────────────────────────────────────────────────────
        # 이중 구성(Dual-component) 점수 공식:
        #   관련성 점수 = 우선순위 + 증거 + 연관성 + 의미유사도 + 기본목표일치
        #   품질 점수   = EvidenceQualityScorer.total (구체성, 실행력 등)
        #   최종 점수 = (1 - 품질_가중치) * 관련성_점수
        #             + 품질_가중치       * 품질_점수
        #             - 부정어_감점
        #
        # 실질적인 관련성 가중치 비중 (각 항목 × 0.70):
        #   우선순위=0.30, 증거=0.18, 연관성=0.08, 의미유사도=0.04, 기본=0.05
        # 실질적인 품질 가중치 비중: 0.30
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
                f"log_type={log_activity_type}"
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
            log_activity_type, "open", "open",
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
            schema_category=log_activity_type,
            goal_domain="open",
            category_hit_strength="unknown",
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
        """단일 ResearchLog의 점수를 매기고 전체 RankedLog 객체를 반환합니다 (관문 기록 포함).

        호출자가 관문 통과 여부 및 카테고리 정보가 필요할 때 score_log() 대신 사용됩니다.
        Stage 2의 LocalExpander에서 이웃 로그들을 재심사할 때 사용됩니다.

        Parameters
        ----------
        skip_semantic_gate:
            True일 경우, use_real_embeddings 설정과 무관하게 Tier 2 의미론적 관문을 건너뜁니다.
            Stage 2 이웃 재심사 시 True로 설정됨: 이웃 로그들은 이미 시간적 인접성과
            어휘 관문을 거쳤으며, 여기서의 dense_score는 Stage 1처럼 정규화된 점수가 아니라
            0.0으로 들어오기 때문에 커트라인을 적용하면 모두 억울하게 탈락하기 때문입니다.
        """
        candidate = CandidateLog(log=log, dense_score=0.0)
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
