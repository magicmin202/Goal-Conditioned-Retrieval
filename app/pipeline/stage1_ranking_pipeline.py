"""Stage 1: Candidate Retrieval → Reranking → Relevance Filtering → Diversity Selection.

Core method: raw goal (no expansion).
Expansion variant: set use_expansion=True (optional baseline).
expanded_terms and negative_terms are passed to the reranker in both cases.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.config import Stage1Config
from app.retrieval.candidate_retrieval import CandidateRetriever, RetrievalMode
from app.retrieval.diversity_selector import DiversitySelector
from app.retrieval.evidence_quality import compute_redundancy_penalty
from app.retrieval.query_expansion import ExpandedQuery, expand_goal_query
from app.retrieval.query_understanding import build_query
from app.retrieval.reranker import GoalConditionedReranker
from app.schemas import CandidateLog, RankedLog, ResearchGoal, ResearchLog

logger = logging.getLogger(__name__)


@dataclass
class Stage1Result:
    goal: ResearchGoal
    candidates: list[CandidateLog]
    ranked_logs: list[RankedLog]
    selected_logs: list[RankedLog]       # admitted anchors — contract: Stage2 input
    query_text: str
    used_expansion: bool = False
    expanded_terms: list[str] = field(default_factory=list)
    negative_terms: list[str] = field(default_factory=list)
    priority_terms: list[str] = field(default_factory=list)
    related_terms: list[str] = field(default_factory=list)
    expanded_query: ExpandedQuery | None = None   # carry for Stage2 neighbor admission
    metadata: dict = field(default_factory=dict)


class Stage1Pipeline:
    """Goal → Candidate Retrieval → Reranking → Relevance Filtering → Diversity."""

    def __init__(
        self,
        config: Stage1Config | None = None,
        use_real_embeddings: bool = False,
    ) -> None:
        from app.retrieval.embedding_provider import get_embedding_provider
        self.config = config or Stage1Config()
        embed_provider = get_embedding_provider(real=use_real_embeddings)
        self._retriever = CandidateRetriever(
            mode=RetrievalMode.HYBRID,
            config=self.config.retrieval,
            candidate_config=self.config.candidate,
            vocab_boost_config=self.config.vocab_boost,
            embedding_provider=embed_provider,
        )
        self._reranker = GoalConditionedReranker(config=self.config.ranker)
        self._selector = DiversitySelector(config=self.config.diversity)
        self._indexed = False

    def index(self, logs: list[ResearchLog]) -> None:
        self._retriever.index(logs)
        self._indexed = True

    def run(
        self,
        goal: ResearchGoal,
        use_expansion: bool | None = None,
        run_label: str = "stage1_standalone",
    ) -> Stage1Result:
        if not self._indexed:
            raise RuntimeError("Call index() before run().")

        expand = (
            use_expansion if use_expansion is not None
            else self.config.query_expansion.enabled
        )

        # ── Run-level diagnostic trace ─────────────────────────────────────────
        logger.info(
            "[Stage1 Run]  label=%s  goal=%s  user=%s\n"
            "  expansion=%s  candidate_size=%d  top_k=%d\n"
            "  admission_threshold=%.3f  schema_enabled=%s",
            run_label, goal.goal_id, goal.user_id,
            expand,
            self.config.retrieval.candidate_size,
            self.config.retrieval.top_k,
            self.config.diversity.relevance_threshold,
            getattr(self.config, "schema_category", None) is not None,
        )

        # 1. Query Understanding
        query_obj = build_query(goal)

        # 2. Optional Query Expansion
        expanded_terms: list[str] = []
        priority_terms: list[str] = []
        related_terms: list[str] = []
        negative_terms: list[str] = []
        expanded_query_obj: ExpandedQuery | None = None

        if expand:
            expanded = expand_goal_query(
                goal, query_obj,
                max_terms=self.config.query_expansion.max_terms,
                mode=self.config.query_expansion.mode,
                use_mock_fallback=self.config.query_expansion.use_mock_fallback,
            )
            active_query = expanded
            expanded_query_obj = expanded
            expanded_terms = expanded.expanded_terms
            priority_terms = expanded.priority_terms
            related_terms = expanded.related_terms
            negative_terms = expanded.negative_terms
            logger.info(
                "Stage1 expansion  goal=%s\n"
                "  priority=%s\n  evidence=%s\n  related=%s\n  negative=%s",
                goal.goal_id, priority_terms, expanded_terms, related_terms, negative_terms,
            )
        else:
            # Even without expansion, use heuristic terms for reranker scoring
            from app.retrieval.query_expansion import _heuristic_expansion
            heuristic = _heuristic_expansion(goal, max_terms=15)
            expanded_terms = heuristic.get("evidence_terms", [])
            priority_terms = heuristic.get("priority_terms", expanded_terms[:5])
            related_terms = heuristic.get("related_terms", [])
            negative_terms = heuristic.get("negative_terms", [])
            active_query = query_obj

        # ── Query trace ────────────────────────────────────────────────────────
        if expand and expanded_query_obj:
            logger.info(
                "[Stage1 Query]  goal=%s  label=%s\n"
                "  bm25_q (first 80): %s\n"
                "  dense_q (first 80): %s",
                goal.goal_id, run_label,
                expanded_query_obj.bm25_query[:80],
                expanded_query_obj.dense_query[:80],
            )
        else:
            logger.info(
                "[Stage1 Query]  goal=%s  label=%s  (no expansion)\n"
                "  canonical (first 80): %s",
                goal.goal_id, run_label,
                query_obj.canonical_text[:80],
            )

        # 3. Candidate Retrieval (+ vocab boost if ExpandedQuery)
        candidates = self._retriever.retrieve(
            active_query, top_n=self.config.retrieval.candidate_size
        )
        logger.info("Stage1: %d candidates  goal=%s", len(candidates), goal.goal_id)

        # ── Candidate diagnostic trace ─────────────────────────────────────────
        logger.info("[Stage1 Candidates top-10]  goal=%s  label=%s", goal.goal_id, run_label)
        for c in candidates[:10]:
            logger.info(
                "  %s  bm25=%.4f  dense=%.4f  hybrid=%.4f  [%s]",
                c.log_id, c.sparse_score, c.dense_score, c.hybrid_score, c.log.title,
            )

        # 4. Goal-Conditioned Reranking (3-tier goal_focus)
        ranked = self._reranker.rank(
            goal, candidates,
            expanded_terms=expanded_terms,
            negative_terms=negative_terms,
            priority_terms=priority_terms,
            related_terms=related_terms,
        )

        # 5. Relevance Filtering — strict admission, no fill-to-k fallback
        # "fewer but correct" > "full but noisy"
        top_k = self.config.retrieval.top_k
        threshold = self.config.diversity.relevance_threshold
        above = [r for r in ranked if r.final_score >= threshold]
        logger.info(
            "Stage1 admission: %d/%d logs admitted  threshold=%.3f  label=%s",
            len(above), len(ranked), threshold, run_label,
        )

        # 5b. Redundancy Penalty (applied greedy, high-score first)
        # Logs that are near-duplicates of already-admitted logs get penalised.
        # This pushes out repeated low-info logs (e.g. "여행 준비물 쇼핑" ×3).
        rdup_cfg = self.config.ranker
        admitted_logs: list = []
        penalised: list[RankedLog] = []
        for r in above:
            pen, reason = compute_redundancy_penalty(
                r.log,
                admitted_logs,
                exact_penalty=rdup_cfg.redundancy_penalty_exact,
                similar_penalty=rdup_cfg.redundancy_penalty_similar,
                similarity_threshold=rdup_cfg.redundancy_similarity_threshold,
            )
            if pen > 0:
                r.redundancy_penalty = round(pen, 4)
                r.final_score = max(0.0, round(r.final_score - pen, 4))
                r.rejection_reason = r.rejection_reason or f"redundancy:{reason}"
                logger.info(
                    "  [REDUNDANCY] %s  penalty=%.2f  reason=%s  new_score=%.4f  [%s]",
                    r.log_id, pen, reason, r.final_score, r.log.title,
                )
            admitted_logs.append(r.log)
            penalised.append(r)

        # Re-sort after redundancy penalty
        penalised.sort(key=lambda x: x.final_score, reverse=True)
        for i, r in enumerate(penalised):
            r.rank = i + 1

        # Per-candidate admission trace (INFO level for debugging)
        for r in ranked[:15]:
            rel = r.relevance_score if hasattr(r, "relevance_score") else 0.0
            qual = r.evidence_quality_score if hasattr(r, "evidence_quality_score") else 0.0
            rpen = r.redundancy_penalty if hasattr(r, "redundancy_penalty") else 0.0
            decision = "ADMIT" if r.final_score >= threshold else "REJECT"
            logger.info(
                "  [%s] %s  score=%.4f  rel=%.4f  qual=%.4f  red_pen=%.3f"
                "  spec=%.3f  act=%.3f  prog=%.3f"
                "  cat=%s(%s)  reason=%s  [%s]",
                decision, r.log_id, r.final_score,
                rel, qual, rpen,
                getattr(r, "specificity_score", 0.0),
                getattr(r, "actionability_score", 0.0),
                getattr(r, "goal_progress_score", 0.0),
                getattr(r, "schema_category", "?"),
                getattr(r, "category_hit_strength", "?"),
                r.rejection_reason or r.admission_reason or "-",
                r.log.title,
            )

        # 6. Diversity-aware Top-K Selection (from admitted only, after redundancy)
        pool = penalised
        selected = self._selector.select(goal, pool, top_k=top_k)

        query_text = (
            active_query.full_text
            if hasattr(active_query, "full_text")
            else active_query.canonical_text
        )

        return Stage1Result(
            goal=goal,
            candidates=candidates,
            ranked_logs=ranked,
            selected_logs=selected,
            query_text=query_text,
            used_expansion=expand,
            expanded_terms=expanded_terms,
            negative_terms=negative_terms,
            priority_terms=priority_terms,
            related_terms=related_terms,
            expanded_query=expanded_query_obj,
            metadata={
                "candidate_size": len(candidates),
                "admitted": len(above),
                "top_k": len(selected),
            },
        )
