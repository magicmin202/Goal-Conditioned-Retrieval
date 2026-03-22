"""Stage 2: Anchor-centered Evidence Consolidation Pipeline.

CONTRACT — Stage 2 is NOT a retrieval stage.
============================================================
FORBIDDEN in Stage 2:
  ✗ CandidateRetriever / global corpus retrieval
  ✗ RetrievalMode.HYBRID_EXPANDED
  ✗ expanded query → broad retrieval
  ✗ re-selecting anchors from corpus
  ✗ DiversitySelector (Stage 1 already did this)

REQUIRED in Stage 2:
  ✓ Accepts Stage1 admitted anchors as fixed input
  ✓ Local temporal expansion around each anchor only
  ✓ Neighbor re-admission via same reranker gate
  ✓ Non-admitted logs NEVER reach the compressor
  ✓ Compressor = summarization only

Flow:
  Stage1Result.selected_logs   (admitted anchors)
       ↓
  [mark anchor_source = "stage1"]
       ↓
  Anchor-centered Local Expansion  (± temporal window)
       ↓
  Neighbor Re-admission Check       (reranker gate)
       ↓
  Admitted Cluster Formation
       ↓
  Cluster Summarization (compressor)
       ↓
  LLM Analysis
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.compression.local_expansion import LocalExpander
from app.compression.temporal_semantic_compressor import TemporalSemanticCompressor
from app.config import AdaptiveMode, AdaptivePolicyConfig, ConsolidationConfig, Stage2Config
from app.llm.analysis import GoalAnalyzer
from app.llm.llm_client import get_llm_client
from app.retrieval.query_expansion import ExpandedQuery
from app.retrieval.reranker import GoalConditionedReranker
from app.schemas import (
    CompressedEvidenceUnit, RankedLog, ResearchGoal, ResearchLog,
)

logger = logging.getLogger(__name__)


def _detect_adaptive_mode(
    logs: list[ResearchLog],
    policy: AdaptivePolicyConfig | None = None,
) -> AdaptiveMode:
    """Infer corpus density mode from log count and date span."""
    p = policy or AdaptivePolicyConfig()
    n = len(logs)
    if n == 0:
        return AdaptiveMode.SMALL

    from datetime import date as _date
    dates = sorted(log.date for log in logs if log.date)
    if len(dates) >= 2:
        try:
            span = (_date.fromisoformat(dates[-1]) - _date.fromisoformat(dates[0])).days
        except (ValueError, TypeError):
            span = 0
    else:
        span = 0

    if n < p.small_log_threshold or span < p.small_span_threshold:
        mode = AdaptiveMode.SMALL
    elif n >= p.large_log_threshold and span >= p.large_span_threshold:
        mode = AdaptiveMode.LARGE
    else:
        mode = AdaptiveMode.STANDARD

    logger.info(
        "AdaptiveMode=%s  logs=%d  span_days=%d",
        mode, n, span,
    )
    return mode


def _temporal_window(mode: AdaptiveMode, cfg: ConsolidationConfig) -> int:
    if mode == AdaptiveMode.SMALL:
        return cfg.local_expansion_window_small
    if mode == AdaptiveMode.LARGE:
        return cfg.local_expansion_window_large
    return cfg.local_expansion_window_standard


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class Stage2Result:
    goal: ResearchGoal
    anchors: list[RankedLog]                  # Stage1 anchors (fixed input)
    evidence_units: list[CompressedEvidenceUnit]
    llm_analysis: str
    query_text: str
    expanded_terms: list[str] = field(default_factory=list)
    negative_terms: list[str] = field(default_factory=list)
    priority_terms: list[str] = field(default_factory=list)
    adaptive_mode: str = AdaptiveMode.STANDARD
    metadata: dict = field(default_factory=dict)
    # Back-compat aliases
    candidates: list = field(default_factory=list)   # always empty in consolidation mode
    ranked_logs: list = field(default_factory=list)  # always empty in consolidation mode
    selected_logs: list[RankedLog] = field(default_factory=list)  # = anchors


# ── Pipeline ──────────────────────────────────────────────────────────────────

class Stage2Pipeline:
    """Anchor-centered Evidence Consolidation Pipeline.

    Stage 2 does NOT do global retrieval.
    It receives Stage 1 admitted anchors and consolidates them.

    Usage:
        # Option A: explicit anchor passing
        s2 = Stage2Pipeline(config, ...)
        s2.index(all_logs)   # stores corpus for temporal expansion only
        result = s2.run(goal, anchors=stage1_result.selected_logs,
                        expanded_query=stage1_result.expanded_query)

        # Option B: convenience wrapper
        result = s2.run_with_stage1(stage1_result)
    """

    def __init__(
        self,
        config: Stage2Config | None = None,
        use_mock_llm: bool = False,
        use_real_embeddings: bool = False,  # kept for API compat; used by reranker embed
    ) -> None:
        self.config = config or Stage2Config()
        # Stage 2 has NO CandidateRetriever — no global retrieval.
        self._reranker = GoalConditionedReranker(config=self.config.ranker)
        self._expander = LocalExpander(
            anchor_relevance_threshold=self.config.consolidation.anchor_admission_threshold,
        )
        self._compressor = TemporalSemanticCompressor()
        self._analyzer = GoalAnalyzer(llm=get_llm_client(mock=use_mock_llm))
        self._all_logs: list[ResearchLog] = []
        self._adaptive_mode: AdaptiveMode = AdaptiveMode.STANDARD
        self._indexed = False

        logger.info(
            "Stage2Pipeline initialized [consolidation_mode=True  NO global retrieval]"
        )

    def index(self, logs: list[ResearchLog]) -> None:
        """Store corpus for temporal neighbor lookup only — no retrieval indexing."""
        self._all_logs = logs
        self._adaptive_mode = _detect_adaptive_mode(logs)
        self._indexed = True
        logger.info(
            "Stage2 indexed %d logs for temporal expansion  mode=%s",
            len(logs), self._adaptive_mode,
        )

    def run(
        self,
        goal: ResearchGoal,
        anchors: list[RankedLog],
        expanded_query: ExpandedQuery | None = None,
    ) -> Stage2Result:
        """Consolidate Stage1 admitted anchors into evidence units.

        Parameters
        ----------
        goal:
            The research goal (used for LLM analysis prompt).
        anchors:
            Stage1 admitted anchors. These are FIXED — Stage2 does not
            re-select or re-retrieve anchors from the corpus.
        expanded_query:
            Vocabulary context from Stage1 expansion, used for neighbor
            admission scoring. If None, neighbor admission uses goal text only.
        """
        if not self._indexed:
            raise RuntimeError("Call index() before run().")
        if not anchors:
            logger.warning("Stage2.run(): no anchors provided — returning empty result")
            return Stage2Result(
                goal=goal, anchors=[], selected_logs=[],
                evidence_units=[], llm_analysis="",
                query_text=goal.query_text,
            )

        cfg = self.config.consolidation
        mode = self._adaptive_mode
        logger.info(
            "Stage2 consolidation  goal=%s  anchors=%d  mode=%s  window=adaptive",
            goal.goal_id, len(anchors), mode,
        )

        # ── Mark anchor source ────────────────────────────────────────────────
        for a in anchors:
            a.anchor_source = "stage1"

        logger.info("Stage2 anchor set (from Stage1):")
        for a in anchors:
            logger.info(
                "  [stage1] %s  date=%s  score=%.4f  reason=%s  [%s]",
                a.log_id, a.log.date, a.final_score, a.admission_reason, a.log.title,
            )

        # ── Extract vocabulary for neighbor admission ─────────────────────────
        exp_terms = expanded_query.expanded_terms if expanded_query else []
        neg_terms = expanded_query.negative_terms if expanded_query else []
        pri_terms = expanded_query.priority_terms if expanded_query else []
        rel_terms = expanded_query.related_terms if expanded_query else []

        # ── Anchor-centered Local Expansion ───────────────────────────────────
        # window = adaptive based on corpus density
        window = _temporal_window(mode, cfg)
        logger.info(
            "Stage2 local expansion  temporal_window=±%dd  "
            "neighbor_admission_threshold=%.3f",
            window, cfg.neighbor_admission_threshold,
        )

        expansion_map = self._expander.expand(
            anchors,
            self._all_logs,
            goal=goal,
            expanded_terms=exp_terms,
            negative_terms=neg_terms,
            priority_terms=pri_terms,
            related_terms=rel_terms,
            reranker=self._reranker,
            temporal_window=window,
            neighbor_admission_threshold=cfg.neighbor_admission_threshold,
            max_neighbors=cfg.max_neighbors_per_anchor,
        )

        # Mark admitted neighbors' source
        for neighbors in expansion_map.values():
            for n in neighbors:
                # Wrap into a quick lookup for logging (neighbors are ResearchLog, not RankedLog)
                pass

        # ── Cluster Summarization (compressor = summarization only) ───────────
        evidence_units = self._compressor.compress(anchors, expansion_map)
        logger.info(
            "Stage2 consolidation complete  anchors=%d  evidence_units=%d  goal=%s",
            len(anchors), len(evidence_units), goal.goal_id,
        )

        # ── LLM Analysis ──────────────────────────────────────────────────────
        analysis = self._analyzer.analyze(goal, evidence_units)

        query_text = expanded_query.dense_query if expanded_query else goal.query_text
        return Stage2Result(
            goal=goal,
            anchors=anchors,
            selected_logs=anchors,   # back-compat
            evidence_units=evidence_units,
            llm_analysis=analysis,
            query_text=query_text,
            expanded_terms=exp_terms,
            negative_terms=neg_terms,
            priority_terms=pri_terms,
            adaptive_mode=mode,
            metadata={
                "adaptive_mode": mode,
                "stage1_anchors": len(anchors),
                "temporal_window_days": window,
                "neighbor_admission_threshold": cfg.neighbor_admission_threshold,
                "evidence_units": len(evidence_units),
                "source": "anchor_consolidation",
            },
        )

    def run_with_stage1(self, stage1_result: "Stage1Result") -> Stage2Result:
        """Convenience method: run Stage2 consolidation from a Stage1Result.

        This is the canonical chain:
            stage1_result = Stage1Pipeline.run(goal)
            stage2_result = Stage2Pipeline.run_with_stage1(stage1_result)

        Stage2 uses stage1_result.selected_logs as fixed anchors.
        Stage2 uses stage1_result.expanded_query for neighbor admission vocabulary.
        """
        logger.info(
            "Stage2.run_with_stage1()  goal=%s  stage1_anchors=%d",
            stage1_result.goal.goal_id, len(stage1_result.selected_logs),
        )
        return self.run(
            goal=stage1_result.goal,
            anchors=stage1_result.selected_logs,
            expanded_query=stage1_result.expanded_query,
        )


# Deferred import to avoid circular at module level
from app.pipeline.stage1_ranking_pipeline import Stage1Result  # noqa: E402
