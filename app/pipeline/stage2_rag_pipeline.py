"""Stage 2: Enhanced Retrieval Pipeline with full RAG.

Goal → Query Understanding → LLM Query Expansion
→ Hybrid Candidate Retrieval → Reranking → Diversity Selection
→ Anchor-based Local Expansion → Temporal/Semantic Compression
→ LLM Analysis
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from app.compression.local_expansion import LocalExpander
from app.compression.temporal_semantic_compressor import TemporalSemanticCompressor
from app.config import Stage2Config
from app.llm.analysis import GoalAnalyzer
from app.llm.llm_client import get_llm_client
from app.retrieval.candidate_retrieval import CandidateRetriever, RetrievalMode
from app.retrieval.diversity_selector import DiversitySelector
from app.retrieval.query_expansion import expand_goal_query
from app.retrieval.query_understanding import build_query
from app.retrieval.reranker import GoalConditionedReranker
from app.schemas import CandidateLog, CompressedEvidenceUnit, RankedLog, ResearchGoal, ResearchLog

logger = logging.getLogger(__name__)


@dataclass
class Stage2Result:
    goal: ResearchGoal
    candidates: list[CandidateLog]
    ranked_logs: list[RankedLog]
    selected_logs: list[RankedLog]
    evidence_units: list[CompressedEvidenceUnit]
    llm_analysis: str
    query_text: str
    metadata: dict = field(default_factory=dict)


class Stage2Pipeline:
    """Full RAG pipeline for Stage 2."""

    def __init__(self, config: Stage2Config | None = None, use_mock_llm: bool = True) -> None:
        self.config = config or Stage2Config()
        self._retriever = CandidateRetriever(
            mode=RetrievalMode.HYBRID_EXPANDED, config=self.config.retrieval
        )
        self._reranker = GoalConditionedReranker(config=self.config.ranker)
        self._selector = DiversitySelector(config=self.config.diversity)
        self._expander = LocalExpander(
            similarity_threshold=self.config.compression.cluster_similarity_threshold
        )
        self._compressor = TemporalSemanticCompressor()
        self._analyzer = GoalAnalyzer(llm=get_llm_client(mock=use_mock_llm))
        self._all_logs: list[ResearchLog] = []
        self._indexed = False

    def index(self, logs: list[ResearchLog]) -> None:
        self._retriever.index(logs)
        self._all_logs = logs
        self._indexed = True

    def run(self, goal: ResearchGoal) -> Stage2Result:
        if not self._indexed:
            raise RuntimeError("Call index() before run().")

        query_obj = build_query(goal)
        expanded = expand_goal_query(
            goal, query_obj,
            max_terms=self.config.query_expansion.max_terms,
            mode=self.config.query_expansion.mode,
        )
        logger.info("Stage2: expanded terms=%s", expanded.expanded_terms)

        candidates = self._retriever.retrieve(expanded, top_n=self.config.retrieval.candidate_size)
        logger.info("Stage2: %d candidates retrieved", len(candidates))

        ranked = self._reranker.rank(goal, candidates)
        selected = self._selector.select(goal, ranked, top_k=self.config.retrieval.top_k)

        expansion_map = self._expander.expand(selected, self._all_logs)
        evidence_units = self._compressor.compress(selected, expansion_map)
        logger.info("Stage2: %d evidence units", len(evidence_units))

        analysis = self._analyzer.analyze(goal, evidence_units)

        return Stage2Result(
            goal=goal,
            candidates=candidates,
            ranked_logs=ranked,
            selected_logs=selected,
            evidence_units=evidence_units,
            llm_analysis=analysis,
            query_text=expanded.full_text,
            metadata={
                "expanded_terms": expanded.expanded_terms,
                "candidate_size": len(candidates),
                "top_k": len(selected),
                "evidence_units": len(evidence_units),
            },
        )
