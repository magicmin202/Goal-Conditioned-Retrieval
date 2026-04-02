"""Pydantic-style dataclass schemas for the research pipeline."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ResearchUser:
    user_id: str
    profile: dict[str, Any] = field(default_factory=dict)
    dataset_version: str = "v1"
    created_at: str = ""


@dataclass
class ResearchGoal:
    goal_id: str
    user_id: str
    title: str
    description: str = ""
    time_horizon: str = "mid_term"
    status: str = "active"
    goal_embedding_text: str = ""
    created_at: str = ""

    @property
    def query_text(self) -> str:
        return f"{self.title} {self.description}".strip()


@dataclass
class ResearchLog:
    log_id: str
    user_id: str
    date: str
    title: str
    content: str = ""
    activity_type: str = "study"
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    created_at: str = ""

    @property
    def full_text(self) -> str:
        return f"{self.title} {self.content}".strip()


@dataclass
class GoalLogLabel:
    label_id: str
    user_id: str
    goal_id: str
    log_id: str
    label: str  # "relevant" | "irrelevant"
    relevance_score: float = 1.0
    label_source: str = "synthetic_rule"


@dataclass
class CandidateLog:
    log: ResearchLog
    sparse_score: float = 0.0
    dense_score: float = 0.0
    hybrid_score: float = 0.0

    @property
    def log_id(self) -> str:
        return self.log.log_id


@dataclass
class RankedLog:
    log: ResearchLog
    rank: int = 0
    semantic_relevance: float = 0.0
    goal_focus: float = 0.0
    evidence_value: float = 0.0
    final_score: float = 0.0
    diversity_score: float = 0.0
    # Explanation trace (populated by reranker)
    matched_priority: list[str] = field(default_factory=list)
    matched_evidence: list[str] = field(default_factory=list)
    matched_related: list[str] = field(default_factory=list)
    matched_negative: list[str] = field(default_factory=list)
    admission_reason: str = ""    # why admitted
    rejection_reason: str = ""    # why rejected / vetoed
    anchor_source: str = ""       # "stage1" | "stage2_neighbor"
    # Schema category trace (populated by reranker)
    schema_category: str = ""           # assigned evidence category ("training", "implementation", …)
    goal_domain: str = ""               # inferred goal domain ("productivity_development", …)
    category_hit_strength: str = ""     # "core" | "supporting" | "none"
    # Evidence quality trace (populated by reranker)
    relevance_score: float = 0.0        # goal lexical + semantic component
    evidence_quality_score: float = 0.0 # quality component total
    specificity_score: float = 0.0
    actionability_score: float = 0.0
    goal_progress_score: float = 0.0    # category value prior
    redundancy_penalty: float = 0.0     # applied post-rank in Stage1Pipeline

    @property
    def log_id(self) -> str:
        return self.log.log_id


@dataclass
class CompressedEvidenceUnit:
    unit_id: str
    anchor_log_ids: list[str]
    summary: str
    date_range: str
    activity_cluster: str
    log_count: int = 1
    temporal_progression: str = ""
