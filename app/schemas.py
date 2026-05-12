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
        """BM25 / lexical matching용 plain text."""
        return f"{self.title} {self.content}".strip()

    @property
    def embedding_text(self) -> str:
        """Dense embedding 전용 field-labeled text.

        JSON 구조가 아니라 자연어형 필드 라벨을 사용한다.
        embedding 모델은 JSON 특수문자를 의미 있는 구조로
        이해하지 않으므로 plain text가 더 적합하다.

        date는 의미 유사도 계산에 노이즈가 될 수 있으므로 제외한다.
        temporal 정보는 local_expansion 등에서 별도로 처리한다.

        activity_type이 unknown인 경우 포함하지 않는다.
        keyword 기반 분류의 오분류가 embedding에 영향을 주는 것을 방지한다.
        """
        topic = self.metadata.get("topic", "")

        parts = [f"title: {self.title}"]

        if self.activity_type and self.activity_type != "unknown":
            parts.append(f"activity_type: {self.activity_type}")

        if topic:
            parts.append(f"topic: {topic}")

        if self.content:
            parts.append(f"content: {self.content}")

        return "\n".join(parts).strip()


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
    dense_score: float = 0.0

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
    gate_mode: str = "direct"           # "direct" | "supporting" | "reject"
    support_context_matched: list[str] = field(default_factory=list)  # terms matched in support gate

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
