"""Experiment configuration for the Goal-Conditioned Retrieval research pipeline."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class FirestoreCollections:
    research_users: str = "research_users"
    research_goals: str = "research_goals"
    research_logs: str = "research_logs"
    research_goal_log_labels: str = "research_goal_log_labels"


@dataclass
class RetrievalConfig:
    candidate_size: int = 30
    top_k: int = 10
    sparse_weight: float = 0.4
    dense_weight: float = 0.6
    rrf_k: int = 60
    random_seed: int = 42


@dataclass
class RankerConfig:
    semantic_weight: float = 0.5
    goal_focus_weight: float = 0.3
    evidence_value_weight: float = 0.2


@dataclass
class DiversityConfig:
    mmr_lambda: float = 0.5
    top_k: int = 10


@dataclass
class QueryExpansionConfig:
    enabled: bool = False
    max_terms: int = 7
    mode: str = "simple"  # "simple" | "structured"


@dataclass
class CompressionConfig:
    cluster_similarity_threshold: float = 0.7
    max_clusters: int = 5


@dataclass
class Stage1Config:
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    ranker: RankerConfig = field(default_factory=RankerConfig)
    diversity: DiversityConfig = field(default_factory=DiversityConfig)
    query_expansion: QueryExpansionConfig = field(
        default_factory=lambda: QueryExpansionConfig(enabled=False)
    )


@dataclass
class Stage2Config:
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    ranker: RankerConfig = field(default_factory=RankerConfig)
    diversity: DiversityConfig = field(default_factory=DiversityConfig)
    query_expansion: QueryExpansionConfig = field(
        default_factory=lambda: QueryExpansionConfig(enabled=True)
    )
    compression: CompressionConfig = field(default_factory=CompressionConfig)


@dataclass
class AppConfig:
    collections: FirestoreCollections = field(default_factory=FirestoreCollections)
    stage1: Stage1Config = field(default_factory=Stage1Config)
    stage2: Stage2Config = field(default_factory=Stage2Config)
    use_mock: bool = True


DEFAULT_CONFIG = AppConfig()
