"""Experiment configuration for the Goal-Conditioned Retrieval research pipeline."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class FirestoreCollections:
    research_users: str = "research_users"
    research_goals: str = "research_goals"
    research_logs: str = "research_logs"
    research_goal_log_labels: str = "research_goal_log_labels"


@dataclass
class GeminiConfig:
    """Gemini API configuration.

    API key is read from GEMINI_API_KEY or GOOGLE_API_KEY environment variable.
    """
    model_name: str = "gemini-2.0-flash"
    api_key_env: str = "GEMINI_API_KEY"
    fallback_env: str = "GOOGLE_API_KEY"
    max_output_tokens: int = 512
    temperature: float = 0.2        # low temperature → deterministic expansion
    use_mock_fallback: bool = True   # fall back to heuristic if API call fails

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env) or os.environ.get(self.fallback_env)


@dataclass
class RetrievalConfig:
    # candidate_size is set dynamically in scripts relative to corpus size
    candidate_size: int = 20
    top_k: int = 10
    sparse_weight: float = 0.4
    dense_weight: float = 0.6
    rrf_k: int = 60
    random_seed: int = 42


@dataclass
class RankerConfig:
    # Weights shifted toward goal relevance
    semantic_weight: float = 0.45
    goal_focus_weight: float = 0.40    # raised: "relevant to goal" > "evidence of effort"
    evidence_value_weight: float = 0.15


@dataclass
class DiversityConfig:
    mmr_lambda: float = 0.6            # higher → favour relevance over diversity in MMR
    top_k: int = 10
    relevance_threshold: float = 0.05  # pre-filter: drop logs below this score before MMR


@dataclass
class QueryExpansionConfig:
    enabled: bool = False
    max_terms: int = 10
    mode: str = "structured"           # "simple" | "structured"
    use_mock_fallback: bool = True


@dataclass
class CompressionConfig:
    cluster_similarity_threshold: float = 0.6
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
    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    stage1: Stage1Config = field(default_factory=Stage1Config)
    stage2: Stage2Config = field(default_factory=Stage2Config)
    use_mock: bool = True


DEFAULT_CONFIG = AppConfig()
