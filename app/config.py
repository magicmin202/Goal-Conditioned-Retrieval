"""Experiment configuration for the Goal-Conditioned Retrieval research pipeline."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum


class AdaptiveMode(str, Enum):
    """Retrieval intensity mode inferred from corpus size and date span."""
    SMALL = "small"       # <15 logs OR span_days <14  → prioritise precision
    STANDARD = "standard"
    LARGE = "large"       # ≥60 logs AND span_days ≥60 → prioritise diversity


@dataclass
class AdaptivePolicyConfig:
    small_log_threshold: int = 15
    small_span_threshold: int = 14
    large_log_threshold: int = 60
    large_span_threshold: int = 60


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
    # Weights: goal_focus is primary signal
    semantic_weight: float = 0.35
    goal_focus_weight: float = 0.50     # primary signal
    evidence_value_weight: float = 0.15
    negative_term_penalty: float = 0.40  # stronger mismatch suppression
    priority_term_boost: float = 0.15   # additive bonus when log matches priority terms

    # goal_focus 3-tier breakdown (must sum to 1.0)
    priority_focus_weight: float = 0.55  # top-signal: priority_terms
    evidence_focus_weight: float = 0.30  # mid-signal: evidence_terms
    related_focus_weight: float = 0.15   # weak-signal: related_terms


@dataclass
class VocabularyBoostConfig:
    """Vocabulary-based score adjustments applied at candidate retrieval level.

    These are additive/subtractive offsets on top of the RRF hybrid score,
    applied before reranking so the best candidates rise earlier.
    """
    priority_term_boost: float = 0.20    # strong positive: priority terms
    evidence_term_boost: float = 0.10    # normal positive: evidence terms
    related_term_boost: float = 0.04     # weak positive: related/expanded terms
    negative_term_penalty: float = 0.15  # penalty per matched negative term/phrase
    phrase_match_multiplier: float = 1.5  # phrase match counts heavier than token match
    remove_generic_terms: bool = True     # strip generic terms from expansion output


@dataclass
class DiversityConfig:
    mmr_lambda: float = 0.6            # default (STANDARD mode)
    mmr_lambda_small: float = 0.85     # SMALL corpus → maximise relevance, weak diversity
    mmr_lambda_large: float = 0.55     # LARGE corpus → more diversity
    top_k: int = 10
    relevance_threshold: float = 0.05  # pre-filter: drop logs below this score before MMR
    pre_mmr_multiplier: int = 3        # keep top (k * multiplier) before MMR


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
    vocab_boost: VocabularyBoostConfig = field(default_factory=VocabularyBoostConfig)


@dataclass
class Stage2Config:
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    ranker: RankerConfig = field(default_factory=RankerConfig)
    diversity: DiversityConfig = field(default_factory=DiversityConfig)
    query_expansion: QueryExpansionConfig = field(
        default_factory=lambda: QueryExpansionConfig(enabled=True)
    )
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    vocab_boost: VocabularyBoostConfig = field(default_factory=VocabularyBoostConfig)


@dataclass
class AppConfig:
    collections: FirestoreCollections = field(default_factory=FirestoreCollections)
    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    stage1: Stage1Config = field(default_factory=Stage1Config)
    stage2: Stage2Config = field(default_factory=Stage2Config)
    adaptive_policy: AdaptivePolicyConfig = field(default_factory=AdaptivePolicyConfig)
    use_mock: bool = True


DEFAULT_CONFIG = AppConfig()
