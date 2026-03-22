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
    # Dual-space weights for HybridRetriever (BM25 + Dense).
    # Sum to 0.85 — the remaining 0.15 is the vocab_boost_weight in CandidateConfig.
    sparse_weight: float = 0.40   # BM25 weight
    dense_weight: float = 0.45    # Dense embedding weight
    rrf_k: int = 60               # kept for backward compat (not used in score-based mode)
    random_seed: int = 42


@dataclass
class CandidateConfig:
    """Dual-space candidate retrieval weights.

    candidate_score = sparse_weight * bm25
                    + dense_weight  * dense
                    + vocab_weight  * weak_vocab_boost

    Weights should sum to ~1.0.
    """
    vocab_boost_weight: float = 0.15   # how much lexicon can shift candidate score


@dataclass
class RankerConfig:
    """Lexical-control reranker weights.

    Dual-space retrieval architecture:
      Stage 1 (candidate) = recall   → semantic 45%, BM25 40%, vocab 15%
      Stage 2 (reranker)  = precision → lexical 90%, semantic 5-10%

    final_score =
        priority_weight   * priority_phrase_score
      + evidence_weight   * evidence_phrase_score
      + related_weight    * related_score
      + action_weight     * action_signal
      + domain_weight     * domain_consistency
      + semantic_weight   * semantic_similarity
      + base_weight       * base_goal_overlap
      - negative_penalty
    """
    # ── Reranker component weights (should sum to ~1.0) ───────────────────────
    priority_weight: float = 0.35       # strongest lexical signal
    evidence_weight: float = 0.20       # direct evidence vocabulary
    related_weight: float = 0.10        # indirect/related vocabulary
    action_weight: float = 0.15         # completion/action keywords
    domain_weight: float = 0.10         # activity_type + metadata consistency
    semantic_weight: float = 0.05       # tie-breaker: dense similarity
    base_weight: float = 0.05           # raw goal-text token overlap

    # ── Negative penalty levels ───────────────────────────────────────────────
    negative_penalty_phrase: float = 0.70   # phrase match in text body
    negative_penalty_token: float = 0.40    # token match
    negative_penalty_title: float = 0.30    # extra if match in title
    negative_daily_penalty: float = 0.20    # activity_type="daily" extra

    # ── Negative veto ─────────────────────────────────────────────────────────
    negative_veto_enabled: bool = True
    negative_veto_dm_threshold: float = 0.70   # dm ≥ this triggers veto
    negative_veto_priority_min: float = 0.05   # unless priority_score ≥ this

    # ── Title weight (phrase matches in title count more) ─────────────────────
    title_weight_multiplier: float = 1.5

    # ── Back-compat aliases (used by pipelines/tests) ─────────────────────────
    @property
    def negative_term_penalty(self) -> float:
        return self.negative_penalty_phrase

    @property
    def priority_term_boost(self) -> float:
        return 0.0   # priority handled within score components, not additive

    @property
    def goal_focus_weight(self) -> float:
        return self.priority_weight + self.evidence_weight + self.related_weight


@dataclass
class VocabularyBoostConfig:
    """Weak vocabulary boost at candidate retrieval level (recall-focused).

    Applied as additive offsets on the hybrid (BM25+Dense) score.
    Intentionally mild — just nudges direction, does not override semantic recall.
    Strong precision control happens in the reranker (see RankerConfig).

    Total vocab_boost ∈ [−1, +1], then scaled by CandidateConfig.vocab_boost_weight.
    """
    # Priority terms (strongest, but still mild at candidate stage)
    priority_phrase_boost: float = 0.50   # phrase hit in full text  → normalized [0,1]
    priority_token_boost: float = 0.30    # token hit in full text
    priority_title_bonus: float = 0.20    # extra if token hit in title

    # Evidence terms
    evidence_phrase_boost: float = 0.30   # phrase match
    evidence_token_boost: float = 0.15    # token match

    # Related terms
    related_token_boost: float = 0.08     # token match (weak)

    # Negative terms (mild penalty — don't over-suppress at recall stage)
    negative_phrase_penalty: float = 0.40  # phrase match
    negative_token_penalty: float = 0.20   # token match

    remove_generic_terms: bool = True

    # Back-compat
    @property
    def priority_term_boost(self) -> float:
        return self.priority_phrase_boost

    @property
    def evidence_term_boost(self) -> float:
        return self.evidence_phrase_boost


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
class ConsolidationConfig:
    """Stage 2 anchor-centered evidence consolidation settings.

    Stage 2 = consolidation only (NOT new retrieval).
    - Only admitted anchors (reranker score >= anchor_admission_threshold) enter.
    - Local expansion is limited to anchor ± temporal window (days).
    - Neighbors must pass reranker re-admission before entering cluster.
    - allow_fewer_than_k=True: fewer correct > full noisy.
    """
    consolidation_mode: bool = True

    # Temporal expansion window (days ± from anchor date)
    local_expansion_window_small: int = 5      # sparse corpus
    local_expansion_window_standard: int = 3   # default
    local_expansion_window_large: int = 2      # dense corpus → tighter

    # Admission thresholds
    anchor_admission_threshold: float = 0.10   # reranker score to be an anchor
    neighbor_admission_threshold: float = 0.08  # slightly lower for neighbors

    # Do NOT fill to top_k with below-threshold logs
    allow_fewer_than_k: bool = True
    max_neighbors_per_anchor: int = 5


@dataclass
class Stage1Config:
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    candidate: CandidateConfig = field(default_factory=CandidateConfig)
    ranker: RankerConfig = field(default_factory=RankerConfig)
    diversity: DiversityConfig = field(
        default_factory=lambda: DiversityConfig(relevance_threshold=0.08)
    )
    query_expansion: QueryExpansionConfig = field(
        default_factory=lambda: QueryExpansionConfig(enabled=False)
    )
    vocab_boost: VocabularyBoostConfig = field(default_factory=VocabularyBoostConfig)


@dataclass
class Stage2Config:
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    candidate: CandidateConfig = field(default_factory=CandidateConfig)
    ranker: RankerConfig = field(default_factory=RankerConfig)
    diversity: DiversityConfig = field(
        default_factory=lambda: DiversityConfig(relevance_threshold=0.10)
    )
    query_expansion: QueryExpansionConfig = field(
        default_factory=lambda: QueryExpansionConfig(enabled=True)
    )
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    consolidation: ConsolidationConfig = field(default_factory=ConsolidationConfig)
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
