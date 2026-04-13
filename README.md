# Goal-Conditioned Retrieval

개인 목표 관리 서비스 **Prologue**에서 누적되는 사용자 행동 로그를 효율적으로 분석하기 위한
**Goal-Conditioned Evidence Retrieval** 연구 시스템.

## Research Problem

**Intent → Personal Timeline Events Retrieval**

> 사용자의 목표(Goal)가 주어졌을 때, 장기간의 개인 행동 로그(Work Logs)에서
> goal-conditioned evidence를 검색·통합·분석하는 문제.

---

## Architecture

```
Goal
 │
 ▼
Stage 1: Candidate Retrieval + Admission
  BM25 (0.40) + Dense (0.45) + VocabBoost (0.15)
  → Activity-Type Gate     (goal/log activity-type 구조적 비호환 → reject)
  → Goal Lexical Gate      (primary signal 없으면 → reject)
  → Tier2 Semantic Gate    (real embeddings 활성 시: dense similarity < 0.50 → reject)
  → Negative Veto          (domain conflict + no positive evidence)
  → Reranker Scoring       (relevance_score + quality_score)
  → Redundancy Penalty     (near-duplicate logs penalised)
  → Admitted Anchors
 │
 ▼
Stage 2: Anchor-centered Evidence Consolidation  ← NOT a retrieval stage
  Fixed anchors (from Stage 1)
  → Temporal Local Expansion (± window days)
  → Neighbor Re-admission (same gate rule)
  → Cluster Summarization
  → LLM Analysis
```

### Stage 1 — Recall + Admission

| 컴포넌트 | 가중치 | 설명 |
|---|---|---|
| BM25 | 0.40 | 어절/형태소 단위 키워드 매칭 |
| Dense Embedding | 0.45 | Mock / SentenceTransformer / Gemini Embedding |
| VocabBoost | 0.15 | priority/evidence/negative 어휘 기반 약한 보정 |

**Admission Gates (순서 보장):**

1. **Activity-Type Gate** — goal과 log의 activity type이 구조적으로 비호환이면 reject
   - 5개 bucket: `learning` · `execution` · `lifestyle` · `planning` · `creative`
   - 비호환 쌍 예시: `(learning, lifestyle)`, `(execution, lifestyle)`, `(lifestyle, creative)`
   - `unknown` 타입은 항상 통과 (보수적 판단 — lexical gate에 위임)

2. **Goal Lexical Gate** — priority/evidence/base 중 하나 이상 매칭이 있어야 admit
   - `pri_score > 0.0 OR ev_score > 0.0 OR base_overlap ≥ 0.04` → `direct` 경로
   - 조건 불충족 시 reject

3. **Tier2 Semantic Gate** *(real embeddings 활성 시만 동작)* — dense cosine similarity < 0.50이면 reject
   - `--real_embeddings` 플래그 사용 시에만 활성화 (Mock 임베딩에서는 비활성)
   - Stage 2 neighbor re-admission 시 `skip_semantic_gate=True` 자동 적용

4. **Negative Veto** — 도메인 충돌 + positive evidence 없을 때만 reject (keyword blacklist 아님)

**gate_mode 필드:**

| gate_mode | 조건 | score_cap |
|---|---|---|
| `direct` | priority/evidence/base 중 하나 이상 매칭 | 없음 |
| `reject` | 위 조건 불충족 | — |

### Stage 2 — Consolidation Only

Stage 2는 새 retrieval을 수행하지 않습니다.

| 규칙 | 설명 |
|---|---|
| **Global retrieval 금지** | Stage 1 anchors가 고정 입력 |
| **Local expansion only** | anchor 기준 ±window days 범위만 확인 |
| **Neighbor re-admission** | 동일한 gate rule 재적용 |
| **Fewer but correct** | top-k 채우기 위한 noisy log 허용 안 함 |

---

## Scoring Formula

### Reranker (precision-focused) — Dual-Component

```
relevance_score =
    0.30 × priority_phrase_score    ← 가장 강한 goal lexical 신호
  + 0.18 × evidence_phrase_score    ← 직접 활동 어휘
  + 0.08 × related_score            ← 간접 연관 어휘
  + 0.04 × semantic_similarity      ← tie-breaker only
  + 0.05 × base_goal_overlap        ← raw goal text overlap
  (각 항목은 relevance_weight=0.70 스케일 후 합산)

quality_score =
    0.25 × specificity              ← 숫자/단위/구체적 표현 포함 여부
  + 0.35 × actionability            ← 완료/실행 동사 vs 탐색/조사 동사
  + 0.25 × goal_progress            ← activity-type quality prior
  + 0.15 × domain_consistency       ← activity-type 기반 생산성 점수

final_score =
    relevance_score                 (×0.70 스케일)
  + quality_weight(0.30) × quality_score
  − negative_penalty
```

**핵심 설계 원칙:** relevance(관련성)와 evidence quality(분석 가치)를 분리.
goal과 관련 있는 로그라도 generic/redundant하면 상위 anchor로 올라오지 않도록 설계.

### Evidence Quality 예시

| 로그 | specificity | actionability | goal_progress | domain_consist | quality_total | 비고 |
|---|---|---|---|---|---|---|
| 방콕 항공권 예약 완료 (28만원) | 0.70 | 0.85 | 0.80 | 0.80 | **0.81** | activity=execution |
| 호텔 예약 확정 | 0.20 | 0.80 | 0.80 | 0.80 | **0.68** | activity=execution |
| 여행 준비물 쇼핑 | 0.10 | 0.60 | 0.20 | 0.55 | **0.38** | activity=planning |
| 짐 준비 | 0.00 | 0.20 | 0.20 | 0.55 | **0.21** | activity=planning |

### Redundancy Penalty (Stage1Pipeline post-rank)

동일/유사 로그가 이미 admitted 된 경우 후순위 로그에 penalty 적용:

| 조건 | 감점 |
|---|---|
| title 완전 일치 | -0.30 |
| title 토큰 유사도 ≥ 60% | -0.15 |

그리디 방식: score 높은 순서로 admitted set에 추가하면서 적용.

### Query Expansion (7-field structured)

| 필드 | 역할 | 사용처 |
|---|---|---|
| `priority_terms` | 핵심 식별 표현 (4-8개) | BM25 query + reranker 최강 신호 |
| `evidence_terms` | 직접 활동 어휘 (8-15개) | BM25 query + reranker 중간 신호 |
| `related_terms` | 간접 연관 표현 (5-10개) | reranker 약한 신호 (dense query 제외) |
| `negative_terms` | 무관 도메인 표현 (8-15개) | penalty + veto (retrieval query 불포함) |
| `core_intents` | 핵심 하위 목표 (3-5개) | dense query 보조 |
| `goal_summary` | 한 문장 요약 | dense query 주체 |
| `goal_activity_types` | 목표의 기대 activity type 리스트 | activity-type gate |

**Dense query** = `goal_summary + core_intents[0]` only
→ lexical expansion을 dense embedding에 주입하면 semantic drift 발생 → 분리 설계

### Activity-Type System

**5개 Activity Bucket (gate + quality prior):**

| Activity Type | 키워드 예시 | quality prior (progression) |
|---|---|---|
| `execution` | 예약, 구매, 완료, booked, implemented | 0.80 |
| `learning` | 공부, 읽기, 강의, study, lecture | 0.55 |
| `planning` | 계획, 준비, 정리, 고민 | 0.30 |
| `creative` | 포트폴리오, 작성, 디자인, 제작 | 0.50 |
| `lifestyle` | 산책, 식사, 카페, 운동 (일상) | 0.10 |

**비호환 Activity-Type 쌍 (`_INCOMPATIBLE_PAIRS`):**

| Goal Activity Type | Log Activity Type | 결과 |
|---|---|---|
| `learning` | `lifestyle` | reject |
| `execution` | `lifestyle` | reject |
| `lifestyle` | `creative` | reject |

> `unknown` 타입은 어떤 goal type과도 compatible — lexical gate에서 최종 판단.

---

## Project Structure

```
app/
  config.py                        # 실험 설정 (가중치, threshold, quality weights 등)
  schemas.py                       # ResearchGoal, ResearchLog, RankedLog, ...

  retrieval/
    query_understanding.py         # Goal → QueryObject 정규화
    query_expansion.py             # 7-field 구조화 어휘 확장 (Gemini / heuristic)
    schema_category.py             # Activity-type gate + quality prior 함수
    evidence_quality.py            # Evidence quality scoring (specificity/actionability/goal_progress)
    sparse_retriever.py            # 순수 Python BM25Okapi (numpy 의존성 없음)
    dense_retriever.py             # Embedding 기반 retrieval
    hybrid_retriever.py            # BM25 × 0.40 + Dense × 0.45 (score-based fusion)
    candidate_retrieval.py         # Hybrid + VocabBoost 통합 진입점
    reranker.py                    # Activity-type-first lexical-control reranker
    diversity_selector.py          # MMR 기반 diversity selection
    embedding_provider.py          # Mock / Gemini / SentenceTransformer 추상화

  compression/
    local_expansion.py             # Anchor-centered temporal local expansion
    temporal_semantic_compressor.py  # Cluster → CompressedEvidenceUnit

  llm/
    llm_client.py                  # LLM 인터페이스 (Mock / Gemini)
    analysis.py                    # Goal progress LLM analysis

  pipeline/
    stage1_ranking_pipeline.py     # Stage 1: retrieval → reranking → admission → redundancy
    stage2_rag_pipeline.py         # Stage 2: consolidation only (NO retrieval)

  evaluation/
    ranking_metrics.py             # Recall@k, Precision@k, F1@k, MRR, nDCG@k, FPR
    rag_metrics.py                 # coverage@k, Goal Alignment, Token Reduction, etc.
    result_writer.py               # JSON 결과 저장 (results/stage1/, results/stage2/)

  utils/
    text_matching.py               # Phrase/token/title-aware matching utilities
    logging_utils.py

scripts/
  run_stage1.py                    # Stage 1 단독 실행 (6 baselines)
  run_stage2.py                    # Stage 1 → Stage 2 체인 실행 (5 baselines)
  aggregate_results.py             # JSON → CSV 집계 (results/*_summary.csv)
  compare_debug_runs.py            # 3-way 비교 디버깅 실험
  generate_synthetic_dataset.py    # 합성 데이터 생성

results/
  stage1/                          # Stage 1 결과 JSON ({goal_id}_{baseline}.json)
  stage2/                          # Stage 2 결과 JSON ({goal_id}_{baseline}.json)
  stage1_summary.csv               # 집계 결과
  stage2_summary.csv               # 집계 결과

.cache/
  embeddings/                      # 임베딩 영구 디스크 캐시
  expansions/                      # Goal별 query expansion 캐시
```

---

## Setup

```bash
# 1. 가상환경 생성 (Nix 환경에서는 절대 경로 사용)
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. 환경 변수 설정
cp .env.example .env   # GEMINI_API_KEY 입력
```

`.env` 파일:
```
GEMINI_API_KEY=your_api_key_here
```

> **Note (Nix 환경)**: `python` 대신 `.venv/bin/python` 또는
> `source .venv/bin/activate` 후 실행.
> Python 경로가 바뀐 경우 `.venv` 재생성:
> `rm -rf .venv && python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt`

---

## Quick Start

```bash
# 데이터 생성 (3 users, ~30 logs each)
.venv/bin/python scripts/generate_synthetic_dataset.py --small

# Stage 1 단독 — no expansion
.venv/bin/python scripts/run_stage1.py --goal_id G-U0001-01 --top_k 5

# Stage 1 단독 — with expansion (Gemini query expansion)
.venv/bin/python scripts/run_stage1.py --goal_id G-U0001-01 --top_k 5 --expand

# Stage 1 → Stage 2 전체 체인
.venv/bin/python scripts/run_stage2.py --goal_id G-U0001-01 --top_k 5

# Mock LLM 사용 (API 키 없이 테스트)
.venv/bin/python scripts/run_stage2.py --goal_id G-U0001-01 --mock

# 특정 user의 첫 번째 목표
.venv/bin/python scripts/run_stage1.py --user_id U0002 --top_k 5 --expand
```

### 진단/디버깅

```bash
# 3-way 비교: no-expand vs expand vs stage2-chain
.venv/bin/python scripts/compare_debug_runs.py --goal_id G-U0001-01

# JSON으로 결과 저장
.venv/bin/python scripts/compare_debug_runs.py --goal_id G-U0001-01 --json_out debug.json
```

**admission trace 로그 확인:**

```bash
.venv/bin/python scripts/run_stage1.py --auto 2>&1 | grep -E "\[ADMIT\]|\[REJECT\]|\[ACTIVITY_GATE\]|\[REDUNDANCY\]"
```

출력 예시:
```
[ACTIVITY_GATE_PASS] L-U0001-0005  log_type=execution  goal_types=['learning', 'creative']  [포트폴리오 구현]
[ACTIVITY_GATE_FAIL] L-U0001-0017  log_type=lifestyle   goal_types=['learning', 'execution']  [아침 산책]
[ADMIT|direct]     L-U0001-0005  score=0.3252  log_type=execution  [포트폴리오 구현]
[REJECT|direct]    L-U0001-0012  reason=domain_conflict_veto(dm=0.70)  [주식 시장 공부]
[REDUNDANCY]       L-U0001-0031  penalty=0.15  reason=similar(0.72)   [여행 준비물 쇼핑]
```

**real embeddings로 Tier2 semantic gate 활성화:**

```bash
# GEMINI_API_KEY 환경변수 필요
.venv/bin/python scripts/run_stage1.py --goal_id G-U0001-01 --top_k 5 --real_embeddings
```

Tier2 gate 활성 시 `semantic_irrelevant(dense=0.3121<0.5)` rejection이 추가됩니다.

---

## Caching

API 호출을 최소화하기 위한 3단계 캐시 구조:

| 캐시 | 경로 | 키 |
|---|---|---|
| 임베딩 | `.cache/embeddings/{model}.json` | SHA256(text)[:16] |
| Query expansion | `.cache/expansions/{goal_id}.json` | goal_id |
| Reranker dense score | `candidate.dense_score` 재사용 | — |

---

## Embedding Providers

| Provider | 설정 | 특징 |
|---|---|---|
| `MockEmbeddingProvider` | 기본 (의존성 없음) | SHA256 기반 결정적 — 실행 간 동일 결과 |
| `SentenceTransformerProvider` | `pip install sentence-transformers` | 로컬 multilingual 모델 |
| `GeminiEmbeddingProvider` | `GEMINI_API_KEY` 설정 | `gemini-embedding-001`, 3072-dim, 한국어 지원 |

---

## Fallback 동작

| 상황 | 동작 |
|---|---|
| `GEMINI_API_KEY` 설정됨 | Gemini Embedding API 사용 |
| `GEMINI_API_KEY` 없음 | Mock 임베딩 (SHA256 기반) |
| Gemini expansion 성공 | API 결과 캐시 후 사용 |
| Gemini expansion 실패 | heuristic 사전 기반 fallback |
| `--mock` 플래그 | LLM analysis를 mock 응답으로 대체 |

---

## Evaluation Metrics

### Stage 1

| 메트릭 | 설명 |
|---|---|
| `Recall@k` | 정답 로그 중 top-k에 포함된 비율 |
| `Precision@k` | top-k 중 정답 비율 (분모=k) |
| `selected_precision` | admitted 로그 중 정답 비율 (분모=admitted 수) |
| `selected_count` | 실제 admitted 로그 수 (top_k 이하) |
| `F1@k` | Precision@k와 Recall@k의 조화 평균 |
| `false_positive_rate` | admitted 로그 중 정답이 아닌 비율 (`1 - selected_precision`) |
| `MRR` | Mean Reciprocal Rank |
| `nDCG@k` | Normalized Discounted Cumulative Gain |
| `diversity_coverage` | selected logs의 activity_type 다양성 |

### Stage 2

| 메트릭 | 설명 |
|---|---|
| `coverage@k` | 정답 로그(`relevance_score ≥ 0.5`) 중 evidence unit의 `anchor_log_ids`에 포함된 비율 |
| `goal_alignment_score` | evidence unit summary가 goal 핵심 어휘와 얼마나 겹치는지 (keyword overlap) |
| `token_reduction_rate` | 원본 로그 대비 evidence unit 압축률 |
| `redundancy_reduction` | evidence unit이 커버하는 로그 수 / 전체 로그 수 |
| `actionability_score` | `temporal_progression`이 있는 evidence unit 비율 |
| `evidence_unit_count` | 생성된 CompressedEvidenceUnit 수 |

---

## Baseline Comparison

### Stage 1 Baselines

| Baseline | Retrieval | Expansion | Lexical Gate |
|---|---|---|---|
| `bm25` | BM25 only | ✗ | ✗ |
| `dense` | Dense only | ✗ | ✗ |
| `hybrid` | BM25 + Dense | ✗ | ✗ |
| `hybrid_expand` | BM25 + Dense | ✓ | ✗ |
| `ours` | BM25 + Dense | ✓ | ✓ |
| `ours_wo_lexical_gate` | BM25 + Dense | ✓ | ✗ |

### Stage 2 Baselines

| Baseline | Stage1 입력 | Compression | LLM |
|---|---|---|---|
| `ours` | ours | ✓ (cluster + summarize) | ✓ |
| `ours_wo_compression` | ours | ✗ (anchor → 1:1 CEU) | ✓ |
| `ours_wo_lexical_gate` | ours_wo_lexical_gate | ✓ | ✓ |
| `raw_llm` | — (retrieval 없음) | ✗ | ✓ (raw logs 전체) |
| `simple_summary` | — (retrieval 없음) | ✗ | ✓ (summarize all) |

### 실행 방법

```bash
# Stage 1 — 단일 baseline
.venv/bin/python scripts/run_stage1.py --goal_id G-U0010-02 --top_k 5 --baseline ours --save_result

# Stage 1 — 전체 6 baselines × 1 goal
for B in bm25 dense hybrid hybrid_expand ours ours_wo_lexical_gate; do
  .venv/bin/python scripts/run_stage1.py --goal_id G-U0010-02 --top_k 5 --baseline $B --save_result
done

# Stage 2 — 단일 baseline
.venv/bin/python scripts/run_stage2.py --goal_id G-U0010-02 --top_k 5 --baseline ours --save_result

# 결과 집계 → CSV
.venv/bin/python scripts/aggregate_results.py --stage all
cat results/stage1_summary.csv
cat results/stage2_summary.csv
```

> `--mock` 플래그를 추가하면 Gemini LLM 호출 없이 Stage 2를 테스트할 수 있습니다.

### Stage 1 결과 (3 goals, top_k=5)

| goal_id | baseline | recall@5 | precision@5 | selected_precision | f1@5 | fpr | ndcg@5 |
|---|---|---|---|---|---|---|---|
| G-U0002-01 | bm25 | 0.417 | 1.000 | 1.000 | 0.588 | 0.000 | 0.906 |
| G-U0002-01 | dense | 0.417 | 1.000 | 1.000 | 0.588 | 0.000 | 0.970 |
| G-U0002-01 | hybrid | 0.417 | 1.000 | 1.000 | 0.588 | 0.000 | 0.947 |
| G-U0002-01 | hybrid_expand | 0.417 | 1.000 | 1.000 | 0.588 | 0.000 | 0.918 |
| G-U0002-01 | **ours** | **0.417** | **1.000** | **1.000** | **0.588** | **0.000** | **0.918** |
| G-U0002-01 | ours_wo_lexical_gate | 0.417 | 1.000 | 1.000 | 0.588 | 0.000 | 0.918 |
| G-U0002-02 | bm25 | 0.417 | 1.000 | 1.000 | 0.588 | 0.000 | 0.933 |
| G-U0002-02 | dense | 0.333 | 0.800 | 1.000 | 0.471 | 0.000 | 0.861 |
| G-U0002-02 | hybrid | 0.417 | 1.000 | 1.000 | 0.588 | 0.000 | 0.945 |
| G-U0002-02 | hybrid_expand | 0.417 | 1.000 | 0.800 | 0.588 | 0.200 | 0.952 |
| G-U0002-02 | **ours** | **0.417** | **1.000** | **1.000** | **0.588** | **0.000** | **0.973** |
| G-U0002-02 | ours_wo_lexical_gate | 0.417 | 1.000 | 0.800 | 0.588 | 0.200 | 0.940 |
| G-U0010-02 | bm25 | 0.300 | 0.600 | 0.600 | 0.400 | 0.400 | 0.586 |
| G-U0010-02 | dense | 0.300 | 0.600 | 0.400 | 0.400 | 0.600 | 0.597 |
| G-U0010-02 | hybrid | 0.300 | 0.600 | 0.400 | 0.400 | 0.600 | 0.599 |
| G-U0010-02 | hybrid_expand | 0.400 | 0.800 | 0.750 | 0.533 | 0.250 | 0.877 |
| G-U0010-02 | **ours** | **0.500** | **1.000** | **1.000** | **0.667** | **0.000** | **0.948** |
| G-U0010-02 | ours_wo_lexical_gate | 0.400 | 0.800 | 0.750 | 0.533 | 0.250 | 0.868 |

> G-U0010-02에서 `ours`가 precision=1.000, fpr=0.000으로 lexical gate 효과가 명확히 드러납니다.

### Stage 2 결과 (3 goals, top_k=5)

| goal_id | baseline | coverage@k | token_reduction | goal_alignment | actionability | eu_count |
|---|---|---|---|---|---|---|
| G-U0002-01 | **ours** | **0.500** | 0.876 | 0.145 | 1.000 | 5 |
| G-U0002-01 | ours_wo_compression | 0.313 | 0.976 | 0.091 | 0.000 | 5 |
| G-U0002-01 | ours_wo_lexical_gate | 0.500 | 0.876 | 0.145 | 1.000 | 5 |
| G-U0002-02 | **ours** | **0.267** | 0.932 | 0.111 | 1.000 | 3 |
| G-U0002-02 | ours_wo_compression | 0.200 | 0.983 | 0.074 | 0.000 | 3 |
| G-U0002-02 | ours_wo_lexical_gate | 0.333 | 0.899 | 0.067 | 1.000 | 5 |
| G-U0010-02 | **ours** | **0.313** | 0.904 | 0.100 | 1.000 | 3 |
| G-U0010-02 | ours_wo_compression | 0.188 | 0.989 | 0.033 | 0.000 | 3 |
| G-U0010-02 | ours_wo_lexical_gate | 0.313 | 0.886 | 0.050 | 1.000 | 4 |

---

## Key Design Decisions

### Relevance와 Evidence Quality는 분리된 문제다

이전 구현에서 relevance(goal 관련성)만으로 admission을 결정했기 때문에,
goal과 관련은 있지만 분석 가치가 낮은 generic 로그가 과도하게 admitted되었습니다.

현재 구현에서 `final_score = relevance_score (70%) + quality_score (30%)`로
두 신호를 명시적으로 분리합니다. quality_score는 specificity, actionability,
activity-type quality prior로 구성됩니다.

### Stage 2는 retrieval 단계가 아니다

이전 구현에서 Stage 2가 전체 corpus를 재검색하면서
Stage 1에서 선별된 evidence와 무관한 로그가 anchor로 올라오는 문제가 있었습니다.

현재 구현에서 Stage 2는 **Stage 1 admitted anchors를 고정 입력**으로 받아
temporal local expansion과 summarization만 수행합니다.

### Activity-Type Gate는 open-domain goal을 지원한다

이전 구현의 hard-coded domain schema (6개 domain)는 schema 밖의 goal
(예: "친구 관계 넓히기")에서 전량 reject가 발생했습니다.

현재 구현은 domain schema를 제거하고 activity-type 호환성만 검사합니다.
`unknown` 타입은 항상 통과시켜 lexical gate가 최종 판단을 담당합니다.
3-layer precision control (activity-type → semantic → lexical)이 schema 없이도
동일한 수준의 precision을 유지합니다.

**검증 결과 (schema 제거 후):**

| goal_id | 케이스 | precision@5 | FPR |
|---|---|---|---|
| G-U0009-04 | 친구 관계 넓히기 (신규) | 1.000 (2 admit) | 0.00 |
| G-U0002-01 | 저비용 해외여행 | 1.000 | 0.00 |
| G-U0002-02 | 투자 공부 | 1.000 | 0.00 |
| G-U0010-02 | 독서 습관 | 1.000 | 0.00 |

### Dense query는 goal_summary만 사용한다

lexical expansion 전체를 dense embedding에 주입하면 embedding centroid가 흐려져
관련 없는 로그가 cosine similarity 기준 상위에 올라오는 semantic drift가 발생합니다.
dense query = `goal_summary + core_intents[0]`으로 최소화합니다.

### Redundancy penalty는 post-rank greedy 방식으로 적용된다

reranker가 각 log를 독립적으로 score하는 구조이므로,
near-duplicate 감지는 ranking 이후 admitted set을 순서대로 구축하면서
그리디하게 적용합니다. 높은 score 로그가 먼저 admitted set에 진입하므로
반복 중 가장 가치 있는 로그 1개는 보존됩니다.

### CEU topic 이름은 anchor title을 우선 사용한다

`TemporalSemanticCompressor`의 cluster topic 이름 결정 우선순위:

1. `anchor.log.title` — 가장 구체적이며 실제 활동 내용 반영
2. `metadata["topic"]` — 합성 데이터 생성 시 부여한 broad topic
3. `activity_type` — 최후 fallback

---

## Known Limitations

### Lexical False Positive (Phrase-Level Matching)

BM25 및 lexical gate 기반 시스템에서 priority/evidence term이
짧은 일반 토큰("완료", "시작", "정리" 등)을 포함할 경우,
해당 토큰이 무관한 로그에서 hit되어 false positive가 발생할 수 있음.

**완화 방법 (구현됨):**
- `score_priority_terms()` (`app/utils/text_matching.py`) — `PriorityTermMatch` 기반 weak-token-filtered phrase matching:
  - **Exact phrase match** (multi-token) → `mode="exact_phrase"`, score=1.0
  - **Core token match** (non-weak token 포함) → `mode="core_token"`, score=0.4
  - **Weak-only token match** → `mode="weak_token_only"`, score=0.0 (**positive 신호 없음**)

**`WEAK_TOKENS` (현재):**
```
완료, 시작, 정리, 계획, 준비, 공부, 하기, 수행, 진행, 실행, 관리, 확인, 검토, 작성
```

**매칭 예시:**

| term | log text | core hit | weak hit | mode | score |
|---|---|---|---|---|---|
| `독서 완료` | `숙소 예약 완료` | ✗ (`독서` miss) | `완료` | `weak_token_only` | **0.0** |
| `독서 완료` | `독서 완료` | — | — | `exact_phrase` | **1.0** |
| `포트폴리오 작성` | `포트폴리오 업로드` | `포트폴리오` | — | `core_token` | **0.4** |

### Activity-Type 분류 한계

activity-type 분류는 keyword 기반이므로 완전히 신규 유형의 log는 `unknown`으로 처리됩니다.
`unknown` log는 lexical + semantic gate가 최종 판단합니다.
LLM 기반 dynamic activity-type 분류는 향후 개선 과제입니다.

**향후 개선 방향:**
- phrase-level learned reranker 도입
- BM25 token scoring 시 IDF 기반 일반 토큰 자동 down-weighting
- LLM 기반 dynamic activity-type classification

---

## Firestore Collections (Production)

```
research_users/{user_id}
research_goals/{goal_id}
research_logs/{log_id}
research_goal_log_labels/{label_id}
```
