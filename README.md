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
Stage 1: Candidate Retrieval + Reranking
  Dense Embedding (Gemini embedding-001)
  → Negative Veto          (domain conflict + no positive evidence → score=0)
  → Reranker Scoring       (relevance_score only)
  → Redundancy Penalty     (near-duplicate logs penalised)
  → Relevance Filtering    (final_score < threshold → drop)
  → Admitted Anchors
 │
 ▼
Stage 2: Anchor-centered Evidence Consolidation  ← NOT a retrieval stage
  Fixed anchors (from Stage 1)
  → Temporal Local Expansion (± window days)
  → Neighbor Re-admission (same reranker)
  → Cluster Summarization
  → LLM Analysis
```

### Stage 1 — Recall + Precision

**Retrieval:** Gemini embedding-001 Dense cosine similarity (Dense-only).

BM25와 VocabBoost는 실험을 통해 제거되었습니다.
Gemini embedding-001이 candidate_recall=0.759을 달성하므로 BM25와 어휘 기반 보정이 불필요합니다.
Precision control은 전적으로 Reranker가 담당합니다.

**Admission Gates:**

1. **Negative Veto** — 도메인 충돌 + positive evidence 없을 때만 reject (keyword blacklist 아님)
   - 조건: `raw_dm ≥ veto_dm_threshold AND pri_score < veto_priority_min`
   - 둘 다 충족해야 reject — 하나라도 빠지면 scoring으로 진행

2. **Relevance Filtering** — ranking 이후 `final_score < threshold`인 log 제거
   - Stage 1: `relevance_threshold_stage1 = 0.08`
   - Stage 2: `relevance_threshold_stage2 = 0.10`

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

### Reranker (precision-focused) — Relevance-Only

```
relevance_score =
    scale × (
        0.35 × priority_phrase_score    ← 가장 강한 goal lexical 신호
      + 0.20 × evidence_phrase_score    ← 직접 활동 어휘
      + 0.10 × related_score            ← 간접 연관 어휘
      + 0.05 × semantic_similarity      ← tie-breaker only
      + 0.05 × base_goal_overlap        ← raw goal text overlap
    )
  where scale = 1.0 / 0.75 = 1.333   ← 최대 relevance_score ≈ 1.0

final_score = relevance_score − negative_penalty
```

**핵심 설계 원칙:**
- Candidate 단계 = semantic recall 확대 (Dense embedding)
- Reranker 단계 = lexical precision 잠금 (phrase/token overlap 중심)
- quality_score는 final_score에서 제거 (goal과 무관한 숫자/동사 여부가 순위를 역전시키는 문제 해결)

**`priority_phrase_score` 계산 방식 (3단계 우선순위):**

| match mode | 조건 | 점수 배율 |
|---|---|---|
| `exact_phrase` | term 문자열이 텍스트에 완전 포함 | `phrase_weight=1.5` |
| `core_token` | term의 핵심 토큰 하나 이상 매칭 | `token_weight=0.4` |
| `weak_token_only` | WEAK_TOKENS만 매칭 | **0.0** (무효화) |

title 매칭 시 `title_multiplier=1.5` 추가 가중.

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
| `priority_terms` | 핵심 식별 표현 (4-8개) | reranker 최강 신호 |
| `evidence_terms` | 직접 활동 어휘 (8-15개) | reranker 중간 신호 |
| `related_terms` | 간접 연관 표현 (5-10개) | reranker 약한 신호 |
| `negative_terms` | 무관 도메인 표현 (8-15개) | penalty + veto |
| `core_intents` | 핵심 하위 목표 (3-5개) | dense query 보조 |
| `goal_summary` | 한 문장 요약 | dense query 주체 |
| `goal_activity_types` | 목표의 기대 activity type 리스트 | activity-type gate |

**Dense query** = `goal_summary + core_intents[0]` only
→ lexical expansion을 dense embedding에 주입하면 semantic drift 발생 → 분리 설계

### Activity-Type System

**6개 Activity Type (scoring 보조 신호):**

| Activity Type | 키워드 예시 | domain_consistency |
|---|---|---|
| `creative` | 포트폴리오, 작성, 디자인, 제작 | 0.85 |
| `execution` | 예약, 구매, 완료, booked, implemented | 0.80 |
| `learning` | 공부, 읽기, 강의, study, lecture | 0.60 |
| `planning` | 계획, 준비, 정리, 고민 | 0.55 |
| `unknown` | 분류 불가 | 0.40 |
| `lifestyle` | 산책, 식사, 카페, 운동 (일상) | 0.25 |

Activity-Type은 hard-reject gate로 사용되지 않으며 scoring 컴포넌트에만 영향을 줍니다.
최종 precision control은 `priority_phrase_score` / `negative_penalty`가 담당합니다.

---

## Project Structure

```
app/
  config.py                        # 실험 설정 (threshold, quality weights 등)
  schemas.py                       # ResearchGoal, ResearchLog, RankedLog, ...

  retrieval/
    query_understanding.py         # Goal → QueryObject 정규화
    query_expansion.py             # 7-field 구조화 어휘 확장 (Gemini / heuristic)
    schema_category.py             # Activity-type gate + quality prior 함수
    evidence_quality.py            # Evidence quality scoring (specificity/actionability/goal_progress)
    dense_retriever.py             # Gemini embedding-001 기반 Dense retrieval
    candidate_retrieval.py         # Dense retrieval 진입점
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
  run_stage1.py                    # Stage 1 단독 실행 (4 baselines)
  run_stage2.py                    # Stage 1 → Stage 2 체인 실행
  aggregate_results.py             # JSON → CSV 집계 (results/*_summary.csv)
  compare_debug_runs.py            # 3-way 비교 디버깅 실험
  compare_retrieval_weights.py     # Dense baseline 비교 실험
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

# Stage 1 단독 — no expansion (dense only)
.venv/bin/python scripts/run_stage1.py --goal_id G-U0001-01 --top_k 5 --baseline dense

# Stage 1 단독 — with expansion (Gemini query expansion + lexical gate)
.venv/bin/python scripts/run_stage1.py --goal_id G-U0001-01 --top_k 5 --baseline ours

# Stage 1 → Stage 2 전체 체인
.venv/bin/python scripts/run_stage2.py --goal_id G-U0001-01 --top_k 5

# Mock LLM 사용 (API 키 없이 테스트)
.venv/bin/python scripts/run_stage2.py --goal_id G-U0001-01 --mock

# 특정 user의 첫 번째 목표
.venv/bin/python scripts/run_stage1.py --user_id U0002 --top_k 5
```

### 진단/디버깅

```bash
# 3-way 비교: no-expand vs expand vs stage2-chain
.venv/bin/python scripts/compare_debug_runs.py --goal_id G-U0001-01

# JSON으로 결과 저장
.venv/bin/python scripts/compare_debug_runs.py --goal_id G-U0001-01 --json_out debug.json
```

**reranker 점수 분해 로그 확인:**

```bash
.venv/bin/python scripts/run_stage1.py --goal_id G-U0001-01 2>&1 | grep -E "DOMAIN-CONFLICT|Reranker Score|NegPenalty"
```

출력 예시:
```
[Reranker Score]  log=L-U0001-0005  [포트폴리오 구현]
  category:        creative
  priority_phrase: 0.800  matched=['포트폴리오']
  evidence_phrase: 0.533  matched=['프로젝트 구현']
  related:         0.120
  semantic:        0.850
  base_overlap:    0.100
  → relevance_score: 0.4533  (scale=1/0.75=1.333)
  neg_penalty:     0.000  (matched=set())
  → final:         0.4533  [rel=0.4533 - pen=0.0000]

DOMAIN-CONFLICT VETO  log=L-U0001-0012  dm=0.70  pri=0.000  matched={'주식'}  [주식 시장 공부]
```

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
| `dense` | Dense only | ✗ | ✗ |
| `dense_expand` | Dense only | ✓ | ✗ |
| `ours` | Dense only | ✓ | ✓ |
| `ours_wo_lexical_gate` | Dense only | ✓ | ✗ |

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

# Stage 1 — 전체 4 baselines × 1 goal
for B in dense dense_expand ours ours_wo_lexical_gate; do
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

### Stage 1 결과 (393 goals, top_k=5, real embeddings)

**전체 평균 (latest):**

| 지표 | 값 |
|---|---|
| `candidate_recall` | 0.759 |
| `selected_precision` | 0.851 |
| `false_positive_rate` | 0.149 |

**대표 goal 상세 (U0002, top_k=5):**

| goal_id | title | recall@5 | precision@5 | selected_precision | fpr | ndcg@5 |
|---|---|---|---|---|---|---|
| G-U0002-01 | 단편 소설 완성하기 | 0.500 | 1.000 | 0.750 | 0.250 | 0.945 |
| G-U0002-02 | 우쿨렐레 독학하기 | 0.500 | 1.000 | 1.000 | 0.000 | 0.958 |
| G-U0002-03 | 영어 회화 실력 높이기 | 0.400 | 0.800 | 0.500 | 0.500 | 0.740 |

> `quality_score` 제거 후 G-U0002-01에서 "악기 30분 연습"이 제거되고 "퇴고 작업"이 1위로 상승.

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

<img width="1089" height="434" alt="image" src="https://github.com/user-attachments/assets/6ead7c1d-5be2-4fa7-bdd0-e05494249ee2" />

---

## Key Design Decisions

### quality_score를 final_score에서 제거했다

이전 구현에서 `final_score = 0.70 * relevance_score + 0.30 * quality_score`를 사용했으나,
quality_score(specificity / actionability / activity-type prior)가 goal 관련성과 무관한 신호로
순위를 역전시키는 문제가 발견되었습니다.

예시 (goal: 단편 소설 완성하기):
- "악기 30분 연습" — relevance=0.039, quality=0.1545 → final=0.2101 → **1위** (오분류)
- "퇴고 작업" — relevance=0.0993, quality=0.0465 → final=0.1458 → **9위** (정답 로그)

변경: `final_score = relevance_score - negative_penalty`
- scale을 `1.0 / total_rel_w = 1.333`으로 올려 relevance_score가 [0, 1.0] 범위를 사용하도록 정규화
- quality_score는 `evidence_quality.py`에 보존 — Stage 2 evidence selection에서 활용 가능

### Stage 2는 retrieval 단계가 아니다

이전 구현에서 Stage 2가 전체 corpus를 재검색하면서
Stage 1에서 선별된 evidence와 무관한 로그가 anchor로 올라오는 문제가 있었습니다.

현재 구현에서 Stage 2는 **Stage 1 admitted anchors를 고정 입력**으로 받아
temporal local expansion과 summarization만 수행합니다.

### Activity-Type는 scoring 보조 신호로만 사용된다

이전 구현에서는 Activity-Type Gate로 hard-reject를 수행했으나,
rule-based keyword 분류의 정확도 한계로 인해 false reject가 발생했습니다.

현재 구현에서 activity-type은 `classify_log_activity_type()`으로 분류 후
`domain_consistency` scoring 컴포넌트에만 사용됩니다 (hard-reject 없음).
Hard-reject는 Negative Veto 하나만 남겨두어 오분류 risk를 최소화합니다.

| activity_type | domain_consistency score |
|---|---|
| `creative` | 0.85 |
| `execution` | 0.80 |
| `learning` | 0.60 |
| `planning` | 0.55 |
| `unknown` | 0.40 |
| `lifestyle` | 0.25 |

### Dense query는 goal_summary만 사용한다

lexical expansion 전체를 dense embedding에 주입하면 embedding centroid가 흐려져
관련 없는 로그가 cosine similarity 기준 상위에 올라오는 semantic drift가 발생합니다.
dense query = `goal_summary + core_intents[0]`으로 최소화합니다.

### BM25와 VocabBoost는 Gemini embedding-001 앞에서 무의미하다

실험 결과 Gemini embedding-001 단독으로 candidate_recall=1.00을 달성합니다.
BM25는 canonical query 기준 corpus의 ~10%에만 nonzero score를 부여하며,
expanded query를 사용해도 Dense가 이미 해당 로그들을 모두 커버합니다.
VocabBoost는 max delta ±0.05로 Dense score 클러스터링 폭(0.93–0.97)에 비해
너무 작아 순위 변동이 없었습니다.

→ Retrieval은 Dense-only, Precision control은 Reranker (lexical gate + scoring)로 완전히 분리합니다.

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

## Limitations & Future Work

### Limitations

---

#### L1. Synthetic Data Only

현재 모든 평가는 자동 생성된 합성 데이터셋(100 users, 393 goals, 7,030 logs)으로만 수행되었다.
합성 데이터는 실제 사용자 로그의 표현 다양성, 오탈자, 맥락 모호성, 비정형 서술을 반영하지 못한다.
특히 label은 rule-based heuristic으로 생성되었으므로, 실제 사용자가 판단하는 goal relevance와
완전히 일치한다고 볼 수 없다.

> **영향:** recall/precision 수치는 합성 데이터 환경에서의 상한선으로 해석해야 하며,
> 실제 서비스 데이터에서는 하락할 가능성이 있다.

---

#### L2. LLM-Generated Lexicon에 대한 의존

Query Expansion 단계에서 생성되는 `priority_terms`, `evidence_terms`, `negative_terms`의 품질은
Gemini API 응답의 일관성에 강하게 의존한다. 동일한 goal에 대해 API를 반복 호출하면 다른 어휘가
생성될 수 있으며(temperature=0.2임에도), 이는 retrieval 결과의 재현성을 저해한다.

현재 expansion cache(`.cache/`)로 완화하고 있으나,
cache miss 시 non-determinism이 발생한다.

> **영향:** 동일 goal에 대해 실행 시점에 따라 top-k 결과가 달라질 수 있다.

---

#### L3. Lexical Gate의 Phrase-Level False Positive

`priority_terms` / `evidence_terms` 중 단일 토큰 term("공부", "완료", "시작" 등)이 포함된 경우,
해당 토큰이 무관한 로그에서 매칭되어 false positive가 발생할 수 있다.

현재 `WEAK_TOKENS` 필터링으로 부분 완화하고 있으나,
phrase 수준 의미를 완전히 포착하지는 못한다.

```
WEAK_TOKENS: 완료, 시작, 정리, 계획, 준비, 공부, 하기, 수행, 진행, 실행, 관리, 확인, 검토, 작성
```

| term | log text | mode | score |
|---|---|---|---|
| `독서 완료` | `숙소 예약 완료` | `weak_token_only` | **0.0** |
| `독서 완료` | `독서 완료` | `exact_phrase` | **1.0** |
| `포트폴리오 작성` | `포트폴리오 업로드` | `core_token` | **0.4** |

> **영향:** false_positive_rate 평균 0.213 중 일부는 이 문제에서 기인한다.

---

#### L4. Activity-Type Gate의 Keyword 의존성

Activity-type 분류는 rule-based keyword matching에 기반하므로,
신규 활동 유형이나 비정형 표현을 가진 log는 `unknown`으로 fallback된다.
`unknown` log는 gate를 통과하여 lexical / semantic 판단에 위임되므로
precision 저하의 원인이 될 수 있다.

또한 현재 gate는 5개의 bucket (`learning` / `execution` / `lifestyle` / `planning` / `creative`)으로
고정되어 있어, goal domain의 다양성을 충분히 반영하지 못한다.

---

#### L5. Stage 2 (Compression & Analysis) 정량 평가 부재

현재 Stage 1의 retrieval 성능(precision / recall / F1 / nDCG)은 체계적으로 측정되었으나,
Stage 2의 Temporal-Semantic Compression 품질과 LLM Analysis의 적절성은
정량적으로 평가되지 않았다.

Compression이 goal-relevant 정보를 손실 없이 요약했는지,
LLM Analysis가 실제로 의미 있는 인사이트를 생성했는지는
사람의 판단(human evaluation) 또는 reference summary 기반 자동 평가가 필요하다.

---

#### L6. Single-Embedding Space의 의미적 한계

현재 Dense Retrieval은 Gemini embedding-001 모델 하나의 embedding space에 의존한다.
Goal의 추상적 의도와 Log의 구체적 행동 표현은 서로 다른 언어적 층위에 있으며,
단일 범용 모델이 이 둘을 동일한 semantic space에 정확히 배치한다는 보장이 없다.

Threshold 실험(t=0.92)에서 recall이 0.39 수준에 머무는 것은 이 한계를 부분적으로 반영한다.

---

#### L7. 단일 사용자 Corpus 가정

현재 pipeline은 한 user의 log corpus를 대상으로 retrieval을 수행한다.
실제 서비스에서는 user 간 log 패턴이 상이하고, corpus 크기도 크게 달라질 수 있다.
100명 × 70개 log의 현재 실험 설정은 실제 서비스의 장기 축적 시나리오를
완전히 재현하지 못한다.

---

### Future Work

---

#### F1. Real User Data Evaluation

합성 데이터를 넘어 실제 Prologue 서비스의 사용자 로그와 목표 데이터로
시스템을 검증하는 것이 최우선 과제다.
실제 데이터에서는 레이블 없이 implicit relevance signal(클릭, 열람 시간 등)을
활용한 평가 방법론도 함께 설계되어야 한다.

---

#### F2. Goal-Log Domain-Adapted Embedding Model

범용 Gemini embedding-001 대신,
goal과 log 쌍 데이터로 fine-tuning된 bi-encoder 모델을 도입한다.
목표 표현(추상적 의도)과 로그 표현(구체적 행동) 간의 asymmetric retrieval 특성을
반영한 학습이 필요하다.

후보 아키텍처:
- `paraphrase-multilingual-MiniLM-L12-v2` fine-tuned on (goal, log) pairs
- Gemini embedding fine-tuning (Vertex AI)

---

#### F3. Learned Reranker (Cross-Encoder)

현재 rule-based reranker를 cross-encoder 기반 learned reranker로 대체한다.
Goal-Log 쌍의 relevance score를 직접 예측하는 모델로,
현재의 weight tuning 한계를 극복하고 context-aware precision을 높일 수 있다.

```
Cross-Encoder: (goal_text, log_text) → relevance_score
```

학습 데이터: 합성 label + 실제 사용자 implicit feedback

---

#### F4. Stage 2 Compression 품질 자동 평가

Temporal-Semantic Compression이 생성한 Evidence Unit의 품질을 측정하기 위해
아래 평가 지표를 도입한다:

- **Compression Ratio**: 입력 log 수 대비 evidence unit 수
- **Coverage Score**: 원본 relevant log의 핵심 정보가 unit에 포함되는 비율
- **Faithfulness**: LLM-as-judge 방식으로 unit이 원본 log를 정확히 요약했는지 검증
- **Goal Alignment**: evidence unit이 goal과 실제로 연관된 내용을 담고 있는지

---

#### F5. LLM Expansion Consistency 강화

Query Expansion의 non-determinism 문제를 해결하기 위해:

1. **Structured Prompt + Output Schema 고정** — JSON schema validation으로 형식 보장
2. **Expansion Cache 확장** — goal_id + description hash 기반 persistent cache
3. **Fallback 품질 개선** — heuristic table을 goal domain별로 세분화
4. **Ensemble Expansion** — 동일 goal에 대해 N회 생성 후 다수결로 stable lexicon 구성

---

#### F6. Activity-Type LLM Classification

Keyword 기반 activity-type 분류를 LLM 기반 dynamic classifier로 교체한다.
Log title + content → activity_type + topic 을 LLM이 직접 분류하면
`unknown` fallback 비율을 줄이고 gate 정확도를 높일 수 있다.

비용 최적화를 위해 소형 모델(Gemini Flash)을 사용하고
분류 결과를 log metadata에 캐싱한다.

---

#### F7. Vector Index 도입 (Production Scale)

현재 in-memory cosine similarity 계산은 소규모 corpus(< 1,000 logs)에 적합하다.
사용자의 로그가 수천~수만 건으로 증가하는 프로덕션 환경에서는
FAISS 또는 Weaviate 기반 approximate nearest neighbor(ANN) index가 필요하다.

`DenseRetriever`의 `EmbeddingProvider` 인터페이스는 이미 교체 가능하도록 설계되어 있다.

---

#### F8. Multi-Goal & Cross-Goal Retrieval

현재 pipeline은 단일 goal에 대해 독립적으로 retrieval을 수행한다.
실제 사용자는 복수의 목표를 동시에 추구하며, 로그 하나가 여러 goal에 걸쳐
관련성을 가질 수 있다.

향후 연구 방향:
- **Goal Priority Weighting**: 사용자가 현재 집중하는 goal에 더 높은 가중치 부여
- **Cross-Goal Deduplication**: 복수 goal에서 공통으로 선택된 log를 효율적으로 처리
- **Goal Dependency Graph**: 하위 goal이 상위 goal의 evidence를 상속하는 구조

---

#### F9. Personalization & Long-term Adaptation

사용자마다 로그 작성 스타일, 어휘 선택, 활동 패턴이 다르다.
개인화된 retrieval 모델은 사용자의 로그 이력을 학습하여
goal-log relevance 판단 기준을 사용자별로 적응시킬 수 있다.

- User-specific expansion lexicon 구축
- Online learning from user feedback (clicked / dismissed evidence)
- Personalized threshold calibration per user

---

## Firestore Collections (Production)

```
research_users/{user_id}
research_goals/{goal_id}
research_logs/{log_id}
research_goal_log_labels/{label_id}
```
