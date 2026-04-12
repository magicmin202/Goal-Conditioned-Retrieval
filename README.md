# Goal-Conditioned Retrieval

개인 목표 관리 시스템에서 누적되는 사용자 행동 로그를 효율적으로 분석하기 위한
**Goal-Conditioned Evidence Retrieval** 시스템.

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
  → Schema Category Gate    (none → immediate reject)
  → Activity-Type Gate      (goal/log activity-type 비호환 → reject)
  → Goal Lexical Gate       (primary signal 없으면 support gate 시도)
      └ Support Gate        (support_context_hit + subdomain_consistent → admit, score_cap=0.12)
  → Tier2 Semantic Gate     (real embeddings 활성 시: dense similarity < 0.50 → reject)
  → Negative Veto           (domain conflict + no positive evidence)
  → Reranker Scoring        (relevance_score + quality_score)
  → Redundancy Penalty      (near-duplicate logs penalised)
  → Admitted Anchors
 │
 ▼
Stage 2: Anchor-centered Evidence Consolidation  ← NOT a retrieval stage
  Fixed anchors (from Stage 1)
  → Temporal Local Expansion (± window days)
  → Neighbor Re-admission (same 3-gate rule)
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
1. **Schema Category Gate** — log의 evidence category가 goal domain과 연관 없으면 즉시 reject
2. **Activity-Type Gate** — goal과 log의 activity type이 구조적으로 비호환이면 reject
   - 5개 bucket: `learning` · `execution` · `lifestyle` · `planning` · `creative`
   - 비호환 쌍 예시: `(learning, lifestyle)`, `(execution, lifestyle)`, `(lifestyle, creative)`
3. **Goal Lexical Gate + Support Gate** — primary signal(priority/evidence/base) 있으면 `direct` 경로, 없으면 support gate 시도
   - **Support Gate**: `support_context_signal` 매칭 + `subdomain_consistent` → `supporting` 경로 admit (score_cap=0.12)
   - 둘 다 해당 없으면 reject
4. **Tier2 Semantic Gate** *(real embeddings 활성 시만 동작)* — dense cosine similarity < 0.50이면 reject
   - `--real_embeddings` 플래그 사용 시에만 활성화 (Mock 임베딩에서는 비활성)
   - Stage 2 neighbor re-admission 시 `skip_semantic_gate=True` 자동 적용 — 이웃 로그는 temporal locality + lexical gate로 이미 제한되며, raw cosine은 Stage 1의 max-normalized scale과 달라 threshold 오적용을 방지
5. **Negative Veto** — 도메인 충돌 + positive evidence 없을 때만 reject (keyword blacklist 아님)

**gate_mode 필드:**

| gate_mode | 조건 | score_cap |
|---|---|---|
| `direct` | priority/evidence/base 중 하나 이상 매칭 | 없음 |
| `supporting` | primary signal 없음 + support_context_hit + subdomain_consistent | 0.12 |
| `reject` | 위 조건 불충족 | — |

### Stage 2 — Consolidation Only

Stage 2는 새 retrieval을 수행하지 않습니다.

| 규칙 | 설명 |
|---|---|
| **Global retrieval 금지** | Stage 1 anchors가 고정 입력 |
| **Local expansion only** | anchor 기준 ±window days 범위만 확인 |
| **Neighbor re-admission** | 동일한 3-gate rule 재적용 |
| **Category hard drop** | `category_hit_strength=none` 로그는 compressor 입력 금지 |
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
  + 0.25 × goal_progress            ← activity-type quality prior (execution > planning > lifestyle)
  + 0.15 × domain_consistency       ← category relevance tier (core=1.0, supporting=0.6, none=0.0)

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
| 방콕 항공권 예약 완료 (28만원) | 0.70 | 0.85 | 0.90 | 1.00 | **0.88** | activity=execution, cat=booking(core) |
| 호텔 예약 확정 | 0.20 | 0.80 | 0.90 | 1.00 | **0.73** | activity=execution, cat=booking(core) |
| 여행 준비물 쇼핑 | 0.10 | 0.60 | 0.50 | 0.60 | **0.49** | activity=execution, cat=logistics(supporting) |
| 짐 준비 | 0.00 | 0.20 | 0.30 | 0.60 | **0.23** | activity=planning, cat=logistics(supporting) |

> **Note**: `goal_progress`는 이제 domain/category lookup 대신 **activity-type quality prior**를 사용합니다.
> `execution` 타입 로그는 높은 prior(0.80–0.90), `planning`은 중간(0.30–0.50), `lifestyle`은 낮은 prior(0.10)를 갖습니다.
> `domain_consistency`는 category relevance tier로 단순화: core=1.0, supporting=0.6, none=0.0.

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
| `goal_activity_types` | 목표의 기대 activity type 리스트 | activity-type gate (Gemini 미사용 시 heuristic 추론) |

**Dense query** = `goal_summary + core_intents[0]` only
→ lexical expansion을 dense embedding에 주입하면 semantic drift 발생 → 분리 설계

### Schema Category System

**Goal Domains (6개):**

| Domain | 감지 키워드 예시 | Core categories | Supporting categories | support_context_signals |
|---|---|---|---|---|
| `productivity_development` | 개발, 취업, 포트폴리오, 개발자 | implementation, problem_solving, debugging | study_progress, planning | — |
| `learning_coding` | 알고리즘, 코테, 코딩테스트 | study_progress, problem_solving | implementation, planning | — |
| `fitness_muscle_gain` | 근육, 벌크, 헬스장, 증량 | training, body_metrics | nutrition, recovery | — |
| `fitness_fat_loss` | 다이어트, 감량, 칼로리, 체지방 | training, nutrition, body_metrics | recovery | — |
| `travel_planning` | 여행, 해외, 항공, 배낭, 저비용 | booking, budgeting | logistics, travel_research, planning | — |
| `language_exam` | 토익, toeic, lc, rc, 리스닝, 리딩, 모의고사 | language_exam_study, mock_test, reading_practice, listening_practice, vocab_building | graduate_admission_prep, career_support | 취업 준비, 대학원 준비, gre, 지도교수, 지원서 |

**Log Categories (20개):**
`training` · `nutrition` · `body_metrics` · `recovery` ·
`implementation` · `debugging` · `study_progress` · `problem_solving` · `planning` ·
`booking` · `budgeting` · `logistics` · `travel_research` ·
`language_exam_study` · `vocab_building` · `listening_practice` · `reading_practice` · `mock_test` ·
`graduate_admission_prep` · `career_support`

**Activity-Type Buckets (goal_progress 및 gate에서 사용):**

| Activity Type | 키워드 예시 | quality prior (progression) |
|---|---|---|
| `execution` | 예약, 구매, 완료, booked, implemented | 0.80 |
| `learning` | 공부, 읽기, 강의, study, lecture | 0.55 |
| `planning` | 계획, 준비, 정리, 고민 | 0.30 |
| `creative` | 포트폴리오, 작성, 디자인, 제작 | 0.50 |
| `lifestyle` | 산책, 식사, 카페, 운동 (일상) | 0.10 |

> 이전의 domain/category 기반 `_CATEGORY_VALUE_PRIORS` 테이블은 activity-type prior로 대체되었습니다.
> goal_progress는 로그가 *어떻게* 수행됐는지(activity type)를 기준으로 산출하며,
> category relevance tier(core/supporting)로 보정합니다.

**비호환 Activity-Type 쌍 (`_INCOMPATIBLE_PAIRS`):**

| Goal Activity Type | Log Activity Type | 결과 |
|---|---|---|
| `learning` | `lifestyle` | reject |
| `execution` | `lifestyle` | reject |
| `lifestyle` | `creative` | reject |

---

## Project Structure

```
app/
  config.py                        # 실험 설정 (가중치, threshold, quality weights 등)
  schemas.py                       # ResearchGoal, ResearchLog, RankedLog, ...

  retrieval/
    query_understanding.py         # Goal → QueryObject 정규화
    query_expansion.py             # 6-field 구조화 어휘 확장 (Gemini / heuristic)
    schema_category.py             # Goal domain + Log category 매핑 + gate logic
    evidence_quality.py            # Evidence quality scoring (specificity/actionability/goal_progress)
    sparse_retriever.py            # 순수 Python BM25Okapi (numpy 의존성 없음)
    dense_retriever.py             # Embedding 기반 retrieval
    hybrid_retriever.py            # BM25 × 0.40 + Dense × 0.45 (score-based fusion)
    candidate_retrieval.py         # Hybrid + VocabBoost 통합 진입점
    reranker.py                    # Category-first lexical-control reranker (relevance + quality)
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
> Python 경로가 바뀐 경우 `.venv` 재생성: `rm -rf .venv && python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt`

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
.venv/bin/python scripts/run_stage1.py --auto 2>&1 | grep -E "\[ADMIT\]|\[REJECT\]|\[TIER1\]|\[REDUNDANCY\]"
```

출력 예시:
```
[TIER1_PASS] L-U0001-0005  log_type=execution  cat=implementation  domain=productivity_development  [포트폴리오 구현]
[TIER1_FAIL] L-U0001-0017  log_type=learning   cat=study_progress  domain=travel_planning  reason=domain_gate(none)  [자료구조 공부]
[ADMIT|direct]     L-U0001-0005  score=0.3252  cat=implementation(core)     [포트폴리오 구현]
[ADMIT|supporting] L-U0001-0008  score=0.0980  cat=graduate_admission_prep  [GRE 대비]  support=['gre']
[REJECT|direct]    L-U0001-0012  reason=domain_conflict_veto(dm=0.70)       [주식 시장 공부]
[REDUNDANCY]       L-U0001-0031  penalty=0.15  reason=similar(0.72)         [여행 준비물 쇼핑]
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
| `selected_precision` | admitted 로그 중 정답 비율 (분모=admitted 수) — `selected_count < top_k`일 때 더 정확한 지표 |
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

> **Goal Alignment 구현 노트**: 한국어 어형 변화(항공권과→항공권, 숙소를→숙소, 예약하고→예약)를
> 처리하기 위해 `rag_metrics._extract_goal_keywords()`에서 조사/어미 제거를 적용합니다.
> 순수 keyword overlap의 이론적 상한은 약 0.3–0.5 수준이며, 의미적 정확도를 높이려면
> LLM judge (`judge_fn(goal, unit) → float`)로 교체해야 합니다 (TODO 기록됨).

> **coverage@k threshold**: 합성 데이터의 `relevance_score`는 0.0–1.0 사이 실수값이므로
> threshold를 `>= 0.5`로 설정합니다 (`relevant` + `partial` 레이블 포함).

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

# Stage 2 — 전체 5 baselines × 전체 goal (실 LLM 사용, GEMINI_API_KEY 필요)
for GOAL in $(목표_ID_목록); do
  for B in ours ours_wo_compression ours_wo_lexical_gate raw_llm simple_summary; do
    .venv/bin/python scripts/run_stage2.py --goal_id $GOAL --top_k 5 --baseline $B --save_result
  done
done

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

> G-U0010-02에서 `ours`가 다른 baselines 대비 precision=1.000, fpr=0.000으로 lexical gate 효과가 명확하게 드러납니다.

### Stage 2 결과 (3 goals, top_k=5)

| goal_id | baseline | coverage@k | token_reduction | goal_alignment | actionability | eu_count |
|---|---|---|---|---|---|---|
| G-U0002-01 | **ours** | **0.500** | 0.876 | 0.145 | 1.000 | 5 |
| G-U0002-01 | ours_wo_compression | 0.313 | 0.976 | 0.091 | 0.000 | 5 |
| G-U0002-01 | ours_wo_lexical_gate | 0.500 | 0.876 | 0.145 | 1.000 | 5 |
| G-U0002-01 | raw_llm | 0.000 | 1.000 | 0.000 | 0.000 | 0 |
| G-U0002-01 | simple_summary | 0.000 | 1.000 | 0.000 | 0.000 | 0 |
| G-U0002-02 | **ours** | **0.267** | 0.932 | 0.111 | 1.000 | 3 |
| G-U0002-02 | ours_wo_compression | 0.200 | 0.983 | 0.074 | 0.000 | 3 |
| G-U0002-02 | ours_wo_lexical_gate | 0.333 | 0.899 | 0.067 | 1.000 | 5 |
| G-U0002-02 | raw_llm | 0.000 | 1.000 | 0.000 | 0.000 | 0 |
| G-U0002-02 | simple_summary | 0.000 | 1.000 | 0.000 | 0.000 | 0 |
| G-U0010-02 | **ours** | **0.313** | 0.904 | 0.100 | 1.000 | 3 |
| G-U0010-02 | ours_wo_compression | 0.188 | 0.989 | 0.033 | 0.000 | 3 |
| G-U0010-02 | ours_wo_lexical_gate | 0.313 | 0.886 | 0.050 | 1.000 | 4 |
| G-U0010-02 | raw_llm | 0.000 | 1.000 | 0.000 | 0.000 | 0 |
| G-U0010-02 | simple_summary | 0.000 | 1.000 | 0.000 | 0.000 | 0 |

> `raw_llm` / `simple_summary`는 retrieval이 없으므로 evidence_unit이 생성되지 않아 coverage@k=0.
> `ours`는 compression을 통해 `ours_wo_compression` 대비 coverage가 높고 actionability=1.0을 유지합니다.

---

## Key Design Decisions

### Relevance와 Evidence Quality는 분리된 문제다

이전 구현에서 relevance(goal 관련성)만으로 admission을 결정했기 때문에,
goal과 관련은 있지만 분석 가치가 낮은 generic 로그가 과도하게 admitted되었습니다.

현재 구현에서 `final_score = relevance_score (70%) + quality_score (30%)`로
두 신호를 명시적으로 분리합니다. quality_score는 specificity, actionability,
category value prior로 구성됩니다.

travel_planning의 경우, `여행 준비물 쇼핑`·`짐 준비` 류는 `logistics(supporting)`으로
admitted되지만, booking/budgeting 대비 낮은 domain_consistency score로 인해
예약·예산 관련 로그보다 자연스럽게 낮은 순위를 갖습니다.

### Stage 2는 retrieval 단계가 아니다

이전 구현에서 Stage 2가 전체 corpus를 재검색하면서
Stage 1에서 선별된 evidence와 무관한 로그가 anchor로 올라오는 문제가 있었습니다.

현재 구현에서 Stage 2는 **Stage 1 admitted anchors를 고정 입력**으로 받아
temporal local expansion과 summarization만 수행합니다.

### Schema category gate는 scoring 이전에 적용된다

`action_signal + domain_consistency`만으로도 goal-agnostic 로그가
admitted되는 문제를 막기 위해, scoring 전에 category gate를 먼저 적용합니다.
`relevance=none`인 로그는 scoring 자체를 수행하지 않습니다.

### Support gate는 간접 증거를 제한적으로 허용한다

primary signal(goal vocabulary match)이 없더라도, 목표와 맥락적으로 연관된 로그
(예: `language_exam` 목표 하에서 `GRE 대비`, `취업 준비` 로그)를 완전히 차단하면
recall이 과도하게 낮아지는 문제가 있습니다.

support gate는 다음 두 조건을 동시에 만족하는 로그를 `supporting` 경로로 admission합니다:
- `support_context_signal` 매칭: 목표 도메인에 정의된 맥락 신호가 로그 텍스트에 포함
- `subdomain_consistent`: 로그의 evidence category가 해당 goal domain의 supporting_categories 소속

score_cap=0.12를 적용해 supporting 로그가 direct evidence보다 위로 올라오지 않도록 제한합니다.

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

이전 구현에서는 `metadata["topic"]`을 우선 사용했기 때문에,
서로 다른 anchor title을 가진 로그들이 동일한 broad topic("여행 예산 정리" ×4)을
반복했습니다. 이제 각 anchor의 실제 title이 CEU 요약에 반영됩니다:

- 이전: `'여행 예산 정리' 관련 활동 3회 수행`
- 이후: `'여행 예산 계획' 관련 활동 3회 수행. 진행: 항공권 비교 검색 → 항공편 탐색 → 여행 예산 계획`

---

## Firestore Collections (Production)

```
research_users/{user_id}
research_goals/{goal_id}
research_logs/{log_id}
research_goal_log_labels/{label_id}
```

---

## Known Limitations

### Lexical False Positive (Phrase-Level Matching)

BM25 및 lexical gate 기반 시스템에서 priority/evidence term이
짧은 일반 토큰("완료", "시작", "정리" 등)을 포함할 경우,
해당 토큰이 무관한 로그에서 hit되어 false positive가 발생할 수 있음.

**예시:**
- priority term: `"독서 완료"`
- false positive log: `"숙소 예약 완료"` → `"완료"` 토큰 단독 매칭

**완화 방법 (구현됨):**
- `score_priority_terms()` / `match_priority_phrase()` (`app/utils/text_matching.py`) — `PriorityTermMatch` 기반 weak-token-filtered phrase matching:
  - **Exact phrase match** (multi-token) → `mode="exact_phrase"`, score=1.0
  - **Core token match** (non-weak token 포함) → `mode="core_token"`, score=0.4 (token_weight)
  - **Weak-only token match** → `mode="weak_token_only"`, score=0.0 (**positive 신호 없음**)
  - **No match** → `mode="none"`, score=0.0
- `WEAK_TOKENS` 집합 (`app/utils/text_matching.py`) — 단독 positive 금지 토큰 목록
- priority / evidence matching 모두 동일한 `score_priority_terms()` 사용 (token_weight=0.4)
- Tier2 semantic gate — real embeddings 사용 시 cosine similarity < 0.50인 로그 reject

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

**향후 개선 방향:**
- phrase-level learned reranker 도입
- BM25 token scoring 시 IDF 기반 일반 토큰 자동 down-weighting
