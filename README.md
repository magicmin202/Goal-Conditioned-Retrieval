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
  → Goal Lexical Gate       (primary signal 없으면 support gate 시도)
      └ Support Gate        (support_context_hit + subdomain_consistent → admit, score_cap=0.12)
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
2. **Goal Lexical Gate + Support Gate** — primary signal(priority/evidence/base) 있으면 `direct` 경로, 없으면 support gate 시도
   - **Support Gate**: `support_context_signal` 매칭 + `subdomain_consistent` → `supporting` 경로 admit (score_cap=0.12)
   - 둘 다 해당 없으면 reject
3. **Negative Veto** — 도메인 충돌 + positive evidence 없을 때만 reject (keyword blacklist 아님)

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
  + 0.25 × goal_progress            ← category value prior (booking > research)
  + 0.15 × domain_consistency       ← goal-domain-aware activity type

final_score =
    relevance_score                 (×0.70 스케일)
  + quality_weight(0.30) × quality_score
  − negative_penalty
```

**핵심 설계 원칙:** relevance(관련성)와 evidence quality(분석 가치)를 분리.
goal과 관련 있는 로그라도 generic/redundant하면 상위 anchor로 올라오지 않도록 설계.

### Evidence Quality 예시

| 로그 | specificity | actionability | goal_progress | quality_total |
|---|---|---|---|---|
| 방콕 항공권 예약 완료 (28만원) | 0.70 | 0.85 | 1.00 | **0.87** |
| 호텔 예약 확정 | 0.20 | 0.80 | 1.00 | **0.67** |
| 여행 준비물 쇼핑 | 0.10 | 0.00 | 0.60 | **0.31** |
| 짐 준비 (조사해봄) | 0.10 | 0.00 | 0.60 | **0.28** |

### Redundancy Penalty (Stage1Pipeline post-rank)

동일/유사 로그가 이미 admitted 된 경우 후순위 로그에 penalty 적용:

| 조건 | 감점 |
|---|---|
| title 완전 일치 | -0.30 |
| title 토큰 유사도 ≥ 60% | -0.15 |

그리디 방식: score 높은 순서로 admitted set에 추가하면서 적용.

### Query Expansion (6-field structured)

| 필드 | 역할 | 사용처 |
|---|---|---|
| `priority_terms` | 핵심 식별 표현 (4-8개) | BM25 query + reranker 최강 신호 |
| `evidence_terms` | 직접 활동 어휘 (8-15개) | BM25 query + reranker 중간 신호 |
| `related_terms` | 간접 연관 표현 (5-10개) | reranker 약한 신호 (dense query 제외) |
| `negative_terms` | 무관 도메인 표현 (8-15개) | penalty + veto (retrieval query 불포함) |
| `core_intents` | 핵심 하위 목표 (3-5개) | dense query 보조 |
| `goal_summary` | 한 문장 요약 | dense query 주체 |

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
| `travel_planning` | 여행, 해외, 항공, 배낭, 저비용 | booking, budgeting, logistics | travel_research, planning | — |
| `language_exam` | 토익, toeic, lc, rc, 리스닝, 리딩, 모의고사 | mock_test, reading_practice, listening_practice, vocab_building | graduate_admission_prep, career_support | 취업 준비, 대학원 준비, gre, 지도교수, 지원서 |

**Log Categories (20개):**
`training` · `nutrition` · `body_metrics` · `recovery` ·
`implementation` · `debugging` · `study_progress` · `problem_solving` · `planning` ·
`booking` · `budgeting` · `logistics` · `travel_research` ·
`language_exam_study` · `vocab_building` · `listening_practice` · `reading_practice` · `mock_test` ·
`graduate_admission_prep` · `career_support`

**Category Value Priors (goal_progress 산출 기준):**

| Domain | 높은 가치 카테고리 | 낮은 가치 카테고리 |
|---|---|---|
| travel_planning | booking(1.0), budgeting(0.8), logistics(0.6) | travel_research(0.25) |
| fitness_muscle_gain | training(1.0), body_metrics(0.9) | recovery(0.4) |
| productivity_development | implementation(1.0), debugging(0.85), problem_solving(0.85) | planning(0.3) |
| learning_coding | problem_solving(1.0), study_progress(0.8) | planning(0.3) |

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
    ranking_metrics.py             # Recall@k, Precision@k, MRR, nDCG@k
    rag_metrics.py                 # Goal Alignment, Token Reduction, etc.

  utils/
    text_matching.py               # Phrase/token/title-aware matching utilities
    logging_utils.py

scripts/
  run_stage1.py                    # Stage 1 단독 실행
  run_stage2.py                    # Stage 1 → Stage 2 체인 실행
  compare_debug_runs.py            # 3-way 비교 디버깅 실험
  generate_synthetic_dataset.py    # 합성 데이터 생성

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
.venv/bin/python scripts/run_stage1.py --auto 2>&1 | grep -E "\[ADMIT\]|\[REJECT\]|\[REDUNDANCY\]"
```

출력 예시:
```
[ADMIT|direct]     L-U0001-0005  score=0.3252  cat=implementation(core)     [포트폴리오 구현]
[ADMIT|supporting] L-U0001-0008  score=0.0980  cat=graduate_admission_prep  [GRE 대비]  support=['gre']
[REJECT|direct]    L-U0001-0012  reason=domain_conflict_veto(dm=0.70)       [주식 시장 공부]
[REJECT|direct]    L-U0001-0017  reason=category_mismatch(study_progress)   [자료구조 공부]
[REDUNDANCY]       L-U0001-0031  penalty=0.15  reason=similar(0.72)         [여행 준비물 쇼핑]
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
| `Precision@k` | top-k 중 정답 비율 |
| `MRR` | Mean Reciprocal Rank |
| `nDCG@k` | Normalized Discounted Cumulative Gain |

### Stage 2

| 메트릭 | 설명 |
|---|---|
| `Goal Alignment` | evidence unit이 goal과 얼마나 관련됐는지 |
| `Token Reduction` | 원본 대비 압축률 |
| `Coverage` | 정답 로그 중 evidence unit에 포함된 비율 |

---

## Key Design Decisions

### Relevance와 Evidence Quality는 분리된 문제다

이전 구현에서 relevance(goal 관련성)만으로 admission을 결정했기 때문에,
goal과 관련은 있지만 분석 가치가 낮은 generic 로그
(`여행 준비물 쇼핑`, `짐 준비`, `조사해봄`)가 과도하게 admitted되었습니다.

현재 구현에서 `final_score = relevance_score (70%) + quality_score (30%)`로
두 신호를 명시적으로 분리합니다. quality_score는 specificity, actionability,
category value prior로 구성됩니다.

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

---

## Firestore Collections (Production)

```
research_users/{user_id}
research_goals/{goal_id}
research_logs/{log_id}
research_goal_log_labels/{label_id}
```
