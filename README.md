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
  → Schema Category Gate  (none → immediate reject)
  → Goal Lexical Gate     (pri=0 AND ev=0 AND base<0.04 → reject)
  → Negative Veto         (domain conflict + no positive evidence)
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
| Dense Embedding | 0.45 | Gemini `gemini-embedding-001` 또는 Mock |
| VocabBoost | 0.15 | priority/evidence/negative 어휘 기반 약한 보정 |

**Admission Gates (순서 보장):**
1. **Schema Category Gate** — log의 evidence category가 goal domain과 연관 없으면 즉시 reject
2. **Goal Lexical Gate** — priority/evidence/base 어휘 신호가 전혀 없으면 reject
3. **Negative Veto** — 도메인 충돌 + positive evidence 없을 때만 reject (keyword blacklist 아님)

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

### Reranker (precision-focused)

```
final_score =
    0.35 × priority_phrase_score    ← 가장 강한 goal lexical 신호
  + 0.20 × evidence_phrase_score    ← 직접 활동 어휘
  + 0.10 × related_score            ← 간접 연관 어휘
  + 0.15 × action_signal            ← 실제 행동/완료 신호
  + 0.10 × domain_consistency       ← schema category 기반 (goal-domain-aware)
  + 0.05 × semantic_similarity      ← tie-breaker only
  + 0.05 × base_goal_overlap        ← raw goal text overlap
  − negative_penalty
```

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

**Goal Domains (4개):**

| Domain | 감지 키워드 예시 | Core categories | Supporting categories |
|---|---|---|---|
| `productivity_development` | 개발, 취업, 포트폴리오, 개발자 | implementation, problem_solving, debugging | study_progress, planning |
| `learning_coding` | 알고리즘, 코테, 코딩테스트 | study_progress, problem_solving | implementation, planning |
| `fitness_muscle_gain` | 근육, 벌크, 헬스장, 증량 | training, body_metrics | nutrition, recovery |
| `fitness_fat_loss` | 다이어트, 감량, 칼로리, 체지방 | training, nutrition, body_metrics | recovery |

**Log Categories (9개):**
`training` · `nutrition` · `body_metrics` · `recovery` · `implementation` · `debugging` · `study_progress` · `problem_solving` · `planning`

---

## Project Structure

```
app/
  config.py                        # 실험 설정 (가중치, threshold, 캐시 경로 등)
  schemas.py                       # ResearchGoal, ResearchLog, RankedLog, ...

  retrieval/
    query_understanding.py         # Goal → QueryObject 정규화
    query_expansion.py             # 6-field 구조화 어휘 확장 (Gemini / heuristic)
    schema_category.py             # Goal domain + Log category 매핑
    sparse_retriever.py            # 순수 Python BM25Okapi (numpy 의존성 없음)
    dense_retriever.py             # Embedding 기반 retrieval
    hybrid_retriever.py            # BM25 × 0.40 + Dense × 0.45
    candidate_retrieval.py         # Hybrid + VocabBoost 통합 진입점
    reranker.py                    # Category-first lexical-control reranker
    diversity_selector.py          # MMR 기반 diversity selection
    embedding_provider.py          # Mock / Gemini / SentenceTransformer 추상화

  compression/
    local_expansion.py             # Anchor-centered temporal local expansion
    temporal_semantic_compressor.py  # Cluster → CompressedEvidenceUnit

  llm/
    llm_client.py                  # LLM 인터페이스 (Mock / Gemini)
    analysis.py                    # Goal progress LLM analysis

  pipeline/
    stage1_ranking_pipeline.py     # Stage 1: retrieval → reranking → admission
    stage2_rag_pipeline.py         # Stage 2: consolidation only (NO retrieval)

  evaluation/
    ranking_metrics.py             # Recall@k, Precision@k, MRR, nDCG@k
    rag_metrics.py                 # Goal Alignment, Token Reduction, etc.

  utils/
    text_matching.py               # Phrase/token/title-aware matching
    logging_utils.py

scripts/
  run_stage1.py                    # Stage 1 단독 실행
  run_stage2.py                    # Stage 1 → Stage 2 체인 실행
  compare_debug_runs.py            # 3-way 비교 디버깅 실험
  generate_synthetic_dataset.py    # 합성 데이터 생성

.cache/
  embeddings/                      # Gemini 임베딩 영구 디스크 캐시
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

---

## Quick Start

```bash
# 데이터 생성 (3 users, ~30 logs each)
.venv/bin/python scripts/generate_synthetic_dataset.py --small

# Stage 1 단독 — no expansion
.venv/bin/python scripts/run_stage1.py --goal_id G-U0001-01 --top_k 5

# Stage 1 단독 — with expansion
.venv/bin/python scripts/run_stage1.py --goal_id G-U0001-01 --top_k 5 --expand

# Stage 1 → Stage 2 전체 체인
.venv/bin/python scripts/run_stage2.py --goal_id G-U0001-01 --top_k 5

# Gemini 실제 임베딩 사용 (GEMINI_API_KEY 필요)
.venv/bin/python scripts/run_stage2.py --goal_id G-U0001-01 --real_embeddings

# Mock LLM 사용 (API 키 없이 테스트)
.venv/bin/python scripts/run_stage2.py --goal_id G-U0001-01 --mock

# 특정 user 전체 목표 확인
.venv/bin/python scripts/run_stage1.py --user_id U0002 --top_k 5 --expand
```

### 진단/디버깅

```bash
# 3-way 비교: no-expand vs expand vs stage2-chain
.venv/bin/python scripts/compare_debug_runs.py --goal_id G-U0001-01

# JSON으로 결과 저장
.venv/bin/python scripts/compare_debug_runs.py --goal_id G-U0001-01 --json_out debug.json
```

`compare_debug_runs.py`는 다음 3가지 실험을 동일 goal에 대해 실행하고 비교합니다:

| 실험 | 설명 |
|---|---|
| A | Stage1 standalone, expansion 없음 |
| B | Stage1 standalone, expansion 있음 |
| C | Stage2 chain 내부 Stage1 (expansion 있음) |

출력에는 **candidate pool diff**, **admission trace**, **가설 판별 결과**가 포함됩니다.

---

## Caching

API 호출을 최소화하기 위한 3단계 캐시 구조:

| 캐시 | 경로 | 키 |
|---|---|---|
| Gemini 임베딩 | `.cache/embeddings/{model}.json` | SHA256(text)[:16] |
| Query expansion | `.cache/expansions/{goal_id}.json` | goal_id |
| Reranker dense score | `candidate.dense_score` 재사용 | — |

캐시 덕분에 2차 실행부터 임베딩 API 호출이 0회입니다.

---

## Embedding Providers

| Provider | 설정 | 특징 |
|---|---|---|
| `MockEmbeddingProvider` | 기본 (의존성 없음) | SHA256 기반 결정적 — 실행 간 동일 결과 |
| `GeminiEmbeddingProvider` | `--real_embeddings` + `GEMINI_API_KEY` | `gemini-embedding-001`, 3072-dim, 한국어 지원 |
| `SentenceTransformerProvider` | `pip install sentence-transformers` | 로컬 multilingual 모델 |

---

## Fallback 동작

| 상황 | 동작 |
|---|---|
| `GEMINI_API_KEY` 설정됨 + `--real_embeddings` | Gemini Embedding API 사용 |
| `GEMINI_API_KEY` 없음 | Mock 임베딩 (hash 기반) |
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

### Stage 2는 retrieval 단계가 아니다

이전 구현에서 Stage 2가 `HYBRID_EXPANDED`로 전체 corpus를 재검색하면서
Stage 1에서 선별된 evidence와 무관한 로그(스트레칭, 요리 등)가 anchor로 올라오는 문제가 있었습니다.

현재 구현에서 Stage 2는 **Stage 1 admitted anchors를 고정 입력**으로 받아
temporal local expansion과 summarization만 수행합니다.

### Schema category gate는 scoring 이전에 적용된다

`action_signal(0.15) + domain_consistency(0.10)` 만으로도 goal-agnostic 로그가
admitted 되는 문제를 막기 위해, scoring 전에 category gate를 먼저 적용합니다.
`relevance=none`인 로그는 scoring 자체를 수행하지 않습니다.

### Dense query는 goal_summary만 사용한다

lexical expansion 전체를 dense embedding에 주입하면 embedding centroid가 흐려져
관련 없는 로그가 cosine similarity 기준 상위에 올라오는 semantic drift가 발생합니다.
dense query = `goal_summary + core_intents[0]`으로 최소화합니다.

### MockEmbeddingProvider는 SHA256 기반이다

Python `hash()`는 `PYTHONHASHSEED`에 의해 실행마다 다른 값을 반환합니다.
현재 구현에서는 `SHA256`을 사용해 실행 간 결정적(deterministic) 임베딩을 보장합니다.

---

## Firestore Collections (Production)

```
research_users/{user_id}
research_goals/{goal_id}
research_logs/{log_id}
research_goal_log_labels/{label_id}
```
