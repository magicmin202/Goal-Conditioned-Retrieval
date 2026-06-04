# Goal-Conditioned Retrieval

개인 목표 관리 서비스에서 누적되는 사용자 행동 로그를 효율적으로 분석하기 위한
**Goal-Conditioned Evidence Retrieval** 시스템.

---

## Architecture

```
Goal
 │
 ▼
[Stage 1] Candidate Retrieval
  Gemini embedding-001 cosine similarity
  dense_threshold = 0.85  (max-scaled)
 │
 ▼
[Stage 1] Reranking
  final_score = scale × (0.70×sem + 0.10×pri + 0.06×ev + 0.03×rel + 0.05×base)
               − negative_penalty
  Negative Veto  (domain conflict + no priority evidence → score=0)
 │
 ▼
[Stage 1] Relevance Filtering
  final_score ≥ 0.674  →  admitted
  dynamic k  (top-k 고정 없음, threshold 통과 전체)
 │
 ▼
[Harness Analysis]
  title / content / time → LLM prompt
  첫 실행: 전체 로그 분석 + 캐시 저장
  이후 실행: prev_summary + new_logs 만 전달
```

> Stage 2 (Local Expansion + Compression)는 현재 비활성화 상태입니다.
> 정보 손실 없이 admitted 로그를 LLM에 직접 전달하는 Harness 방식을 사용합니다.

---

## Scoring Formula

```
scale       = 1 / (0.70 + 0.10 + 0.06 + 0.03 + 0.05) = 1.064

relevance_score = scale × (
    0.70 × semantic_similarity      ← Gemini cosine (dominant)
  + 0.10 × priority_phrase_score   ← goal-specific key phrases
  + 0.06 × evidence_phrase_score   ← direct evidence vocabulary
  + 0.03 × related_score           ← indirect/related vocabulary
  + 0.05 × base_goal_overlap       ← raw goal-text token overlap
)

final_score = max(0, relevance_score − negative_penalty)
```

### Matching weights (실험 검증, Step 2)

| match type | 가중치 |
|---|---|
| `exact_phrase` (multi-token 구문 완전 포함) | `1.0` |
| `core_token` (핵심 토큰 매칭) | `1.0` |
| `weak_token_only` (완료/정리 등 범용어만 매칭) | `0.0` (무효) |

- title 위치 가중치 없음 (title_multiplier 제거)
- phrase : token = 1 : 1 (Step 2 실험 결과 P(relevant\|phrase)=1.0, P(relevant\|token)=0.835 → 1.2:1 이하)

### Negative Veto

```
veto 조건 (AND):
  raw_dm ≥ 0.70  (domain conflict 강도)
  pri_score < 0.05  (priority evidence 없음)

제약: phrase_penalty(0.70) ≥ veto_dm_threshold(0.70)
  → single phrase match 하나로 veto 발동 보장
```

### Negative Penalty 값

| 신호 | 패널티 |
|---|---|
| multi-token phrase match (body) | +0.70 |
| single-token match (body) | +0.40 |
| activity_type == "daily" | +0.20 (noise 보정, 반드시 > 0) |

---

## Validated Parameters (Step 1~5)

| 파라미터 | 값 | 검증 방법 |
|---|---|---|
| `dense_threshold` | **0.85** | Step 5 / recall 우선 설계 |
| `semantic_weight` | **0.70** | Step 3 — solo recall 0.757 |
| `priority_weight` | **0.10** | Step 3 |
| `evidence_weight` | **0.06** | Step 3 |
| `related_weight` | **0.03** | Step 3 |
| `base_weight` | **0.05** | Step 3 |
| `phrase_weight = token_weight` | **1.0 : 1.0** | Step 2 |
| `negative_penalty_phrase` | **0.70** | Step 4 (veto 제약 충족) |
| `negative_penalty_token` | **0.40** | Step 4 |
| `negative_daily_penalty` | **0.20** | Step 4 (> 0 필수) |
| `relevance_threshold` | **0.674** | Step 5 — recall=precision 교차점 |

---

## Harness Analysis

Relevance Filtering 통과 로그를 LLM에 직접 전달하는 캐시 기반 분석 시스템.

```
.harness/{user_id}/{goal_id}.json
  analyzed_log_ids  이미 분석된 로그 ID 집합
  analysis_summary  마지막 LLM 분석 결과
  last_updated      마지막 업데이트 시각
  log_count         누적 분석 로그 수
```

| 상황 | LLM 입력 | 프롬프트 |
|---|---|---|
| 첫 실행 | 전체 admitted 로그 | 초기 분석 |
| 이후 실행 (신규 로그 있음) | prev_summary + new_logs | 업데이트 분석 |
| 이후 실행 (신규 로그 없음) | — (캐시 반환) | LLM 호출 없음 |

로그 포맷 (title / content / time):
```
[2026-03-15] 토익 단어 암기
오늘 500개 단어 암기 목록 완성. 반복 학습 진행.
```

---

## Project Structure

```
app/
  config.py                         # 파라미터 설정 (검증된 값 포함)
  schemas.py                        # ResearchGoal, ResearchLog, RankedLog, ...

  retrieval/
    query_understanding.py          # Goal → QueryObject
    query_expansion.py              # 구조화 어휘 확장 (Gemini / heuristic)
    dense_retriever.py              # Gemini embedding-001 Dense retrieval
    candidate_retrieval.py          # Dense retrieval 진입점
    reranker.py                     # Lexical-control reranker + Negative Veto
    diversity_selector.py           # MMR (현재 비활성화)
    embedding_provider.py           # Mock / Gemini / SentenceTransformer

  pipeline/
    stage1_ranking_pipeline.py      # Stage 1: retrieval → reranking → filtering
                                    #   disable_redundancy=True (기본)
                                    #   disable_diversity=True  (기본, dynamic k)
    stage2_rag_pipeline.py          # Stage 2: consolidation (현재 미사용)

  harness/
    store.py                        # HarnessStore — 캐시 I/O
    prompts.py                      # 초기/업데이트 프롬프트 템플릿
    analyzer.py                     # HarnessAnalyzer — LLM 호출 + 캐시 관리

  llm/
    llm_client.py                   # LLM 인터페이스 (Mock / Gemini)
    analysis.py                     # Goal progress 분석 (Stage 2용, 현재 미사용)

  compression/
    local_expansion.py              # Stage 2 temporal expansion (현재 미사용)
    temporal_semantic_compressor.py # Stage 2 compressor (현재 미사용)

  evaluation/
    ranking_metrics.py              # Recall@selected, Precision@selected, F1, nDCG, MRR
    result_writer.py                # JSON 결과 저장

  utils/
    text_matching.py                # Phrase/token matching utilities

validation/
  step1_negative_penalty/           # negative penalty 수치 근거 실험
  step2_phrase_token/               # phrase:token 비율 검증
  step3_relevance_weights/          # relevance score 가중치 검증
  step4_negative_penalty/           # 새 가중치 기준 재검증
  step5_relevance_threshold/        # threshold sweep + 교차점 분석
  step6_cosine_redundancy/          # embedding 기반 redundancy 실험

scripts/
  run_stage1.py                     # Stage 1 + Harness Analysis 실행
  fix_labels.py                     # 레이블 교차 오염 수정
  regenerate_labels.py              # 레이블 재생성 (참고용)

data/synthetic/
  goals.json / logs.json / labels.json  # 합성 데이터 (100 users, 393 goals, 7030 logs)

.cache/
  embeddings/                       # 임베딩 영구 디스크 캐시 (SHA256 키)
  expansions/                       # Goal별 query expansion 캐시

.harness/
  {user_id}/{goal_id}.json          # Harness 분석 캐시
```

---

## Setup

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`.env`:
```
GEMINI_API_KEY=your_api_key_here
```

---

## Quick Start

```bash
# Stage 1 단독 실행
.venv/bin/python scripts/run_stage1.py --goal_id G-U0003-05 --baseline ours

# Stage 1 + Harness Analysis (첫 실행 → 전체 로그 LLM 분석)
.venv/bin/python scripts/run_stage1.py --goal_id G-U0003-05 --baseline ours --harness

# Harness Analysis (이후 실행 → 신규 로그만 전달)
.venv/bin/python scripts/run_stage1.py --goal_id G-U0003-05 --baseline ours --harness

# Harness 캐시 초기화 후 재분석
.venv/bin/python scripts/run_stage1.py --goal_id G-U0003-05 --baseline ours --harness --clear_harness
```

---

## Embedding Providers

| Provider | 조건 | 특징 |
|---|---|---|
| `GeminiEmbeddingProvider` | `GEMINI_API_KEY` 설정 | `gemini-embedding-001`, 3072-dim, 한국어 지원, 비대칭(doc/query 분리) |
| `MockEmbeddingProvider` | API 키 없음 (fallback) | SHA256 기반 결정적 벡터 (semantic 의미 없음) |

비대칭 임베딩:
- 로그 인덱싱: `task_type=RETRIEVAL_DOCUMENT`
- 쿼리 인코딩: `task_type=RETRIEVAL_QUERY`

---

## Evaluation Metrics

| 메트릭 | 설명 |
|---|---|
| `recall@selected` | relevant 로그 중 admitted 집합에 포함된 비율 |
| `precision@selected` | admitted 로그 중 relevant 비율 |
| `f1@selected` | recall · precision 조화 평균 |
| `ndcg@selected` | Normalized DCG (admitted 전체 기준) |
| `false_positive_rate` | admitted 중 irrelevant 비율 |
| `mrr` | Mean Reciprocal Rank |
| `candidate_recall` | dense retrieval 단계 recall |

> Stage 1 평가는 top-k 고정이 아닌 admitted 집합 전체를 기준으로 합니다.

---

## Data

합성 데이터 생성 방식: LLM 기반 로그 텍스트 렌더링 + rule-based 레이블 생성.

레이블 교차 오염 수정 (`scripts/fix_labels.py`):
- hobby 카테고리 (사진/요리/드로잉) 간 cross-contamination 632개 수정
- 수정 전: relevant=3,930 / 수정 후: relevant=3,298 (partial로 downgrade)

| 레이블 | 의미 |
|---|---|
| `relevant` | 해당 goal을 위해 생성된 로그 (도메인 일치 확인됨) |
| `partial` | 동일 카테고리 내 다른 도메인 또는 간접 연관 |
| `irrelevant` | noise 로그 또는 완전히 다른 도메인 |

---

## Key Design Decisions

### semantic_weight = 0.70 (dominant)

Step 3 실험에서 semantic 단독 recall = 0.757로 모든 lexical 컴포넌트를 압도.
lexical (pri/ev/rel/base) 합산 = 0.24로 threshold 경계에서 보조 역할만 수행.

### dynamic k (Relevance Filtering 기반)

top-k = 10 고정 대신 threshold=0.674 통과 로그 전체를 선택.
교차점(recall=precision≈0.779) 기준으로 F1 최적화.
Redundancy Penalty와 MMR Diversity Selection은 비활성화.

### phrase:token = 1:1

Step 2 실험에서 P(relevant|phrase)=1.0, P(relevant|token)=0.835.
이론적 최적 비율=1.2:1이나 실험 F1 차이 < 0.013으로 equal weight 채택.
token_only relevant 로그 722개 — token_weight=0 시 recall 급락.

### Harness Analysis

Stage 2 (Compression + LLM)를 제거하고 admitted 로그를 LLM에 직접 전달.
첫 실행 후 캐시에 저장 → 이후 신규 로그만 전달하여 토큰 최소화.

---

## Limitations

- **Synthetic data only**: 실제 사용자 로그의 표현 다양성·비정형성 미반영
- **LLM-generated lexicon**: Query Expansion non-determinism (expansion cache로 완화)
- **Single embedding space**: 목표(추상)와 로그(구체) 간 의미적 층위 차이
- **Label quality**: rule-based 생성 레이블, 실제 사용자 relevance 판단과 완전 일치 불보장
