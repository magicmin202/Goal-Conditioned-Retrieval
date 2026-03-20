# Goal-Conditioned-Retrieval

개인 목표 관리 시스템에서 누적되는 사용자 행동 로그 데이터를 효율적으로 분석하기 위한 Goal-Conditioned Retrieval 기반 AI 분석 시스템을 설계하고 구현하는 것

## Research Problem

**Intent → Personal Timeline Events Retrieval**

> 사용자의 목표(Goal)가 주어졌을 때, 장기간의 개인 행동 로그(Work Logs)에서 관련 이벤트를 검색하는 문제

## Pipeline

```
Goal → Retrieval → Ranking → Compression → LLM
```

| Stage | Description |
|---|---|
| **Stage 1** | Retrieval + Ranking component 연구 |
| **Stage 2** | Full RAG (Retrieval + Compression + LLM Analysis) |

## Project Structure

```
app/
  config.py                   # Experiment config (top_k, weights, etc.)
  schemas.py                  # ResearchGoal, ResearchLog, CandidateLog, ...
  firestore_loader.py         # Firebase Admin SDK + Firestore client
  repository.py               # Data access layer (research_* collections)
  main.py                     # Smoke test entry point

  retrieval/
    query_understanding.py    # Goal → QueryObject normalization
    query_expansion.py        # LLM query expansion (mock / Gemini)
    sparse_retriever.py       # BM25 / TF sparse retrieval
    dense_retriever.py        # Embedding-based retrieval (mock / sentence-transformers)
    hybrid_retriever.py       # Sparse + Dense via RRF
    candidate_retrieval.py    # Unified retrieval entry point
    reranker.py               # Goal-Conditioned Evidence Ranker
    diversity_selector.py     # MMR-based diversity selection

  compression/
    local_expansion.py        # Anchor-based local log expansion
    temporal_semantic_compressor.py  # Compress clusters → CompressedEvidenceUnit

  llm/
    llm_client.py             # LLM interface (Mock / Gemini TODO)
    analysis.py               # Goal progress analysis

  pipeline/
    stage1_ranking_pipeline.py
    stage2_rag_pipeline.py

  evaluation/
    ranking_metrics.py        # Recall@k, Precision@k, MRR, nDCG@k
    rag_metrics.py            # Goal Alignment, Token Reduction, etc.

  utils/
    text_utils.py
    date_utils.py
    logging_utils.py

scripts/
  run_stage1.py
  run_stage2.py
```

## Setup

```bash
pip install -r requirements.txt
```

### Gemini API 설정

query expansion (Stage 2 My Method, Stage 1 expansion variant)에 Gemini API를 사용합니다.

```bash
export GEMINI_API_KEY=your_api_key_here
# 또는
export GOOGLE_API_KEY=your_api_key_here
```

API 키가 없는 경우 자동으로 **heuristic fallback** (goal-specific 키워드 사전)을 사용합니다.
연구 목적으로는 fallback만으로도 테스트 가능합니다.

config 변경 (`app/config.py`):
```python
GeminiConfig(
    model_name="gemini-2.0-flash",  # Gemini 모델
    use_mock_fallback=True,          # API 실패 시 heuristic fallback
)
```

### Firestore 설정 (선택)

실제 Firestore 연결이 필요한 경우:
```
serviceAccountKey.json   ← 프로젝트 루트에 배치 (never committed)
```

## Quick Start

```bash
# 1. 데이터 생성 (small mode — 3 users, ~30 logs each)
python scripts/generate_synthetic_dataset.py --small

# 2. Stage 1 — core method (raw goal query)
python scripts/run_stage1.py --auto --top_k 5

# 3. Stage 1 — query expansion variant (heuristic or Gemini)
python scripts/run_stage1.py --auto --top_k 5 --expand

# 4. Stage 2 — full RAG pipeline (Gemini expansion + compression + LLM)
python scripts/run_stage2.py --auto --top_k 5

# 특정 user/goal 지정
python scripts/run_stage1.py --user_id U0001 --goal_id G-U0001-01 --top_k 5
```

### fallback 동작

| 상황 | 동작 |
|---|---|
| GEMINI_API_KEY 설정됨 | Gemini API로 goal-specific expansion 생성 |
| API 키 없음 | heuristic 사전 기반 expansion (goal-specific 키워드) |
| API 호출 실패 | `use_mock_fallback=True`이면 heuristic으로 자동 전환 |

## Baselines (Stage 1)

| Method | Description |
|---|---|
| Dense Similarity | Dense retrieval only |
| Hybrid Similarity | Sparse + Dense RRF |
| LLM Query Expansion + Hybrid | Query expansion variant |
| Cross-encoder Reranking | Reranking baseline |
| **Goal-Conditioned (Proposed)** | Hybrid + Goal-Conditioned Reranker + MMR |

## Extending

**Real embeddings** — replace `_mock_embed` in `app/retrieval/dense_retriever.py`:
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
return model.encode(text).tolist()
```

**Gemini query expansion** — set API key and it activates automatically:
```bash
export GEMINI_API_KEY=your_key
python scripts/run_stage2.py --auto  # uses real Gemini expansion
```

**Gemini LLM analysis** — replace mock in `app/llm/llm_client.py`:
```python
pipeline = Stage2Pipeline(config=cfg, use_mock_llm=False)
```

## Firestore Collections (Research)

```
research_users/{user_id}
research_goals/{goal_id}
research_logs/{log_id}
research_goal_log_labels/{label_id}
```
