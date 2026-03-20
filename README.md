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

For Firestore access, place the service account key at the project root:
```
serviceAccountKey.json   ← never committed (.gitignore)
```

## Quick Start (Mock Data)

```bash
# Stage 1 — core method (raw goal)
python scripts/run_stage1.py --top_k 5

# Stage 1 — query expansion variant
python scripts/run_stage1.py --top_k 5 --expand

# Stage 2 — full RAG pipeline
python scripts/run_stage2.py --top_k 5
```

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

**Gemini LLM** — implement `GeminiLLMClient` in `app/llm/llm_client.py`:
```python
import google.generativeai as genai
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
```

## Firestore Collections (Research)

```
research_users/{user_id}
research_goals/{goal_id}
research_logs/{log_id}
research_goal_log_labels/{label_id}
```
