#!/usr/bin/env python3
"""Dense candidate retrieval 비교 도구.

Query input에 따른 dense retrieval 결과를 비교한다.
Reranker 없음 — dense_score 기준 threshold만으로 선택.

비교 대상:
  dense_raw      raw goal text → dense retrieval
  dense_expanded Gemini-expanded query → dense retrieval

사용법:
    .venv/bin/python scripts/compare_dense_hybrid.py --goal_id G-U0001-01
    .venv/bin/python scripts/compare_dense_hybrid.py --goal_id G-U0001-01 --dense_threshold 0.92
    .venv/bin/python scripts/compare_dense_hybrid.py --goal_id G-U0001-01 --dense_threshold 0.80
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

from app.utils.logging_utils import setup_logging
setup_logging(level="WARNING")
import logging
logging.getLogger("app.retrieval.embedding_provider").setLevel(logging.ERROR)

from app.data_generation.export_utils import load_dataset_from_json
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.embedding_provider import get_embedding_provider
from app.retrieval.query_expansion import expand_goal_query
from app.retrieval.query_understanding import build_query
from app.schemas import CandidateLog, ResearchGoal, ResearchLog

logger = logging.getLogger(__name__)

_DATA_DIR = "data/synthetic"


@dataclass
class RetrievalResult:
    """Pure candidate retrieval result — no reranker."""
    candidates: list[CandidateLog]   # threshold 통과한 것들
    all_scored: list[CandidateLog]   # 전체 corpus score 순
    query_text: str
    expanded_terms: list[str] = field(default_factory=list)


def load_data():
    _, goals, logs, labels = load_dataset_from_json(_DATA_DIR)
    return goals, logs, labels


def list_goals(goals: list[ResearchGoal]) -> None:
    print("\n사용 가능한 goal_id 목록:")
    print(f"  {'goal_id':<16} {'유저':<8} {'목표'}")
    print("  " + "─" * 50)
    for g in goals:
        print(f"  {g.goal_id:<16} {g.user_id:<8} {g.title}")


def run_retrieval(
    goal: ResearchGoal,
    user_logs: list[ResearchLog],
    use_expansion: bool,
    retriever: DenseRetriever,
    dense_threshold: float | None,
) -> RetrievalResult:
    """Dense-only retrieval. No reranker."""
    query_obj = build_query(goal)

    if use_expansion:
        expanded = expand_goal_query(
            goal, query_obj,
            max_terms=15,
            use_mock_fallback=True,
        )
        query_text = (
            expanded.dense_query
            if hasattr(expanded, "dense_query")
            else expanded.full_text
        )
        expanded_terms = expanded.expanded_terms
    else:
        query_text = query_obj.canonical_text
        expanded_terms = []

    # Score all corpus logs
    all_scored = retriever.retrieve(query_text, top_n=len(user_logs))

    # Apply threshold on normalized dense_score
    if dense_threshold is not None:
        candidates = [c for c in all_scored if c.dense_score >= dense_threshold]
    else:
        candidates = list(all_scored)

    return RetrievalResult(
        candidates=candidates,
        all_scored=all_scored,
        query_text=query_text,
        expanded_terms=expanded_terms,
    )


def print_comparison(
    goal: ResearchGoal,
    results: dict[str, RetrievalResult],
    labels,
    user_logs: list[ResearchLog],
    dense_threshold: float | None,
) -> None:
    thr_str = f"dense_threshold={dense_threshold}" if dense_threshold is not None else "threshold=None (전체)"
    print("\n" + "=" * 64)
    print(f"  목표: {goal.title}  ({goal.goal_id})")
    print(f"  corpus: {len(user_logs)} logs  |  {thr_str}")
    print(f"  ※ reranker 없음 — dense_score 기준 선택")
    print("=" * 64)

    label_map = {lb.log_id: lb.label for lb in labels}
    total_relevant = sum(1 for lb in labels if lb.label == "relevant")
    total_partial  = sum(1 for lb in labels if lb.label == "partial")
    total_rel_all  = total_relevant + total_partial

    # ── 선택된 로그 ──────────────────────────────────────────────────────────
    print(f"\n{'─'*64}")
    for bl, res in results.items():
        n = len(res.candidates)
        print(f"\n[{bl.upper()} — {n}개 선택]")
        print(f"  query: {res.query_text[:72]}")
        for c in res.candidates:
            lbl = label_map.get(c.log_id, "?")
            mark = "✓" if lbl == "relevant" else ("△" if lbl == "partial" else "✗")
            print(f"  {mark}  {c.dense_score:.4f}  {c.log.date}  {c.log.title}")

    # ── 지표 비교 ─────────────────────────────────────────────────────────────
    def _metrics(res: RetrievalResult) -> dict:
        sel_ids = {c.log_id for c in res.candidates}
        hit_all = sum(
            1 for lb in labels
            if lb.log_id in sel_ids and lb.label in ("relevant", "partial")
        )
        n = len(res.candidates)
        recall    = round(hit_all / total_rel_all, 4) if total_rel_all else 0.0
        precision = round(hit_all / n, 4) if n else 0.0
        fp = n - hit_all
        fpr = round(fp / n, 4) if n else 0.0
        f1 = round(
            2 * precision * recall / (precision + recall), 4
        ) if (precision + recall) else 0.0
        return {"selected": n, "recall": recall, "precision": precision,
                "f1": f1, "fpr": fpr}

    bl_keys = list(results.keys())
    m_all = {bl: _metrics(res) for bl, res in results.items()}

    print(f"\n{'─'*64}")
    col_headers = "  ".join(f"{bl:>16}" for bl in bl_keys)
    print(f"  {'지표':<24} {col_headers}  {'winner':>14}")
    print(f"  {'─'*70}")

    rows = [
        ("selected_count", "selected", False),
        ("recall",         "recall",   False),
        ("precision",      "precision",False),
        ("f1",             "f1",       False),
        ("false_pos_rate", "fpr",      True),
    ]
    for label, key, lower_better in rows:
        vals = {bl: m_all[bl][key] for bl in bl_keys}
        winner = min(vals, key=vals.get) if lower_better else max(vals, key=vals.get)
        if key == "selected":
            val_str = "  ".join(f"{int(vals[bl]):>16d}" for bl in bl_keys)
        else:
            val_str = "  ".join(f"{vals[bl]:>16.4f}" for bl in bl_keys)
        print(f"  {label:<24} {val_str}  {winner:>14}")

    print(f"\n  relevant={total_relevant}  partial={total_partial}  (둘 다 hit 처리)")
    print(f"  ✓=relevant  △=partial  ✗=irrelevant")
    print("=" * 64 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dense candidate retrieval 비교")
    parser.add_argument("--goal_id", default=None, help="비교할 goal_id (생략 시 대화형 입력)")
    parser.add_argument(
        "--dense_threshold", type=float, default=None,
        help="dense_score threshold (e.g. 0.92). 생략 시 corpus 전체 사용.",
    )
    args = parser.parse_args()

    print("데이터 로딩 중...", end=" ", flush=True)
    goals, logs, labels = load_data()
    print("완료")

    # 공유 DenseRetriever — 임베딩은 goal별로 한 번만 인덱싱
    provider = get_embedding_provider()   # auto: Gemini if key set, else mock
    retriever = DenseRetriever(doc_provider=provider)

    goal_map = {g.goal_id: g for g in goals}

    while True:
        if args.goal_id:
            goal_id = args.goal_id.strip()
        else:
            list_goals(goals)
            print()
            goal_id = input("goal_id 입력 (종료: q): ").strip()

        if goal_id.lower() in ("q", "quit", "exit", ""):
            print("종료합니다.")
            break

        if goal_id not in goal_map:
            print(f"[오류] '{goal_id}' 를 찾을 수 없습니다.")
            if args.goal_id:
                break
            continue

        goal = goal_map[goal_id]
        user_logs = [l for l in logs if l.user_id == goal.user_id]
        user_labels = [
            lb for lb in labels
            if lb.user_id == goal.user_id and lb.goal_id == goal.goal_id
        ]

        print(f"\n'{goal.title}' ({goal.user_id}, {len(user_logs)} logs) — 인덱싱 중...",
              end=" ", flush=True)
        retriever.index(user_logs)
        print("완료")

        baselines = {
            "dense_raw":      False,   # use_expansion=False
            "dense_expanded": True,    # use_expansion=True (Gemini)
        }

        results: dict[str, RetrievalResult] = {}
        for bl, use_exp in baselines.items():
            print(f"  [{bl}] 실행 중...", end=" ", flush=True)
            try:
                results[bl] = run_retrieval(
                    goal, user_logs,
                    use_expansion=use_exp,
                    retriever=retriever,
                    dense_threshold=args.dense_threshold,
                )
                print("완료")
            except Exception as e:
                print(f"오류: {e}")

        if results:
            print_comparison(goal, results, user_labels, user_logs, args.dense_threshold)

        if args.goal_id:
            break

        again = input("다른 goal을 테스트하시겠습니까? (y/n): ").strip().lower()
        if again != "y":
            print("종료합니다.")
            break


if __name__ == "__main__":
    main()
