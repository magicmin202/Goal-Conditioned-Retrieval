"""
Retrieval 실험 실행 엔트리포인트.

사용 예:
    python run_experiment.py --user_id <uid> --project_id <pid>

향후 추가될 baseline:
    --baseline dense_retrieval
    --baseline hybrid_retrieval
    --baseline llm_query_expansion
    --baseline llm_reranking
    --baseline goal_conditioned  (proposed method)
"""

import argparse
import logging
import sys
from collections import Counter

from config.firebase_config import get_firestore_client
from loaders.firestore_loader import get_project_logs, get_user_goal_projects

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 인자 파싱
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prologue Retrieval 실험 실행기",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--user_id", required=True, help="Firestore uid")
    parser.add_argument("--project_id", required=True, help="goal_projects document id")
    # 향후 baseline 선택을 위한 플레이스홀더
    parser.add_argument(
        "--baseline",
        default=None,
        choices=[
            "dense_retrieval",
            "hybrid_retrieval",
            "llm_query_expansion",
            "llm_reranking",
            "goal_conditioned",
        ],
        help="실행할 Retrieval baseline (현재 미구현)",
    )
    return parser


# ---------------------------------------------------------------------------
# 실험 실행
# ---------------------------------------------------------------------------

def run(user_id: str, project_id: str) -> None:
    # 1. Firestore 연결 확인
    logger.info("Firestore 연결 중...")
    get_firestore_client()
    logger.info("Firestore 연결 성공")

    # 2. Goal project 로드
    logger.info("user=%s의 goal_projects 로드 중...", user_id)
    projects = get_user_goal_projects(user_id)

    target_project = next(
        (p for p in projects if p.get("_doc_id") == project_id), None
    )
    if target_project is None:
        logger.warning(
            "project_id=%s를 goal_projects에서 찾을 수 없습니다. "
            "project_id를 그대로 사용합니다.",
            project_id,
        )
        project_title = project_id
    else:
        project_title = (
            target_project.get("title")
            or target_project.get("name")
            or project_id
        )

    print(f"\nLoaded goal project: {project_title}")

    # 3. 로그 로드
    logger.info("project=%s의 logs 로드 중...", project_id)
    logs = get_project_logs(user_id, project_id)

    # 4. 통계 출력
    total = len(logs)
    type_counts = Counter(log.get("type", "unknown") for log in logs)

    print(f"Total logs: {total}")
    print(f"Work logs: {type_counts.get('work_log', 0)}")

    if total > 0:
        print("\n[Log type breakdown]")
        for log_type, count in sorted(type_counts.items()):
            print(f"  {log_type}: {count}")

        empty_text = sum(1 for log in logs if not log.get("text"))
        if empty_text:
            logger.warning("text 필드가 비어있는 로그: %d건", empty_text)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.baseline is not None:
        logger.warning(
            "--baseline=%s 는 아직 구현되지 않았습니다. 데이터 로딩만 실행합니다.",
            args.baseline,
        )

    try:
        run(user_id=args.user_id, project_id=args.project_id)
    except Exception as exc:
        logger.error("실험 실행 중 오류 발생: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
