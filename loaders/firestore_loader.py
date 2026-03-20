"""
Prologue Firestore 데이터 로더.

실제 Firestore 경로:
  users/{uid}/goal_projects/{projectId}
  users/{uid}/goal_archives/{projectId}
  users/{uid}/calendar_events/{docId}
  users/{uid}/chat_threads/{threadId}/messages
  users/{uid}/work_logs/{YYYY-MM-DD}
  users/{uid}/entry_responses/{docId}

로그 데이터는 아래 정규화 포맷으로 반환한다:
  {
      "log_id": str,
      "date":   str,   # "YYYY-MM-DD" 또는 ISO 문자열
      "text":   str,
      "type":   "work_log | calendar_event | chat_message | entry_response",
      "source": str,   # Firestore 컬렉션명
  }
"""

import logging
from datetime import date, datetime
from typing import Any

from google.cloud.firestore_v1 import DocumentSnapshot

from config.firebase_config import get_firestore_client

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 내부 유틸리티
# ---------------------------------------------------------------------------

def _user_ref(user_id: str):
    """users/{uid} DocumentReference를 반환한다."""
    return get_firestore_client().collection("users").document(user_id)


def _snapshot_to_dict(snapshot: DocumentSnapshot) -> dict[str, Any]:
    """DocumentSnapshot → dict 변환 (doc_id 포함)."""
    data = snapshot.to_dict() or {}
    data["_doc_id"] = snapshot.id
    return data


def _extract_work_log_text(data: dict[str, Any]) -> str:
    """work_log 문서에서 text 필드를 방어적으로 조합한다.

    탐색 우선순위: title → content → tasks 내 텍스트
    """
    parts: list[str] = []

    title = data.get("title")
    if isinstance(title, str) and title.strip():
        parts.append(title.strip())

    content = data.get("content")
    if isinstance(content, str) and content.strip():
        parts.append(content.strip())

    tasks = data.get("tasks")
    if isinstance(tasks, list):
        for task in tasks:
            if isinstance(task, dict):
                task_text = task.get("text") or task.get("title") or task.get("content")
                if isinstance(task_text, str) and task_text.strip():
                    parts.append(task_text.strip())
            elif isinstance(task, str) and task.strip():
                parts.append(task.strip())

    return " | ".join(parts) if parts else ""


def _coerce_date_str(value: Any) -> str:
    """Firestore Timestamp / datetime / date / str → "YYYY-MM-DD" 문자열."""
    if hasattr(value, "date"):          # Firestore Timestamp or datetime
        dt = value.date() if isinstance(value, datetime) else value
        return str(dt)
    if isinstance(value, date):
        return str(value)
    if isinstance(value, str):
        return value
    return ""


# ---------------------------------------------------------------------------
# 핵심 공개 함수
# ---------------------------------------------------------------------------

def get_user_goal_projects(user_id: str) -> list[dict[str, Any]]:
    """해당 user의 활성 goal_projects 목록을 반환한다.

    goal_archives도 함께 조회하려면 include_archives=True 옵션을 추가하면 된다.
    현재 단계에서는 goal_projects만 로드한다.
    """
    user_ref = _user_ref(user_id)
    projects: list[dict[str, Any]] = []

    try:
        docs = user_ref.collection("goal_projects").stream()
        for doc in docs:
            projects.append(_snapshot_to_dict(doc))
        logger.info("user=%s goal_projects %d건 로드 완료", user_id, len(projects))
    except Exception as exc:
        logger.error("goal_projects 로드 실패 (user=%s): %s", user_id, exc)
        raise

    return projects


def get_project_logs(
    user_id: str,
    project_id: str,
) -> list[dict[str, Any]]:
    """특정 project와 연관된 work_logs를 정규화하여 반환한다.

    work_logs 문서의 project_id 필드(또는 projectId)로 필터링한다.
    필드명이 다를 경우를 대비해 두 가지 필드명을 모두 시도한다.
    """
    user_ref = _user_ref(user_id)
    normalized: list[dict[str, Any]] = []

    for field_name in ("project_id", "projectId"):
        try:
            query = (
                user_ref.collection("work_logs")
                .where(field_name, "==", project_id)
            )
            docs = list(query.stream())
            if docs:
                logger.info(
                    "user=%s project=%s work_logs %d건 (field=%s)",
                    user_id, project_id, len(docs), field_name,
                )
                for doc in docs:
                    normalized.append(_normalize_work_log(doc))
                return normalized
        except Exception as exc:
            logger.warning(
                "work_logs 쿼리 실패 (field=%s, user=%s): %s",
                field_name, user_id, exc,
            )

    logger.info(
        "user=%s project=%s에 해당하는 work_logs 없음",
        user_id, project_id,
    )
    return normalized


def get_logs_by_date_range(
    user_id: str,
    start_date: str | date,
    end_date: str | date,
) -> list[dict[str, Any]]:
    """날짜 범위로 work_logs를 정규화하여 반환한다.

    work_logs 컬렉션의 문서 ID가 "YYYY-MM-DD" 형식이므로
    문자열 대소비교로 범위 필터링한다.

    Args:
        user_id:    Firestore uid
        start_date: 시작일 ("YYYY-MM-DD" 문자열 또는 date 객체)
        end_date:   종료일 ("YYYY-MM-DD" 문자열 또는 date 객체, 포함)
    """
    start_str = str(start_date)
    end_str = str(end_date)

    user_ref = _user_ref(user_id)
    normalized: list[dict[str, Any]] = []

    try:
        docs = (
            user_ref.collection("work_logs")
            .order_by("__name__")
            .start_at({"__name__": user_ref.collection("work_logs").document(start_str)})
            .end_at({"__name__": user_ref.collection("work_logs").document(end_str)})
            .stream()
        )
        for doc in docs:
            normalized.append(_normalize_work_log(doc))
        logger.info(
            "user=%s date_range=%s~%s work_logs %d건 로드",
            user_id, start_str, end_str, len(normalized),
        )
    except Exception as exc:
        # cursor 기반 쿼리가 실패하면 전체 로드 후 클라이언트 필터링으로 폴백
        logger.warning("cursor 기반 날짜 범위 쿼리 실패, 폴백 사용: %s", exc)
        try:
            all_docs = user_ref.collection("work_logs").stream()
            for doc in all_docs:
                if start_str <= doc.id <= end_str:
                    normalized.append(_normalize_work_log(doc))
            logger.info(
                "user=%s 폴백 date_range=%s~%s work_logs %d건 로드",
                user_id, start_str, end_str, len(normalized),
            )
        except Exception as fallback_exc:
            logger.error("work_logs 폴백 로드 실패 (user=%s): %s", user_id, fallback_exc)
            raise

    return normalized


# ---------------------------------------------------------------------------
# 정규화 함수
# ---------------------------------------------------------------------------

def _normalize_work_log(doc: DocumentSnapshot) -> dict[str, Any]:
    data = _snapshot_to_dict(doc)
    return {
        "log_id": doc.id,
        "date": _coerce_date_str(data.get("date") or doc.id),
        "text": _extract_work_log_text(data),
        "type": "work_log",
        "source": "work_logs",
    }


def _normalize_calendar_event(doc: DocumentSnapshot) -> dict[str, Any]:
    data = _snapshot_to_dict(doc)
    text_parts = [
        data.get("title", ""),
        data.get("description", ""),
    ]
    return {
        "log_id": doc.id,
        "date": _coerce_date_str(data.get("start") or data.get("date", "")),
        "text": " | ".join(p for p in text_parts if p),
        "type": "calendar_event",
        "source": "calendar_events",
    }


def _normalize_chat_message(doc: DocumentSnapshot, thread_id: str) -> dict[str, Any]:
    data = _snapshot_to_dict(doc)
    return {
        "log_id": f"{thread_id}/{doc.id}",
        "date": _coerce_date_str(data.get("created_at") or data.get("timestamp", "")),
        "text": data.get("content") or data.get("text", ""),
        "type": "chat_message",
        "source": "chat_threads/messages",
    }


def _normalize_entry_response(doc: DocumentSnapshot) -> dict[str, Any]:
    data = _snapshot_to_dict(doc)
    return {
        "log_id": doc.id,
        "date": _coerce_date_str(data.get("created_at") or data.get("date", "")),
        "text": data.get("response") or data.get("content", ""),
        "type": "entry_response",
        "source": "entry_responses",
    }


# ---------------------------------------------------------------------------
# 보조 함수 (확장용)
# ---------------------------------------------------------------------------

def get_calendar_events(user_id: str) -> list[dict[str, Any]]:
    """user의 calendar_events를 정규화하여 반환한다."""
    user_ref = _user_ref(user_id)
    results: list[dict[str, Any]] = []
    try:
        for doc in user_ref.collection("calendar_events").stream():
            results.append(_normalize_calendar_event(doc))
        logger.info("user=%s calendar_events %d건 로드", user_id, len(results))
    except Exception as exc:
        logger.error("calendar_events 로드 실패 (user=%s): %s", user_id, exc)
        raise
    return results


def get_chat_messages(user_id: str, thread_id: str) -> list[dict[str, Any]]:
    """특정 chat_thread의 messages를 정규화하여 반환한다."""
    user_ref = _user_ref(user_id)
    results: list[dict[str, Any]] = []
    try:
        messages_ref = (
            user_ref.collection("chat_threads")
            .document(thread_id)
            .collection("messages")
        )
        for doc in messages_ref.stream():
            results.append(_normalize_chat_message(doc, thread_id))
        logger.info(
            "user=%s thread=%s messages %d건 로드", user_id, thread_id, len(results)
        )
    except Exception as exc:
        logger.error(
            "chat_threads/messages 로드 실패 (user=%s, thread=%s): %s",
            user_id, thread_id, exc,
        )
        raise
    return results


def get_entry_responses(user_id: str) -> list[dict[str, Any]]:
    """user의 entry_responses를 정규화하여 반환한다."""
    user_ref = _user_ref(user_id)
    results: list[dict[str, Any]] = []
    try:
        for doc in user_ref.collection("entry_responses").stream():
            results.append(_normalize_entry_response(doc))
        logger.info("user=%s entry_responses %d건 로드", user_id, len(results))
    except Exception as exc:
        logger.error("entry_responses 로드 실패 (user=%s): %s", user_id, exc)
        raise
    return results
