"""Firebase Admin SDK initialization and Firestore client provider."""
from __future__ import annotations
import logging
import os
from typing import Any

import firebase_admin
from firebase_admin import credentials, firestore

logger = logging.getLogger(__name__)
_app: firebase_admin.App | None = None


def _resolve_credential_path() -> str:
    env_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if env_path and os.path.isfile(env_path):
        return env_path
    default = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "serviceAccountKey.json"
    )
    if os.path.isfile(default):
        return default
    raise FileNotFoundError(
        "serviceAccountKey.json not found. "
        "Set GOOGLE_APPLICATION_CREDENTIALS or place it at project root."
    )


def initialize_firebase() -> firebase_admin.App:
    """Initialize Firebase Admin SDK (idempotent)."""
    global _app
    if _app is not None:
        return _app
    try:
        _app = firebase_admin.get_app()
        return _app
    except ValueError:
        pass
    cred = credentials.Certificate(_resolve_credential_path())
    _app = firebase_admin.initialize_app(cred)
    logger.info("Firebase Admin SDK initialized.")
    return _app


def get_firestore_client() -> firestore.Client:
    initialize_firebase()
    return firestore.client()


def batch_get_docs(
    client: firestore.Client,
    collection: str,
    filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Fetch all docs from a collection with optional equality filters."""
    ref = client.collection(collection)
    query = ref
    if filters:
        for field_name, value in filters.items():
            query = query.where(field_name, "==", value)
    docs = []
    for doc in query.stream():
        data = doc.to_dict() or {}
        data.setdefault("_doc_id", doc.id)
        docs.append(data)
    return docs


def write_doc(
    client: firestore.Client,
    collection: str,
    doc_id: str,
    data: dict[str, Any],
    merge: bool = True,
) -> None:
    client.collection(collection).document(doc_id).set(data, merge=merge)
    logger.debug("Written doc %s/%s", collection, doc_id)
