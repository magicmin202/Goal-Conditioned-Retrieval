"""
Firebase Admin SDK 초기화 및 Firestore 클라이언트 제공 모듈.

serviceAccountKey.json 경로는 환경변수 GOOGLE_APPLICATION_CREDENTIALS 또는
프로젝트 루트의 serviceAccountKey.json 파일을 순서대로 탐색합니다.
"""

import logging
import os

import firebase_admin
from firebase_admin import credentials, firestore

logger = logging.getLogger(__name__)

_app: firebase_admin.App | None = None


def _resolve_credential_path() -> str:
    """서비스 계정 키 파일 경로를 결정한다."""
    env_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if env_path and os.path.isfile(env_path):
        return env_path

    default_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "serviceAccountKey.json"
    )
    if os.path.isfile(default_path):
        return default_path

    raise FileNotFoundError(
        "serviceAccountKey.json을 찾을 수 없습니다. "
        "프로젝트 루트에 파일을 두거나 GOOGLE_APPLICATION_CREDENTIALS 환경변수를 설정하세요."
    )


def initialize_firebase() -> firebase_admin.App:
    """Firebase Admin SDK를 초기화하고 앱 인스턴스를 반환한다.

    이미 초기화된 경우 기존 앱을 재사용한다(멱등 보장).
    """
    global _app

    if _app is not None:
        return _app

    # firebase_admin 내부 registry에 이미 등록된 경우도 재사용
    try:
        _app = firebase_admin.get_app()
        logger.debug("기존 Firebase 앱 인스턴스를 재사용합니다.")
        return _app
    except ValueError:
        pass

    cred_path = _resolve_credential_path()
    cred = credentials.Certificate(cred_path)
    _app = firebase_admin.initialize_app(cred)
    logger.info("Firebase Admin SDK 초기화 완료: %s", cred_path)
    return _app


def get_firestore_client() -> firestore.Client:
    """초기화된 Firestore 클라이언트를 반환한다."""
    initialize_firebase()
    return firestore.client()
