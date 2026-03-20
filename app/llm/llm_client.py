"""LLM client interface.

GeminiLLMClient is the default path when GEMINI_API_KEY is set.
MockLLMClient is used when the key is absent or use_mock_fallback=True.

Uses google-genai SDK (REST-based, no gRPC dependency).
"""
from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)

# Auto-load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class BaseLLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str: ...


class MockLLMClient(BaseLLMClient):
    """Returns deterministic placeholder responses."""

    def generate(self, prompt: str, **kwargs: Any) -> str:
        logger.debug("MockLLMClient.generate (prompt_len=%d)", len(prompt))
        return (
            "[MOCK LLM RESPONSE]\n"
            "목표 진행 분석: 사용자는 목표 관련 활동을 꾸준히 수행하고 있습니다.\n"
            "추천 액션: 현재 페이스를 유지하며 다음 단계로 진행하세요.\n"
        )


class GeminiLLMClient(BaseLLMClient):
    """Gemini API client via google-genai SDK (REST, no gRPC).

    Set GEMINI_API_KEY environment variable before use.
    Install: pip install google-genai
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "gemini-2.0-flash",
        temperature: float = 0.2,
        max_output_tokens: int = 512,
    ) -> None:
        try:
            from google import genai
            from google.genai import types
        except ImportError as e:
            raise ImportError(
                "google-genai is not installed. Run: pip install google-genai"
            ) from e

        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise ValueError(
                "Gemini API key not found. Set GEMINI_API_KEY environment variable."
            )

        self._client = genai.Client(api_key=key)
        self._model_name = model_name
        self._config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        logger.info("GeminiLLMClient initialized (model=%s)", model_name)

    def generate(self, prompt: str, **kwargs: Any) -> str:
        from google.genai import types
        resp = self._client.models.generate_content(
            model=self._model_name,
            contents=prompt,
            config=self._config,
        )
        return resp.text


def get_llm_client(mock: bool = True, config=None) -> BaseLLMClient:
    """Factory: returns GeminiLLMClient when mock=False and API key is available."""
    if mock:
        return MockLLMClient()

    from app.config import DEFAULT_CONFIG
    cfg = config or DEFAULT_CONFIG.gemini

    api_key = cfg.api_key
    if not api_key:
        logger.warning("GEMINI_API_KEY not set. Falling back to MockLLMClient.")
        return MockLLMClient()

    try:
        return GeminiLLMClient(
            api_key=api_key,
            model_name=cfg.model_name,
            temperature=cfg.temperature,
            max_output_tokens=cfg.max_output_tokens,
        )
    except Exception as exc:
        logger.warning("GeminiLLMClient init failed (%s). Falling back to mock.", exc)
        return MockLLMClient()
