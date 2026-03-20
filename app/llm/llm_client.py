"""LLM client interface.

MockLLMClient is the default. Replace with GeminiLLMClient when ready.
"""
from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str: ...


class MockLLMClient(BaseLLMClient):
    """Returns deterministic placeholder responses.

    TODO: Replace with GeminiLLMClient when GEMINI_API_KEY is available.
    """

    def generate(self, prompt: str, **kwargs: Any) -> str:
        logger.debug("MockLLMClient.generate called (prompt_len=%d)", len(prompt))
        return (
            "[MOCK LLM RESPONSE]\n"
            "목표 진행 분석: 사용자는 목표 관련 활동을 꾸준히 수행하고 있습니다.\n"
            "추천 액션: 현재 페이스를 유지하며 다음 단계로 진행하세요.\n"
        )


class GeminiLLMClient(BaseLLMClient):
    """Gemini API client.

    TODO: Implement when GEMINI_API_KEY is available.

        import google.generativeai as genai
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)
        return response.text
    """

    def __init__(self, api_key: str | None = None) -> None:
        raise NotImplementedError("GeminiLLMClient is not yet implemented.")

    def generate(self, prompt: str, **kwargs: Any) -> str:
        raise NotImplementedError


def get_llm_client(mock: bool = True) -> BaseLLMClient:
    if mock:
        return MockLLMClient()
    raise NotImplementedError("Only mock mode is supported. Implement GeminiLLMClient first.")
