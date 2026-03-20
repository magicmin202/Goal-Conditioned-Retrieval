"""Text utility functions."""
from __future__ import annotations
import re


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\w가-힣]+", text.lower())


def truncate(text: str, max_chars: int = 500) -> str:
    return text[:max_chars] + "..." if len(text) > max_chars else text


def clean_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
