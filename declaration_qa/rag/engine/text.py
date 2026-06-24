"""Text helpers for RAG engines."""

from __future__ import annotations

import re
from typing import Any


def normalize_text(value: Any) -> str:
    """Normalize text to compact whitespace."""
    return re.sub(r"\s+", " ", str(value or "").strip())


def tokenize(text: str) -> list[str]:
    """Tokenize text for local retrieval scoring."""
    normalized = normalize_text(text).lower()
    try:
        import jieba

        tokens = [token.strip() for token in jieba.lcut(normalized) if token.strip()]
    except Exception:
        tokens = re.findall(r"[\w\u4e00-\u9fff]+", normalized)
    expanded = []
    for token in tokens:
        expanded.append(token)
        if len(token) >= 4 and re.search(r"[\u4e00-\u9fff]", token):
            expanded.extend(token[index : index + 2] for index in range(len(token) - 1))
    return [token for token in expanded if token]


def unique_ordered_strings(values: list[str]) -> list[str]:
    """Return non-empty strings in first-seen order."""
    seen = set()
    output = []
    for value in values:
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def safe_int(value: Any) -> int:
    """Convert a value to int, defaulting to 0."""
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
