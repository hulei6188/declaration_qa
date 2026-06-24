"""Vector helpers for RAG engines."""

from __future__ import annotations

import hashlib
import math
from typing import Iterable

from declaration_qa.rag.engine.constants import EMBEDDING_DIMS
from declaration_qa.rag.engine.text import tokenize


def cosine(left: Iterable[float], right: Iterable[float]) -> float:
    """Return cosine dot product for normalized vectors."""
    left_values = list(left)
    right_values = list(right)
    return sum(a * b for a, b in zip(left_values, right_values))


def embed_text(text: str) -> list[float]:
    """Generate a local hashing embedding."""
    values = [0.0 for _ in range(EMBEDDING_DIMS)]
    for token in tokenize(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:2], "big") % EMBEDDING_DIMS
        sign = 1.0 if digest[2] % 2 == 0 else -1.0
        values[index] += sign
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]
