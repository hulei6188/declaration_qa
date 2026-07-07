"""向量化辅助函数。"""

from __future__ import annotations

import hashlib
import math
from typing import Iterable


def hashing_embedding(text: str, *, dimension: int) -> list[float]:
    """使用哈希技巧生成归一化文本向量。"""
    from declaration_qa.rag.engine.text import tokenize

    values = [0.0 for _ in range(dimension)]
    for token in tokenize(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:2], "big") % dimension
        sign = 1.0 if digest[2] % 2 == 0 else -1.0
        values[index] += sign
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]


def cosine(left: Iterable[float], right: Iterable[float]) -> float:
    """计算两个向量的余弦点积分数。"""
    left_values = list(left)
    right_values = list(right)
    return sum(a * b for a, b in zip(left_values, right_values))
