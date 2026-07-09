"""RAG 引擎文本辅助函数。"""

from __future__ import annotations

import re
from typing import Any


def normalize_text(value: Any) -> str:
    """将文本空白归一化为紧凑格式。"""
    return re.sub(r"\s+", " ", str(value or "").strip())


def tokenize(text: str) -> list[str]:
    """为本地检索评分分词。"""
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
    """按首次出现顺序返回非空字符串。"""
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
    """将值转换为整数，默认返回 0。"""
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
