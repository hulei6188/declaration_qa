"""聊天模型流式内容抽取。"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def stream_chat_model_content(chunks: Iterable[Any]) -> Iterable[str]:
    """从聊天模型流式结果中抽取文本片段。"""
    for chunk in chunks:
        content = getattr(chunk, "content", chunk)
        if isinstance(content, list):
            for item in content:
                text = chat_content_part_text(item)
                if text:
                    yield text
            continue
        text = str(content or "")
        if text:
            yield text


def chat_content_part_text(value: Any) -> str:
    """将聊天内容片段转换为纯文本。"""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return str(value.get("text") or value.get("content") or "")
    return str(value or "")

