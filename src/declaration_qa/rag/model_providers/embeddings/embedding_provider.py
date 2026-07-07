"""向量化 provider 协议。"""

from __future__ import annotations

from typing import Protocol


class EmbeddingProvider(Protocol):
    """文本向量化提供方协议。"""

    name: str
    dimension: int

    def embed_query(self, text: str) -> list[float]:
        """为单条查询文本生成向量表示。"""
        ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """为多条文档文本生成向量表示。"""
        ...
