"""向量化 provider 协议。"""

from __future__ import annotations

from typing import Protocol

from declaration_qa.core.domain import KnowledgeChunk


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


class ChunkEmbeddingProvider(EmbeddingProvider, Protocol):
    """支持按 chunk 元数据选择文本/图片向量化的 provider 协议。"""

    def embed_chunks(self, chunks: list[KnowledgeChunk]) -> list[list[float]]:
        """为知识库 chunk 生成向量表示。"""
        ...
