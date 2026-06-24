"""Hash embedding provider."""

from __future__ import annotations

from declaration_qa.rag.model_providers.embeddings.utils import hashing_embedding


class HashEmbeddingProvider:
    """基于哈希特征的本地向量化实现。"""

    name = "hashing"

    def __init__(self, *, dimension: int = 96) -> None:
        """初始化HashEmbeddingProvider。"""
        self.dimension = dimension

    def embed_query(self, text: str) -> list[float]:
        """为单条查询文本生成向量表示。"""
        return hashing_embedding(text, dimension=self.dimension)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """为多条文档文本生成向量表示。"""
        return [self.embed_query(text) for text in texts]
