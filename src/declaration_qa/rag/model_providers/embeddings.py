"""文本向量化提供方。"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Iterable, Protocol


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


class OpenAIEmbeddingProvider:
    """基于 OpenAI Embeddings API 的向量化实现。"""
    name = "openai"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "",
        dimension: int | None = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        """初始化OpenAIEmbeddingProvider。"""
        if not api_key:
            raise ValueError("OpenAI embedding provider requires an API key")
        from langchain_openai import OpenAIEmbeddings

        kwargs: dict[str, Any] = {
            "api_key": api_key,
            "model": model,
            "timeout": timeout_seconds,
            "max_retries": 1,
            "retry_min_seconds": 1,
            "retry_max_seconds": 2,
        }
        if base_url:
            kwargs["base_url"] = base_url
        if dimension:
            kwargs["dimensions"] = dimension
        if "dashscope.aliyuncs.com" in base_url:
            kwargs["check_embedding_ctx_length"] = False
        self._embeddings = OpenAIEmbeddings(**kwargs)
        self.dimension = int(dimension or 1536)

    def embed_query(self, text: str) -> list[float]:
        """为单条查询文本生成向量表示。"""
        return [float(value) for value in self._embeddings.embed_query(text)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """为多条文档文本生成向量表示。"""
        return [[float(value) for value in item] for item in self._embeddings.embed_documents(texts)]


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

