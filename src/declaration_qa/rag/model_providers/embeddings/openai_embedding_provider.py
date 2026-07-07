"""兼容 OpenAI 的向量化 provider。"""

from __future__ import annotations

from typing import Any


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
