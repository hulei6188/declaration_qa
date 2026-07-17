"""RAG 引擎的向量化辅助函数。"""

from __future__ import annotations

import random
import time
from typing import Any

from gov_service_agent.core.domain import KnowledgeChunk


def embed_chunk_batch(
    embedding_provider: Any,
    texts: list[str],
    *,
    batch_size: int | None = None,
    max_retries: int = 3,
    base_delay_seconds: float = 1.0,
) -> list[list[float]]:
    """按批创建向量，并使用有限重试和随机抖动。"""
    vectors: list[list[float]] = []
    effective_batch_size = embedding_batch_size(embedding_provider, batch_size)
    for index in range(0, len(texts), effective_batch_size):
        vectors.extend(
            embed_batch_with_retry(
                embedding_provider,
                texts[index : index + effective_batch_size],
                max_retries=max_retries,
                base_delay_seconds=base_delay_seconds,
            )
        )
    return vectors


def embed_knowledge_chunk_batch(
    embedding_provider: Any,
    chunks: list[KnowledgeChunk],
    *,
    batch_size: int | None = None,
    max_retries: int = 3,
    base_delay_seconds: float = 1.0,
) -> list[list[float]]:
    """按 chunk 元数据创建向量，支持多模态 provider。"""
    if hasattr(embedding_provider, "embed_chunks"):
        return embed_chunk_objects_with_retry(
            embedding_provider,
            chunks,
            max_retries=max_retries,
            base_delay_seconds=base_delay_seconds,
        )
    return embed_chunk_batch(
        embedding_provider,
        [chunk.content for chunk in chunks],
        batch_size=batch_size,
        max_retries=max_retries,
        base_delay_seconds=base_delay_seconds,
    )


def embedding_batch_size(embedding_provider: Any, requested_batch_size: int | None = None) -> int:
    """返回当前向量化 provider 的安全批大小。"""
    if requested_batch_size is not None:
        return max(1, int(requested_batch_size))
    provider_batch_size = getattr(embedding_provider, "document_batch_size", None)
    if provider_batch_size is not None:
        return max(1, int(provider_batch_size))
    return 10


def embed_batch_with_retry(
    embedding_provider: Any,
    texts: list[str],
    *,
    max_retries: int,
    base_delay_seconds: float,
) -> list[list[float]]:
    """使用指数退避调用向量化 provider。"""
    attempt = 0
    while True:
        try:
            return embedding_provider.embed_documents(texts)
        except Exception:
            attempt += 1
            if attempt > max_retries:
                raise
            delay = max(0.1, base_delay_seconds) * (2 ** (attempt - 1))
            time.sleep(delay + random.uniform(0, 0.25))


def embed_chunk_objects_with_retry(
    embedding_provider: Any,
    chunks: list[KnowledgeChunk],
    *,
    max_retries: int,
    base_delay_seconds: float,
) -> list[list[float]]:
    """使用指数退避调用支持 chunk 感知的向量化 provider。"""
    attempt = 0
    while True:
        try:
            return embedding_provider.embed_chunks(chunks)
        except Exception:
            attempt += 1
            if attempt > max_retries:
                raise
            delay = max(0.1, base_delay_seconds) * (2 ** (attempt - 1))
            time.sleep(delay + random.uniform(0, 0.25))
