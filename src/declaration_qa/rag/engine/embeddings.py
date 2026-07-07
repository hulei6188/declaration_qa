"""RAG 引擎的向量化辅助函数。"""

from __future__ import annotations

import random
import time
from typing import Any


def embed_chunk_batch(
    embedding_provider: Any,
    texts: list[str],
    *,
    batch_size: int = 32,
    max_retries: int = 3,
    base_delay_seconds: float = 1.0,
) -> list[list[float]]:
    """按批创建向量，并使用有限重试和随机抖动。"""
    vectors: list[list[float]] = []
    for index in range(0, len(texts), batch_size):
        vectors.extend(
            embed_batch_with_retry(
                embedding_provider,
                texts[index : index + batch_size],
                max_retries=max_retries,
                base_delay_seconds=base_delay_seconds,
            )
        )
    return vectors


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
