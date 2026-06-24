"""Embedding helpers for RAG engines."""

from __future__ import annotations

from typing import Any


def embed_chunk_batch(embedding_provider: Any, texts: list[str], *, batch_size: int = 10) -> list[list[float]]:
    """Create embeddings in batches."""
    vectors: list[list[float]] = []
    for index in range(0, len(texts), batch_size):
        vectors.extend(embedding_provider.embed_documents(texts[index : index + batch_size]))
    return vectors
