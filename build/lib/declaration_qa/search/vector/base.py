"""向量索引协议。"""

from __future__ import annotations

from typing import Any, Protocol

from declaration_qa.core.domain import KnowledgeChunk


class VectorIndex(Protocol):
    """向量索引读写协议。"""
    backend: str

    def upsert(self, chunk: KnowledgeChunk, vector: list[float]) -> None:
        """写入或更新向量索引记录。"""
        ...

    def delete_by_doc(self, doc_id: str) -> None:
        """按文档 ID 删除向量索引记录。"""
        ...

    def search(self, vector: list[float], filters: dict[str, Any], *, limit: int) -> list[tuple[str, float]]:
        """按向量和过滤条件搜索相似片段。"""
        ...

    def search_batch(self, vectors: list[list[float]], filters: dict[str, Any], *, limit: int) -> list[list[tuple[str, float]]]:
        """按多条向量和过滤条件批量搜索相似片段。"""
        ...

    def clear(self) -> None:
        """清空当前存储中的全部数据。"""
        ...

    def flush(self) -> None:
        """后端支持时刷写待提交写入。"""
        ...

