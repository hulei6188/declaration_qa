"""关键词检索索引协议。"""

from __future__ import annotations

from typing import Any, Protocol

from gov_service_agent.core.domain import KnowledgeChunk


class KeywordIndex(Protocol):
    """关键词检索索引协议。"""

    backend: str

    def upsert_chunks(self, chunks: list[KnowledgeChunk]) -> None:
        """将可检索的 chunk 写入关键词索引。"""
        ...

    def delete_by_doc(self, doc_id: str) -> None:
        """删除指定文档下已经索引的全部 chunk。"""
        ...

    def search(self, query: str, filters: dict[str, Any], *, limit: int) -> list[tuple[str, float]]:
        """按关键词相关性检索，返回 chunk id 及归一化分数。"""
        ...

    def search_many(
        self,
        query_specs: list[dict[str, Any]],
        filters: dict[str, Any],
        limits: list[int],
    ) -> dict[int, list[tuple[str, float]]]:
        """批量关键词检索，返回每个 query index 对应的 chunk id 和分数。"""
        ...

    def clear(self) -> None:
        """清空全部关键词索引数据。"""
        ...
