"""进程内向量索引。"""

from __future__ import annotations

from typing import Any

from declaration_qa.core.domain import KnowledgeChunk


class MemoryVectorIndex:
    """进程内向量索引实现。"""
    backend = "memory"

    def __init__(self) -> None:
        """初始化MemoryVectorIndex。"""
        self._items: dict[str, tuple[list[float], KnowledgeChunk]] = {}

    def upsert(self, chunk: KnowledgeChunk, vector: list[float]) -> None:
        """写入或更新向量索引记录。"""
        self._items[chunk.id] = (list(vector), chunk)

    def delete_by_doc(self, doc_id: str) -> None:
        """按文档 ID 删除向量索引记录。"""
        for chunk_id, (_, chunk) in list(self._items.items()):
            if chunk.doc_id == doc_id:
                self._items.pop(chunk_id, None)

    def search(self, vector: list[float], filters: dict[str, Any], *, limit: int) -> list[tuple[str, float]]:
        """按向量和过滤条件搜索相似片段。"""
        from declaration_qa.rag.engine.scoring import matches_filter
        from declaration_qa.rag.engine.vectors import cosine

        hits = []
        for chunk_id, (item_vector, chunk) in self._items.items():
            if chunk.chunk_type != "child" or not matches_filter(chunk, filters):
                continue
            score = cosine(vector, item_vector)
            if score > 0:
                hits.append((chunk_id, float(score)))
        return sorted(hits, key=lambda item: item[1], reverse=True)[:limit]

    def search_batch(self, vectors: list[list[float]], filters: dict[str, Any], *, limit: int) -> list[list[tuple[str, float]]]:
        """按多条向量批量搜索相似片段。"""
        return [self.search(vector, filters, limit=limit) for vector in vectors]

    def clear(self) -> None:
        """清空当前存储中的全部数据。"""
        self._items.clear()

    def flush(self) -> None:
        """内存索引写入立即可见。"""
        return None

