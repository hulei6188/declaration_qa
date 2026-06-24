"""Milvus 向量索引实现。"""

from __future__ import annotations

from typing import Any

from declaration_qa.core.domain import KnowledgeChunk
from declaration_qa.search.vector.memory import MemoryVectorIndex
from declaration_qa.search.vector.metadata import (
    MILVUS_JSON_METADATA_FIELD,
    MILVUS_TEXT_METADATA_FIELDS,
    milvus_filter_expr,
    milvus_metadata_row,
)


class MilvusVectorIndex:
    """基于 Milvus 的向量索引实现。"""
    backend = "milvus"

    def __init__(
        self,
        *,
        uri: str,
        collection_name: str,
        token: str = "",
        dimension: int = 96,
        allow_fallback: bool = True,
    ) -> None:
        """初始化MilvusVectorIndex。"""
        from pymilvus import MilvusClient

        kwargs = {"uri": uri}
        if token:
            kwargs["token"] = token
        self._client = MilvusClient(**kwargs)
        self.collection_name = collection_name
        self.dimension = dimension
        self.allow_fallback = bool(allow_fallback)
        self._fallback = MemoryVectorIndex() if self.allow_fallback else None
        self._available = True
        try:
            self._ensure_collection()
        except Exception:
            if not self.allow_fallback:
                raise
            self._available = False

    def upsert(self, chunk: KnowledgeChunk, vector: list[float]) -> None:
        """写入或更新向量索引记录。"""
        if self._fallback is not None:
            self._fallback.upsert(chunk, vector)
        if not self._available:
            return
        row = {
            "id": chunk.id,
            "doc_id": chunk.doc_id,
            "vector": list(vector),
            **milvus_metadata_row(chunk),
        }
        try:
            self._client.upsert(collection_name=self.collection_name, data=[row])
        except Exception:
            if self._fallback is None:
                raise
            self._available = False

    def delete_by_doc(self, doc_id: str) -> None:
        """按文档 ID 删除向量索引记录。"""
        if self._fallback is not None:
            self._fallback.delete_by_doc(doc_id)
        if not self._available:
            return
        safe = str(doc_id).replace('"', '\\"')
        try:
            self._client.delete(collection_name=self.collection_name, filter=f'doc_id == "{safe}"')
        except Exception:
            if self._fallback is None:
                raise
            self._available = False

    def search(self, vector: list[float], filters: dict[str, Any], *, limit: int) -> list[tuple[str, float]]:
        """按向量和过滤条件搜索相似片段。"""
        if not self._available:
            if self._fallback is None:
                raise RuntimeError("Milvus vector index is unavailable and in-memory fallback is disabled")
            return self._fallback.search(vector, filters, limit=limit)
        expr = milvus_filter_expr(filters)
        try:
            results = self._client.search(
                collection_name=self.collection_name,
                data=[list(vector)],
                limit=limit,
                filter=expr or "",
                output_fields=["id"],
            )
        except Exception:
            if self._fallback is None:
                raise
            self._available = False
            return self._fallback.search(vector, filters, limit=limit)
        hits = []
        for hit in results[0] if results else []:
            entity = hit.get("entity") or {}
            chunk_id = str(entity.get("id") or hit.get("id") or "")
            score = float(hit.get("distance") or hit.get("score") or 0.0)
            if chunk_id:
                hits.append((chunk_id, score))
        return hits

    def search_batch(self, vectors: list[list[float]], filters: dict[str, Any], *, limit: int) -> list[list[tuple[str, float]]]:
        """按多条向量和过滤条件批量搜索相似片段。"""
        if not vectors:
            return []
        if not self._available:
            if self._fallback is None:
                raise RuntimeError("Milvus vector index is unavailable and in-memory fallback is disabled")
            return self._fallback.search_batch(vectors, filters, limit=limit)
        expr = milvus_filter_expr(filters)
        try:
            results = self._client.search(
                collection_name=self.collection_name,
                data=[list(vector) for vector in vectors],
                limit=limit,
                filter=expr or "",
                output_fields=["id"],
            )
        except Exception:
            if self._fallback is None:
                raise
            self._available = False
            return self._fallback.search_batch(vectors, filters, limit=limit)
        output: list[list[tuple[str, float]]] = []
        for result in results or []:
            hits = []
            for hit in result:
                entity = hit.get("entity") or {}
                chunk_id = str(entity.get("id") or hit.get("id") or "")
                score = float(hit.get("distance") or hit.get("score") or 0.0)
                if chunk_id:
                    hits.append((chunk_id, score))
            output.append(hits)
        while len(output) < len(vectors):
            output.append([])
        return output

    def clear(self) -> None:
        """清空当前存储中的全部数据。"""
        if self._fallback is not None:
            self._fallback.clear()
        if not self._available:
            return
        try:
            if self._client.has_collection(self.collection_name):
                self._client.drop_collection(self.collection_name)
            self._ensure_collection()
        except Exception:
            if self._fallback is None:
                raise
            self._available = False

    def flush(self) -> None:
        """Flush Milvus writes so immediately-following searches can see them."""
        if self._fallback is not None:
            self._fallback.flush()
        if not self._available:
            return
        try:
            flush = getattr(self._client, "flush", None)
            if callable(flush):
                flush(collection_name=self.collection_name)
        except TypeError:
            self._client.flush(self.collection_name)
        except Exception:
            if self._fallback is None:
                raise
            self._available = False

    def _ensure_collection(self) -> None:
        """确保 Milvus 集合和索引已创建。"""
        from pymilvus import DataType

        if self._client.has_collection(self.collection_name):
            if not self._collection_schema_is_current():
                self._client.drop_collection(self.collection_name)
            else:
                return
        schema = self._client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=180)
        schema.add_field("doc_id", DataType.VARCHAR, max_length=120)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self.dimension)
        for field_name, max_length in MILVUS_TEXT_METADATA_FIELDS.items():
            schema.add_field(field_name, DataType.VARCHAR, max_length=max_length, nullable=True)
        schema.add_field(MILVUS_JSON_METADATA_FIELD, DataType.JSON, nullable=True)
        index_params = self._client.prepare_index_params()
        index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
        self._client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )

    def _collection_schema_is_current(self) -> bool:
        """判断 Milvus 集合结构是否符合当前版本。"""
        description = self._client.describe_collection(self.collection_name)
        fields = {str(field.get("name") or ""): field for field in description.get("fields", [])}
        expected_fields = {"id", "doc_id", "vector", MILVUS_JSON_METADATA_FIELD, *MILVUS_TEXT_METADATA_FIELDS}
        if set(fields) != expected_fields:
            return False
        for field in description.get("fields", []):
            if field.get("name") == "id" and field.get("is_primary"):
                field_type = field.get("type")
                field_type_name = str(getattr(field_type, "name", "") or field_type or "").upper()
                return field_type_name in {"VARCHAR", "STRING"} or field_type_name.endswith(".VARCHAR")
        return False

