from __future__ import annotations

from datetime import date, datetime
from typing import Any, Protocol

from declaration_qa.domain import KnowledgeChunk


MILVUS_TEXT_METADATA_FIELDS: dict[str, int] = {
    "region_code": 32,
    "region_key": 32,
    "region_name": 120,
    "policy_version": 80,
    "source_version": 120,
    "official_source_url": 1024,
    "catalog_name": 240,
    "catalog_subitem_name": 240,
    "item_code": 120,
    "item_name": 240,
    "item_type": 80,
    "theme_code": 80,
    "service_object": 120,
    "basic_code": 120,
    "agency_name": 240,
    "ssqd_code": 120,
    "ssqd_id": 120,
    "chunking_strategy": 40,
    "effective_start_time": 32,
    "effective_end_time": 32,
}
MILVUS_JSON_METADATA_FIELD = "metadata"


class VectorIndex(Protocol):
    backend: str

    def upsert(self, chunk: KnowledgeChunk, vector: list[float]) -> None:
        ...

    def delete_by_doc(self, doc_id: str) -> None:
        ...

    def search(self, vector: list[float], filters: dict[str, Any], *, limit: int) -> list[tuple[str, float]]:
        ...

    def clear(self) -> None:
        ...


class MemoryVectorIndex:
    backend = "memory"

    def __init__(self) -> None:
        self._items: dict[str, tuple[list[float], KnowledgeChunk]] = {}

    def upsert(self, chunk: KnowledgeChunk, vector: list[float]) -> None:
        self._items[chunk.id] = (list(vector), chunk)

    def delete_by_doc(self, doc_id: str) -> None:
        for chunk_id, (_, chunk) in list(self._items.items()):
            if chunk.doc_id == doc_id:
                self._items.pop(chunk_id, None)

    def search(self, vector: list[float], filters: dict[str, Any], *, limit: int) -> list[tuple[str, float]]:
        from declaration_qa.rag_engine import cosine

        hits = []
        for chunk_id, (item_vector, chunk) in self._items.items():
            if chunk.chunk_type != "child":
                continue
            score = cosine(vector, item_vector)
            if score > 0:
                hits.append((chunk_id, float(score)))
        return sorted(hits, key=lambda item: item[1], reverse=True)[:limit]

    def clear(self) -> None:
        self._items.clear()


class MilvusVectorIndex:
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
        from pymilvus import MilvusClient

        kwargs = {"uri": uri}
        if token:
            kwargs["token"] = token
        self._client = MilvusClient(**kwargs)
        self.collection_name = collection_name
        self.dimension = dimension
        self._fallback = MemoryVectorIndex()
        self._available = True
        try:
            self._ensure_collection()
        except Exception:
            if not allow_fallback:
                raise
            self._available = False

    def upsert(self, chunk: KnowledgeChunk, vector: list[float]) -> None:
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
            if not hasattr(self, "_fallback"):
                raise
            self._available = False

    def delete_by_doc(self, doc_id: str) -> None:
        self._fallback.delete_by_doc(doc_id)
        if not self._available:
            return
        safe = str(doc_id).replace('"', '\\"')
        try:
            self._client.delete(collection_name=self.collection_name, filter=f'doc_id == "{safe}"')
        except Exception:
            if not hasattr(self, "_fallback"):
                raise
            self._available = False

    def search(self, vector: list[float], filters: dict[str, Any], *, limit: int) -> list[tuple[str, float]]:
        if not self._available:
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
            if not hasattr(self, "_fallback"):
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

    def clear(self) -> None:
        self._fallback.clear()
        if not self._available:
            return
        try:
            if self._client.has_collection(self.collection_name):
                self._client.drop_collection(self.collection_name)
            self._ensure_collection()
        except Exception:
            if not hasattr(self, "_fallback"):
                raise
            self._available = False

    def _ensure_collection(self) -> None:
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


def milvus_metadata_row(chunk: KnowledgeChunk) -> dict[str, Any]:
    metadata = dict(chunk.metadata or {})
    values = {
        "region_code": chunk.region_code or metadata.get("region_code") or "",
        "region_key": metadata.get("region_key") or chunk.region_code or "",
        "region_name": metadata.get("region_name") or "",
        "policy_version": chunk.policy_version or metadata.get("policy_version") or "",
        "source_version": metadata.get("source_version") or chunk.policy_version or "",
        "official_source_url": metadata.get("official_source_url") or chunk.source_url or "",
        "catalog_name": metadata.get("catalog_name") or "",
        "catalog_subitem_name": metadata.get("catalog_subitem_name") or "",
        "item_code": metadata.get("item_code") or "",
        "item_name": metadata.get("item_name") or "",
        "item_type": metadata.get("item_type") or "",
        "theme_code": metadata.get("theme_code") or "",
        "service_object": metadata.get("service_object") or "",
        "basic_code": metadata.get("basic_code") or "",
        "agency_name": metadata.get("agency_name") or "",
        "ssqd_code": metadata.get("ssqd_code") or "",
        "ssqd_id": metadata.get("ssqd_id") or "",
        "chunking_strategy": metadata.get("chunking_strategy") or "",
        "effective_start_time": metadata.get("effective_start_time") or "",
        "effective_end_time": metadata.get("effective_end_time") or "",
    }
    return {
        **{
            field_name: truncate_milvus_text(values.get(field_name), max_length)
            for field_name, max_length in MILVUS_TEXT_METADATA_FIELDS.items()
        },
        MILVUS_JSON_METADATA_FIELD: milvus_json_value(metadata),
    }


def truncate_milvus_text(value: Any, max_length: int) -> str:
    text = "" if value is None else str(value)
    return text[:max_length]


def milvus_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): milvus_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [milvus_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [milvus_json_value(item) for item in value]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def milvus_filter_expr(filters: dict[str, Any]) -> str:
    return ""
