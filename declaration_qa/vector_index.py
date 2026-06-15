from __future__ import annotations

from typing import Any, Protocol

from declaration_qa.domain import KnowledgeChunk


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
        from declaration_qa.rag_engine import cosine, matches_filter

        hits = []
        for chunk_id, (item_vector, chunk) in self._items.items():
            if chunk.chunk_type != "child" or not matches_filter(chunk, filters):
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
            "biz_type": chunk.biz_type,
            "region_code": chunk.region_code,
            "policy_version": chunk.policy_version,
            "source_type": chunk.source_type,
            "material_code": chunk.material_code,
            "reject_code": chunk.reject_code,
            "audit_node": chunk.audit_node,
            "status": chunk.status,
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
        if not self._client.has_collection(self.collection_name):
            self._client.create_collection(
                collection_name=self.collection_name,
                dimension=self.dimension,
                metric_type="COSINE",
                auto_id=False,
            )


def milvus_filter_expr(filters: dict[str, Any]) -> str:
    parts = []
    for key in ["biz_type", "region_code", "policy_version", "source_type", "material_code", "reject_code", "audit_node", "status"]:
        value = filters.get(key)
        if not value:
            continue
        if isinstance(value, list):
            values = ", ".join(f'"{str(item).replace(chr(34), chr(92) + chr(34))}"' for item in value)
            parts.append(f"{key} in [{values}]")
        else:
            safe = str(value).replace('"', '\\"')
            parts.append(f'{key} == "{safe}"')
    return " and ".join(parts)
