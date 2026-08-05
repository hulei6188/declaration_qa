from threading import Lock, get_ident
from time import sleep

import pytest

from gov_service_agent.core.domain import KnowledgeChunk, KnowledgeDocument, RetrievedSource
from gov_service_agent.rag.engine.external_index import ExternalIndexRAGEngine


def test_rebuild_does_not_mutate_external_indexes_when_embedding_fails():
    document = KnowledgeDocument(id="doc-safe", title="安全重建", content="")
    child = KnowledgeChunk(
        id="chunk-safe",
        doc_id=document.id,
        parent_id="",
        chunk_type="child",
        title=document.title,
        content="向量生成失败时必须保留旧索引。",
    )

    class FailingEmbeddingProvider:
        name = "failing"
        dimension = 2

        def embed_chunks(self, chunks):
            raise RuntimeError("embedding unavailable")

    class RecordingVectorIndex:
        backend = "milvus"

        def __init__(self):
            self.mutations = []

        def clear(self):
            self.mutations.append(("clear", ""))

        def delete_by_doc(self, doc_id):
            self.mutations.append(("delete", doc_id))

        def upsert(self, chunk, vector):
            self.mutations.append(("upsert", chunk.id))

    class RecordingKeywordIndex:
        backend = "elasticsearch"

        def __init__(self):
            self.mutations = []

        def clear(self):
            self.mutations.append(("clear", ""))

        def delete_by_doc(self, doc_id):
            self.mutations.append(("delete", doc_id))

        def upsert_chunks(self, chunks):
            self.mutations.append(("upsert", [chunk.id for chunk in chunks]))

    vector_index = RecordingVectorIndex()
    keyword_index = RecordingKeywordIndex()
    engine = ExternalIndexRAGEngine(
        vector_index=vector_index,
        keyword_index=keyword_index,
        chunk_store=object(),
        embedding_provider=FailingEmbeddingProvider(),
    )

    with pytest.raises(RuntimeError, match="embedding unavailable"):
        engine.rebuild_indexed_data([document], [child])

    assert vector_index.mutations == []
    assert keyword_index.mutations == []


def test_rebuild_replaces_documents_without_globally_clearing_indexes():
    old_document = KnowledgeDocument(id="doc-old", title="旧文档", content="")
    new_document = KnowledgeDocument(id="doc-new", title="新文档", content="")
    child = KnowledgeChunk(
        id="chunk-new",
        doc_id=new_document.id,
        parent_id="",
        chunk_type="child",
        title=new_document.title,
        content="新的可检索内容。",
    )

    class EmbeddingProvider:
        name = "recording"
        dimension = 2

        def embed_chunks(self, chunks):
            return [[1.0, 0.0] for _ in chunks]

    class RecordingVectorIndex:
        backend = "milvus"

        def __init__(self):
            self.mutations = []

        def clear(self):
            raise AssertionError("rebuild must not globally clear the vector index")

        def delete_by_doc(self, doc_id):
            self.mutations.append(("delete", doc_id))

        def upsert(self, chunk, vector):
            self.mutations.append(("upsert", chunk.id, vector))

        def flush(self):
            self.mutations.append(("flush",))

    class RecordingKeywordIndex:
        backend = "elasticsearch"

        def __init__(self):
            self.mutations = []

        def clear(self):
            raise AssertionError("rebuild must not globally clear the keyword index")

        def delete_by_doc(self, doc_id):
            self.mutations.append(("delete", doc_id))

        def upsert_chunks(self, chunks):
            self.mutations.append(("upsert", [chunk.id for chunk in chunks]))

    vector_index = RecordingVectorIndex()
    keyword_index = RecordingKeywordIndex()
    engine = ExternalIndexRAGEngine(
        vector_index=vector_index,
        keyword_index=keyword_index,
        chunk_store=object(),
        embedding_provider=EmbeddingProvider(),
    )
    engine.load_indexed_data([old_document], [])

    indexed = engine.rebuild_indexed_data([new_document], [child])

    assert indexed == 1
    assert vector_index.mutations == [
        ("delete", new_document.id),
        ("upsert", child.id, [1.0, 0.0]),
        ("delete", old_document.id),
        ("flush",),
    ]
    assert keyword_index.mutations == [
        ("delete", new_document.id),
        ("upsert", [child.id]),
        ("delete", old_document.id),
    ]


def test_rebuild_refuses_an_accidental_empty_snapshot():
    existing_document = KnowledgeDocument(id="doc-existing", title="现有文档", content="")

    class UnusedEmbeddingProvider:
        name = "unused"
        dimension = 2

        def embed_chunks(self, chunks):
            raise AssertionError("empty snapshot must be rejected before embedding")

    class MutationRejectingIndex:
        backend = "external"

        def clear(self):
            raise AssertionError("empty snapshot must not clear indexes")

        def delete_by_doc(self, doc_id):
            raise AssertionError("empty snapshot must not delete documents")

        def upsert(self, chunk, vector):
            raise AssertionError("empty snapshot must not write documents")

        def upsert_chunks(self, chunks):
            raise AssertionError("empty snapshot must not write documents")

    engine = ExternalIndexRAGEngine(
        vector_index=MutationRejectingIndex(),
        keyword_index=MutationRejectingIndex(),
        chunk_store=object(),
        embedding_provider=UnusedEmbeddingProvider(),
    )
    engine.load_indexed_data([existing_document], [])

    with pytest.raises(RuntimeError, match="empty knowledge snapshot"):
        engine.rebuild_indexed_data([], [])


def test_remote_retrieval_hydrates_chunk_ids_without_memory_candidates():
    child = KnowledgeChunk(
        id="chunk-remote-child",
        doc_id="doc-remote",
        parent_id="chunk-remote-parent",
        chunk_type="child",
        title="远程召回测试",
        content="申请表需要申请人自备并在线上传。",
        metadata={
            "chunk_type": "material_item",
            "parent_section_id": "doc-remote#申请材料",
            "node_title": "申请表",
            "node_order": 1,
        },
    )
    parent = KnowledgeChunk(
        id="chunk-remote-parent",
        doc_id="doc-remote",
        parent_id="",
        chunk_type="parent",
        title="远程召回测试",
        content="父级上下文：申请表需要申请人自备并在线上传。",
    )
    sibling = KnowledgeChunk(
        id="chunk-remote-sibling",
        doc_id="doc-remote",
        parent_id="chunk-remote-parent",
        chunk_type="child",
        title="远程召回测试",
        content="身份证件通过共享核验。",
        metadata={
            "chunk_type": "material_item",
            "parent_section_id": "doc-remote#申请材料",
            "node_title": "身份证件",
            "node_order": 0,
        },
    )

    class RemoteVectorIndex:
        backend = "milvus"

        def __init__(self):
            self.search_calls = []
            self.batch_calls = []

        def upsert(self, chunk, vector):
            raise AssertionError("remote retrieval should not upsert")

        def delete_by_doc(self, doc_id):
            pass

        def search(self, vector, filters, *, limit):
            self.search_calls.append({"filters": dict(filters), "limit": limit})
            return [(child.id, 0.95)]

        def search_batch(self, vectors, filters, *, limit):
            self.batch_calls.append({"count": len(vectors), "filters": dict(filters), "limit": limit})
            return [[(child.id, 0.95)] for _ in vectors]

        def clear(self):
            pass

    class RemoteKeywordIndex:
        backend = "elasticsearch"

        def __init__(self):
            self.search_calls = []

        def upsert_chunks(self, chunks):
            raise AssertionError("remote retrieval should not index")

        def delete_by_doc(self, doc_id):
            pass

        def search(self, query, filters, *, limit):
            self.search_calls.append({"query": query, "filters": dict(filters), "limit": limit})
            return [(sibling.id, 0.8)]

        def clear(self):
            pass

    class RemoteChunkStore:
        def __init__(self):
            self.chunks = {child.id: child, parent.id: parent, sibling.id: sibling}
            self.chunk_calls = []
            self.parent_calls = []

        def get_chunks_by_ids(self, chunk_ids):
            self.chunk_calls.append(list(chunk_ids))
            return [self.chunks[chunk_id] for chunk_id in chunk_ids if chunk_id in self.chunks]

        def get_parent_chunks(self, parent_ids):
            self.parent_calls.append(list(parent_ids))
            return [self.chunks[chunk_id] for chunk_id in parent_ids if chunk_id in self.chunks]

    store = RemoteChunkStore()
    vector_index = RemoteVectorIndex()
    keyword_index = RemoteKeywordIndex()
    engine = ExternalIndexRAGEngine(
        vector_index=vector_index,
        keyword_index=keyword_index,
        chunk_store=store,
    )

    result = engine.retrieve("申请表需要哪些材料", {}, top_k=1)

    assert engine.chunk_count == 0
    assert result.sources
    assert result.status["candidate_source"] == "external_indexes"
    assert result.status["vector_backend"] == "milvus"
    assert result.status["channels"]["dense"] == 1
    assert result.status["channels"]["bm25"] == 1
    assert store.chunk_calls
    assert store.parent_calls == [[parent.id]]

    many_results = engine.retrieve_many(
        [
            {"text": "申请表材料", "top_k": 1},
            {"text": "身份证材料", "top_k": 1},
        ],
        {"region_code": "340000"},
    )

    assert len(many_results) == 2
    assert engine.chunk_count == 0
    assert vector_index.batch_calls == [{"count": 2, "filters": {"region_code": "340000", "doc_status": "published"}, "limit": 30}]
    assert [call["filters"] for call in keyword_index.search_calls[-2:]] == [
        {"region_code": "340000", "doc_status": "published"},
        {"region_code": "340000", "doc_status": "published"},
    ]
    assert all(result.status["candidate_source"] == "external_indexes" for result in many_results)
    assert all(result.status["batch_mode"] is True for result in many_results)


def test_remote_multi_query_retrieval_batches_io_and_reranks_each_query():
    child_a = KnowledgeChunk(
        id="chunk-rewrite-a",
        doc_id="doc-rewrite",
        parent_id="",
        chunk_type="child",
        title="改写召回测试",
        content="办理时限为10个工作日。",
    )
    child_b = KnowledgeChunk(
        id="chunk-rewrite-b",
        doc_id="doc-rewrite",
        parent_id="",
        chunk_type="child",
        title="改写召回测试",
        content="承诺办结时限为5个工作日。",
    )

    class BatchVectorIndex:
        backend = "milvus"

        def __init__(self):
            self.batch_calls = []

        def upsert(self, chunk, vector):
            pass

        def delete_by_doc(self, doc_id):
            pass

        def search(self, vector, filters, *, limit):
            raise AssertionError("single dense search should not be used")

        def search_batch(self, vectors, filters, *, limit):
            self.batch_calls.append({"count": len(vectors), "filters": dict(filters), "limit": limit})
            return [[(child_a.id, 0.9)], [(child_b.id, 0.88)]]

        def clear(self):
            pass

    class BatchKeywordIndex:
        backend = "elasticsearch"

        def __init__(self):
            self.search_many_calls = []
            self.search_calls = []

        def upsert_chunks(self, chunks):
            pass

        def delete_by_doc(self, doc_id):
            pass

        def search(self, query, filters, *, limit):
            self.search_calls.append({"query": query, "filters": dict(filters), "limit": limit})
            raise AssertionError("single BM25 search should not be used")

        def search_many(self, query_specs, filters, limits):
            self.search_many_calls.append(
                {
                    "queries": [spec["query"] for spec in query_specs],
                    "filters": dict(filters),
                    "limits": list(limits),
                }
            )
            return {
                int(query_specs[0]["index"]): [(child_b.id, 0.7)],
                int(query_specs[1]["index"]): [(child_a.id, 0.6)],
            }

        def clear(self):
            pass

    class ChunkStore:
        def __init__(self):
            self.chunks = {child_a.id: child_a, child_b.id: child_b}
            self.chunk_calls = []

        def get_chunks_by_ids(self, chunk_ids):
            self.chunk_calls.append(list(chunk_ids))
            return [self.chunks[chunk_id] for chunk_id in chunk_ids if chunk_id in self.chunks]

        def get_parent_chunks(self, parent_ids):
            return []

    class CountingReranker:
        name = "counting"

        def __init__(self):
            self.calls = []
            self.active_calls = 0
            self.max_active_calls = 0
            self.lock = Lock()

        def rerank(self, query, sources, filters, *, top_n):
            with self.lock:
                self.calls.append({"query": query, "count": len(sources), "top_n": top_n})
                self.active_calls += 1
                self.max_active_calls = max(self.max_active_calls, self.active_calls)
            try:
                sleep(0.05)
                return [
                    RetrievedSource(
                        chunk=source.chunk,
                        score=source.score,
                        dense_score=source.dense_score,
                        bm25_score=source.bm25_score,
                        rerank_score=1.0 - index * 0.1,
                        retrieval_channel=f"{source.retrieval_channel}+rerank",
                    )
                    for index, source in enumerate(sources[:top_n])
                ]
            finally:
                with self.lock:
                    self.active_calls -= 1

    class RecordingCache:
        backend = "recording"

        def __init__(self):
            self.write_thread_ids = []
            self.items = {}

        def get_json(self, key):
            return self.items.get(key)

        def set_json(self, key, value, *, ttl_seconds):
            self.write_thread_ids.append(get_ident())
            self.items[key] = dict(value)

        def clear(self):
            self.items.clear()

    vector_index = BatchVectorIndex()
    keyword_index = BatchKeywordIndex()
    chunk_store = ChunkStore()
    reranker = CountingReranker()
    cache_store = RecordingCache()
    caller_thread_id = get_ident()
    engine = ExternalIndexRAGEngine(
        vector_index=vector_index,
        keyword_index=keyword_index,
        chunk_store=chunk_store,
        reranker=reranker,
        cache_store=cache_store,
    )

    results = engine.retrieve_many(
        [
            {"id": "rq1", "text": "办理时限", "strategy": "time_limit", "top_k": 2},
            {"id": "rq2", "text": "承诺办结时限", "strategy": "promise_limit", "top_k": 2},
        ],
        {"region_code": "340000"},
    )

    assert vector_index.batch_calls == [{"count": 2, "filters": {"region_code": "340000", "doc_status": "published"}, "limit": 30}]
    assert keyword_index.search_many_calls == [
        {
            "queries": ["办理时限", "承诺办结时限"],
            "filters": {"region_code": "340000", "doc_status": "published"},
            "limits": [30, 30],
        }
    ]
    assert keyword_index.search_calls == []
    assert sorted(call["query"] for call in reranker.calls) == ["办理时限", "承诺办结时限"]
    assert all(call["top_n"] == 2 for call in reranker.calls)
    assert reranker.max_active_calls == 2
    assert len(results) == 2
    assert all(result.status["batch_mode"] is True for result in results)
    assert all(result.status["channels"]["rerank"] == 2 for result in results)
    assert all(result.status["candidate_source"] == "external_indexes" for result in results)
    assert len(chunk_store.chunk_calls) == 1
    assert cache_store.write_thread_ids == [caller_thread_id, caller_thread_id]
