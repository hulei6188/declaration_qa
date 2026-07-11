from declaration_qa.core.domain import KnowledgeChunk, RetrievedSource
from declaration_qa.rag.engine.external_index import ExternalIndexRAGEngine


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
    assert vector_index.batch_calls == [{"count": 2, "filters": {"region_code": "340000", "doc_status": "published"}, "limit": 8}]
    assert [call["filters"] for call in keyword_index.search_calls[-2:]] == [
        {"region_code": "340000", "doc_status": "published"},
        {"region_code": "340000", "doc_status": "published"},
    ]
    assert all(result.status["candidate_source"] == "external_indexes" for result in many_results)
    assert all(result.status["batch_mode"] is True for result in many_results)


def test_remote_rewrite_retrieval_batches_bm25_and_reranks_once():
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

        def rerank(self, query, sources, filters, *, top_n):
            self.calls.append({"query": query, "count": len(sources), "top_n": top_n})
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

    vector_index = BatchVectorIndex()
    keyword_index = BatchKeywordIndex()
    chunk_store = ChunkStore()
    reranker = CountingReranker()
    engine = ExternalIndexRAGEngine(
        vector_index=vector_index,
        keyword_index=keyword_index,
        chunk_store=chunk_store,
        reranker=reranker,
    )

    result = engine.retrieve_rewrites(
        "用户问办理要多久",
        [
            {"id": "rq1", "text": "办理时限", "strategy": "time_limit", "top_k": 2},
            {"id": "rq2", "text": "承诺办结时限", "strategy": "promise_limit", "top_k": 2},
        ],
        {"region_code": "340000"},
        top_k=2,
    )

    assert vector_index.batch_calls == [{"count": 2, "filters": {"region_code": "340000", "doc_status": "published"}, "limit": 8}]
    assert keyword_index.search_many_calls == [
        {
            "queries": ["办理时限", "承诺办结时限"],
            "filters": {"region_code": "340000", "doc_status": "published"},
            "limits": [8, 8],
        }
    ]
    assert keyword_index.search_calls == []
    assert len(reranker.calls) == 1
    assert reranker.calls[0]["query"] == "用户问办理要多久"
    assert result.status["retrieval_mode"] == "batch_agent_query_fused"
    assert result.status["channels"]["hit_lists"] == 4
    assert result.status["channels"]["rerank"] == 2
    assert result.status["candidate_source"] == "external_indexes"
    assert result.status["queries"][0]["channels"] == {"dense": 1, "bm25": 1}
    assert len(chunk_store.chunk_calls) == 1
