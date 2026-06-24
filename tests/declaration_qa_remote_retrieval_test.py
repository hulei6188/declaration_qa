from declaration_qa.core.domain import KnowledgeChunk
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
            self.section_calls = []

        def get_chunks_by_ids(self, chunk_ids):
            self.chunk_calls.append(list(chunk_ids))
            return [self.chunks[chunk_id] for chunk_id in chunk_ids if chunk_id in self.chunks]

        def get_parent_chunks(self, parent_ids):
            self.parent_calls.append(list(parent_ids))
            return [self.chunks[chunk_id] for chunk_id in parent_ids if chunk_id in self.chunks]

        def get_structured_section_chunks(self, parent_section_ids, target_chunk_type):
            self.section_calls.append((list(parent_section_ids), target_chunk_type))
            return [sibling, child]

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
    assert store.section_calls == [(["doc-remote#申请材料"], "material_item")]
    assert any(source.chunk.id == sibling.id for source in result.sources)

    many_results = engine.retrieve_many(
        [
            {"text": "申请表材料", "top_k": 1},
            {"text": "身份证材料", "top_k": 1},
        ],
        {"region_code": "340000"},
    )

    assert len(many_results) == 2
    assert engine.chunk_count == 0
    assert vector_index.batch_calls == [{"count": 2, "filters": {"region_code": "340000"}, "limit": 8}]
    assert [call["filters"] for call in keyword_index.search_calls[-2:]] == [{"region_code": "340000"}, {"region_code": "340000"}]
    assert all(result.status["candidate_source"] == "external_indexes" for result in many_results)
    assert all(result.status["batch_mode"] is True for result in many_results)
