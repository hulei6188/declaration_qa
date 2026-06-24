from __future__ import annotations

from declaration_qa.core.domain import KnowledgeChunk, KnowledgeDocument
from declaration_qa.indexing.service.reindex import process_reindex


def test_process_reindex_reuses_persisted_chunks_when_available():
    document = KnowledgeDocument(id="doc-1", title="持久化重建", content="重新切分不应被调用")
    chunk = KnowledgeChunk(
        id="persisted-chunk-id",
        doc_id=document.id,
        parent_id="",
        chunk_type="child",
        title=document.title,
        content="PostgreSQL 已持久化 chunk",
    )

    class FakeStore:
        def list_indexed_knowledge(self):
            return [document], [chunk]

    class FakeRagEngine:
        def __init__(self):
            self.rebuild_indexed_payload = None

        def rebuild_indexed_data(self, documents, chunks):
            self.rebuild_indexed_payload = (documents, chunks)
            return len(chunks)

        def rebuild(self, documents):
            raise AssertionError("reindex should use persisted chunks when the store provides them")

    class FakeWorkflow:
        def __init__(self):
            self.audit_store = FakeStore()
            self.rag_engine = FakeRagEngine()

    workflow = FakeWorkflow()

    result = process_reindex(workflow)

    assert result["documents_indexed"] == 1
    assert result["chunks_indexed"] == 1
    assert workflow.rag_engine.rebuild_indexed_payload == ([document], [chunk])
