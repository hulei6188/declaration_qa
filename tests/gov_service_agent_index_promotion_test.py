from dataclasses import replace

from gov_service_agent.core.domain import KnowledgeChunk, KnowledgeDocument
from gov_service_agent.rag.engine.external_index import ExternalIndexRAGEngine
from gov_service_agent.storage.cache import InMemoryCacheStore


def test_external_index_promotion_reuses_staged_vectors_without_embedding_again():
    class EmbeddingProvider:
        name = "test"

        def __init__(self):
            self.calls = 0

        def embed_documents(self, texts):
            self.calls += 1
            return [[0.1, 0.2] for _ in texts]

    class VectorIndex:
        backend = "memory"

        def __init__(self):
            self.deleted = []
            self.upserts = []

        def delete_by_doc(self, doc_id):
            self.deleted.append(doc_id)

        def upsert(self, chunk, vector):
            self.upserts.append((chunk, vector))

        def flush(self):
            pass

    class KeywordIndex:
        backend = "memory"

        def __init__(self):
            self.deleted = []
            self.upserts = []

        def delete_by_doc(self, doc_id):
            self.deleted.append(doc_id)

        def upsert_chunks(self, chunks):
            self.upserts.append(list(chunks))

    class ChunkStore:
        def get_chunks_by_ids(self, chunk_ids):
            return []

        def get_parent_chunks(self, parent_ids):
            return []

    provider = EmbeddingProvider()
    vector_index = VectorIndex()
    keyword_index = KeywordIndex()
    engine = ExternalIndexRAGEngine(
        vector_index=vector_index,
        keyword_index=keyword_index,
        chunk_store=ChunkStore(),
        cache_store=InMemoryCacheStore(),
        embedding_provider=provider,
    )
    staging = KnowledgeDocument(id="doc-1", title="测试", content="", status="indexing")
    published = KnowledgeDocument(id="doc-1", title="测试", content="", status="published")
    staging_chunk = KnowledgeChunk(
        id="chunk-1",
        doc_id="doc-1",
        parent_id="",
        chunk_type="child",
        title="测试",
        content="测试内容",
        embedding_id="embedding-1",
        status="draft",
    )

    engine.index_chunks(staging, [staging_chunk])
    engine.promote_indexed_chunks(published, [replace(staging_chunk, status="published")])

    assert provider.calls == 1
    assert vector_index.deleted == ["doc-1"]
    assert len(vector_index.upserts) == 2
    assert vector_index.upserts[-1][0].status == "published"
    assert len(keyword_index.upserts) == 2
