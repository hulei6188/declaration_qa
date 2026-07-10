from declaration_qa.core.domain import KnowledgeChunk, RetrievedSource
from declaration_qa.rag.engine.parent_context import expand_parent_context


def test_expand_parent_context_dedupes_sources_with_same_parent() -> None:
    parent = KnowledgeChunk(
        id="doc-1:p:0",
        doc_id="doc-1",
        parent_id="",
        chunk_type="parent",
        title="取水许可办事指南 > 申请材料 > 5. 取水许可申请书",
        content="完整父级内容",
    )
    child_a = KnowledgeChunk(
        id="doc-1:p:0:c:0",
        doc_id="doc-1",
        parent_id=parent.id,
        chunk_type="child",
        title=parent.title,
        content="子片段 A",
    )
    child_b = KnowledgeChunk(
        id="doc-1:p:0:c:1",
        doc_id="doc-1",
        parent_id=parent.id,
        chunk_type="child",
        title=parent.title,
        content="子片段 B",
    )
    sources = [
        RetrievedSource(chunk=child_a, score=0.8, rerank_score=0.8, retrieval_channel="dense"),
        RetrievedSource(chunk=child_b, score=0.9, rerank_score=0.9, retrieval_channel="bm25"),
    ]

    expanded = expand_parent_context(sources, {parent.id: parent})

    assert len(expanded) == 1
    assert expanded[0].chunk.content == parent.content
    assert expanded[0].score == 0.9
    assert expanded[0].chunk.metadata["retrieved_child_chunk_id"] == child_b.id
    assert expanded[0].chunk.metadata["retrieved_child_chunk_ids"] == [child_a.id, child_b.id]
    assert expanded[0].chunk.metadata["parent_context_merged_child_count"] == 2
    assert expanded[0].retrieval_channel == "dense+bm25"
