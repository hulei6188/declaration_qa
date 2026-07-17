from gov_service_agent.core.domain import KnowledgeChunk, RetrievedSource
from gov_service_agent.rag.engine.parent_context import dedupe_asset_sources, expand_parent_context


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


def test_expand_parent_context_dedupes_sources_from_same_asset() -> None:
    original = KnowledgeChunk(
        id="doc-1:asset:flowchart",
        doc_id="doc-1",
        parent_id="",
        chunk_type="child",
        title="流程图",
        content="流程图视觉摘要和 OCR 预览",
        metadata={
            "chunk_type": "attachment_image",
            "modality": "image",
            "asset_id": "asset-flowchart",
            "asset_url": "/assets/asset-flowchart",
        },
    )
    ocr_a = KnowledgeChunk(
        id="doc-1:asset:flowchart:ocr:0",
        doc_id="doc-1",
        parent_id="",
        chunk_type="child",
        title="流程图 OCR",
        content="OCR 片段 A",
        metadata={
            "chunk_type": "attachment_image_ocr",
            "modality": "text",
            "source_asset_id": "asset-flowchart",
        },
    )
    ocr_b = KnowledgeChunk(
        id="doc-1:asset:flowchart:ocr:1",
        doc_id="doc-1",
        parent_id="",
        chunk_type="child",
        title="流程图 OCR",
        content="OCR 片段 B",
        metadata={
            "chunk_type": "attachment_image_ocr",
            "modality": "text",
            "source_asset_id": "asset-flowchart",
        },
    )
    sources = [
        RetrievedSource(chunk=ocr_a, score=0.95, dense_score=0.8, rerank_score=0.95, retrieval_channel="dense"),
        RetrievedSource(chunk=original, score=0.9, bm25_score=0.9, rerank_score=0.9, retrieval_channel="bm25"),
        RetrievedSource(chunk=ocr_b, score=0.92, dense_score=0.92, rerank_score=0.92, retrieval_channel="dense"),
    ]

    expanded = expand_parent_context(sources, {})

    assert len(expanded) == 1
    assert expanded[0].chunk.id == original.id
    assert expanded[0].score == 0.95
    assert expanded[0].dense_score == 0.92
    assert expanded[0].bm25_score == 0.9
    assert expanded[0].rerank_score == 0.95
    assert expanded[0].retrieval_channel == "dense+bm25"
    assert expanded[0].chunk.metadata["asset_group_id"] == "asset-flowchart"
    assert expanded[0].chunk.metadata["asset_merged_chunk_ids"] == [ocr_a.id, original.id, ocr_b.id]
    assert expanded[0].chunk.metadata["asset_merged_chunk_count"] == 3
    assert expanded[0].chunk.metadata["asset_merged_semantic_types"] == ["attachment_image_ocr", "attachment_image"]


def test_dedupe_asset_sources_merges_hits_from_separate_retrieval_channels() -> None:
    original = KnowledgeChunk(
        id="doc-1:asset:image",
        doc_id="doc-1",
        parent_id="",
        chunk_type="child",
        title="材料图片",
        content="原图证据",
        metadata={"chunk_type": "attachment_image", "modality": "image", "asset_id": "asset-1"},
    )
    ocr = KnowledgeChunk(
        id="doc-1:asset:image:ocr:0",
        doc_id="doc-1",
        parent_id="",
        chunk_type="child",
        title="材料图片 OCR",
        content="OCR 证据",
        metadata={"chunk_type": "attachment_image_ocr", "modality": "text", "source_asset_id": "asset-1"},
    )

    deduped = dedupe_asset_sources(
        [
            RetrievedSource(chunk=original, score=0.8, rerank_score=0.8, retrieval_channel="image_dense"),
            RetrievedSource(chunk=ocr, score=0.9, rerank_score=0.9, retrieval_channel="elasticsearch_bm25"),
        ]
    )

    assert len(deduped) == 1
    assert deduped[0].chunk.id == original.id
    assert deduped[0].score == 0.9
    assert deduped[0].chunk.metadata["asset_merged_chunk_count"] == 2
    assert deduped[0].retrieval_channel == "image_dense+elasticsearch_bm25"
