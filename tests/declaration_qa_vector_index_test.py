from __future__ import annotations

from datetime import date

from declaration_qa.domain import KnowledgeChunk
from declaration_qa.vector_index import MILVUS_TEXT_METADATA_FIELDS, milvus_filter_expr, milvus_metadata_row


def test_milvus_metadata_row_keeps_observable_metadata_fields():
    chunk = KnowledgeChunk(
        id="chunk-1",
        doc_id="doc-1",
        parent_id="",
        chunk_type="child",
        title="测试事项",
        content="content",
        region_code="340000",
        policy_version="2025",
        source_url="https://example.test/source",
        audit_node="材料初审",
        status="published",
        effective_start_time=date(2026, 1, 1),
        metadata={
            "region_key": "340000000000",
            "region_name": "安徽省",
            "catalog_name": "户政",
            "item_code": "ITEM001",
            "item_name": "居住证办理",
            "official_source_url": "https://example.test/official",
            "chunking_strategy": "parent_child",
            "nested": {"start": date(2026, 1, 1)},
        },
    )

    row = milvus_metadata_row(chunk)

    assert row["region_code"] == "340000"
    assert row["region_key"] == "340000000000"
    assert row["region_name"] == "安徽省"
    assert row["policy_version"] == "2025"
    assert row["official_source_url"] == "https://example.test/official"
    assert row["catalog_name"] == "户政"
    assert row["item_code"] == "ITEM001"
    assert row["item_name"] == "居住证办理"
    assert row["chunking_strategy"] == "parent_child"
    assert row["metadata"]["nested"]["start"] == "2026-01-01"


def test_milvus_metadata_row_supplies_all_text_fields_with_defaults():
    row = milvus_metadata_row(
        KnowledgeChunk(
            id="chunk-1",
            doc_id="doc-1",
            parent_id="",
            chunk_type="child",
            title="测试事项",
            content="content",
        )
    )

    for field_name in MILVUS_TEXT_METADATA_FIELDS:
        assert field_name in row
        assert row[field_name] == ""
    assert row["metadata"] == {}


def test_milvus_metadata_row_truncates_long_text_fields():
    row = milvus_metadata_row(
        KnowledgeChunk(
            id="chunk-1",
            doc_id="doc-1",
            parent_id="",
            chunk_type="child",
            title="测试事项",
            content="content",
            metadata={"item_name": "事项" * 200},
        )
    )

    assert len(row["item_name"]) == MILVUS_TEXT_METADATA_FIELDS["item_name"]


def test_milvus_metadata_fields_do_not_enable_vector_filtering():
    assert milvus_filter_expr({"biz_type": "residence_permit", "region_code": "340000"}) == ""
