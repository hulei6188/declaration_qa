from __future__ import annotations

from datetime import date
import sys
from types import SimpleNamespace

from declaration_qa.core.domain import KnowledgeChunk
from declaration_qa.search.vector import MILVUS_TEXT_METADATA_FIELDS, MilvusVectorIndex, milvus_filter_expr, milvus_metadata_row


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
    assert row["audit_node"] == "材料初审"
    assert row["status"] == "published"
    assert row["doc_status"] == "published"
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
        if field_name in {"status", "doc_status"}:
            assert row[field_name] == "published"
        else:
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


def test_milvus_filter_expr_uses_supported_metadata_fields_only():
    expr = milvus_filter_expr(
        {
            "biz_type": "residence_permit",
            "item_code": "ITEM001",
            "region_code": ["340000", "340000000000"],
            "audit_node": "材料初审",
            "doc_status": "published",
        }
    )

    assert (
        expr
        == 'item_code == "ITEM001" and region_code in ["340000", "340000000000"] and audit_node == "材料初审" and doc_status == "published"'
    )


def test_milvus_vector_index_does_not_keep_memory_fallback_when_disabled(monkeypatch):
    class FakeSchema:
        def add_field(self, *args, **kwargs):
            pass

    class FakeIndexParams:
        def add_index(self, *args, **kwargs):
            pass

    class FakeMilvusClient:
        instances = []

        def __init__(self, **kwargs):
            self.upserts = []
            self.flushes = []
            FakeMilvusClient.instances.append(self)

        def has_collection(self, collection_name):
            return False

        def create_schema(self, **kwargs):
            return FakeSchema()

        def prepare_index_params(self):
            return FakeIndexParams()

        def create_collection(self, **kwargs):
            pass

        def upsert(self, **kwargs):
            self.upserts.append(kwargs)

        def flush(self, **kwargs):
            self.flushes.append(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "pymilvus",
        SimpleNamespace(
            MilvusClient=FakeMilvusClient,
            DataType=SimpleNamespace(VARCHAR="VARCHAR", FLOAT_VECTOR="FLOAT_VECTOR", JSON="JSON"),
        ),
    )
    chunk = KnowledgeChunk(
        id="chunk-1",
        doc_id="doc-1",
        parent_id="",
        chunk_type="child",
        title="测试事项",
        content="content",
    )

    index = MilvusVectorIndex(uri="http://milvus.test", collection_name="chunks", dimension=2, allow_fallback=False)
    index.upsert(chunk, [0.2, 0.8])
    index.flush()

    assert index._fallback is None
    assert FakeMilvusClient.instances[0].upserts[0]["data"][0]["id"] == "chunk-1"
    assert FakeMilvusClient.instances[0].flushes == [{"collection_name": "chunks"}]
