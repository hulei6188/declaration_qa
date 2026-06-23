from __future__ import annotations

from declaration_qa.domain import KnowledgeChunk
from declaration_qa.keyword_index import ElasticsearchKeywordIndex, elasticsearch_chunk_document, elasticsearch_filter_clauses


def test_elasticsearch_chunk_document_contains_filter_fields():
    chunk = KnowledgeChunk(
        id="chunk-1",
        doc_id="doc-1",
        parent_id="",
        chunk_type="child",
        title="居住证办理",
        content="需要申请表。",
        audit_node="材料初审",
        status="published",
        metadata={
            "region_code": "340000",
            "catalog_name": "户政",
            "item_name": "居住证办理",
        },
    )

    document = elasticsearch_chunk_document(chunk)

    assert document["region_code"] == "340000"
    assert document["catalog_name"] == "户政"
    assert document["item_name"] == "居住证办理"
    assert document["audit_node"] == "材料初审"
    assert document["status"] == "published"
    assert document["doc_status"] == "published"


def test_elasticsearch_filter_clauses_push_down_supported_filters():
    clauses = elasticsearch_filter_clauses(
        {
            "biz_type": "residence_permit",
            "region_code": ["340000", "340000000000"],
            "itemName": "居住证办理",
            "audit_node": "材料初审",
            "doc_status": "published",
            "as_of": "2026-01-01",
        }
    )

    assert clauses == [
        {"term": {"chunk_type": "child"}},
        {"terms": {"region_code": ["340000", "340000000000"]}},
        {"term": {"item_name.keyword": "居住证办理"}},
        {"term": {"audit_node": "材料初审"}},
        {"term": {"status": "published"}},
    ]


def test_elasticsearch_bulk_payload_is_utf8_encoded(monkeypatch):
    captured = {}

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

    class FakeRequests:
        def head(self, *args, **kwargs):
            return FakeResponse()

        def post(self, *args, **kwargs):
            captured["args"] = args
            captured.update(kwargs)
            return FakeResponse()

    monkeypatch.setattr("declaration_qa.keyword_index.import_requests", lambda: FakeRequests())
    index = ElasticsearchKeywordIndex(url="http://es.test", index_name="chunks")
    chunk = KnowledgeChunk(
        id="chunk-1",
        doc_id="doc-1",
        parent_id="",
        chunk_type="child",
        title="远程测试",
        content="申请表需要申请人自备。",
    )

    index.upsert_chunks([chunk])

    assert captured["args"][0] == "http://es.test/_bulk?refresh=true"
    assert isinstance(captured["data"], bytes)
    assert "远程测试".encode("utf-8") in captured["data"]
