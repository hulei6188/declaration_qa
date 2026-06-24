from __future__ import annotations

from declaration_qa.core.domain import KnowledgeChunk
from declaration_qa.search.keyword import ElasticsearchKeywordIndex, elasticsearch_chunk_document, elasticsearch_filter_clauses


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
            "item_name": "居住证办理",
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


def test_elasticsearch_bulk_payload_uses_sdk_client(monkeypatch):
    captured = {}

    class FakeIndices:
        def exists(self, *, index):
            captured["exists_index"] = index
            return True

    class FakeElasticsearch:
        def __init__(self, url, **kwargs):
            captured["client_url"] = url
            captured["client_kwargs"] = kwargs
            self.indices = FakeIndices()

    class FakeNotFoundError(Exception):
        pass

    def fake_bulk(client, actions, **kwargs):
        captured["bulk_client"] = client
        captured["bulk_actions"] = actions
        captured["bulk_kwargs"] = kwargs
        return 1, []

    monkeypatch.setattr(
        "declaration_qa.search.keyword.import_elasticsearch",
        lambda: (FakeElasticsearch, FakeNotFoundError, fake_bulk),
    )
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

    assert captured["client_url"] == "http://es.test"
    assert captured["client_kwargs"] == {"request_timeout": 10.0}
    assert captured["exists_index"] == "chunks"
    assert captured["bulk_client"] is index._client
    assert captured["bulk_kwargs"] == {"refresh": True}
    assert captured["bulk_actions"] == [
        {
            "_op_type": "index",
            "_index": "chunks",
            "_id": "chunk-1",
            "_source": {
                "doc_id": "doc-1",
                "chunk_type": "child",
                "title": "远程测试",
                "content": "申请表需要申请人自备。",
                "region_code": "",
                "region_key": "",
                "policy_version": "",
                "source_version": "",
                "catalog_name": "",
                "catalog_subitem_name": "",
                "item_code": "",
                "item_name": "",
                "item_type": "",
                "ssqd_code": "",
                "ssqd_id": "",
                "theme_code": "",
                "service_object": "",
                "audit_node": "",
                "status": "published",
                "doc_status": "published",
                "metadata": {},
            },
        }
    ]
