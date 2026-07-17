import json

from gov_service_agent.core.config import GovServiceAgentSettings
from gov_service_agent.indexing.attachments.chunks import build_attachment_chunk_specs
from gov_service_agent.indexing.attachments.models import AttachmentRef, ExtractedAttachment
from gov_service_agent.indexing.document_parsing.factory import build_document_parser, parse_document_bytes
from gov_service_agent.indexing.document_parsing.mineru_parser import (
    MinerUDocumentParser,
    mineru_chunk_specs,
)


def test_mineru_content_items_become_deduplicated_chonkie_chunks():
    items = [
        {
            "type": "text",
            "text": "申请材料",
            "text_level": 1,
            "page_idx": 0,
            "bbox": [10, 20, 300, 60],
        },
        {
            "type": "text",
            "text": "提交营业执照或身份证明。",
            "page_idx": 0,
            "bbox": [10, 80, 800, 120],
        },
        {
            "type": "text",
            "text": "提交营业执照或身份证明。",
            "page_idx": 0,
            "bbox": [10, 80, 800, 120],
        },
        {
            "type": "table",
            "table_caption": ["材料清单"],
            "table_body": "<table><tr><td>材料</td><td>要求</td></tr></table>",
            "page_idx": 1,
            "bbox": [20, 100, 900, 600],
        },
    ]

    specs = mineru_chunk_specs(items)

    assert len(specs) == 2
    assert all(spec["chunk_type"] == "child" for spec in specs)
    assert all("parent_id" not in spec and "parent_id_suffix" not in spec for spec in specs)
    table_specs = [spec for spec in specs if spec["metadata"]["chonkie_chunk_kind"] == "table"]
    assert len(table_specs) == 1
    assert table_specs[0]["metadata"]["chunk_type"] == "chonkie_table"
    assert table_specs[0]["metadata"]["table_source"] == "mineru_native"
    assert "材料清单" in table_specs[0]["content"]


def test_mineru_parser_calls_api_and_reads_native_outputs(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "status": "completed",
                "results": {
                    "guide": {
                        "md_content": "# 取水许可\n\n办理材料说明",
                        "content_list": json.dumps(
                            [
                                {
                                    "type": "text",
                                    "text": "取水许可",
                                    "text_level": 1,
                                    "page_idx": 0,
                                    "bbox": [0, 0, 900, 80],
                                },
                                {
                                    "type": "table",
                                    "table_caption": ["申请材料"],
                                    "table_body": "<table><tr><td>申请表</td><td>原件</td></tr></table>",
                                    "page_idx": 0,
                                    "bbox": [10, 100, 900, 700],
                                },
                            ],
                            ensure_ascii=False,
                        ),
                        "images": {},
                    }
                },
            }

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["files"] = kwargs["files"]
        captured["data"] = dict(kwargs["data"])
        captured["timeout"] = kwargs["timeout"]
        return FakeResponse()

    monkeypatch.setattr("gov_service_agent.indexing.document_parsing.mineru_parser.requests.post", fake_post)
    parser = MinerUDocumentParser(timeout_seconds=120, backend="pipeline", model_source="modelscope")

    parsed = parser.parse(filename="guide.pdf", content_type="application/pdf", data=b"%PDF fake")

    assert parsed.parser_name == "mineru"
    assert parsed.metadata["chonkie_chunk_count"] == 1
    assert len(parsed.metadata["chonkie_chunk_specs"]) == 1
    assert [block.block_type for block in parsed.blocks] == ["text", "table"]
    assert captured["url"] == "http://127.0.0.1:8011/file_parse"
    assert captured["files"][0][1][0] == "guide.pdf"
    assert captured["data"]["backend"] == "pipeline"
    assert captured["data"]["lang_list"] == "ch"
    assert captured["data"]["formula_enable"] == "false"
    assert captured["data"]["return_content_list"] == "true"
    assert captured["timeout"] == (10, 120)


def test_mineru_factory_receives_runtime_settings():
    parser = build_document_parser(
        provider="mineru",
        fallback_enabled=False,
        mineru_api_url="http://127.0.0.1:8011",
        mineru_backend="pipeline",
        mineru_method="ocr",
        mineru_language="ch",
        mineru_model_source="modelscope",
        mineru_timeout_seconds=5400,
    )

    assert isinstance(parser, MinerUDocumentParser)
    assert parser.api_url == "http://127.0.0.1:8011"
    assert parser.method == "ocr"
    assert parser.formula_enabled is False
    assert parser.timeout_seconds == 5400


def test_markdown_uses_native_fallback_when_mineru_does_not_support_it():
    parsed = parse_document_bytes(
        filename="item_28.md",
        content_type="text/markdown",
        data="# 取水许可\n\n申请材料".encode("utf-8"),
        parser_provider="auto",
        parser_order="mineru,native",
    )

    assert parsed.parser_name == "native"
    assert "取水许可" in parsed.text
    assert "document_parser_attempt_errors" not in parsed.metadata


def test_attachment_reuses_mineru_chunks_without_table_semantic_expansion():
    ref = AttachmentRef(
        index=0,
        name="取水申请表.pdf",
        url="https://example.test/form.pdf",
        extension="pdf",
        section_path="申请材料",
        section_title="申请材料",
        node_title="取水申请表",
        parent_section_id="doc#申请材料",
    )
    extracted = ExtractedAttachment(
        ref=ref,
        text="申请材料表格",
        content_type="application/pdf",
        sha256="abc123",
        metadata={"semantic_chunk_type": "attachment_form", "document_parser": "mineru"},
        chunk_specs=[
            {
                "id_suffix": "chonkie:table:2-2:0:abc",
                "chunk_type": "child",
                "content": "申请人 | 取水用途",
                "metadata": {
                    "chunk_type": "chonkie_table",
                    "chunking_strategy": "chonkie",
                    "chonkie_chunk_kind": "table",
                    "mineru_item_type": "table",
                    "page_no": 2,
                    "indexable": True,
                },
            }
        ],
    )

    specs = build_attachment_chunk_specs([extracted], document_id="doc", document_metadata={})

    assert len(specs) == 1
    assert specs[0]["metadata"]["chunk_type"] == "attachment_chonkie_table"
    assert specs[0]["metadata"]["attachment_page_no"] == 2
    assert "parent_id_suffix" not in specs[0]
    assert "table_semantic" not in specs[0]["id_suffix"]


def test_index_timeout_is_longer_than_mineru_timeout():
    settings = GovServiceAgentSettings(
        _env_file=None,
        GOV_SERVICE_AGENT_MINERU_TIMEOUT_SECONDS=3600,
        GOV_SERVICE_AGENT_INDEX_JOB_TIMEOUT_SECONDS=7200,
    )

    assert settings.document_parser_order == "mineru,native"
    assert settings.index_job_timeout_seconds > settings.mineru_timeout_seconds
