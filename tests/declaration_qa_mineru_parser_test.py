import json
import types

from declaration_qa.core.config import DeclarationQASettings
from declaration_qa.indexing.attachments.chunks import build_attachment_chunk_specs
from declaration_qa.indexing.attachments.models import AttachmentRef, ExtractedAttachment
from declaration_qa.indexing.document_parsing.factory import build_document_parser, parse_document_bytes
from declaration_qa.indexing.document_parsing.mineru_parser import (
    MinerUDocumentParser,
    mineru_chunk_specs,
)


def test_mineru_content_items_are_one_to_one_and_exact_duplicates_are_removed():
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

    assert len(specs) == 3
    assert all(spec["chunk_type"] == "child" for spec in specs)
    assert all("parent_id" not in spec and "parent_id_suffix" not in spec for spec in specs)
    table_specs = [spec for spec in specs if spec["metadata"]["mineru_item_type"] == "table"]
    assert len(table_specs) == 1
    assert table_specs[0]["metadata"]["chunk_type"] == "mineru_table"
    assert table_specs[0]["metadata"]["table_source"] == "mineru_native"
    assert "材料清单" in table_specs[0]["content"]


def test_mineru_parser_runs_pipeline_with_modelscope_and_reads_native_outputs(monkeypatch, tmp_path):
    captured = {}

    monkeypatch.setattr(
        "declaration_qa.indexing.document_parsing.mineru_parser.shutil.which",
        lambda value: "C:/tools/mineru.exe",
    )

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        output_dir = tmp_path / "mineru-output"
        actual_output = command[command.index("-o") + 1]
        output_dir = type(output_dir)(actual_output) / "result" / "auto"
        output_dir.mkdir(parents=True)
        (output_dir / "document.md").write_text("# 取水许可\n\n办理材料说明", encoding="utf-8")
        (output_dir / "document_content_list.json").write_text(
            json.dumps(
                [
                    {"type": "text", "text": "取水许可", "text_level": 1, "page_idx": 0, "bbox": [0, 0, 900, 80]},
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
            encoding="utf-8",
        )
        return types.SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("declaration_qa.indexing.document_parsing.mineru_parser.subprocess.run", fake_run)
    parser = MinerUDocumentParser(timeout_seconds=120, backend="pipeline", model_source="modelscope")

    parsed = parser.parse(filename="guide.pdf", content_type="application/pdf", data=b"%PDF fake")

    assert parsed.parser_name == "mineru"
    assert parsed.metadata["mineru_chunk_count"] == 2
    assert len(parsed.metadata["mineru_chunk_specs"]) == 2
    assert [block.block_type for block in parsed.blocks] == ["text", "table"]
    assert captured["command"][captured["command"].index("-b") + 1] == "pipeline"
    assert captured["command"][captured["command"].index("-l") + 1] == "ch"
    assert captured["env"]["MINERU_MODEL_SOURCE"] == "modelscope"


def test_mineru_factory_receives_runtime_settings():
    parser = build_document_parser(
        provider="mineru",
        fallback_enabled=False,
        mineru_cli_path="custom-mineru",
        mineru_backend="pipeline",
        mineru_method="ocr",
        mineru_language="ch",
        mineru_model_source="modelscope",
        mineru_timeout_seconds=5400,
    )

    assert isinstance(parser, MinerUDocumentParser)
    assert parser.cli_path == "custom-mineru"
    assert parser.method == "ocr"
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
    assert parsed.metadata["document_parser_attempt_errors"][0].startswith("mineru:")


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
                "id_suffix": "mineru:2:table",
                "chunk_type": "child",
                "content": "申请人 | 取水用途",
                "metadata": {
                    "chunk_type": "mineru_table",
                    "chunking_strategy": "mineru",
                    "mineru_item_type": "table",
                    "page_no": 2,
                    "indexable": True,
                },
            }
        ],
    )

    specs = build_attachment_chunk_specs([extracted], document_id="doc", document_metadata={})

    assert len(specs) == 1
    assert specs[0]["metadata"]["chunk_type"] == "attachment_mineru_table"
    assert specs[0]["metadata"]["attachment_page_no"] == 2
    assert "parent_id_suffix" not in specs[0]
    assert "table_semantic" not in specs[0]["id_suffix"]


def test_index_timeout_is_longer_than_mineru_timeout():
    settings = DeclarationQASettings(
        _env_file=None,
        DECLARATION_QA_MINERU_TIMEOUT_SECONDS=3600,
        DECLARATION_QA_INDEX_JOB_TIMEOUT_SECONDS=7200,
    )

    assert settings.document_parser_order == "mineru,native"
    assert settings.index_job_timeout_seconds > settings.mineru_timeout_seconds
