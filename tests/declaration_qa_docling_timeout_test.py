import logging
import sys
import types

from declaration_qa.core.config import DeclarationQASettings
from declaration_qa.indexing.attachments.extractor import attachment_parser_metadata
from declaration_qa.indexing.document_parsing.docling_parser import (
    DoclingDocumentParser,
    docling_chunk_specs,
    docling_picture_blocks,
    docling_table_lookup,
    resolve_docling_accelerator_device,
)
from declaration_qa.indexing.document_parsing.factory import CompositeDocumentParser, build_document_parser, parse_document_bytes
from declaration_qa.indexing.document_parsing.models import DocumentParseError, ParsedDocument


def test_docling_pdf_converter_receives_document_timeout(monkeypatch):
    captured = {}

    class FakePdfPipelineOptions:
        def __init__(self, **kwargs):
            captured["pipeline_options"] = kwargs

    class FakeAcceleratorOptions:
        def __init__(self, **kwargs):
            captured["accelerator_options"] = kwargs

    class FakePdfFormatOption:
        def __init__(self, **kwargs):
            captured["format_option"] = kwargs

    class FakeImageFormatOption:
        def __init__(self, **kwargs):
            captured["image_format_option"] = kwargs

    class FakeDocumentConverter:
        def __init__(self, **kwargs):
            captured["converter"] = kwargs

    fake_input_format = types.SimpleNamespace(IMAGE="image", PDF="pdf")
    monkeypatch.setitem(
        sys.modules,
        "docling.document_converter",
        types.SimpleNamespace(
            DocumentConverter=FakeDocumentConverter,
            ImageFormatOption=FakeImageFormatOption,
            PdfFormatOption=FakePdfFormatOption,
        ),
    )
    monkeypatch.setitem(sys.modules, "docling.datamodel.base_models", types.SimpleNamespace(InputFormat=fake_input_format))
    monkeypatch.setitem(
        sys.modules,
        "docling.datamodel.pipeline_options",
        types.SimpleNamespace(AcceleratorOptions=FakeAcceleratorOptions, PdfPipelineOptions=FakePdfPipelineOptions),
    )
    monkeypatch.setattr("declaration_qa.indexing.document_parsing.docling_parser.is_docling_cuda_available", lambda: True)

    parser = DoclingDocumentParser(
        document_timeout_seconds=3600,
        accelerator_device="cuda",
        accelerator_num_threads=6,
        layout_batch_size=8,
        table_batch_size=4,
        ocr_batch_size=2,
    )
    parser._document_converter(filename="sample.pdf", content_type="application/pdf")

    assert captured["accelerator_options"] == {"num_threads": 6, "device": "cuda"}
    assert captured["pipeline_options"] == {
        "document_timeout": 3600.0,
        "accelerator_options": captured["pipeline_options"]["accelerator_options"],
        "layout_batch_size": 8,
        "table_batch_size": 4,
        "ocr_batch_size": 2,
        "generate_picture_images": True,
    }
    assert isinstance(captured["format_option"]["pipeline_options"], FakePdfPipelineOptions)
    assert "pdf" in captured["converter"]["format_options"]


def test_docling_image_converter_receives_accelerator_options(monkeypatch):
    captured = {}

    class FakePdfPipelineOptions:
        def __init__(self, **kwargs):
            captured["pipeline_options"] = kwargs

    class FakeAcceleratorOptions:
        def __init__(self, **kwargs):
            captured["accelerator_options"] = kwargs

    class FakePdfFormatOption:
        def __init__(self, **kwargs):
            captured["pdf_format_option"] = kwargs

    class FakeImageFormatOption:
        def __init__(self, **kwargs):
            captured["image_format_option"] = kwargs

    class FakeDocumentConverter:
        def __init__(self, **kwargs):
            captured["converter"] = kwargs

    fake_input_format = types.SimpleNamespace(IMAGE="image", PDF="pdf")
    monkeypatch.setitem(
        sys.modules,
        "docling.document_converter",
        types.SimpleNamespace(
            DocumentConverter=FakeDocumentConverter,
            ImageFormatOption=FakeImageFormatOption,
            PdfFormatOption=FakePdfFormatOption,
        ),
    )
    monkeypatch.setitem(sys.modules, "docling.datamodel.base_models", types.SimpleNamespace(InputFormat=fake_input_format))
    monkeypatch.setitem(
        sys.modules,
        "docling.datamodel.pipeline_options",
        types.SimpleNamespace(AcceleratorOptions=FakeAcceleratorOptions, PdfPipelineOptions=FakePdfPipelineOptions),
    )
    monkeypatch.setattr("declaration_qa.indexing.document_parsing.docling_parser.is_docling_cuda_available", lambda: True)

    parser = DoclingDocumentParser(document_timeout_seconds=3600, accelerator_device="cuda", layout_batch_size=8)
    parser._document_converter(filename="sample.png", content_type="image/png")

    assert captured["accelerator_options"] == {"num_threads": 4, "device": "cuda"}
    assert captured["pipeline_options"]["layout_batch_size"] == 8
    assert isinstance(captured["image_format_option"]["pipeline_options"], FakePdfPipelineOptions)
    assert "image" in captured["converter"]["format_options"]
    assert "pdf_format_option" not in captured


def test_docling_cuda_device_falls_back_to_cpu_when_unavailable(monkeypatch, caplog):
    monkeypatch.setattr("declaration_qa.indexing.document_parsing.docling_parser.is_docling_cuda_available", lambda: False)

    with caplog.at_level(logging.WARNING):
        resolved = resolve_docling_accelerator_device("cuda")

    assert resolved == "cpu"
    assert "falling back to CPU" in caplog.text


def test_docling_converter_is_reused_for_same_input_format(monkeypatch):
    parser = DoclingDocumentParser()
    created = []

    def fake_create(*, filename, content_type):
        converter = object()
        created.append(converter)
        return converter

    monkeypatch.setattr(parser, "_document_converter", fake_create)

    first, first_lock = parser._cached_document_converter(filename="a.pdf", content_type="application/pdf")
    second, second_lock = parser._cached_document_converter(filename="b.pdf", content_type="application/pdf")

    assert first is second
    assert first_lock is second_lock
    assert len(created) == 1


def test_docling_picture_is_exported_with_caption_location_and_text():
    from PIL import Image

    child = types.SimpleNamespace(text="申请、受理、审查、决定")
    child_ref = types.SimpleNamespace(resolve=lambda document: child)
    bbox = types.SimpleNamespace(l=10, t=20, r=210, b=120, to_top_left_origin=lambda: bbox)
    picture = types.SimpleNamespace(
        self_ref="#/pictures/0",
        label=types.SimpleNamespace(value="picture"),
        children=[child_ref],
        annotations=[],
        prov=[types.SimpleNamespace(page_no=3, bbox=bbox)],
        get_image=lambda document: Image.new("RGB", (200, 100), "white"),
        caption_text=lambda document: "图1 办理流程",
    )

    blocks = docling_picture_blocks(types.SimpleNamespace(pictures=[picture]))

    assert len(blocks) == 1
    assert blocks[0].block_type == "picture"
    assert blocks[0].binary_data.startswith(b"\x89PNG")
    assert blocks[0].text == "申请、受理、审查、决定"
    assert blocks[0].page_no == 3
    assert blocks[0].bbox == [10.0, 20.0, 210.0, 120.0]
    assert blocks[0].metadata["picture_caption"] == "图1 办理流程"


def test_docling_timeout_flows_from_parser_factory():
    parser = build_document_parser(
        provider="docling",
        fallback_enabled=False,
        docling_document_timeout_seconds=5400,
        docling_accelerator_device="gpu",
        docling_accelerator_num_threads=5,
        docling_layout_batch_size=8,
        docling_table_batch_size=4,
        docling_ocr_batch_size=2,
    )

    assert isinstance(parser, DoclingDocumentParser)
    assert parser.document_timeout_seconds == 5400
    assert parser.accelerator_device == "cuda"
    assert parser.accelerator_num_threads == 5
    assert parser.layout_batch_size == 8
    assert parser.table_batch_size == 4
    assert parser.ocr_batch_size == 2


def test_index_timeout_is_longer_than_docling_timeout():
    settings = DeclarationQASettings(
        _env_file=None,
        DECLARATION_QA_DOCLING_DOCUMENT_TIMEOUT_SECONDS=3600,
        DECLARATION_QA_INDEX_JOB_TIMEOUT_SECONDS=7200,
    )

    assert settings.docling_document_timeout_seconds == 3600
    assert settings.index_job_timeout_seconds == 7200
    assert settings.index_job_timeout_seconds > settings.docling_document_timeout_seconds


def test_attachment_metadata_drops_docling_chunk_specs():
    metadata = attachment_parser_metadata(
        {
            "document_parser": "docling",
            "docling_chunk_count": 2,
            "docling_chunk_specs": [{"content": "large internal chunk payload"}],
        }
    )

    assert metadata == {
        "document_parser": "docling",
        "docling_chunk_count": 2,
    }


def test_composite_parser_logs_failure_before_fallback(caplog):
    class FailingParser:
        name = "docling"

        def parse(self, *, filename, content_type, data):
            raise DocumentParseError("Docling parse failed: RuntimeError: test failure")

    class FallbackParser:
        name = "native"

        def parse(self, *, filename, content_type, data):
            return ParsedDocument(
                filename=filename,
                content_type=content_type,
                text="fallback result",
                parser_name=self.name,
            )

    parser = CompositeDocumentParser([FailingParser(), FallbackParser()])

    with caplog.at_level(logging.WARNING):
        result = parser.parse(filename="sample.pdf", content_type="application/pdf", data=b"pdf")

    assert result.parser_name == "native"
    assert "parser=docling" in caplog.text
    assert "next_parser=native" in caplog.text
    assert "RuntimeError: test failure" in caplog.text


def test_docling_chunker_allows_documents_without_text_chunks(monkeypatch):
    class EmptyHybridChunker:
        def chunk(self, document):
            return []

    monkeypatch.setattr("docling_core.transforms.chunker.HybridChunker", EmptyHybridChunker)

    assert docling_chunk_specs(types.SimpleNamespace(tables=[])) == []


def test_docling_table_caption_method_is_called_instead_of_serialized():
    cell = types.SimpleNamespace(text="申请材料")
    table = types.SimpleNamespace(
        self_ref="#/tables/0",
        data=types.SimpleNamespace(grid=[[cell]], table_cells=[]),
        caption_text=lambda document: f"{document.name}清单",
        prov=[],
    )

    lookup = docling_table_lookup(types.SimpleNamespace(name="取水许可", tables=[table]))

    assert lookup["#/tables/0"]["table_json"]["title"] == "取水许可清单"
    assert "bound method" not in lookup["#/tables/0"]["table_text"]


def test_auto_parser_uses_project_ocr_first_for_images():
    class FakeOcrClient:
        def __init__(self):
            self.calls = 0

        def extract_text(self, *, filename, content_type, data):
            self.calls += 1
            return "申请、受理、审查、决定"

    ocr_client = FakeOcrClient()

    result = parse_document_bytes(
        filename="1.png",
        content_type="image/png",
        data=b"fake image",
        ocr_client=ocr_client,
        parser_provider="auto",
        parser_order="docling,marker,native",
    )

    assert result.parser_name == "native"
    assert result.text == "申请、受理、审查、决定"
    assert ocr_client.calls == 1
