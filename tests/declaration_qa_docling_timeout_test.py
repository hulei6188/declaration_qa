import sys
import types

from declaration_qa.core.config import DeclarationQASettings
from declaration_qa.indexing.attachments.extractor import attachment_parser_metadata
from declaration_qa.indexing.document_parsing.docling_parser import DoclingDocumentParser
from declaration_qa.indexing.document_parsing.factory import build_document_parser


def test_docling_pdf_converter_receives_document_timeout(monkeypatch):
    captured = {}

    class FakePdfPipelineOptions:
        def __init__(self, **kwargs):
            captured["pipeline_options"] = kwargs

    class FakePdfFormatOption:
        def __init__(self, **kwargs):
            captured["format_option"] = kwargs

    class FakeDocumentConverter:
        def __init__(self, **kwargs):
            captured["converter"] = kwargs

    fake_input_format = types.SimpleNamespace(PDF="pdf")
    monkeypatch.setitem(
        sys.modules,
        "docling.document_converter",
        types.SimpleNamespace(DocumentConverter=FakeDocumentConverter, PdfFormatOption=FakePdfFormatOption),
    )
    monkeypatch.setitem(sys.modules, "docling.datamodel.base_models", types.SimpleNamespace(InputFormat=fake_input_format))
    monkeypatch.setitem(
        sys.modules,
        "docling.datamodel.pipeline_options",
        types.SimpleNamespace(PdfPipelineOptions=FakePdfPipelineOptions),
    )

    parser = DoclingDocumentParser(document_timeout_seconds=3600)
    parser._document_converter(filename="sample.pdf", content_type="application/pdf")

    assert captured["pipeline_options"] == {"document_timeout": 3600.0}
    assert isinstance(captured["format_option"]["pipeline_options"], FakePdfPipelineOptions)
    assert "pdf" in captured["converter"]["format_options"]


def test_docling_timeout_flows_from_parser_factory():
    parser = build_document_parser(
        provider="docling",
        fallback_enabled=False,
        docling_document_timeout_seconds=5400,
    )

    assert isinstance(parser, DoclingDocumentParser)
    assert parser.document_timeout_seconds == 5400


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
