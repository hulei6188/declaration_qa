import io
import sys
import types
import zipfile

from declaration_qa.indexing.attachments.form_fields import extract_form_fields
from declaration_qa.indexing.document_parsing.legacy_doc import DOCX_CONTENT_TYPE, LegacyDocConversionResult
from declaration_qa.indexing.documents import HttpOcrClient, parse_docx, parse_document_bytes, parse_pdf
from declaration_qa.indexing.service.workflow_index import chunk_storage_status


WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def test_parse_docx_preserves_headings_and_tables():
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="{WORD_NS}">
  <w:body>
    <w:p>
      <w:pPr><w:pStyle w:val="Heading1"/></w:pPr>
      <w:r><w:t>创业补贴申报指南</w:t></w:r>
    </w:p>
    <w:p><w:r><w:t>一、申请条件</w:t></w:r></w:p>
    <w:p><w:r><w:t>符合条件人员可线上申报。</w:t></w:r></w:p>
    <w:tbl>
      <w:tr>
        <w:tc><w:p><w:r><w:t>材料名称</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>要求</w:t></w:r></w:p></w:tc>
      </w:tr>
      <w:tr>
        <w:tc><w:p><w:r><w:t>身份证件</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>原件扫描件</w:t></w:r></w:p></w:tc>
      </w:tr>
    </w:tbl>
  </w:body>
</w:document>
"""
    data = make_docx(xml)

    text = parse_docx(data)

    assert "# 创业补贴申报指南" in text
    assert "## 一、申请条件" in text
    assert "符合条件人员可线上申报。" in text
    assert "| 材料名称 | 要求 |" in text
    assert "| --- | --- |" in text
    assert "| 身份证件 | 原件扫描件 |" in text


def test_parse_pdf_preserves_page_boundaries_and_layout_fallback(monkeypatch):
    class LayoutPage:
        def __init__(self, text):
            self.text = text

        def extract_text(self, **kwargs):
            if kwargs.get("extraction_mode") == "layout":
                return self.text
            return "plain fallback"

    class FallbackPage:
        def extract_text(self, **kwargs):
            if kwargs.get("extraction_mode") == "layout":
                return ""
            return "第二页办理流程"

    class FakePdfReader:
        def __init__(self, _stream):
            self.pages = [LayoutPage("第一页申请材料"), FallbackPage()]

    monkeypatch.setitem(sys.modules, "pypdf", types.SimpleNamespace(PdfReader=FakePdfReader))

    text = parse_pdf(b"%PDF fake bytes")

    assert "## PDF 第 1 页" in text
    assert "第一页申请材料" in text
    assert "## PDF 第 2 页" in text
    assert "第二页办理流程" in text


def test_parse_xls_with_calamine_preserves_table_for_form_fields(monkeypatch):
    class FakeSheet:
        def to_python(self):
            return [
                ["  国家重点保护野生植物采集申请表 ", "", "", ""],
                ["申请单位、个人", "", "", ""],
                ["联系人", "", "联系电话", ""],
                ["采集目的", "", "", ""],
                ["物种中文名", "学名", "保护级别", "采集数量"],
            ]

    class FakeWorkbook:
        sheet_names = ["Sheet1", "Sheet2"]

        def get_sheet_by_name(self, name):
            return FakeSheet() if name == "Sheet1" else types.SimpleNamespace(to_python=lambda: [])

        def close(self):
            return None

    monkeypatch.setitem(
        sys.modules,
        "python_calamine",
        types.SimpleNamespace(load_workbook=lambda _stream: FakeWorkbook()),
    )

    parsed = parse_document_bytes(
        filename="国家重点保护野生植物采集申请表.xls",
        content_type="application/vnd.ms-excel",
        data=b"fake-xls",
    )

    assert "## 工作表：Sheet1" in parsed.text
    assert "### 国家重点保护野生植物采集申请表" in parsed.text
    assert "| 申请单位、个人 |" in parsed.text
    assert "| 联系人 |" in parsed.text
    assert "| --- | --- | --- | --- |" in parsed.text
    assert "Sheet2" not in parsed.text

    field_names = {field.field_name for field in extract_form_fields(parsed.text)}
    assert {"申请单位、个人", "联系人", "联系电话", "采集目的", "物种中文名", "保护级别"}.issubset(field_names)


def test_legacy_doc_is_converted_to_docx_before_parsing(monkeypatch):
    captured = {}
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="{WORD_NS}">
  <w:body>
    <w:p><w:r><w:t>退出现役残疾军人集中供养申请书</w:t></w:r></w:p>
    <w:p><w:r><w:t>申请人应填写本人基本情况。</w:t></w:r></w:p>
  </w:body>
</w:document>
"""

    def fake_convert(**kwargs):
        captured.update(kwargs)
        return LegacyDocConversionResult(
            filename="退出现役残疾军人集中供养申请书.docx",
            content_type=DOCX_CONTENT_TYPE,
            data=make_docx(xml),
            converter_command="fake-soffice",
        )

    monkeypatch.setattr(
        "declaration_qa.indexing.document_parsing.factory.convert_legacy_doc_to_docx",
        fake_convert,
    )

    parsed = parse_document_bytes(
        filename="退出现役残疾军人集中供养申请书.doc",
        content_type="application/msword",
        data=b"legacy-doc-bytes",
        parser_provider="native",
        legacy_doc_converter_command="fake-soffice",
        legacy_doc_conversion_timeout_seconds=9,
    )

    assert "退出现役残疾军人集中供养申请书" in parsed.text
    assert "申请人应填写本人基本情况" in parsed.text
    assert parsed.filename == "退出现役残疾军人集中供养申请书.doc"
    assert parsed.content_type == "application/msword"
    assert parsed.metadata["legacy_doc_converted"] is True
    assert parsed.metadata["legacy_doc_converted_filename"] == "退出现役残疾军人集中供养申请书.docx"
    assert parsed.metadata["document_parser"] == "native"
    assert captured["filename"] == "退出现役残疾军人集中供养申请书.doc"
    assert captured["data"] == b"legacy-doc-bytes"
    assert captured["converter_command"] == "fake-soffice"
    assert captured["timeout_seconds"] == 9


def test_chunk_storage_status_keeps_staging_rows_database_safe():
    assert chunk_storage_status("indexing") == "draft"
    assert chunk_storage_status("draft") == "draft"
    assert chunk_storage_status("published") == "published"
    assert chunk_storage_status("deprecated") == "deprecated"


def test_dashscope_qwen_ocr_uses_compatible_chat_endpoint(monkeypatch):
    captured = {}

    class FakeResponse:
        status_code = 200
        text = "{}"

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "```html\n<html><body><p>识别出的文字</p></body></html>\n```"}}]}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["headers"] = kwargs["headers"]
        captured["json"] = kwargs["json"]
        return FakeResponse()

    monkeypatch.setattr("declaration_qa.indexing.documents.requests.post", fake_post)
    client = HttpOcrClient(
        url="https://dashscope.aliyuncs.com/api/v1",
        api_key="sk-test",
        model="qwen-vl-ocr-2025-11-20",
    )

    text = client.extract_text(filename="page.png", content_type="image/png", data=b"fake-png")

    assert text == "识别出的文字"
    assert captured["url"] == "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    assert captured["json"]["model"] == "qwen-vl-ocr-2025-11-20"
    content = captured["json"]["messages"][0]["content"]
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert "只输出识别到的正文" in content[1]["text"]


def make_docx(document_xml: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("word/document.xml", document_xml)
    return buffer.getvalue()
