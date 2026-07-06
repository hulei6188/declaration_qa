import io
import sys
import types
import zipfile

from declaration_qa.indexing.documents import parse_docx, parse_pdf
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


def test_chunk_storage_status_keeps_staging_rows_database_safe():
    assert chunk_storage_status("indexing") == "draft"
    assert chunk_storage_status("draft") == "draft"
    assert chunk_storage_status("published") == "published"
    assert chunk_storage_status("deprecated") == "deprecated"


def make_docx(document_xml: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("word/document.xml", document_xml)
    return buffer.getvalue()
