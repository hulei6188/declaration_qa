import hashlib
import io
import zipfile
from pathlib import Path

from declaration_qa.core.config import DeclarationQASettings
from declaration_qa.core.domain import KnowledgeDocument
from declaration_qa.indexing.attachments.downloader import content_type_from_extension
from declaration_qa.indexing.attachments.extractor import attachment_text_quality_error
from declaration_qa.indexing.attachments.models import DownloadedAttachment
from declaration_qa.indexing.attachments.parser import parse_markdown_attachments
from declaration_qa.indexing.attachments.pipeline import build_attachment_chunks_for_document
from declaration_qa.rag.engine.chunking import split_document
from declaration_qa.rag.engine.in_memory import InMemoryRAGEngine


WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def test_parse_item_28_markdown_attachments_with_section_context():
    content = Path(
        "materials/curated/declaration_qa/2026-06-15-ahzwfw-real-40-topics/item_28.md"
    ).read_text(encoding="utf-8")

    refs = parse_markdown_attachments(content, document_id="item_28_official_ahzwfw")

    assert len(refs) == 7
    assert {ref.extension for ref in refs} == {"pdf", "docx", "png"}
    form_ref = next(ref for ref in refs if ref.name == "空白样表-取水许可申请书.docx")
    assert form_ref.role == "blank_form"
    assert form_ref.material_name == "取水许可申请书"
    assert form_ref.section_title == "申请材料"
    assert form_ref.parent_section_id.startswith("item_28_official_ahzwfw#")
    flow_ref = next(ref for ref in refs if ref.name == "1.png")
    assert flow_ref.role == "flowchart"
    assert flow_ref.section_title == "办理流程"


def test_attachment_pipeline_builds_form_chunks_and_keeps_failed_image_non_blocking(monkeypatch, tmp_path):
    content = """
# 取水许可办事指南

## 申请材料

### 5. 取水许可申请书
样本文件：
- 示例样表-取水许可申请书.docx: https://example.test/form.docx

## 办理流程

### 流程图
- 1.png: https://example.test/flow.png
"""

    def fake_download(self, ref):
        data = make_docx() if ref.extension == "docx" else b"fake image"
        return DownloadedAttachment(
            ref=ref,
            data=data,
            content_type=content_type_from_extension(ref.extension),
            cache_path=str(tmp_path / ref.name),
            sha256=hashlib.sha256(data).hexdigest(),
        )

    monkeypatch.setattr("declaration_qa.indexing.attachments.pipeline.AttachmentDownloader.download", fake_download)
    monkeypatch.setenv("DECLARATION_QA_ATTACHMENT_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("DECLARATION_QA_ATTACHMENT_INDEXING_ENABLED", "true")
    monkeypatch.delenv("DECLARATION_QA_OCR_API_URL", raising=False)
    settings = DeclarationQASettings()

    result = build_attachment_chunks_for_document(
        document_id="doc-water",
        content=content,
        metadata={"item_name": "取水许可", "chunking_strategy": "hierarchical"},
        settings=settings,
    )

    assert result.total_count == 2
    assert result.success_count == 1
    assert result.failed_count == 1
    assert result.chunk_specs
    spec = result.chunk_specs[0]
    assert spec["metadata"]["chunk_type"] == "attachment_form"
    assert spec["metadata"]["attachment_role"] == "sample_file"
    assert spec["metadata"]["material_name"] == "取水许可申请书"
    assert "取水用途" in spec["content"]
    assert "制水供水" in spec["content"]


def test_attachment_pipeline_deduplicates_same_file_content(monkeypatch, tmp_path):
    content = """
# 取水许可办事指南

## 申请材料

### 5. 取水许可申请书
空表文件：
- 空白样表-取水许可申请书.docx: https://example.test/form-a.docx
- 空白样表-取水许可申请书.docx: https://example.test/form-b.docx
"""

    def fake_download(self, ref):
        data = make_docx()
        return DownloadedAttachment(
            ref=ref,
            data=data,
            content_type=content_type_from_extension(ref.extension),
            cache_path=str(tmp_path / ref.name),
            sha256=hashlib.sha256(data).hexdigest(),
        )

    monkeypatch.setattr("declaration_qa.indexing.attachments.pipeline.AttachmentDownloader.download", fake_download)
    monkeypatch.setenv("DECLARATION_QA_ATTACHMENT_CACHE_DIR", str(tmp_path))
    settings = DeclarationQASettings(_env_file=None)

    result = build_attachment_chunks_for_document(
        document_id="doc-water",
        content=content,
        metadata={"item_name": "取水许可", "chunking_strategy": "hierarchical"},
        settings=settings,
    )

    assert result.total_count == 2
    assert result.success_count == 1
    assert result.failed_count == 0
    assert result.skipped_count == 1
    assert result.chunk_specs


def test_split_document_indexes_derived_attachment_chunks_as_child_siblings():
    document = KnowledgeDocument(
        id="doc-water",
        title="取水许可",
        content="# 取水许可办事指南\n\n## 申请材料\n\n### 5. 取水许可申请书\n正文材料说明",
        metadata={"chunking_strategy": "hierarchical", "item_name": "取水许可"},
        derived_chunks=[
            {
                "id_suffix": "att:0:abc:0",
                "content": "附件：空白样表-取水许可申请书.docx\n\n字段：取水用途、取水工程类型",
                "source_url": "https://example.test/form.docx",
                "metadata": {
                    "chunk_type": "attachment_form",
                    "section_title": "申请材料",
                    "node_title": "5. 取水许可申请书",
                    "parent_section_id": "doc-water#申请材料",
                    "material_name": "取水许可申请书",
                    "attachment_name": "空白样表-取水许可申请书.docx",
                },
            }
        ],
    )

    chunks = split_document(document)

    derived = next(chunk for chunk in chunks if chunk.metadata.get("derived_chunk"))
    assert derived.chunk_type == "child"
    assert derived.source_url == "https://example.test/form.docx"
    assert derived.metadata["chunk_type"] == "attachment_form"
    assert derived.metadata["parent_section_id"] == "doc-water#申请材料"
    assert derived.metadata["sibling_ids"]


def test_form_question_backfills_attachment_form_from_material_section():
    engine = InMemoryRAGEngine()
    document = KnowledgeDocument(
        id="doc-water-form",
        title="取水许可",
        content="# 取水许可办事指南\n\n## 申请材料\n\n### 1. 取水许可申请书\n申请书需要申请人自备。",
        metadata={"chunking_strategy": "hierarchical", "item_name": "取水许可"},
        derived_chunks=[
            {
                "id_suffix": "att:0:abc:0",
                "content": "附件：空白样表-取水许可申请书.docx\n\n字段：取水用途、取水工程类型、水源、取水地点",
                "source_url": "https://example.test/form.docx",
                "metadata": {
                    "chunk_type": "attachment_form",
                    "section_title": "申请材料",
                    "node_title": "1. 取水许可申请书",
                    "parent_section_id": "doc-water-form#申请材料",
                    "material_name": "取水许可申请书",
                    "attachment_name": "空白样表-取水许可申请书.docx",
                    "attachment_role": "blank_form",
                    "indexable": True,
                },
            }
        ],
    )

    engine.index_document(document)
    result = engine.retrieve("申请书需要申请人自备，怎么填", {}, top_k=1)

    attachment_sources = [
        source
        for source in result.sources
        if source.chunk.metadata.get("chunk_type") == "attachment_form"
    ]
    assert attachment_sources
    assert "取水用途" in attachment_sources[0].chunk.content
    assert "attachment_form" in result.status["structured_backfill"]["target_chunk_types"]


def test_attachment_quality_gate_rejects_pdf_mojibake():
    bad_pdf_text = "## PDF 第 1 页\n" + "\x8b\x95\xaa\x89E\x8c\x94\x82\xca" * 5
    good_form_text = "取水许可申请书\n\n| 字段 | 内容 |\n| --- | --- |\n| 取水用途 | □制水供水 □生活用水 |"

    assert attachment_text_quality_error(bad_pdf_text, "pdf") == "low_quality_extracted_text"
    assert attachment_text_quality_error(good_form_text, "docx") == ""


def make_docx() -> bytes:
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="{WORD_NS}">
  <w:body>
    <w:p><w:r><w:t>取水许可申请书</w:t></w:r></w:p>
    <w:tbl>
      <w:tr>
        <w:tc><w:p><w:r><w:t>取水用途（可多选）</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>□制水供水 □原水供水 □生活用水 □工业用水</w:t></w:r></w:p></w:tc>
      </w:tr>
      <w:tr>
        <w:tc><w:p><w:r><w:t>取水工程类型</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>□闸 □坝 □渠道 □水泵 □水井</w:t></w:r></w:p></w:tc>
      </w:tr>
    </w:tbl>
  </w:body>
</w:document>
"""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("word/document.xml", xml)
    return buffer.getvalue()
