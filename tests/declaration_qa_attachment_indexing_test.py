import hashlib
import io
import sys
import threading
import time
import types
import zipfile
from pathlib import Path

from declaration_qa.core.config import DeclarationQASettings
from declaration_qa.core.domain import KnowledgeDocument
from declaration_qa.indexing.attachments.chunks import build_attachment_chunk_specs
from declaration_qa.indexing.attachments.assets import extract_visual_assets
from declaration_qa.indexing.attachments.downloader import content_type_from_extension
from declaration_qa.indexing.attachments.extractor import attachment_text_quality_error, extract_attachment
from declaration_qa.indexing.attachments.form_fields import extract_form_fields
from declaration_qa.indexing.attachments.models import AttachmentAsset, AttachmentRef, DownloadedAttachment, ExtractedAttachment
from declaration_qa.indexing.attachments.parser import parse_markdown_attachments
from declaration_qa.indexing.attachments.pipeline import AttachmentIndexingResult, build_attachment_chunks_for_document
from declaration_qa.indexing.document_parsing.models import ParsedBlock, ParsedDocument
from declaration_qa.rag.engine.chunking import split_document
from declaration_qa.rag.engine.in_memory import InMemoryRAGEngine
from declaration_qa.indexing.service.documents import build_uploaded_file_asset_specs, process_document_index


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
    assert "\\_" not in flow_ref.url
    assert "wKhoJmCU_oSAPG8PABGK" in flow_ref.url


def test_parse_attachment_when_parser_merges_following_label_onto_url():
    content = """
## 申请材料

### 1. 水资源论证报告
- 示例样表-水资源论证报告样本.pdf: https://example.test/report.pdf?attrname=report.pdf 空表文件：
- 空白样表-取水许可申请书.docx: https://example.test/form.docx?attrname=form.docx
"""

    refs = parse_markdown_attachments(content, document_id="doc-water")

    assert [ref.name for ref in refs] == [
        "示例样表-水资源论证报告样本.pdf",
        "空白样表-取水许可申请书.docx",
    ]
    assert refs[0].url == "https://example.test/report.pdf?attrname=report.pdf"


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
    spec = next(spec for spec in result.chunk_specs if spec["metadata"]["chunk_type"] == "attachment_form")
    assert spec["metadata"]["chunk_type"] == "attachment_form"
    assert spec["metadata"]["attachment_role"] == "sample_file"
    assert spec["metadata"]["material_name"] == "取水许可申请书"
    assert "取水用途" in spec["content"]
    assert "制水供水" in spec["content"]
    field_specs = [
        spec
        for spec in result.chunk_specs
        if spec["metadata"].get("chunk_type") == "form_field"
    ]
    fields_by_name = {spec["metadata"]["field_name"]: spec for spec in field_specs}
    assert {"取水用途", "取水工程类型"} <= set(fields_by_name)
    water_use = fields_by_name["取水用途"]
    assert water_use["metadata"]["field_type"] == "checkbox"
    assert water_use["metadata"]["field_multiple"] is True
    assert water_use["metadata"]["field_options"] == ["制水供水", "原水供水", "生活用水", "工业用水"]
    assert "表单字段：取水用途" in water_use["content"]


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


def test_attachment_pipeline_extracts_independent_files_in_parallel(monkeypatch):
    content = """
# 测试
- 附件A.docx: https://example.test/a.docx
- 附件B.docx: https://example.test/b.docx
- 附件C.docx: https://example.test/c.docx
"""
    lock = threading.Lock()
    active = 0
    max_active = 0

    def fake_download(self, ref):
        data = ref.name.encode("utf-8")
        return DownloadedAttachment(
            ref=ref,
            data=data,
            content_type=content_type_from_extension(ref.extension),
            cache_path=ref.name,
            sha256=hashlib.sha256(data).hexdigest(),
        )

    def fake_extract(downloaded, **kwargs):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.08)
        with lock:
            active -= 1
        return ExtractedAttachment(
            ref=downloaded.ref,
            text=f"{downloaded.ref.name} 正文内容",
            content_type=downloaded.content_type,
            cache_path=downloaded.cache_path,
            sha256=downloaded.sha256,
            status="success",
        )

    monkeypatch.setattr("declaration_qa.indexing.attachments.pipeline.AttachmentDownloader.download", fake_download)
    monkeypatch.setattr("declaration_qa.indexing.attachments.pipeline.extract_attachment", fake_extract)
    monkeypatch.setattr("declaration_qa.indexing.attachments.pipeline.build_shared_document_parser", lambda *args, **kwargs: object())
    settings = DeclarationQASettings(_env_file=None, DECLARATION_QA_ATTACHMENT_MAX_WORKERS=3)

    result = build_attachment_chunks_for_document(
        document_id="parallel-test",
        content=content,
        metadata={},
        settings=settings,
    )

    assert result.success_count == 3
    assert max_active >= 2


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


def test_attachment_non_table_text_uses_hierarchical_parent_child_chunks():
    extracted = ExtractedAttachment(
        ref=AttachmentRef(
            index=0,
            name="空白样表-取水许可申请书.docx",
            url="https://example.test/water-form.docx",
            extension="docx",
            section_path="取水许可办事指南 > 申请材料 > 5. 取水许可申请书",
            section_title="申请材料",
            node_title="5. 取水许可申请书",
            parent_section_id="doc-water#申请材料",
            material_name="取水许可申请书",
            role="blank_form",
        ),
        text=(
            "填表说明\n\n"
            "- 申请人：申请取水的单位或个人名称，应与注册登记信息一致。\n\n"
            "- 取水用途：指取水的使用范围、方面，主要包括制水供水、原水供水、生活用水、工业用水、农业用水、生态用水等。"
            "按实际取水用途填写，该取水同时具有多种用途时，应逐项点选。\n\n"
            "- 计量方式：包括管道计量方式、明渠计量方式、其他折算方式。管道计量可选择机械水表、电子远传水表、"
            "电磁流量计、超声波流量计等。明渠计量可选择水工建筑物法、堰槽、规则断面单水位推流等。"
        ),
        content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        metadata={"semantic_chunk_type": "attachment_text", "text_length": 260},
    )

    specs = build_attachment_chunk_specs(
        [extracted],
        document_id="doc-water",
        document_metadata={"item_name": "取水许可", "chunking_strategy": "hierarchical"},
        max_chunk_size=160,
        overlap=20,
    )
    document = KnowledgeDocument(
        id="doc-water",
        title="取水许可",
        content="# 取水许可办事指南\n\n## 申请材料\n\n正文",
        metadata={"chunking_strategy": "hierarchical", "item_name": "取水许可"},
        derived_chunks=specs,
    )

    chunks = split_document(document)
    text_chunks = [chunk for chunk in chunks if chunk.metadata.get("chunk_type") == "attachment_text"]
    parent_chunks = [chunk for chunk in text_chunks if chunk.chunk_type == "parent"]
    child_chunks = [chunk for chunk in text_chunks if chunk.chunk_type == "child"]

    assert parent_chunks
    assert child_chunks
    assert all(chunk.metadata["attachment_text_chunking_strategy"] == "hierarchical" for chunk in text_chunks)
    assert all(chunk.metadata.get("indexable") is False for chunk in parent_chunks)
    assert all(chunk.metadata.get("indexable") is True for chunk in child_chunks)
    assert any(chunk.parent_id in {parent.id for parent in parent_chunks} for chunk in child_chunks)


def test_attachment_pdf_ocr_text_chunks_include_page_metadata():
    extracted = ExtractedAttachment(
        ref=AttachmentRef(
            index=0,
            name="示例样表-水资源论证报告样本.pdf",
            url="https://example.test/report.pdf",
            extension="pdf",
            section_path="取水许可办事指南 > 申请材料 > 建设项目水资源论证报告书（表）",
            section_title="申请材料",
            node_title="建设项目水资源论证报告书（表）",
            parent_section_id="doc-water#申请材料",
            material_name="建设项目水资源论证报告书（表）",
            role="sample_file",
        ),
        text=(
            "## OCR PDF 第 1 页\n"
            "水资源论证报告封面\n\n"
            "## OCR PDF 第 2 页\n"
            "目录\n1. 总论\n2. 建设项目概况"
        ),
        content_type="application/pdf",
        metadata={"semantic_chunk_type": "attachment_text", "extraction_method": "pdf_ocr_fallback"},
    )

    specs = build_attachment_chunk_specs(
        [extracted],
        document_id="doc-water",
        document_metadata={"item_name": "取水许可", "chunking_strategy": "hierarchical"},
        max_chunk_size=900,
        overlap=20,
    )

    child_specs = [spec for spec in specs if spec.get("chunk_type") == "child"]
    page_metadata = {spec["metadata"]["attachment_page_no"]: spec["metadata"] for spec in child_specs}
    assert set(page_metadata) == {1, 2}
    assert page_metadata[1]["attachment_page_range"] == "1"
    assert page_metadata[2]["attachment_page_numbers"] == [2]


def test_attachment_parent_child_chunk_records_cross_page_range():
    extracted = ExtractedAttachment(
        ref=AttachmentRef(
            index=0,
            name="示例样表-水资源论证报告样本.pdf",
            url="https://example.test/report.pdf",
            extension="pdf",
            section_path="取水许可办事指南 > 申请材料 > 建设项目水资源论证报告书（表）",
            section_title="申请材料",
            node_title="建设项目水资源论证报告书（表）",
            parent_section_id="doc-water#申请材料",
            material_name="建设项目水资源论证报告书（表）",
            role="sample_file",
        ),
        text="## OCR PDF 第 1 页\n第一页内容\n\n## OCR PDF 第 2 页\n第二页内容",
        content_type="application/pdf",
        metadata={"semantic_chunk_type": "attachment_text", "extraction_method": "pdf_ocr_fallback"},
    )

    specs = build_attachment_chunk_specs(
        [extracted],
        document_id="doc-water",
        document_metadata={"item_name": "取水许可", "chunking_strategy": "parent_child"},
        max_chunk_size=900,
        overlap=20,
    )

    parent_spec = next(spec for spec in specs if spec.get("chunk_type") == "parent")
    assert parent_spec["metadata"]["attachment_page_numbers"] == [1, 2]
    assert parent_spec["metadata"]["attachment_page_range"] == "1-2"
    assert "attachment_page_no" not in parent_spec["metadata"]


def test_attachment_text_child_retrieval_backfills_parent_context():
    parent_content = (
        "附件：空白样表-取水许可申请书.docx\n"
        "所属章节：取水许可办事指南 > 申请材料 > 5. 取水许可申请书\n\n"
        "填表说明\n"
        "申请人应填写申请取水的单位或个人名称。\n"
        "取水用途应填写取水的使用范围、方面，包括制水供水、原水供水、生活用水、工业用水、农业用水、生态用水。"
    )
    document = KnowledgeDocument(
        id="doc-water-text",
        title="取水许可",
        content="# 取水许可办事指南\n\n## 申请材料\n\n### 5. 取水许可申请书\n申请书需要申请人自备。",
        metadata={"chunking_strategy": "hierarchical", "item_name": "取水许可"},
        derived_chunks=[
            {
                "id_suffix": "att_text_parent:0:fill:0:0",
                "parent_id": "doc-water-text:att_text_parent:0:fill:0:0",
                "chunk_type": "parent",
                "content": parent_content,
                "source_url": "https://example.test/form.docx",
                "metadata": {
                    "chunk_type": "attachment_text",
                    "attachment_name": "空白样表-取水许可申请书.docx",
                    "attachment_text_chunking_strategy": "hierarchical",
                    "section_title": "申请材料",
                    "node_title": "填表说明",
                    "parent_section_id": "doc-water-text#申请材料",
                    "indexable": False,
                },
            },
            {
                "id_suffix": "att_text_child:0:fill:0:0:0",
                "parent_id": "doc-water-text:att_text_parent:0:fill:0:0",
                "chunk_type": "child",
                "content": "取水用途应填写取水的使用范围、方面，包括制水供水、原水供水、生活用水、工业用水。",
                "source_url": "https://example.test/form.docx",
                "metadata": {
                    "chunk_type": "attachment_text",
                    "attachment_name": "空白样表-取水许可申请书.docx",
                    "attachment_text_chunking_strategy": "hierarchical",
                    "section_title": "申请材料",
                    "node_title": "填表说明",
                    "parent_section_id": "doc-water-text#申请材料",
                    "indexable": True,
                },
            },
        ],
    )

    engine = InMemoryRAGEngine()
    engine.index_document(document)
    result = engine.retrieve("制水供水 原水供水 生活用水 工业用水", {}, top_k=1)

    assert any(source.chunk.metadata.get("expanded_parent_chunk_id") for source in result.sources)
    assert any("申请人应填写" in source.chunk.content for source in result.sources)
    assert any(source.chunk.metadata.get("chunk_type") == "attachment_text" for source in result.sources)


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
    result = engine.retrieve("取水用途怎么填", {}, top_k=3)

    attachment_sources = [
        source
        for source in result.sources
        if source.chunk.metadata.get("chunk_type") == "attachment_form"
    ]
    assert attachment_sources
    assert "取水用途" in attachment_sources[0].chunk.content
    assert "structured_backfill" not in result.status


def test_specific_form_field_question_retrieves_field_without_backfilling_form():
    engine = InMemoryRAGEngine()
    document = KnowledgeDocument(
        id="doc-water-field",
        title="取水许可",
        content="# 取水许可办事指南\n\n## 申请材料\n\n### 1. 取水许可申请书\n申请书需要申请人自备。",
        metadata={"chunking_strategy": "hierarchical", "item_name": "取水许可"},
        derived_chunks=[
            {
                "id_suffix": "field:0:water-use:0",
                "content": (
                    "表单字段：取水用途\n"
                    "所属表单：取水许可申请书\n"
                    "字段类型：勾选项\n"
                    "是否多选：是\n"
                    "可选项：制水供水、原水供水、生活用水、工业用水"
                ),
                "source_url": "https://example.test/form.docx",
                "metadata": {
                    "chunk_type": "form_field",
                    "section_title": "申请材料",
                    "node_title": "表单字段：取水用途",
                    "node_order": 1,
                    "parent_section_id": "doc-water-field#申请材料",
                    "material_name": "取水许可申请书",
                    "attachment_name": "空白样表-取水许可申请书.docx",
                    "field_name": "取水用途",
                    "field_type": "checkbox",
                    "field_options": ["制水供水", "原水供水", "生活用水", "工业用水"],
                    "indexable": True,
                },
            },
            {
                "id_suffix": "field:0:project-type:1",
                "content": "表单字段：取水工程类型\n可选项：闸、坝、渠道、水泵、水井",
                "source_url": "https://example.test/form.docx",
                "metadata": {
                    "chunk_type": "form_field",
                    "section_title": "申请材料",
                    "node_title": "表单字段：取水工程类型",
                    "node_order": 2,
                    "parent_section_id": "doc-water-field#申请材料",
                    "material_name": "取水许可申请书",
                    "field_name": "取水工程类型",
                    "field_options": ["闸", "坝", "渠道", "水泵", "水井"],
                    "indexable": True,
                },
            },
            {
                "id_suffix": "att:0:form:0",
                "content": "附件：空白样表-取水许可申请书.docx\n\n| 取水用途（可多选） | □制水供水 □原水供水 □生活用水 □工业用水 |",
                "source_url": "https://example.test/form.docx",
                "metadata": {
                    "chunk_type": "attachment_form",
                    "section_title": "申请材料",
                    "node_title": "1. 取水许可申请书",
                    "parent_section_id": "doc-water-field#申请材料",
                    "material_name": "取水许可申请书",
                    "attachment_name": "空白样表-取水许可申请书.docx",
                    "indexable": True,
                },
            },
        ],
    )

    engine.index_document(document)
    result = engine.retrieve("取水用途有哪些选项", {}, top_k=1)

    field_sources = [
        source
        for source in result.sources
        if source.chunk.metadata.get("chunk_type") == "form_field"
    ]
    form_sources = [
        source
        for source in result.sources
        if source.chunk.metadata.get("chunk_type") == "attachment_form"
    ]
    assert [source.chunk.metadata.get("field_name") for source in field_sources] == ["取水用途"]
    assert "制水供水" in field_sources[0].chunk.content
    assert not form_sources
    assert "structured_backfill" not in result.status


def test_attachment_quality_gate_rejects_pdf_mojibake():
    bad_pdf_text = "## PDF 第 1 页\n" + "\x8b\x95\xaa\x89E\x8c\x94\x82\xca" * 5
    good_form_text = "取水许可申请书\n\n| 字段 | 内容 |\n| --- | --- |\n| 取水用途 | □制水供水 □生活用水 |"

    assert attachment_text_quality_error(bad_pdf_text, "pdf") == "low_quality_extracted_text"
    assert attachment_text_quality_error(good_form_text, "docx") == ""


def test_extract_form_fields_from_markdown_table_rows():
    fields = extract_form_fields(
        "| 取水用途（可多选） | □制水供水 □原水供水 □生活用水 □工业用水 |\n"
        "| --- | --- |\n"
        "| 申请取水起始时间 | 年 月 日 |\n"
        "| 申请单位盖章 |  |\n"
        "| 联系人 |  | 联系电话 |  |\n"
        "| 物种中文名 | 学名 | 保护级别 | 采集数量 |\n"
    )

    fields_by_name = {field.field_name: field for field in fields}
    assert fields_by_name["取水用途"].field_type == "checkbox"
    assert fields_by_name["取水用途"].multiple is True
    assert fields_by_name["取水用途"].options == ["制水供水", "原水供水", "生活用水", "工业用水"]
    assert fields_by_name["申请取水起始时间"].field_type == "date"
    assert fields_by_name["申请单位盖章"].field_type == "signature"
    assert "联系电话" in fields_by_name
    assert "学名" in fields_by_name
    assert "保护级别" in fields_by_name


def test_docx_embedded_image_generates_visual_asset_chunk(tmp_path):
    downloaded = make_docx_with_media_downloaded()

    class FakeOcrClient:
        def extract_text(self, *, filename, content_type, data):
            assert filename == "image1.png"
            assert content_type == "image/png"
            return "流程图图片文字"

    class FakeVisionSummaryClient:
        model = "qwen3-vl-plus"

        def summarize_image(self, *, filename, content_type, data, ocr_text):
            assert filename == "image1.png"
            assert ocr_text == "流程图图片文字"
            return "这是一张取水许可办理流程图，包含受理、承办和决定节点。"

    extracted = extract_attachment(
        downloaded,
        ocr_client=FakeOcrClient(),
        vision_summary_client=FakeVisionSummaryClient(),
        asset_dir=tmp_path,
    )

    assert extracted.status == "success"
    assert len(extracted.assets) == 1
    assert extracted.assets[0].asset_kind == "docx_embedded_image"
    assert extracted.assets[0].summary.startswith("这是一张取水许可办理流程图")
    assert extracted.assets[0].metadata["visual_summary_status"] == "success"
    assert Path(extracted.assets[0].path).is_file()

    specs = build_attachment_chunk_specs([extracted], document_id="doc-media", document_metadata={})
    image_specs = [spec for spec in specs if spec["metadata"].get("chunk_type") == "attachment_image"]

    assert len(image_specs) == 1
    assert image_specs[0]["metadata"]["modality"] == "image"
    assert image_specs[0]["metadata"]["asset_url"].startswith("/api/declaration-qa/assets/")
    assert image_specs[0]["metadata"]["visual_summary"].startswith("这是一张取水许可办理流程图")
    assert "视觉摘要：" in image_specs[0]["content"]
    assert "OCR文本：流程图图片文字" in image_specs[0]["content"]
    assert "流程图图片文字" in image_specs[0]["content"]


def test_image_asset_reuses_parser_ocr_text(monkeypatch, tmp_path):
    data = b"fake image bytes"
    ref = AttachmentRef(
        index=0,
        name="1.png",
        url="https://example.test/1.png",
        extension="png",
        section_path="办理流程 > 流程图",
        section_title="办理流程",
        node_title="流程图",
        parent_section_id="doc-flow#办理流程",
        role="flowchart",
    )
    downloaded = DownloadedAttachment(
        ref=ref,
        data=data,
        content_type="image/png",
        cache_path="",
        sha256=hashlib.sha256(data).hexdigest(),
    )
    monkeypatch.setattr(
        "declaration_qa.indexing.attachments.extractor.parse_document_bytes",
        lambda **kwargs: types.SimpleNamespace(
            text="申请、受理、审查、决定",
            content_type="image/png",
            metadata={},
            blocks=[],
        ),
    )

    class FailOnDuplicateOcr:
        def extract_text(self, **kwargs):
            raise AssertionError("image OCR must not run twice")

    extracted = extract_attachment(downloaded, ocr_client=FailOnDuplicateOcr(), asset_dir=tmp_path)

    assert extracted.status == "success"
    assert extracted.assets[0].text == "申请、受理、审查、决定"

    specs = build_attachment_chunk_specs([extracted], document_id="doc-flow", document_metadata={})

    assert len(specs) == 1
    assert specs[0]["metadata"]["chunk_type"] == "attachment_image"
    assert specs[0]["metadata"]["ocr_text"] == "申请、受理、审查、决定"


def test_pdf_page_assets_generate_page_image_chunks(monkeypatch, tmp_path):
    downloaded = make_pdf_downloaded(data=b"%PDF page asset")
    monkeypatch.setattr(
        "declaration_qa.indexing.attachments.extractor.parse_document_bytes",
        lambda **kwargs: types.SimpleNamespace(
            text="## PDF 第 1 页\n第一页表单内容\n\n## PDF 第 2 页\n第二页说明内容",
            content_type="application/pdf",
        ),
    )
    monkeypatch.setitem(sys.modules, "fitz", fake_fitz_module())

    extracted = extract_attachment(downloaded, asset_dir=tmp_path, pdf_ocr_max_pages=2, pdf_ocr_dpi=144)

    assert extracted.status == "success"
    assert len(extracted.assets) == 2
    assert {asset.page_no for asset in extracted.assets} == {1, 2}

    specs = build_attachment_chunk_specs([extracted], document_id="doc-pdf", document_metadata={})
    page_specs = [spec for spec in specs if spec["metadata"].get("chunk_type") == "page_image"]

    assert len(page_specs) == 2
    assert page_specs[0]["metadata"]["modality"] == "image"
    assert page_specs[0]["metadata"]["attachment_page_no"] == 1
    assert page_specs[0]["metadata"]["asset_kind"] == "pdf_page_image"
    assert "第一页表单内容" in page_specs[0]["content"]


def test_mineru_pdf_picture_is_indexed_without_full_page_duplicate(monkeypatch, tmp_path):
    downloaded = make_pdf_downloaded(data=b"%PDF mineru picture")
    picture = ParsedBlock(
        block_type="picture",
        text="申请、受理、审查、决定",
        page_no=2,
        bbox=[10.0, 20.0, 210.0, 120.0],
        binary_data=b"fake png bytes",
        content_type="image/png",
        metadata={"picture_ref": "#/pictures/0", "picture_caption": "图1 办理流程"},
    )
    monkeypatch.setattr(
        "declaration_qa.indexing.attachments.assets.extract_pdf_page_assets",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("page image fallback must not run")),
    )

    assets = extract_visual_assets(
        downloaded,
        asset_dir=tmp_path,
        extracted_text="正文",
        parsed_blocks=[picture],
        pdf_page_fallback=False,
    )

    assert len(assets) == 1
    assert assets[0].asset_kind == "pdf_embedded_picture"
    assert assets[0].page_no == 2
    assert assets[0].bbox == [10.0, 20.0, 210.0, 120.0]
    assert assets[0].text == "申请、受理、审查、决定"
    assert assets[0].metadata["picture_caption"] == "图1 办理流程"


def test_mineru_pdf_without_picture_does_not_create_page_images(monkeypatch, tmp_path):
    downloaded = make_pdf_downloaded(data=b"%PDF text only")
    monkeypatch.setattr(
        "declaration_qa.indexing.attachments.assets.extract_pdf_page_assets",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("page image fallback must not run")),
    )

    assets = extract_visual_assets(
        downloaded,
        asset_dir=tmp_path,
        extracted_text="纯文本 PDF",
        parsed_blocks=[],
        pdf_page_fallback=False,
    )

    assert assets == []


def test_uploaded_pdf_mineru_picture_becomes_image_chunk(tmp_path):
    picture = ParsedBlock(
        block_type="picture",
        text="流程图文字",
        page_no=1,
        bbox=[1.0, 2.0, 100.0, 80.0],
        binary_data=b"picture png bytes",
        content_type="image/png",
        metadata={"picture_ref": "#/pictures/0", "picture_caption": "办理流程图"},
    )
    settings = DeclarationQASettings(
        _env_file=None,
        DECLARATION_QA_ATTACHMENT_ASSET_DIR=str(tmp_path),
        DECLARATION_QA_IMAGE_SUMMARY_ENABLED=False,
    )

    specs = build_uploaded_file_asset_specs(
        document_id="doc-uploaded-pdf",
        filename="guide.pdf",
        content_type="application/pdf",
        data=b"%PDF uploaded",
        extracted_text="办理指南正文",
        parsed_blocks=[picture],
        parser_name="mineru",
        title="办理指南",
        source_url="",
        document_metadata={"chunking_strategy": "mineru"},
        settings=settings,
        ocr_client=None,
    )

    assert len(specs) == 1
    assert specs[0]["metadata"]["chunk_type"] == "attachment_image"
    assert specs[0]["metadata"]["asset_kind"] == "pdf_embedded_picture"
    assert specs[0]["metadata"]["attachment_page_no"] == 1
    assert "图片标题：办理流程图" in specs[0]["content"]
    assert "OCR文本：流程图文字" in specs[0]["content"]


def test_long_picture_ocr_is_split_into_text_chunks(tmp_path):
    long_ocr = "申请材料包括身份证明、营业执照和授权委托书。" * 80
    picture = ParsedBlock(
        block_type="picture",
        text=long_ocr,
        page_no=4,
        bbox=[1.0, 2.0, 100.0, 80.0],
        binary_data=b"picture png bytes",
        content_type="image/png",
        metadata={"picture_ref": "#/pictures/1", "picture_caption": "材料清单"},
    )
    settings = DeclarationQASettings(
        _env_file=None,
        DECLARATION_QA_ATTACHMENT_ASSET_DIR=str(tmp_path),
        DECLARATION_QA_IMAGE_SUMMARY_ENABLED=False,
    )

    specs = build_uploaded_file_asset_specs(
        document_id="doc-long-picture",
        filename="guide.pdf",
        content_type="application/pdf",
        data=b"%PDF uploaded",
        extracted_text="办理指南正文",
        parsed_blocks=[picture],
        parser_name="mineru",
        title="办理指南",
        source_url="",
        document_metadata={"chunking_strategy": "mineru"},
        settings=settings,
        ocr_client=None,
    )

    image_spec = next(spec for spec in specs if spec["metadata"]["chunk_type"] == "attachment_image")
    ocr_specs = [spec for spec in specs if spec["metadata"]["chunk_type"] == "attachment_image_ocr"]
    assert len(ocr_specs) >= 2
    assert image_spec["metadata"]["ocr_text_chunk_count"] == len(ocr_specs)
    assert "OCR文本预览：" in image_spec["content"]
    assert all(spec["metadata"]["modality"] == "text" for spec in ocr_specs)
    assert all(spec["metadata"]["source_asset_id"] == image_spec["metadata"]["asset_id"] for spec in ocr_specs)
    assert all("asset_id" not in spec["metadata"] for spec in ocr_specs)


def test_long_visual_summary_is_split_into_text_chunks(tmp_path):
    ref = AttachmentRef(
        index=0,
        name="guide.pdf",
        url="https://example.test/guide.pdf",
        extension="pdf",
        section_path="办理流程",
        section_title="办理流程",
        node_title="流程图",
        parent_section_id="doc#办理流程",
    )
    asset = AttachmentAsset(
        asset_id="asset-summary",
        asset_kind="pdf_embedded_picture",
        modality="image",
        path=str(tmp_path / "picture.png"),
        content_type="image/png",
        filename="picture.png",
        summary="该图展示申请、受理、审查和决定的完整办理流程。" * 80,
    )
    extracted = ExtractedAttachment(ref=ref, text="", content_type="application/pdf", assets=[asset])

    specs = build_attachment_chunk_specs([extracted], document_id="doc-summary", document_metadata={})

    image_spec = next(spec for spec in specs if spec["metadata"]["chunk_type"] == "attachment_image")
    summary_specs = [spec for spec in specs if spec["metadata"]["chunk_type"] == "attachment_image_summary"]
    assert len(summary_specs) >= 2
    assert image_spec["metadata"]["visual_summary_chunk_count"] == len(summary_specs)
    assert "视觉摘要预览：" in image_spec["content"]
    assert all(spec["metadata"]["source_asset_id"] == "asset-summary" for spec in summary_specs)


def test_uploaded_pdf_picture_flows_into_document_index(monkeypatch, tmp_path):
    upload_path = tmp_path / "guide.pdf"
    upload_path.write_bytes(b"%PDF uploaded")
    picture = ParsedBlock(
        block_type="picture",
        text="流程图文字",
        page_no=2,
        bbox=[1.0, 2.0, 100.0, 80.0],
        binary_data=b"picture png bytes",
        content_type="image/png",
        metadata={"picture_ref": "#/pictures/0", "picture_caption": "办理流程图"},
    )
    parsed = ParsedDocument(
        filename="guide.pdf",
        content_type="application/pdf",
        text="# 办理指南\n正文",
        parser_name="mineru",
        blocks=[picture],
        metadata={
            "mineru_chunk_specs": [
                {
                    "id_suffix": "mineru:0",
                    "chunk_type": "child",
                    "content": "办理指南正文",
                    "metadata": {"chunk_type": "mineru_text"},
                }
            ]
        },
    )
    captured = {}
    monkeypatch.setattr("declaration_qa.indexing.service.documents.parse_document_bytes", lambda **kwargs: parsed)
    monkeypatch.setattr(
        "declaration_qa.indexing.service.documents.build_attachment_chunks_for_document",
        lambda **kwargs: AttachmentIndexingResult([], 0, 0, 0),
    )

    def fake_index(workflow, document, **kwargs):
        captured["document"] = document
        return len(document.derived_chunks)

    monkeypatch.setattr("declaration_qa.indexing.service.documents.index_document_for_workflow", fake_index)
    settings = DeclarationQASettings(
        _env_file=None,
        DECLARATION_QA_ATTACHMENT_ASSET_DIR=str(tmp_path / "assets"),
        DECLARATION_QA_IMAGE_SUMMARY_ENABLED=False,
    )

    result = process_document_index(
        {
            "document_id": "doc-upload-flow",
            "input_type": "file",
            "file_path": str(upload_path),
            "filename": "guide.pdf",
            "content_type": "application/pdf",
            "title": "办理指南",
        },
        workflow=object(),
        settings=settings,
    )

    image_specs = [
        spec
        for spec in captured["document"].derived_chunks
        if (spec.get("metadata") or {}).get("chunk_type") == "attachment_image"
    ]
    assert result["chunks_indexed"] == 2
    assert len(image_specs) == 1
    assert image_specs[0]["metadata"]["asset_kind"] == "pdf_embedded_picture"
    assert captured["document"].metadata["uploaded_visual_asset_count"] == 1


def test_pdf_low_quality_text_falls_back_to_page_ocr(monkeypatch):
    downloaded = make_pdf_downloaded()

    monkeypatch.setattr(
        "declaration_qa.indexing.attachments.extractor.parse_document_bytes",
        lambda **kwargs: types.SimpleNamespace(text="## PDF 第 1 页\n" + "\x8b\x95\xaa\x89E\x8c\x94\x82\xca" * 5),
    )
    monkeypatch.setitem(sys.modules, "fitz", fake_fitz_module())

    class FakeOcrClient:
        def __init__(self):
            self.calls = []

        def extract_text(self, *, filename, content_type, data):
            self.calls.append({"filename": filename, "content_type": content_type, "bytes": len(data)})
            return "取水许可申请书\n取水用途：制水供水。"

    ocr_client = FakeOcrClient()

    extracted = extract_attachment(downloaded, ocr_client=ocr_client, pdf_ocr_max_pages=2, pdf_ocr_dpi=144)

    assert extracted.status == "success"
    assert "取水用途" in extracted.text
    assert extracted.metadata["extraction_method"] == "pdf_ocr_fallback"
    assert extracted.metadata["native_extraction_error"] == "low_quality_extracted_text"
    assert extracted.metadata["pdf_ocr_rendered_page_count"] == 2
    assert len(ocr_client.calls) == 2


def test_pdf_low_quality_text_records_missing_ocr_client(monkeypatch):
    downloaded = make_pdf_downloaded()
    monkeypatch.setattr(
        "declaration_qa.indexing.attachments.extractor.parse_document_bytes",
        lambda **kwargs: types.SimpleNamespace(text="## PDF 第 1 页\n" + "\x8b\x95\xaa\x89E\x8c\x94\x82\xca" * 5),
    )

    extracted = extract_attachment(downloaded, ocr_client=None)

    assert extracted.status == "failed"
    assert "low_quality_extracted_text" in extracted.error_reason
    assert "pdf_ocr_client_not_configured" in extracted.error_reason
    assert extracted.metadata["pdf_ocr_fallback_attempted"] is True


def test_pdf_ocr_fallback_records_missing_render_backend(monkeypatch):
    downloaded = make_pdf_downloaded()
    monkeypatch.setattr(
        "declaration_qa.indexing.attachments.extractor.parse_document_bytes",
        lambda **kwargs: types.SimpleNamespace(text="## PDF 第 1 页\n" + "\x8b\x95\xaa\x89E\x8c\x94\x82\xca" * 5),
    )
    monkeypatch.setitem(sys.modules, "fitz", None)

    class FakeOcrClient:
        def extract_text(self, *, filename, content_type, data):
            return "不会调用到这里"

    extracted = extract_attachment(downloaded, ocr_client=FakeOcrClient())

    assert extracted.status == "failed"
    assert "pdf_render_backend_unavailable" in extracted.error_reason
    assert extracted.metadata["pdf_ocr_fallback_attempted"] is True


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


def make_docx_with_media_downloaded() -> DownloadedAttachment:
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="{WORD_NS}">
  <w:body>
    <w:p><w:r><w:t>内嵌图片材料</w:t></w:r></w:p>
  </w:body>
</w:document>
"""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("word/document.xml", xml)
        archive.writestr("word/media/image1.png", b"fake image bytes")
    data = buffer.getvalue()
    ref = AttachmentRef(
        index=7,
        name="含图材料.docx",
        url="https://example.test/media.docx",
        extension="docx",
        section_path="申请材料 > 图示材料",
        section_title="申请材料",
        node_title="图示材料",
        parent_section_id="doc-media#申请材料",
        material_name="图示材料",
        role="sample_file",
    )
    return DownloadedAttachment(
        ref=ref,
        data=data,
        content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        cache_path="",
        sha256=hashlib.sha256(data).hexdigest(),
    )


def make_pdf_downloaded(data: bytes = b"%PDF-1.4 fake") -> DownloadedAttachment:
    ref = AttachmentRef(
        index=0,
        name="示例样表-取水许可申请书.pdf",
        url="https://example.test/form.pdf",
        extension="pdf",
        section_path="取水许可办事指南 > 申请材料 > 5. 取水许可申请书",
        section_title="申请材料",
        node_title="5. 取水许可申请书",
        parent_section_id="doc-water#申请材料",
        material_name="取水许可申请书",
        role="sample_file",
    )
    return DownloadedAttachment(
        ref=ref,
        data=data,
        content_type="application/pdf",
        cache_path="",
        sha256=hashlib.sha256(data).hexdigest(),
    )


def fake_fitz_module():
    class Matrix:
        def __init__(self, x_scale, y_scale):
            self.x_scale = x_scale
            self.y_scale = y_scale

    class Pixmap:
        def tobytes(self, image_type):
            assert image_type == "png"
            return b"fake png bytes"

    class Page:
        def get_pixmap(self, *, matrix, alpha):
            assert matrix.x_scale == 2
            assert matrix.y_scale == 2
            assert alpha is False
            return Pixmap()

    class Document:
        def __len__(self):
            return 3

        def load_page(self, page_index):
            assert page_index in {0, 1}
            return Page()

        def close(self):
            pass

    return types.SimpleNamespace(Matrix=Matrix, open=lambda **kwargs: Document())
