"""Docling 文档解析器适配。"""

from __future__ import annotations

import hashlib
from typing import Any

from declaration_qa.indexing.document_parsing.models import DocumentParseError, ParsedDocument
from declaration_qa.indexing.document_parsing.structured import (
    metadata_summary,
    object_to_plain_data,
    parsed_document_from_markdown,
    temp_document_file,
)
from declaration_qa.indexing.document_parsing.text import (
    compact_table_context,
    html_table,
    table_json,
    table_text,
)
from declaration_qa.indexing.document_parsing.table_semantics import (
    build_table_semantic_chunk_specs,
    coerce_table_index,
    table_semantic_id_prefix,
)


class DoclingDocumentParser:
    """基于 Docling 的本地结构化解析器。"""

    name = "docling"

    def __init__(
        self,
        *,
        document_timeout_seconds: int = 3600,
        accelerator_device: str = "auto",
        accelerator_num_threads: int = 4,
        layout_batch_size: int = 4,
        table_batch_size: int = 4,
        ocr_batch_size: int = 4,
    ) -> None:
        """初始化 Docling，并设置单个 PDF 的处理时限。"""
        self.document_timeout_seconds = max(1, int(document_timeout_seconds or 3600))
        self.accelerator_device = normalize_docling_accelerator_device(accelerator_device)
        self.accelerator_num_threads = max(1, int(accelerator_num_threads or 4))
        self.layout_batch_size = max(1, int(layout_batch_size or 4))
        self.table_batch_size = max(1, int(table_batch_size or 4))
        self.ocr_batch_size = max(1, int(ocr_batch_size or 4))

    def parse(self, *, filename: str, content_type: str, data: bytes) -> ParsedDocument:
        """调用 Docling 并转换为标准解析结果。"""
        try:
            from docling.document_converter import DocumentConverter
        except ImportError as exc:
            raise DocumentParseError("docling is not installed") from exc

        with temp_document_file(filename=filename, data=data) as path:
            try:
                result = self._document_converter(filename=filename, content_type=content_type).convert(str(path))
                document = getattr(result, "document", result)
                markdown = export_document_markdown(document)
                structured_payload = object_to_plain_data(document)
                chunk_specs = docling_chunk_specs(document)
            except Exception as exc:
                raise DocumentParseError(f"Docling parse failed: {type(exc).__name__}: {str(exc)[:240]}") from exc
        if not markdown.strip():
            raise DocumentParseError("Docling returned empty markdown")
        return parsed_document_from_markdown(
            filename=filename,
            content_type=content_type,
            parser_name=self.name,
            markdown=markdown,
            metadata={
                "parser_output_format": "markdown",
                "docling_chunker": "hybrid",
                "docling_chunk_count": len(chunk_specs),
                "docling_chunk_specs": chunk_specs,
                **metadata_summary(structured_payload),
            },
        )

    def _document_converter(self, *, filename: str, content_type: str):
        """为 PDF/图片创建带显式 pipeline 配置的转换器，其他格式使用默认转换器。"""
        from docling.document_converter import DocumentConverter

        suffix = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
        normalized_content_type = str(content_type or "").lower()
        is_pdf = suffix == "pdf" or normalized_content_type == "application/pdf"
        is_image = suffix in {"png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"} or normalized_content_type.startswith("image/")
        if not is_pdf and not is_image:
            return DocumentConverter()
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import AcceleratorOptions, PdfPipelineOptions
        from docling.document_converter import ImageFormatOption, PdfFormatOption

        pipeline_options = PdfPipelineOptions(
            document_timeout=float(self.document_timeout_seconds),
            accelerator_options=AcceleratorOptions(
                num_threads=self.accelerator_num_threads,
                device=self.accelerator_device,
            ),
            layout_batch_size=self.layout_batch_size,
            table_batch_size=self.table_batch_size,
            ocr_batch_size=self.ocr_batch_size,
        )
        format_options = {}
        if is_pdf:
            format_options[InputFormat.PDF] = PdfFormatOption(pipeline_options=pipeline_options)
        if is_image:
            format_options[InputFormat.IMAGE] = ImageFormatOption(pipeline_options=pipeline_options)
        return DocumentConverter(
            format_options=format_options
        )


def export_document_markdown(document: Any) -> str:
    """从 Docling 文档对象导出 Markdown。"""
    for method_name in ("export_to_markdown", "export_to_text"):
        method = getattr(document, method_name, None)
        if callable(method):
            try:
                return str(method() or "")
            except Exception:
                continue
    return str(document or "")


def docling_chunk_specs(document: Any) -> list[dict[str, Any]]:
    """使用 Docling HybridChunker 生成主文档 chunk 描述。"""
    try:
        from docling_core.transforms.chunker import HybridChunker
    except ImportError as exc:
        raise DocumentParseError("docling-core chunker is not installed") from exc
    chunker = HybridChunker()
    table_lookup = docling_table_lookup(document)
    specs: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunker.chunk(document)):
        content = normalize_docling_chunk_text(chunker.serialize(chunk))
        if not content:
            continue
        metadata = docling_chunk_metadata(chunk)
        metadata.update(docling_table_metadata(metadata, table_lookup))
        semantic_type = docling_semantic_chunk_type(metadata)
        if semantic_type == "docling_table" and metadata.get("table_json"):
            specs.extend(docling_table_chunk_specs(index=index, content=content, metadata=metadata))
            continue
        specs.append(
            {
                "id_suffix": docling_chunk_suffix(index, content, metadata),
                "chunk_type": "child",
                "content": content,
                "metadata": {
                    **metadata,
                    "chunk_type": semantic_type,
                    "chunking_strategy": "docling",
                    "docling_chunker": "hybrid",
                    "docling_chunk_index": index,
                    "indexable": True,
                },
            }
        )
    return specs


def docling_table_chunk_specs(*, index: int, content: str, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """为 Docling 原生表格构建语义化父子 chunk。"""
    table_index = coerce_table_index(metadata.get("table_index"), index)
    id_prefix = table_semantic_id_prefix(
        "docling",
        table_index,
        metadata.get("section_path"),
        ",".join(metadata.get("docling_item_refs") or []),
        content[:300],
    )
    base_metadata = {
        **metadata,
        "chunking_strategy": "docling",
        "docling_chunker": "hybrid",
        "docling_chunk_index": index,
        "source_table_chunk_type": "docling_table",
    }
    source_lines = [
        "主文档表格",
        f"所属章节：{metadata.get('section_path') or 'Document'}",
        f"Docling chunk：{index}",
    ]
    return build_table_semantic_chunk_specs(
        id_prefix=id_prefix,
        table_index=table_index,
        table_json_value=metadata.get("table_json"),
        table_text_value=str(metadata.get("table_text") or content or ""),
        table_html_value=str(metadata.get("table_html") or ""),
        base_metadata=base_metadata,
        source_lines=source_lines,
        table_source=str(metadata.get("table_source") or "docling_native"),
    )


def docling_table_lookup(document: Any) -> dict[str, dict[str, Any]]:
    """从 DoclingDocument 中抽取原生表格结构。"""
    output: dict[str, dict[str, Any]] = {}
    for table_index, table in enumerate(list(getattr(document, "tables", None) or [])):
        rows = docling_table_rows(table)
        if not rows:
            continue
        cells = docling_table_cells(table)
        json_value = table_json(
            rows,
            cells=cells,
            title=str(getattr(table, "caption_text", "") or ""),
            page_no=docling_table_page_no(table),
            bbox=docling_table_bbox(table),
        )
        output[str(getattr(table, "self_ref", "") or f"#/tables/{table_index}")] = {
            "table_index": table_index,
            "table_text": table_text(rows, title=str(getattr(table, "caption_text", "") or "")),
            "table_html": html_table(rows),
            "table_json": json_value,
            "llm_table_context": compact_table_context(json_value),
            "table_row_count": len(rows),
            "table_column_count": max((len(row) for row in rows), default=0),
        }
    return output


def docling_table_rows(table: Any) -> list[list[str]]:
    """把 Docling TableItem 转成二维文本行。"""
    data = getattr(table, "data", None)
    grid = getattr(data, "grid", None)
    rows = []
    for row in list(grid or []):
        cells = [str(getattr(cell, "text", "") or "").strip() for cell in list(row or [])]
        if any(cells):
            rows.append(cells)
    return rows


def docling_table_cells(table: Any) -> list[dict[str, Any]]:
    """提取 Docling 表格单元格坐标和跨行跨列信息。"""
    data = getattr(table, "data", None)
    cells = []
    for cell in list(getattr(data, "table_cells", None) or []):
        cells.append(
            {
                "row": int(getattr(cell, "start_row_offset_idx", 0) or 0),
                "col": int(getattr(cell, "start_col_offset_idx", 0) or 0),
                "rowspan": int(getattr(cell, "row_span", 1) or 1),
                "colspan": int(getattr(cell, "col_span", 1) or 1),
                "text": str(getattr(cell, "text", "") or "").strip(),
                "column_header": bool(getattr(cell, "column_header", False)),
                "row_header": bool(getattr(cell, "row_header", False)),
                "bbox": object_to_plain_data(getattr(cell, "bbox", None)),
            }
        )
    return cells


def docling_table_page_no(table: Any) -> int | None:
    """返回 Docling 表格所在页码。"""
    for prov in list(getattr(table, "prov", None) or []):
        try:
            number = int(getattr(prov, "page_no", 0) or 0)
        except (TypeError, ValueError):
            continue
        if number > 0:
            return number
    return None


def docling_table_bbox(table: Any) -> list[float] | None:
    """返回 Docling 表格 bbox。"""
    for prov in list(getattr(table, "prov", None) or []):
        bbox = object_to_plain_data(getattr(prov, "bbox", None))
        if isinstance(bbox, list):
            return bbox
    return None


def docling_table_metadata(metadata: dict[str, Any], table_lookup: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """根据 Docling item ref 为表格 chunk 补充结构化表格元数据。"""
    refs = list(metadata.get("docling_item_refs") or [])
    table_refs = [ref for ref in refs if ref in table_lookup]
    if not table_refs:
        return {}
    table_payload = table_lookup[table_refs[0]]
    return {
        "table_index": table_payload["table_index"],
        "table_ref": table_refs[0],
        "table_text": table_payload["table_text"],
        "table_html": table_payload["table_html"],
        "table_json": table_payload["table_json"],
        "llm_table_context": table_payload["llm_table_context"],
        "table_row_count": table_payload["table_row_count"],
        "table_column_count": table_payload["table_column_count"],
        "table_source": "docling_native",
    }


def docling_chunk_metadata(chunk: Any) -> dict[str, Any]:
    """提取 Docling chunk 的紧凑元数据。"""
    meta = getattr(chunk, "meta", None)
    headings = [str(item).strip() for item in list(getattr(meta, "headings", None) or []) if str(item).strip()]
    doc_items = list(getattr(meta, "doc_items", None) or [])
    labels = unique_ordered_strings(docling_item_label(item) for item in doc_items)
    refs = unique_ordered_strings(docling_item_ref(item) for item in doc_items)
    page_numbers = unique_page_numbers(doc_items)
    metadata: dict[str, Any] = {
        "section_path": " > ".join(headings) if headings else "Document",
        "heading_path": headings,
        "section_title": headings[-2] if len(headings) >= 2 else (headings[-1] if headings else "Document"),
        "node_title": headings[-1] if headings else "Document",
        "parent_section_id": "docling:" + hashlib.sha1(" > ".join(headings).encode("utf-8")).hexdigest()[:12],
        "docling_labels": labels,
        "docling_item_refs": refs[:20],
    }
    if page_numbers:
        metadata.update(
            {
                "attachment_page_numbers": page_numbers,
                "attachment_page_start": min(page_numbers),
                "attachment_page_end": max(page_numbers),
                "attachment_page_range": ",".join(str(item) for item in page_numbers),
            }
        )
    return metadata


def docling_item_label(item: Any) -> str:
    """返回 Docling doc item 的标签。"""
    label = getattr(item, "label", "")
    return str(getattr(label, "value", label) or "")


def docling_item_ref(item: Any) -> str:
    """返回 Docling doc item 的 self_ref。"""
    return str(getattr(item, "self_ref", "") or "")


def unique_page_numbers(doc_items: list[Any]) -> list[int]:
    """从 Docling provenance 中提取页码。"""
    output: list[int] = []
    for item in doc_items:
        for prov in list(getattr(item, "prov", None) or []):
            page_no = getattr(prov, "page_no", None)
            try:
                number = int(page_no)
            except (TypeError, ValueError):
                continue
            if number > 0 and number not in output:
                output.append(number)
    return output


def docling_semantic_chunk_type(metadata: dict[str, Any]) -> str:
    """根据 Docling 标签推断语义 chunk 类型。"""
    labels = set(metadata.get("docling_labels") or [])
    if "table" in labels:
        return "docling_table"
    if "picture" in labels:
        return "docling_picture"
    if "list_item" in labels:
        return "docling_list"
    return "docling_text"


def docling_chunk_suffix(index: int, content: str, metadata: dict[str, Any]) -> str:
    """生成稳定的 Docling chunk 后缀。"""
    key = "|".join(
        [
            str(index),
            str(metadata.get("section_path") or ""),
            ",".join(metadata.get("docling_item_refs") or []),
            content[:500],
        ]
    )
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    return f"docling:{index}:{digest}"


def normalize_docling_chunk_text(text: str) -> str:
    """清理 Docling chunk 序列化文本。"""
    return "\n".join(line.rstrip() for line in str(text or "").splitlines() if line.strip()).strip()


def normalize_docling_accelerator_device(value: str) -> str:
    """标准化 Docling accelerator device。"""
    normalized = str(value or "auto").strip().lower().replace("_", "-")
    aliases = {
        "gpu": "cuda",
        "nvidia": "cuda",
        "default": "auto",
        "": "auto",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"auto", "cpu", "cuda", "mps", "xpu"}:
        return "auto"
    return normalized


def unique_ordered_strings(values) -> list[str]:
    """按首次出现顺序返回非空字符串。"""
    seen = set()
    output = []
    for value in values:
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output
