"""项目内置文档解析器。"""

from __future__ import annotations

import csv
import io
import logging
import re
import zipfile
from datetime import date, datetime, time
from xml.etree import ElementTree

from gov_service_agent.indexing.document_parsing.models import DocumentParseError, ParsedBlock, ParsedDocument
from gov_service_agent.indexing.document_parsing.ocr import HttpOcrClient
from gov_service_agent.indexing.document_parsing.text import (
    clean_inline_text,
    decode_text,
    escape_table_cell,
    html_table,
    markdown_table,
    markdown_table_row,
    normalize_extracted_text,
    normalized_table_rows,
    compact_table_context,
    table_json,
    table_text,
)

WORD_NAMESPACE = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
WORD_ATTR_VALUE = f"{{{WORD_NAMESPACE}}}val"


class NativeDocumentParser:
    """不依赖外部文档智能服务的内置解析器。"""

    name = "native"

    def __init__(self, *, ocr_client: HttpOcrClient | None = None) -> None:
        """初始化原生解析器。"""
        self.ocr_client = ocr_client

    def parse(self, *, filename: str, content_type: str, data: bytes) -> ParsedDocument:
        """根据文件类型解析上传文档文本。"""
        if not data:
            raise DocumentParseError("Uploaded file is empty")
        suffix = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
        normalized_content_type = (content_type or "").lower()

        blocks: list[ParsedBlock] = []
        if suffix == "csv" or normalized_content_type in {"text/csv", "application/csv"}:
            text, blocks = parse_csv(data)
        elif suffix in {"txt", "md", "markdown", "json", "xml"} or normalized_content_type.startswith("text/"):
            text = decode_text(data)
            blocks = [ParsedBlock(block_type="text", text=text)]
        elif suffix == "pdf" or normalized_content_type == "application/pdf":
            text, blocks = parse_pdf(data)
        elif suffix == "docx" or normalized_content_type.endswith("wordprocessingml.document"):
            text, blocks = parse_docx(data)
        elif is_spreadsheet_file(suffix, normalized_content_type):
            text, blocks = parse_spreadsheet(data)
        elif suffix in {"png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"} or normalized_content_type.startswith("image/"):
            if not self.ocr_client:
                raise DocumentParseError("Image OCR client is not configured")
            text = self.ocr_client.extract_text(filename=filename, content_type=content_type, data=data)
            blocks = [ParsedBlock(block_type="image_ocr", text=text, metadata={"source_modality": "image"})]
        else:
            raise DocumentParseError(f"Unsupported document type: {filename or content_type}")

        normalized = normalize_extracted_text(text)
        blocks = normalize_blocks(blocks)
        if not normalized:
            raise DocumentParseError("No text could be extracted from uploaded file")
        return ParsedDocument(
            filename=filename,
            content_type=normalized_content_type,
            text=normalized,
            parser_name=self.name,
            blocks=blocks,
            metadata=parse_metadata(self.name, blocks),
        )


def parse_csv(data: bytes) -> tuple[str, list[ParsedBlock]]:
    """将 CSV 内容解析为可检索文本和表格块。"""
    rows = [[cell.strip() for cell in row] for row in csv.reader(io.StringIO(decode_text(data)))]
    normalized_rows = normalized_table_rows(rows)
    text = markdown_table(normalized_rows)
    return text, table_blocks(normalized_rows, source="csv")


def parse_pdf(data: bytes) -> tuple[str, list[ParsedBlock]]:
    """从 PDF 字节中提取页面文本，并保留页码边界。"""
    suppressed_loggers = [logging.getLogger(name) for name in ("pypdf", "pypdf._cmap")]
    previous_levels = [(logger, logger.level) for logger in suppressed_loggers]
    for logger in suppressed_loggers:
        logger.setLevel(logging.CRITICAL)
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        page_texts = []
        blocks: list[ParsedBlock] = []
        for page_no, page in enumerate(reader.pages, start=1):
            text = extract_pdf_page_text(page)
            if text.strip():
                page_text = f"## PDF 第 {page_no} 页\n{text.strip()}"
                page_texts.append(page_text)
                blocks.append(ParsedBlock(block_type="page_text", text=page_text, page_no=page_no))
        return "\n\n".join(page_texts), blocks
    except DocumentParseError:
        raise
    except Exception as exc:
        raise DocumentParseError("Invalid PDF document") from exc
    finally:
        for logger, level in previous_levels:
            logger.setLevel(level)


def extract_pdf_page_text(page) -> str:
    """优先使用 pypdf layout 模式，兼容不支持该参数的版本。"""
    try:
        text = page.extract_text(extraction_mode="layout") or ""
    except TypeError:
        text = page.extract_text() or ""
    except Exception:
        text = page.extract_text() or ""
    if not text.strip():
        text = page.extract_text() or ""
    return str(text)


def parse_docx(data: bytes) -> tuple[str, list[ParsedBlock]]:
    """从 DOCX 文档中提取结构化 Markdown 文本。"""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            xml = archive.read("word/document.xml")
    except (KeyError, zipfile.BadZipFile) as exc:
        raise DocumentParseError("Invalid DOCX document") from exc
    try:
        root = ElementTree.fromstring(xml)
    except ElementTree.ParseError as exc:
        raise DocumentParseError("Invalid DOCX document") from exc

    text_blocks = []
    parsed_blocks: list[ParsedBlock] = []
    table_index = 0
    for block in iter_docx_body_blocks(root):
        if local_name(block.tag) == "p":
            text = format_docx_paragraph(block)
            if text:
                text_blocks.append(text)
                parsed_blocks.append(ParsedBlock(block_type="text", text=text))
        elif local_name(block.tag) == "tbl":
            rows = docx_table_rows(block)
            table = markdown_table(rows)
            if table:
                text_blocks.append(table)
                parsed_blocks.extend(table_blocks(rows, source="docx", table_index=table_index))
                table_index += 1
    return "\n\n".join(text_blocks), parsed_blocks


def iter_docx_body_blocks(root):
    """按正文顺序遍历 DOCX 段落和表格。"""
    body = next((node for node in root.iter() if local_name(node.tag) == "body"), None)
    if body is None:
        return
    for child in body:
        if local_name(child.tag) in {"p", "tbl"}:
            yield child


def format_docx_paragraph(paragraph: ElementTree.Element) -> str:
    """将 DOCX 段落转换成适合层级切块的 Markdown 行。"""
    text = clean_inline_text(docx_paragraph_text(paragraph))
    if not text:
        return ""
    style = docx_paragraph_style(paragraph)
    heading_level = infer_docx_heading_level(text, style)
    if heading_level:
        return f"{'#' * heading_level} {text}"
    if docx_paragraph_is_numbered(paragraph) and not text.startswith(("-", "•")):
        return f"- {text}"
    return text


def docx_table_rows(table: ElementTree.Element) -> list[list[str]]:
    """将 DOCX 表格转换为二维单元格。"""
    rows = []
    for row in direct_children(table, "tr"):
        cells = []
        for cell in direct_children(row, "tc"):
            texts = [
                clean_inline_text(docx_paragraph_text(paragraph))
                for paragraph in cell.iter()
                if local_name(paragraph.tag) == "p"
            ]
            cells.append(escape_table_cell(" ".join(text for text in texts if text)))
        if any(cells):
            rows.append(cells)
    return normalized_table_rows(rows)


def docx_paragraph_text(paragraph: ElementTree.Element) -> str:
    """读取段落内文本，同时保留制表符和换行符。"""
    parts = []
    for node in paragraph.iter():
        name = local_name(node.tag)
        if name == "t":
            parts.append(node.text or "")
        elif name == "tab":
            parts.append("\t")
        elif name in {"br", "cr"}:
            parts.append("\n")
    return "".join(parts)


def docx_paragraph_style(paragraph: ElementTree.Element) -> str:
    """读取段落样式名称。"""
    for prop in paragraph.iter():
        if local_name(prop.tag) == "pStyle":
            return str(prop.attrib.get(WORD_ATTR_VALUE) or "")
    return ""


def docx_paragraph_is_numbered(paragraph: ElementTree.Element) -> bool:
    """判断段落是否为 Word 编号/项目符号列表。"""
    for prop in paragraph.iter():
        if local_name(prop.tag) == "numPr":
            return True
    return False


def infer_docx_heading_level(text: str, style: str) -> int:
    """结合 Word 样式和政务公文编号推断标题层级。"""
    normalized_style = style.replace(" ", "").replace("_", "").replace("-", "").lower()
    match = re.search(r"heading([1-6])", normalized_style)
    if match:
        return int(match.group(1))
    if normalized_style in {"title", "标题"}:
        return 1
    if re.match(r"^第[一二三四五六七八九十百千万0-9]+[章节部分]", text):
        return 1
    if len(text) <= 80 and re.match(r"^[一二三四五六七八九十百千万]+[、.．]", text):
        return 2
    if len(text) <= 80 and re.match(r"^[（(][一二三四五六七八九十百千万]+[）)]", text):
        return 3
    return 0


def direct_children(node: ElementTree.Element, name: str) -> list[ElementTree.Element]:
    """读取指定本地标签名的直接子节点。"""
    return [child for child in node if local_name(child.tag) == name]


def local_name(tag: str) -> str:
    """返回 XML QName 的本地标签名。"""
    return tag.rsplit("}", 1)[-1]


def is_spreadsheet_file(suffix: str, content_type: str) -> bool:
    """判断上传文件是否是表格文档。"""
    if suffix in {"xls", "xlsx", "xlsm", "ods"}:
        return True
    return content_type in {
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-excel.sheet.macroenabled.12",
        "application/vnd.oasis.opendocument.spreadsheet",
    } or content_type.endswith("spreadsheetml.sheet")


def parse_spreadsheet(data: bytes) -> tuple[str, list[ParsedBlock]]:
    """通过 python-calamine 从表格文档中提取工作表内容。"""
    try:
        from python_calamine import load_workbook
    except ImportError as exc:
        raise DocumentParseError("python-calamine is required to parse spreadsheet documents") from exc

    workbook = None
    try:
        workbook = load_workbook(io.BytesIO(data))
        blocks = []
        parsed_blocks: list[ParsedBlock] = []
        for sheet_index, sheet_name in enumerate(workbook.sheet_names):
            sheet = workbook.get_sheet_by_name(sheet_name)
            table, rows = format_spreadsheet_table(sheet.to_python())
            if table:
                title = clean_inline_text(str(sheet_name))
                blocks.append(f"## 工作表：{title}\n\n{table}")
                parsed_blocks.extend(
                    table_blocks(
                        rows,
                        source="spreadsheet",
                        table_index=sheet_index,
                        metadata={"sheet_name": title},
                    )
                )
        return "\n\n".join(blocks), parsed_blocks
    except Exception as exc:
        raise DocumentParseError("Invalid spreadsheet document") from exc
    finally:
        if workbook is not None and hasattr(workbook, "close"):
            workbook.close()


def format_spreadsheet_table(rows: list[list[object]]) -> tuple[str, list[list[str]]]:
    """将工作表行列内容转换为标准 Markdown 表格。"""
    normalized_rows = normalize_spreadsheet_rows(rows)
    if not normalized_rows:
        return "", []

    title = ""
    first_row_non_empty = non_empty_cells(normalized_rows[0])
    if len(first_row_non_empty) == 1 and spreadsheet_title_like(first_row_non_empty[0]):
        title = first_row_non_empty[0]
        normalized_rows = normalized_rows[1:]

    if not normalized_rows:
        return (f"### {title}" if title else ""), []

    table = markdown_table(normalized_rows)
    if title:
        table = f"### {title}\n\n{table}"
    return table, normalized_rows


def normalize_spreadsheet_rows(rows: list[list[object]]) -> list[list[str]]:
    """清理工作表行列空白，同时保留行内列位置。"""
    cleaned_rows = []
    for row in rows:
        cells = [spreadsheet_cell_text(cell) for cell in row]
        if any(cells):
            cleaned_rows.append(cells)
    return normalized_table_rows(cleaned_rows)


def spreadsheet_cell_text(value: object) -> str:
    """将单元格值转换为适合检索的文本。"""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, date):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, time):
        return value.strftime("%H:%M:%S")
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return escape_table_cell(str(value).replace("\xa0", " "))


def non_empty_cells(row: list[str]) -> list[str]:
    """返回一行里的非空单元格。"""
    return [cell for cell in row if cell.strip()]


def spreadsheet_title_like(text: str) -> bool:
    """判断单单元格行是否像表格标题。"""
    normalized = clean_inline_text(text)
    return len(normalized) <= 80 and any(token in normalized for token in ["表", "申请书", "报告", "协议书"])


def table_blocks(
    rows: list[list[str]],
    *,
    source: str,
    table_index: int = 0,
    metadata: dict[str, object] | None = None,
) -> list[ParsedBlock]:
    """将表格二维结构转换为标准表格块。"""
    if not rows:
        return []
    json_value = table_json(rows)
    return [
        ParsedBlock(
            block_type="table",
            text=table_text(rows),
            html=html_table(rows),
            json_value=json_value,
            metadata={
                "table_source": source,
                "table_index": table_index,
                "row_count": len(rows),
                "column_count": max(len(row) for row in rows),
                "table_text_format": "natural_language",
                "llm_table_context": compact_table_context(json_value),
                **(metadata or {}),
            },
        )
    ]


def normalize_blocks(blocks: list[ParsedBlock]) -> list[ParsedBlock]:
    """清理 block 文本，剔除空内容。"""
    normalized_blocks = []
    for block in blocks:
        text = normalize_extracted_text(block.text)
        if not text and not block.html and block.json_value in (None, "", [], {}):
            continue
        normalized_blocks.append(
            ParsedBlock(
                block_type=block.block_type,
                text=text,
                html=block.html,
                json_value=block.json_value,
                page_no=block.page_no,
                bbox=block.bbox,
                metadata=block.metadata,
            )
        )
    return normalized_blocks


def parse_metadata(parser_name: str, blocks: list[ParsedBlock]) -> dict[str, object]:
    """生成统一解析诊断元数据。"""
    table_count = sum(1 for block in blocks if block.block_type == "table")
    return {
        "document_parser": parser_name,
        "parsed_block_count": len(blocks),
        "parsed_table_count": table_count,
        "parsed_has_structured_tables": table_count > 0,
    }
