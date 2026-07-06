"""上传文档的文本抽取和格式解析工具。"""

from __future__ import annotations

import csv
import io
import re
import zipfile
from dataclasses import dataclass
from html import unescape
from xml.etree import ElementTree

import requests

WORD_NAMESPACE = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
WORD_ATTR_VALUE = f"{{{WORD_NAMESPACE}}}val"


class DocumentParseError(ValueError):
    """文档解析失败时抛出的业务异常。"""
    pass


@dataclass(frozen=True)
class ParsedDocument:
    """上传文档解析后的标准文本结果。"""
    filename: str
    content_type: str
    text: str


class HttpOcrClient:
    """通过外部 HTTP OCR 服务提取图片文本。"""
    def __init__(self, *, url: str, api_key: str = "", model: str = "", timeout_seconds: int = 30) -> None:
        """初始化HttpOcrClient。"""
        self.url = url
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds

    def extract_text(self, *, filename: str, content_type: str, data: bytes) -> str:
        """调用 OCR 服务从文件字节中提取文本。"""
        if not self.url:
            raise DocumentParseError("Image OCR requires DECLARATION_QA_OCR_API_URL")
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        form_data = {"model": self.model} if self.model else None
        response = requests.post(
            self.url,
            headers=headers,
            data=form_data,
            files={"file": (filename, data, content_type or "application/octet-stream")},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict):
            data_payload = payload.get("data") if isinstance(payload.get("data"), dict) else payload
            text = data_payload.get("text") or data_payload.get("content") or data_payload.get("result")
            if text:
                return str(text)
        raise DocumentParseError("OCR response does not contain extracted text")


def parse_document_bytes(
    *,
    filename: str,
    content_type: str,
    data: bytes,
    ocr_client: HttpOcrClient | None = None,
) -> ParsedDocument:
    """根据文件类型解析上传文档文本。"""
    if not data:
        raise DocumentParseError("Uploaded file is empty")
    suffix = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    content_type = (content_type or "").lower()

    if suffix in {"txt", "md", "markdown", "json", "xml"} or content_type.startswith("text/"):
        text = decode_text(data)
    elif suffix == "csv" or content_type in {"text/csv", "application/csv"}:
        text = parse_csv(data)
    elif suffix == "pdf" or content_type == "application/pdf":
        text = parse_pdf(data)
    elif suffix == "docx" or content_type.endswith("wordprocessingml.document"):
        text = parse_docx(data)
    elif suffix == "xlsx" or content_type.endswith("spreadsheetml.sheet"):
        text = parse_xlsx(data)
    elif suffix in {"png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"} or content_type.startswith("image/"):
        if not ocr_client:
            raise DocumentParseError("Image OCR client is not configured")
        text = ocr_client.extract_text(filename=filename, content_type=content_type, data=data)
    else:
        raise DocumentParseError(f"Unsupported document type: {filename or content_type}")

    normalized = normalize_extracted_text(text)
    if not normalized:
        raise DocumentParseError("No text could be extracted from uploaded file")
    return ParsedDocument(filename=filename, content_type=content_type, text=normalized)


def decode_text(data: bytes) -> str:
    """按常见编码解码文本文件内容。"""
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def parse_csv(data: bytes) -> str:
    """将 CSV 内容解析为可检索文本。"""
    text = decode_text(data)
    rows = csv.reader(io.StringIO(text))
    return "\n".join(" | ".join(cell.strip() for cell in row) for row in rows)


def parse_pdf(data: bytes) -> str:
    """从 PDF 字节中提取页面文本，并保留页码边界。"""
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        pages = []
        for page_no, page in enumerate(reader.pages, start=1):
            text = extract_pdf_page_text(page)
            if text.strip():
                pages.append(f"## PDF 第 {page_no} 页\n{text.strip()}")
        return "\n\n".join(pages)
    except DocumentParseError:
        raise
    except Exception as exc:
        raise DocumentParseError("Invalid PDF document") from exc


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


def parse_docx(data: bytes) -> str:
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

    blocks = []
    for block in iter_docx_body_blocks(root):
        if local_name(block.tag) == "p":
            text = format_docx_paragraph(block)
            if text:
                blocks.append(text)
        elif local_name(block.tag) == "tbl":
            table = format_docx_table(block)
            if table:
                blocks.append(table)
    return "\n\n".join(blocks)


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


def format_docx_table(table: ElementTree.Element) -> str:
    """将 DOCX 表格转换为 Markdown 表格，保留行列关系。"""
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
    if not rows:
        return ""

    width = max(len(row) for row in rows)
    padded_rows = [row + [""] * (width - len(row)) for row in rows]
    lines = [markdown_table_row(padded_rows[0])]
    lines.append(markdown_table_row(["---"] * width))
    lines.extend(markdown_table_row(row) for row in padded_rows[1:])
    return "\n".join(lines)


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


def clean_inline_text(text: str) -> str:
    """清理段落内多余空白。"""
    text = re.sub(r"[ \t\r\f\v]+", " ", text or "")
    text = re.sub(r"\n+", " ", text)
    return text.strip()


def escape_table_cell(text: str) -> str:
    """转义 Markdown 表格单元格。"""
    return clean_inline_text(text).replace("|", "\\|")


def markdown_table_row(values: list[str]) -> str:
    """拼接 Markdown 表格行。"""
    return "| " + " | ".join(values) + " |"


def parse_xlsx(data: bytes) -> str:
    """从 XLSX 工作表中提取单元格文本。"""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            shared_strings = read_xlsx_shared_strings(archive)
            sheet_names = [name for name in archive.namelist() if name.startswith("xl/worksheets/sheet") and name.endswith(".xml")]
            lines = []
            for sheet_name in sheet_names:
                root = ElementTree.fromstring(archive.read(sheet_name))
                for row in root.iter():
                    if not row.tag.endswith("}row"):
                        continue
                    values = []
                    for cell in row:
                        if not cell.tag.endswith("}c"):
                            continue
                        cell_type = cell.attrib.get("t", "")
                        value = next((node.text for node in cell if node.tag.endswith("}v")), "") or ""
                        if cell_type == "s" and value.isdigit():
                            value = shared_strings[int(value)] if int(value) < len(shared_strings) else value
                        values.append(value)
                    if values:
                        lines.append(" | ".join(values))
    except (KeyError, zipfile.BadZipFile, ElementTree.ParseError) as exc:
        raise DocumentParseError("Invalid XLSX document") from exc
    return "\n".join(lines)


def read_xlsx_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    """读取 XLSX 共享字符串表。"""
    try:
        root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    values = []
    for item in root.iter():
        if item.tag.endswith("}si"):
            values.append("".join(node.text or "" for node in item.iter() if node.tag.endswith("}t")))
    return values


def normalize_extracted_text(text: str) -> str:
    """清理并压缩抽取文本中的空白字符。"""
    text = unescape(str(text or ""))
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
