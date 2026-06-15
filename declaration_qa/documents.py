from __future__ import annotations

import csv
import io
import re
import zipfile
from dataclasses import dataclass
from html import unescape
from xml.etree import ElementTree

import requests


class DocumentParseError(ValueError):
    pass


@dataclass(frozen=True)
class ParsedDocument:
    filename: str
    content_type: str
    text: str


class HttpOcrClient:
    def __init__(self, *, url: str, api_key: str = "", timeout_seconds: int = 30) -> None:
        self.url = url
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def extract_text(self, *, filename: str, content_type: str, data: bytes) -> str:
        if not self.url:
            raise DocumentParseError("Image OCR requires DECLARATION_QA_OCR_API_URL")
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = requests.post(
            self.url,
            headers=headers,
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
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def parse_csv(data: bytes) -> str:
    text = decode_text(data)
    rows = csv.reader(io.StringIO(text))
    return "\n".join(" | ".join(cell.strip() for cell in row) for row in rows)


def parse_pdf(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def parse_docx(data: bytes) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            xml = archive.read("word/document.xml")
    except (KeyError, zipfile.BadZipFile) as exc:
        raise DocumentParseError("Invalid DOCX document") from exc
    root = ElementTree.fromstring(xml)
    texts = [node.text or "" for node in root.iter() if node.tag.endswith("}t")]
    return "\n".join(texts)


def parse_xlsx(data: bytes) -> str:
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
    text = unescape(str(text or ""))
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
