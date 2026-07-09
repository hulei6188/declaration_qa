"""PDF 附件 OCR fallback。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from declaration_qa.indexing.documents import DocumentParseError, HttpOcrClient, ParsedBlock, normalize_extracted_text


@dataclass(frozen=True)
class PdfOcrResult:
    """PDF 页面 OCR 结果。"""

    text: str
    page_count: int
    rendered_page_count: int
    ocr_success_count: int
    ocr_failed_pages: list[dict[str, Any]] = field(default_factory=list)
    blocks: list[ParsedBlock] = field(default_factory=list)


def extract_pdf_pages_with_ocr(
    *,
    filename: str,
    data: bytes,
    ocr_client: HttpOcrClient | None,
    max_pages: int,
    dpi: int,
) -> PdfOcrResult:
    """将 PDF 页面渲染为图片后调用 OCR。"""
    if not ocr_client:
        raise DocumentParseError("pdf_ocr_client_not_configured")
    if max_pages <= 0:
        raise DocumentParseError("pdf_ocr_page_limit_is_zero")
    fitz = import_pymupdf()
    document = open_pdf(fitz, data)
    try:
        page_count = len(document)
        render_count = min(page_count, max_pages)
        page_texts: list[str] = []
        blocks: list[ParsedBlock] = []
        failed_pages: list[dict[str, Any]] = []
        for page_index in range(render_count):
            try:
                image_data = render_page_png(fitz, document, page_index, dpi=dpi)
                text = ocr_client.extract_text(
                    filename=page_image_filename(filename, page_index),
                    content_type="image/png",
                    data=image_data,
                )
                normalized = normalize_extracted_text(text)
                if normalized:
                    page_no = page_index + 1
                    page_text = f"## OCR PDF 第 {page_no} 页\n{normalized}"
                    page_texts.append(page_text)
                    blocks.append(ParsedBlock(block_type="page_image_ocr", text=page_text, page_no=page_no))
            except Exception as exc:
                failed_pages.append({"page": page_index + 1, "error": page_ocr_error(exc)})
        if not page_texts:
            raise DocumentParseError(pdf_ocr_no_text_reason(failed_pages))
        return PdfOcrResult(
            text="\n\n".join(page_texts),
            page_count=page_count,
            rendered_page_count=render_count,
            ocr_success_count=len(page_texts),
            ocr_failed_pages=failed_pages,
            blocks=blocks,
        )
    finally:
        close_pdf(document)


def import_pymupdf():
    """导入 PyMuPDF 渲染后端。"""
    try:
        import fitz

        return fitz
    except Exception as exc:
        raise DocumentParseError("pdf_render_backend_unavailable: install PyMuPDF") from exc


def open_pdf(fitz, data: bytes):
    """打开 PDF 字节流。"""
    try:
        return fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise DocumentParseError("pdf_render_failed") from exc


def render_page_png(fitz, document, page_index: int, *, dpi: int) -> bytes:
    """将单页 PDF 渲染成 PNG。"""
    scale = max(72, dpi) / 72
    page = document.load_page(page_index)
    pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    return pixmap.tobytes("png")


def close_pdf(document) -> None:
    """关闭 PDF 文档对象。"""
    close = getattr(document, "close", None)
    if callable(close):
        close()


def page_image_filename(filename: str, page_index: int) -> str:
    """构建传给 OCR 服务的页面图片文件名。"""
    stem = Path(filename or "document.pdf").stem or "document"
    return f"{stem}-page-{page_index + 1}.png"


def page_ocr_error(exc: Exception) -> str:
    """压缩单页 OCR 失败原因。"""
    message = str(exc or "").strip()
    prefix = type(exc).__name__
    if not message:
        return prefix
    return f"{prefix}: {message[:240]}"


def pdf_ocr_no_text_reason(failed_pages: list[dict[str, Any]]) -> str:
    """构建 PDF OCR 没有产出文本时的诊断原因。"""
    if not failed_pages:
        return "pdf_ocr_fallback_produced_no_text"
    details = "; ".join(f"page {item.get('page')}: {item.get('error')}" for item in failed_pages[:3])
    return f"pdf_ocr_fallback_produced_no_text: {details}"
