"""附件文本抽取和语义分类。"""

from __future__ import annotations

from pathlib import Path

from declaration_qa.indexing.attachments.assets import extract_visual_assets
from declaration_qa.indexing.attachments.models import DownloadedAttachment, ExtractedAttachment
from declaration_qa.indexing.attachments.pdf_ocr import extract_pdf_pages_with_ocr
from declaration_qa.indexing.documents import DocumentParseError, HttpOcrClient, HttpVisionSummaryClient, parse_document_bytes


def extract_attachment(
    downloaded: DownloadedAttachment,
    *,
    ocr_client: HttpOcrClient | None = None,
    vision_summary_client: HttpVisionSummaryClient | None = None,
    parser_provider: str = "auto",
    parser_order: str = "",
    parser_fallback_enabled: bool = True,
    llamaparse_api_key: str = "",
    llamaparse_base_url: str = "",
    llamaparse_result_type: str = "markdown",
    pdf_ocr_fallback_enabled: bool = True,
    pdf_ocr_max_pages: int = 8,
    pdf_ocr_dpi: int = 180,
    asset_dir: Path | None = None,
) -> ExtractedAttachment:
    """抽取附件文本，失败时返回可观测的失败结果。"""
    ref = downloaded.ref
    try:
        parsed = parse_document_bytes(
            filename=ref.name,
            content_type=downloaded.content_type,
            data=downloaded.data,
            ocr_client=ocr_client,
            parser_provider=parser_provider,
            parser_order=parser_order,
            parser_fallback_enabled=parser_fallback_enabled,
            llamaparse_api_key=llamaparse_api_key,
            llamaparse_base_url=llamaparse_base_url,
            llamaparse_result_type=llamaparse_result_type,
        )
        text = parsed.text
        quality_error = attachment_text_quality_error(text, ref.extension)
        if quality_error:
            if ref.extension.lower() == "pdf" and pdf_ocr_fallback_enabled:
                return extract_pdf_attachment_with_ocr(
                    downloaded,
                    ocr_client=ocr_client,
                    vision_summary_client=vision_summary_client,
                    native_error=quality_error,
                    max_pages=pdf_ocr_max_pages,
                    dpi=pdf_ocr_dpi,
                    asset_dir=asset_dir,
                )
            return failed_attachment(downloaded, reason=quality_error)
        metadata = analyze_extracted_attachment(text=text, extension=ref.extension, role=ref.role)
        metadata.update(parsed.metadata)
        assets, asset_metadata = safe_extract_visual_assets(
            downloaded,
            asset_dir=asset_dir,
            extracted_text=text,
            ocr_client=ocr_client,
            vision_summary_client=vision_summary_client,
            pdf_max_pages=pdf_ocr_max_pages,
            pdf_dpi=pdf_ocr_dpi,
        )
        metadata.update(asset_metadata)
        return ExtractedAttachment(
            ref=ref,
            text=text,
            content_type=parsed.content_type or downloaded.content_type,
            cache_path=downloaded.cache_path,
            sha256=downloaded.sha256,
            status="success",
            metadata=metadata,
            assets=assets,
            blocks=parsed.blocks,
        )
    except DocumentParseError as exc:
        if ref.extension.lower() == "pdf" and pdf_ocr_fallback_enabled:
            return extract_pdf_attachment_with_ocr(
                downloaded,
                ocr_client=ocr_client,
                vision_summary_client=vision_summary_client,
                native_error=str(exc) or type(exc).__name__,
                max_pages=pdf_ocr_max_pages,
                dpi=pdf_ocr_dpi,
                asset_dir=asset_dir,
            )
        return failed_attachment(downloaded, reason=str(exc) or type(exc).__name__)
    except Exception as exc:
        return failed_attachment(downloaded, reason=type(exc).__name__)


def extract_pdf_attachment_with_ocr(
    downloaded: DownloadedAttachment,
    *,
    ocr_client: HttpOcrClient | None,
    vision_summary_client: HttpVisionSummaryClient | None = None,
    native_error: str,
    max_pages: int,
    dpi: int,
    asset_dir: Path | None = None,
) -> ExtractedAttachment:
    """在 PDF 原生抽取不可用时使用页面 OCR。"""
    try:
        result = extract_pdf_pages_with_ocr(
            filename=downloaded.ref.name,
            data=downloaded.data,
            ocr_client=ocr_client,
            max_pages=max_pages,
            dpi=dpi,
        )
    except DocumentParseError as exc:
        return failed_attachment(
            downloaded,
            reason=f"{native_error}; {str(exc) or type(exc).__name__}",
            metadata={
                "native_extraction_error": native_error,
                "pdf_ocr_fallback_attempted": True,
                "pdf_ocr_error": str(exc) or type(exc).__name__,
            },
        )
    quality_error = attachment_text_quality_error(result.text, "pdf")
    if quality_error:
        return failed_attachment(
            downloaded,
            reason=f"{native_error}; pdf_ocr_{quality_error}",
            metadata={
                "native_extraction_error": native_error,
                "pdf_ocr_fallback_attempted": True,
                "pdf_ocr_error": quality_error,
            },
        )
    metadata = analyze_extracted_attachment(text=result.text, extension="pdf", role=downloaded.ref.role)
    metadata.update(
        {
            "extraction_method": "pdf_ocr_fallback",
            "native_extraction_error": native_error,
            "pdf_ocr_fallback_attempted": True,
            "pdf_page_count": result.page_count,
            "pdf_ocr_rendered_page_count": result.rendered_page_count,
            "pdf_ocr_success_count": result.ocr_success_count,
            "pdf_ocr_failed_pages": result.ocr_failed_pages,
        }
    )
    assets, asset_metadata = safe_extract_visual_assets(
        downloaded,
        asset_dir=asset_dir,
        extracted_text=result.text,
        ocr_client=ocr_client,
        vision_summary_client=vision_summary_client,
        pdf_max_pages=max_pages,
        pdf_dpi=dpi,
    )
    metadata.update(asset_metadata)
    return ExtractedAttachment(
        ref=downloaded.ref,
        text=result.text,
        content_type=downloaded.content_type,
        cache_path=downloaded.cache_path,
        sha256=downloaded.sha256,
        status="success",
        metadata=metadata,
        assets=assets,
        blocks=result.blocks,
    )


def safe_extract_visual_assets(
    downloaded: DownloadedAttachment,
    *,
    asset_dir: Path | None,
    extracted_text: str,
    ocr_client: HttpOcrClient | None,
    vision_summary_client: HttpVisionSummaryClient | None,
    pdf_max_pages: int,
    pdf_dpi: int,
) -> tuple[list, dict[str, object]]:
    """抽取视觉资产，失败时只记录诊断信息。"""
    try:
        assets = extract_visual_assets(
            downloaded,
            asset_dir=asset_dir,
            extracted_text=extracted_text,
            ocr_client=ocr_client,
            vision_summary_client=vision_summary_client,
            pdf_max_pages=pdf_max_pages,
            pdf_dpi=pdf_dpi,
        )
        if not assets:
            return [], {"visual_asset_count": 0}
        return assets, {
            "visual_asset_count": len(assets),
            "visual_asset_kinds": sorted({asset.asset_kind for asset in assets}),
        }
    except Exception as exc:
        return [], {
            "visual_asset_count": 0,
            "visual_asset_error": f"{type(exc).__name__}: {str(exc)[:240]}",
        }


def failed_attachment(downloaded: DownloadedAttachment, *, reason: str, metadata: dict[str, object] | None = None) -> ExtractedAttachment:
    """构建附件抽取失败结果。"""
    return ExtractedAttachment(
        ref=downloaded.ref,
        text="",
        content_type=downloaded.content_type,
        cache_path=downloaded.cache_path,
        sha256=downloaded.sha256,
        status="failed",
        error_reason=reason,
        metadata={"extraction_error": reason, **(metadata or {})},
    )


def analyze_extracted_attachment(*, text: str, extension: str, role: str) -> dict[str, object]:
    """根据抽取文本和文件类型生成可检索元数据。"""
    normalized = str(text or "")
    has_table = "|" in normalized and "---" in normalized
    has_checkbox = any(token in normalized for token in ["□", "☑", "√", "勾选", "多选"])
    has_blank = any(token in normalized for token in ["____", "  ", "年 月 日", "申请人", "联系人"])
    is_image = extension.lower() in {"png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"}
    semantic_type = "attachment_text"
    if role in {"blank_form", "sample_file"} and (has_table or has_checkbox or has_blank or extension == "docx"):
        semantic_type = "attachment_form"
    elif is_image:
        semantic_type = "attachment_image_ocr"
    return {
        "semantic_chunk_type": semantic_type,
        "has_table": has_table,
        "has_checkbox": has_checkbox,
        "has_blank": has_blank,
        "text_length": len(normalized),
    }


def attachment_text_quality_error(text: str, extension: str) -> str:
    """识别明显乱码或不可用的附件抽取文本。"""
    normalized = str(text or "")
    visible_chars = [char for char in normalized if not char.isspace()]
    if not visible_chars:
        return "empty_extracted_text"
    bad_control_count = sum(1 for char in visible_chars if is_bad_control_char(char))
    replacement_count = normalized.count("\ufffd")
    bad_ratio = (bad_control_count + replacement_count) / max(1, len(visible_chars))
    if bad_control_count >= 5 and bad_ratio >= 0.01:
        return "low_quality_extracted_text"
    if extension.lower() == "pdf" and replacement_count >= 5 and bad_ratio >= 0.01:
        return "low_quality_extracted_text"
    return ""


def is_bad_control_char(char: str) -> bool:
    """判断字符是否属于常见 PDF 编码乱码控制区。"""
    codepoint = ord(char)
    if char in {"\n", "\r", "\t"}:
        return False
    return codepoint < 32 or 0x7F <= codepoint <= 0x9F
