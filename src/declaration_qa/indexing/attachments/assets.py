"""附件视觉资产抽取与落盘。"""

from __future__ import annotations

import hashlib
import io
import re
import zipfile
from pathlib import Path

from declaration_qa.indexing.attachments.models import AttachmentAsset, DownloadedAttachment
from declaration_qa.indexing.attachments.pdf_ocr import close_pdf, import_pymupdf, open_pdf, render_page_png
from declaration_qa.indexing.document_parsing.models import ParsedBlock
from declaration_qa.indexing.documents import HttpOcrClient, HttpVisionSummaryClient, normalize_extracted_text
from declaration_qa.observability.langfuse import observation, record_error


IMAGE_CONTENT_TYPES = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
    "bmp": "image/bmp",
    "tif": "image/tiff",
    "tiff": "image/tiff",
}


def extract_visual_assets(
    downloaded: DownloadedAttachment,
    *,
    asset_dir: Path | None,
    extracted_text: str,
    ocr_client: HttpOcrClient | None = None,
    vision_summary_client: HttpVisionSummaryClient | None = None,
    pdf_max_pages: int = 8,
    pdf_dpi: int = 144,
    parsed_blocks: list[ParsedBlock] | None = None,
    pdf_page_fallback: bool = True,
) -> list[AttachmentAsset]:
    """按附件类型抽取可展示的视觉资产。"""
    if not asset_dir:
        return []
    extension = downloaded.ref.extension.lower().strip(".")
    if extension == "pdf":
        picture_assets = extract_parsed_picture_assets(
            downloaded,
            parsed_blocks=parsed_blocks or [],
            asset_dir=asset_dir,
            ocr_client=ocr_client,
            vision_summary_client=vision_summary_client,
        )
        if picture_assets:
            return picture_assets
        if not pdf_page_fallback:
            return []
        return extract_pdf_page_assets(
            downloaded,
            asset_dir=asset_dir,
            extracted_text=extracted_text,
            vision_summary_client=vision_summary_client,
            max_pages=pdf_max_pages,
            dpi=pdf_dpi,
        )
    if extension == "docx":
        return extract_docx_media_assets(
            downloaded,
            asset_dir=asset_dir,
            ocr_client=ocr_client,
            vision_summary_client=vision_summary_client,
        )
    if extension in IMAGE_CONTENT_TYPES:
        return [
            persist_original_image_asset(
                downloaded,
                asset_dir=asset_dir,
                extracted_text=extracted_text,
                ocr_client=ocr_client,
                vision_summary_client=vision_summary_client,
            )
        ]
    return []


def extract_parsed_picture_assets(
    downloaded: DownloadedAttachment,
    *,
    parsed_blocks: list[ParsedBlock],
    asset_dir: Path,
    ocr_client: HttpOcrClient | None,
    vision_summary_client: HttpVisionSummaryClient | None,
) -> list[AttachmentAsset]:
    """将解析器输出的独立 picture 块落盘为视觉资产。"""
    assets: list[AttachmentAsset] = []
    picture_blocks = [block for block in parsed_blocks if block.block_type == "picture" and block.binary_data]
    for picture_index, block in enumerate(picture_blocks):
        metadata = dict(block.metadata or {})
        picture_ref = str(metadata.get("picture_ref") or picture_index)
        asset_id = stable_asset_id(downloaded, "parsed-picture", picture_ref)
        filename = f"{Path(downloaded.ref.name).stem or 'document'}-picture-{picture_index + 1}.png"
        path = write_asset(asset_dir, asset_id=asset_id, extension="png", data=block.binary_data)
        text = normalize_extracted_text(block.text)
        if not text:
            text = ocr_image_asset(
                ocr_client,
                filename=filename,
                content_type=block.content_type or "image/png",
                data=block.binary_data,
            )
        caption = normalize_extracted_text(metadata.get("picture_caption") or "")
        summary_context = "\n".join(part for part in [caption, text] if part)
        summary, summary_metadata = summarize_image_asset(
            vision_summary_client,
            filename=filename,
            content_type=block.content_type or "image/png",
            data=block.binary_data,
            ocr_text=summary_context,
        )
        assets.append(
            AttachmentAsset(
                asset_id=asset_id,
                asset_kind="pdf_embedded_picture",
                modality="image",
                path=str(path),
                content_type=block.content_type or "image/png",
                filename=filename,
                text=text,
                summary=summary,
                page_no=block.page_no,
                bbox=block.bbox,
                metadata={
                    **metadata,
                    "picture_caption": caption,
                    "picture_ocr_source": "docling" if block.text.strip() else ("external_ocr" if text else "none"),
                    **summary_metadata,
                },
            )
        )
    return assets


def extract_pdf_page_assets(
    downloaded: DownloadedAttachment,
    *,
    asset_dir: Path,
    extracted_text: str,
    vision_summary_client: HttpVisionSummaryClient | None,
    max_pages: int,
    dpi: int,
) -> list[AttachmentAsset]:
    """将 PDF 页面渲染为页面图片资产。"""
    if max_pages <= 0:
        return []
    fitz = import_pymupdf()
    document = open_pdf(fitz, downloaded.data)
    try:
        page_texts = pdf_page_texts(extracted_text)
        page_count = len(document)
        render_count = min(page_count, max_pages)
        assets: list[AttachmentAsset] = []
        for page_index in range(render_count):
            page_no = page_index + 1
            image_data = render_page_png(fitz, document, page_index, dpi=dpi)
            asset_id = stable_asset_id(downloaded, "pdf-page", str(page_no))
            path = write_asset(asset_dir, asset_id=asset_id, extension="png", data=image_data)
            page_text = page_texts.get(page_no, "")
            summary, summary_metadata = summarize_image_asset(
                vision_summary_client,
                filename=f"{Path(downloaded.ref.name).stem or 'document'}-page-{page_no}.png",
                content_type="image/png",
                data=image_data,
                ocr_text=page_text,
            )
            assets.append(
                AttachmentAsset(
                    asset_id=asset_id,
                    asset_kind="pdf_page_image",
                    modality="image",
                    path=str(path),
                    content_type="image/png",
                    filename=f"{Path(downloaded.ref.name).stem or 'document'}-page-{page_no}.png",
                    text=page_text,
                    summary=summary,
                    page_no=page_no,
                    bbox=None,
                    metadata={
                        "pdf_page_count": page_count,
                        "pdf_render_dpi": dpi,
                        **summary_metadata,
                    },
                )
            )
        return assets
    finally:
        close_pdf(document)


def extract_docx_media_assets(
    downloaded: DownloadedAttachment,
    *,
    asset_dir: Path,
    ocr_client: HttpOcrClient | None,
    vision_summary_client: HttpVisionSummaryClient | None,
) -> list[AttachmentAsset]:
    """抽取 DOCX 内嵌图片资产。"""
    try:
        archive = zipfile.ZipFile(io.BytesIO(downloaded.data))
    except zipfile.BadZipFile:
        return []
    assets: list[AttachmentAsset] = []
    with archive:
        media_names = sorted(name for name in archive.namelist() if name.startswith("word/media/") and not name.endswith("/"))
        for media_index, media_name in enumerate(media_names):
            extension = Path(media_name).suffix.lower().lstrip(".")
            if extension not in IMAGE_CONTENT_TYPES:
                continue
            data = archive.read(media_name)
            asset_id = stable_asset_id(downloaded, "docx-media", str(media_index), media_name)
            path = write_asset(asset_dir, asset_id=asset_id, extension=extension, data=data)
            content_type = IMAGE_CONTENT_TYPES[extension]
            text = ocr_image_asset(
                ocr_client,
                filename=Path(media_name).name,
                content_type=content_type,
                data=data,
            )
            summary, summary_metadata = summarize_image_asset(
                vision_summary_client,
                filename=Path(media_name).name,
                content_type=content_type,
                data=data,
                ocr_text=text,
            )
            assets.append(
                AttachmentAsset(
                    asset_id=asset_id,
                    asset_kind="docx_embedded_image",
                    modality="image",
                    path=str(path),
                    content_type=content_type,
                    filename=Path(media_name).name,
                    text=text,
                    summary=summary,
                    metadata={
                        "docx_media_path": media_name,
                        "docx_media_index": media_index,
                        **summary_metadata,
                    },
                )
            )
    return assets


def persist_original_image_asset(
    downloaded: DownloadedAttachment,
    *,
    asset_dir: Path,
    extracted_text: str,
    ocr_client: HttpOcrClient | None,
    vision_summary_client: HttpVisionSummaryClient | None,
) -> AttachmentAsset:
    """将图片附件本身登记为视觉资产。"""
    extension = downloaded.ref.extension.lower().strip(".") or "png"
    content_type = downloaded.content_type or IMAGE_CONTENT_TYPES.get(extension, "application/octet-stream")
    asset_id = stable_asset_id(downloaded, "image-original")
    path = write_asset(asset_dir, asset_id=asset_id, extension=extension, data=downloaded.data)
    text = normalize_extracted_text(extracted_text)
    if not text:
        text = ocr_image_asset(
            ocr_client,
            filename=downloaded.ref.name,
            content_type=content_type,
            data=downloaded.data,
        )
    summary, summary_metadata = summarize_image_asset(
        vision_summary_client,
        filename=downloaded.ref.name,
        content_type=content_type,
        data=downloaded.data,
        ocr_text=text,
    )
    return AttachmentAsset(
        asset_id=asset_id,
        asset_kind="original_image",
        modality="image",
        path=str(path),
        content_type=content_type,
        filename=downloaded.ref.name,
        text=text,
        summary=summary,
        metadata=summary_metadata,
    )


def ocr_image_asset(
    ocr_client: HttpOcrClient | None,
    *,
    filename: str,
    content_type: str,
    data: bytes,
) -> str:
    """对图片资产执行 OCR，失败时不阻断主入库。"""
    if not ocr_client:
        return ""
    try:
        return normalize_extracted_text(ocr_client.extract_text(filename=filename, content_type=content_type, data=data))
    except Exception:
        return ""


def summarize_image_asset(
    client: HttpVisionSummaryClient | None,
    *,
    filename: str,
    content_type: str,
    data: bytes,
    ocr_text: str,
) -> tuple[str, dict[str, object]]:
    """生成图片视觉摘要，并保留可观测的执行状态。"""
    if not client:
        return "", {"visual_summary_status": "skipped", "visual_summary_model": ""}
    with observation(
        "index.image_summary",
        input={"filename": filename, "content_type": content_type, "bytes": len(data), "ocr_text_length": len(ocr_text)},
        metadata={"filename": filename, "model": client.model},
        as_type="generation",
        model=client.model,
    ) as span:
        try:
            summary = normalize_extracted_text(
                client.summarize_image(
                    filename=filename,
                    content_type=content_type,
                    data=data,
                    ocr_text=ocr_text,
                )
            )
            span.update(output={"summary": summary, "summary_length": len(summary)})
            return summary, {
                "visual_summary_status": "success",
                "visual_summary_model": client.model,
            }
        except Exception as exc:
            record_error(span, exc, stage="index.image_summary", extra={"filename": filename, "model": client.model})
            return "", {
                "visual_summary_status": "failed",
                "visual_summary_model": client.model,
                "visual_summary_error": f"{type(exc).__name__}: {str(exc)[:240]}",
            }


def write_asset(asset_dir: Path, *, asset_id: str, extension: str, data: bytes) -> Path:
    """按稳定 ID 写入资产文件。"""
    asset_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{asset_id}.{extension.lower().strip('.') or 'bin'}"
    path = asset_dir / filename
    if not path.exists():
        path.write_bytes(data)
    return path


def stable_asset_id(downloaded: DownloadedAttachment, *parts: str) -> str:
    """生成稳定的视觉资产 ID。"""
    raw = "|".join([downloaded.sha256, downloaded.ref.url, downloaded.ref.name, *parts])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:24]


def pdf_page_texts(text: str) -> dict[int, str]:
    """从 PDF/OCR 文本中按页码拆出页面说明。"""
    normalized = str(text or "")
    matches = list(re.finditer(r"^##\s*(?:OCR\s*)?PDF\s*第\s*([0-9]+)\s*页\s*$", normalized, flags=re.IGNORECASE | re.MULTILINE))
    if not matches:
        return {}
    pages: dict[int, str] = {}
    for index, match in enumerate(matches):
        page_no = int(match.group(1))
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized)
        pages[page_no] = normalized[start:end].strip()
    return pages


def asset_url(asset_id: str, path: str) -> str:
    """返回资产的 API 访问地址。"""
    suffix = Path(path).suffix.lower()
    return f"/api/declaration-qa/assets/{asset_id}{suffix}"
