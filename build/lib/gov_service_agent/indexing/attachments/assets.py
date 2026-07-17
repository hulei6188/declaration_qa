"""附件视觉资产抽取与落盘。"""

from __future__ import annotations

import hashlib
import io
import re
import zipfile
from pathlib import Path

from gov_service_agent.indexing.attachments.models import AttachmentAsset, DownloadedAttachment
from gov_service_agent.indexing.attachments.pdf_ocr import close_pdf, import_pymupdf, open_pdf, render_page_png
from gov_service_agent.indexing.documents import HttpOcrClient, normalize_extracted_text


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
    pdf_max_pages: int = 8,
    pdf_dpi: int = 144,
) -> list[AttachmentAsset]:
    """按附件类型抽取可展示的视觉资产。"""
    if not asset_dir:
        return []
    extension = downloaded.ref.extension.lower().strip(".")
    if extension == "pdf":
        return extract_pdf_page_assets(
            downloaded,
            asset_dir=asset_dir,
            extracted_text=extracted_text,
            max_pages=pdf_max_pages,
            dpi=pdf_dpi,
        )
    if extension == "docx":
        return extract_docx_media_assets(
            downloaded,
            asset_dir=asset_dir,
            ocr_client=ocr_client,
        )
    if extension in IMAGE_CONTENT_TYPES:
        return [persist_original_image_asset(downloaded, asset_dir=asset_dir, ocr_client=ocr_client)]
    return []


def extract_pdf_page_assets(
    downloaded: DownloadedAttachment,
    *,
    asset_dir: Path,
    extracted_text: str,
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
            assets.append(
                AttachmentAsset(
                    asset_id=asset_id,
                    asset_kind="pdf_page_image",
                    modality="image",
                    path=str(path),
                    content_type="image/png",
                    filename=f"{Path(downloaded.ref.name).stem or 'document'}-page-{page_no}.png",
                    text=page_texts.get(page_no, ""),
                    page_no=page_no,
                    bbox=None,
                    metadata={
                        "pdf_page_count": page_count,
                        "pdf_render_dpi": dpi,
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
            assets.append(
                AttachmentAsset(
                    asset_id=asset_id,
                    asset_kind="docx_embedded_image",
                    modality="image",
                    path=str(path),
                    content_type=content_type,
                    filename=Path(media_name).name,
                    text=text,
                    metadata={
                        "docx_media_path": media_name,
                        "docx_media_index": media_index,
                    },
                )
            )
    return assets


def persist_original_image_asset(
    downloaded: DownloadedAttachment,
    *,
    asset_dir: Path,
    ocr_client: HttpOcrClient | None,
) -> AttachmentAsset:
    """将图片附件本身登记为视觉资产。"""
    extension = downloaded.ref.extension.lower().strip(".") or "png"
    content_type = downloaded.content_type or IMAGE_CONTENT_TYPES.get(extension, "application/octet-stream")
    asset_id = stable_asset_id(downloaded, "image-original")
    path = write_asset(asset_dir, asset_id=asset_id, extension=extension, data=downloaded.data)
    text = ocr_image_asset(
        ocr_client,
        filename=downloaded.ref.name,
        content_type=content_type,
        data=downloaded.data,
    )
    return AttachmentAsset(
        asset_id=asset_id,
        asset_kind="original_image",
        modality="image",
        path=str(path),
        content_type=content_type,
        filename=downloaded.ref.name,
        text=text,
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
