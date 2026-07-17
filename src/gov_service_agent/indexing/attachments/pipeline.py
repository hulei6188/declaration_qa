"""知识库附件入库流水线。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from gov_service_agent.core.config import GovServiceAgentSettings
from gov_service_agent.indexing.attachments.chunks import build_attachment_chunk_specs
from gov_service_agent.indexing.attachments.downloader import (
    AttachmentDownloadError,
    AttachmentDownloader,
    AttachmentDownloadSettings,
)
from gov_service_agent.indexing.attachments.extractor import extract_attachment
from gov_service_agent.indexing.attachments.models import AttachmentRef, DownloadedAttachment, ExtractedAttachment
from gov_service_agent.indexing.attachments.parser import parse_markdown_attachments
from gov_service_agent.indexing.document_parsing.factory import build_document_parser, is_image_document, native_first_parser_order
from gov_service_agent.indexing.documents import HttpOcrClient, HttpVisionSummaryClient
from gov_service_agent.observability.langfuse import observation, record_error, safe_value


@dataclass(frozen=True)
class AttachmentIndexingResult:
    """附件入库流水线结果。"""

    chunk_specs: list[dict[str, Any]]
    total_count: int
    success_count: int
    failed_count: int
    skipped_count: int = 0
    failures: list[dict[str, str]] = field(default_factory=list)


def build_attachment_chunks_for_document(
    *,
    document_id: str,
    content: str,
    metadata: dict[str, Any],
    settings: GovServiceAgentSettings,
    stage_callback: Callable[[str, int, str, dict[str, Any]], None] | None = None,
) -> AttachmentIndexingResult:
    """解析、下载并抽取 Markdown 文档中的官方附件。"""
    stage_callback = stage_callback or (lambda *args, **kwargs: None)
    if not settings.attachment_indexing_enabled:
        return AttachmentIndexingResult(chunk_specs=[], total_count=0, success_count=0, failed_count=0, skipped_count=0)
    refs = parse_markdown_attachments(content, document_id=document_id)
    if not refs:
        return AttachmentIndexingResult(chunk_specs=[], total_count=0, success_count=0, failed_count=0, skipped_count=0)
    max_count = max(0, int(settings.attachment_max_per_document))
    selected_refs = refs[:max_count] if max_count else []
    skipped_count = max(0, len(refs) - len(selected_refs))
    stage_callback(
        "attachments",
        32,
        "Document attachments discovered",
        {"attachment_count": len(refs), "selected_count": len(selected_refs), "skipped_count": skipped_count},
    )
    downloader = AttachmentDownloader(
        AttachmentDownloadSettings(
            cache_dir=Path(settings.attachment_cache_dir),
            timeout_seconds=float(settings.attachment_download_timeout_seconds),
            max_bytes=int(settings.attachment_max_bytes),
            retries=int(settings.attachment_download_retries),
        )
    )
    ocr_client = (
        HttpOcrClient(url=settings.ocr_api_url, api_key=settings.ocr_api_key, model=settings.ocr_model)
        if settings.ocr_api_url
        else None
    )
    vision_summary_client = build_vision_summary_client(settings)
    general_parser = build_shared_document_parser(settings, ocr_client=ocr_client)
    image_parser = build_shared_document_parser(settings, ocr_client=ocr_client, image_first=True)
    max_workers = min(max(1, int(settings.attachment_max_workers)), max(1, len(selected_refs)))
    extracted_items: list[ExtractedAttachment] = []
    failures: list[dict[str, str]] = []
    seen_attachment_keys: set[tuple[str, str, str]] = set()
    duplicate_skipped_count = 0
    with observation(
        "index.attachments",
        input={"document_id": document_id, "attachment_count": len(selected_refs)},
        metadata={"document_id": document_id, "attachment_count": len(selected_refs)},
    ) as span:
        def download_one(ref: AttachmentRef) -> tuple[AttachmentRef, DownloadedAttachment | None, dict[str, str] | None]:
            try:
                with observation(
                    "index.attachment_download",
                    input={"name": ref.name, "url": ref.url, "role": ref.role},
                    metadata={"document_id": document_id, "attachment_type": ref.extension, "attachment_role": ref.role},
                ) as download_span:
                    downloaded = downloader.download(ref)
                    download_span.update(output={"bytes": len(downloaded.data), "cache_path": downloaded.cache_path})
            except AttachmentDownloadError as exc:
                failure = {"name": ref.name, "url": ref.url, "stage": "download", "reason": str(exc)}
                record_error(span, exc, stage="index.attachment_download", extra=failure)
                return ref, None, failure
            return ref, downloaded, None

        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="attachment-download") as executor:
            download_results = list(executor.map(download_one, selected_refs))

        unique_downloads: list[DownloadedAttachment] = []
        for ref, downloaded, failure in download_results:
            if failure is not None:
                failures.append(failure)
                continue
            assert downloaded is not None
            attachment_key = (downloaded.sha256, ref.material_name or ref.name, ref.role)
            if attachment_key in seen_attachment_keys:
                duplicate_skipped_count += 1
                continue
            seen_attachment_keys.add(attachment_key)
            unique_downloads.append(downloaded)

        def extract_one(downloaded: DownloadedAttachment) -> ExtractedAttachment:
            ref = downloaded.ref
            with observation(
                "index.attachment_extract",
                input={"name": ref.name, "type": ref.extension, "role": ref.role, "bytes": len(downloaded.data)},
                metadata={"document_id": document_id, "attachment_type": ref.extension, "attachment_role": ref.role},
            ) as extract_span:
                extracted = extract_attachment(
                    downloaded,
                    ocr_client=ocr_client,
                    vision_summary_client=vision_summary_client,
                    parser_provider=settings.document_parser_provider,
                    parser_order=settings.document_parser_order,
                    parser_fallback_enabled=settings.document_parser_fallback_enabled,
                    llamaparse_api_key=settings.llamaparse_api_key,
                    llamaparse_base_url=settings.llamaparse_base_url,
                    llamaparse_result_type=settings.llamaparse_result_type,
                    mineru_api_url=settings.mineru_api_url,
                    mineru_backend=settings.mineru_backend,
                    mineru_method=settings.mineru_method,
                    mineru_language=settings.mineru_language,
                    mineru_model_source=settings.mineru_model_source,
                    mineru_formula_enabled=settings.mineru_formula_enabled,
                    mineru_timeout_seconds=settings.mineru_timeout_seconds,
                    chonkie_text_max_chars=settings.chonkie_text_max_chars,
                    chonkie_text_min_chars=settings.chonkie_text_min_chars,
                    chonkie_table_max_chars=settings.chonkie_table_max_chars,
                    legacy_doc_conversion_enabled=settings.legacy_doc_conversion_enabled,
                    legacy_doc_converter_command=settings.legacy_doc_converter_command,
                    legacy_doc_conversion_timeout_seconds=settings.legacy_doc_conversion_timeout_seconds,
                    pdf_ocr_fallback_enabled=bool(settings.attachment_pdf_ocr_fallback_enabled),
                    pdf_ocr_max_pages=int(settings.attachment_pdf_ocr_max_pages),
                    pdf_ocr_dpi=int(settings.attachment_pdf_ocr_dpi),
                    asset_dir=Path(settings.attachment_asset_dir),
                    document_parser=(
                        image_parser
                        if is_image_document(ref.name, downloaded.content_type)
                        else general_parser
                    ),
                )
                extract_span.update(
                    output={
                        "status": extracted.status,
                        "text_length": len(extracted.text),
                        "asset_count": len(extracted.assets),
                        "visual_summary_success_count": sum(
                            1 for asset in extracted.assets if asset.metadata.get("visual_summary_status") == "success"
                        ),
                        "visual_summary_failed_count": sum(
                            1 for asset in extracted.assets if asset.metadata.get("visual_summary_status") == "failed"
                        ),
                        "semantic_chunk_type": extracted.metadata.get("semantic_chunk_type") or "",
                        "extraction_method": extracted.metadata.get("extraction_method") or "native",
                        "error_reason": extracted.error_reason,
                    }
                )
                return extracted

        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="attachment-extract") as executor:
            extracted_items = list(executor.map(extract_one, unique_downloads))
        for extracted in extracted_items:
            if extracted.status != "success":
                ref = extracted.ref
                failures.append({"name": ref.name, "url": ref.url, "stage": "extract", "reason": extracted.error_reason})
        chunk_specs = build_attachment_chunk_specs(extracted_items, document_id=document_id, document_metadata=metadata)
        success_count = sum(1 for item in extracted_items if item.status == "success" and (item.text.strip() or item.assets))
        failed_count = len(selected_refs) - duplicate_skipped_count - success_count
        total_skipped_count = skipped_count + duplicate_skipped_count
        output = {
            "attachment_count": len(refs),
            "selected_count": len(selected_refs),
            "success_count": success_count,
            "failed_count": failed_count,
            "skipped_count": total_skipped_count,
            "duplicate_skipped_count": duplicate_skipped_count,
            "chunk_count": len(chunk_specs),
            "failures": failures[:10],
        }
        span.update(output=safe_value(output))
        stage_callback("attachments", 34, "Document attachments extracted", output)
    return AttachmentIndexingResult(
        chunk_specs=chunk_specs,
        total_count=len(refs),
        success_count=success_count,
        failed_count=failed_count,
        skipped_count=total_skipped_count,
        failures=failures,
    )


def build_shared_document_parser(
    settings: GovServiceAgentSettings,
    *,
    ocr_client: HttpOcrClient | None,
    image_first: bool = False,
):
    """为一个索引任务构建可复用解析器，避免每个附件重复初始化模型。"""
    order = settings.document_parser_order
    if image_first:
        order = native_first_parser_order(order)
    return build_document_parser(
        provider=settings.document_parser_provider,
        order=order,
        fallback_enabled=settings.document_parser_fallback_enabled,
        ocr_client=ocr_client,
        llamaparse_api_key=settings.llamaparse_api_key,
        llamaparse_base_url=settings.llamaparse_base_url,
        llamaparse_result_type=settings.llamaparse_result_type,
        mineru_api_url=settings.mineru_api_url,
        mineru_backend=settings.mineru_backend,
        mineru_method=settings.mineru_method,
        mineru_language=settings.mineru_language,
        mineru_model_source=settings.mineru_model_source,
        mineru_formula_enabled=settings.mineru_formula_enabled,
        mineru_timeout_seconds=settings.mineru_timeout_seconds,
        chonkie_text_max_chars=settings.chonkie_text_max_chars,
        chonkie_text_min_chars=settings.chonkie_text_min_chars,
        chonkie_table_max_chars=settings.chonkie_table_max_chars,
        legacy_doc_conversion_enabled=settings.legacy_doc_conversion_enabled,
        legacy_doc_converter_command=settings.legacy_doc_converter_command,
        legacy_doc_conversion_timeout_seconds=settings.legacy_doc_conversion_timeout_seconds,
    )


def build_vision_summary_client(settings: GovServiceAgentSettings) -> HttpVisionSummaryClient | None:
    """根据配置构建视觉摘要客户端。"""
    if not settings.image_summary_enabled:
        return None
    url = settings.image_summary_api_url or settings.ocr_api_url
    api_key = settings.image_summary_api_key or settings.ocr_api_key
    if not url or not api_key or not settings.image_summary_model:
        return None
    return HttpVisionSummaryClient(
        url=url,
        api_key=api_key,
        model=settings.image_summary_model,
        timeout_seconds=settings.image_summary_timeout_seconds,
        max_tokens=settings.image_summary_max_tokens,
    )
