"""知识库附件入库流水线。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from declaration_qa.core.config import DeclarationQASettings
from declaration_qa.indexing.attachments.chunks import build_attachment_chunk_specs
from declaration_qa.indexing.attachments.downloader import (
    AttachmentDownloadError,
    AttachmentDownloader,
    AttachmentDownloadSettings,
)
from declaration_qa.indexing.attachments.extractor import extract_attachment
from declaration_qa.indexing.attachments.models import ExtractedAttachment
from declaration_qa.indexing.attachments.parser import parse_markdown_attachments
from declaration_qa.indexing.documents import HttpOcrClient
from declaration_qa.observability.langfuse import observation, record_error, safe_value


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
    settings: DeclarationQASettings,
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
    extracted_items: list[ExtractedAttachment] = []
    failures: list[dict[str, str]] = []
    seen_attachment_keys: set[tuple[str, str, str]] = set()
    duplicate_skipped_count = 0
    with observation(
        "index.attachments",
        input={"document_id": document_id, "attachment_count": len(selected_refs)},
        metadata={"document_id": document_id, "attachment_count": len(selected_refs)},
    ) as span:
        for ref in selected_refs:
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
                failures.append(failure)
                record_error(span, exc, stage="index.attachment_download", extra=failure)
                continue
            attachment_key = (downloaded.sha256, ref.material_name or ref.name, ref.role)
            if attachment_key in seen_attachment_keys:
                duplicate_skipped_count += 1
                continue
            seen_attachment_keys.add(attachment_key)
            with observation(
                "index.attachment_extract",
                input={"name": ref.name, "type": ref.extension, "role": ref.role, "bytes": len(downloaded.data)},
                metadata={"document_id": document_id, "attachment_type": ref.extension, "attachment_role": ref.role},
            ) as extract_span:
                extracted = extract_attachment(downloaded, ocr_client=ocr_client)
                extracted_items.append(extracted)
                extract_span.update(
                    output={
                        "status": extracted.status,
                        "text_length": len(extracted.text),
                        "semantic_chunk_type": extracted.metadata.get("semantic_chunk_type") or "",
                        "error_reason": extracted.error_reason,
                    }
                )
                if extracted.status != "success":
                    failures.append({"name": ref.name, "url": ref.url, "stage": "extract", "reason": extracted.error_reason})
        chunk_specs = build_attachment_chunk_specs(extracted_items, document_id=document_id, document_metadata=metadata)
        success_count = sum(1 for item in extracted_items if item.status == "success" and item.text.strip())
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
