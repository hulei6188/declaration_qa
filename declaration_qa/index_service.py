from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from declaration_qa.config import DeclarationQASettings
from declaration_qa.documents import HttpOcrClient, parse_document_bytes
from declaration_qa.domain import KnowledgeDocument
from declaration_qa.indexing import IndexJobStore
from declaration_qa.retrieval import sanitize_metadata
from declaration_qa.workflow import DeclarationQAWorkflow


def index_document_for_workflow(workflow: DeclarationQAWorkflow, document: KnowledgeDocument) -> int:
    chunks_indexed = workflow.rag_engine.index_document(document)
    if hasattr(workflow.audit_store, "persist_document"):
        workflow.audit_store.persist_document(document, workflow.rag_engine.document_chunks(document))
    return chunks_indexed


def process_index_job(
    job_uid: str,
    *,
    workflow: DeclarationQAWorkflow,
    job_store: IndexJobStore,
    settings: DeclarationQASettings,
) -> dict[str, Any]:
    item = job_store.get_index_job(job_uid)
    if not item:
        raise ValueError(f"Index job not found: {job_uid}")
    job_type = str(item.get("job_type") or "")
    payload = dict(item.get("payload") or {})
    job_store.update_index_job(job_uid=job_uid, status="running", progress=10, message="Index worker started")
    try:
        if job_type == "document_index":
            result = process_document_index(payload, workflow=workflow, settings=settings)
        elif job_type == "knowledge_base_reindex":
            result = process_reindex(workflow)
        elif job_type == "document_delete":
            result = process_document_delete(payload, workflow=workflow)
        else:
            raise ValueError(f"Unsupported index job type: {job_type}")
        return job_store.update_index_job(
            job_uid=job_uid,
            status="succeeded",
            progress=100,
            message=result.get("message", "Index job succeeded"),
            payload={**payload, **result},
        )
    except Exception as exc:
        return job_store.update_index_job(
            job_uid=job_uid,
            status="failed",
            progress=100,
            error_message=str(exc),
            payload=payload,
        )


def process_document_index(
    payload: dict[str, Any],
    *,
    workflow: DeclarationQAWorkflow,
    settings: DeclarationQASettings,
) -> dict[str, Any]:
    input_type = str(payload.get("input_type") or "text")
    if input_type == "file":
        file_path = Path(str(payload.get("file_path") or ""))
        if not file_path.exists():
            raise FileNotFoundError(f"Uploaded file is missing: {file_path}")
        data = file_path.read_bytes()
        ocr_client = (
            HttpOcrClient(url=settings.ocr_api_url, api_key=settings.ocr_api_key, model=settings.ocr_model)
            if settings.ocr_api_url
            else None
        )
        parsed = parse_document_bytes(
            filename=str(payload.get("filename") or file_path.name),
            content_type=str(payload.get("content_type") or "application/octet-stream"),
            data=data,
            ocr_client=ocr_client,
        )
        content = parsed.text
        filename = parsed.filename
        content_type = parsed.content_type
    else:
        content = str(payload.get("content") or "")
        filename = str(payload.get("filename") or "")
        content_type = str(payload.get("content_type") or "text/plain")
    title = str(payload.get("title") or filename or payload["document_id"])
    source_url = str(payload.get("source_url") or "")
    parsed_metadata = extract_document_metadata(content, title=title, source_url=source_url)
    metadata = {**parsed_metadata, **sanitize_metadata(dict(payload.get("metadata") or {}))}
    for key in ["region_code", "policy_version", "effective_start_time", "effective_end_time"]:
        if payload.get(key) not in ("", None) and key not in metadata:
            metadata[key] = payload.get(key)
    if metadata.get("region_code") and not metadata.get("region_key"):
        metadata["region_key"] = metadata["region_code"]
    if metadata.get("policy_version") and not metadata.get("source_version"):
        metadata["source_version"] = metadata["policy_version"]
    document = KnowledgeDocument(
        id=str(payload["document_id"]),
        title=title,
        filename=filename,
        content_type=content_type,
        content=content,
        region_code=str(metadata.get("region_code") or metadata.get("region_key") or ""),
        policy_version=str(metadata.get("policy_version") or metadata.get("source_version") or ""),
        source_url=source_url or str(metadata.get("official_source_url") or metadata.get("source_url") or ""),
        audit_node=str(payload.get("audit_node") or ""),
        status=str(payload.get("status") or "published"),
        version_group_uid=str(payload.get("version_group_uid") or payload["document_id"]),
        version_no=int(payload.get("version_no") or 1),
        rollout_percent=max(0, min(100, int(payload.get("rollout_percent") if payload.get("rollout_percent") is not None else 100))),
        change_note=str(payload.get("change_note") or ""),
        effective_start_time=parse_optional_date(metadata.get("effective_start_time")),
        effective_end_time=parse_optional_date(metadata.get("effective_end_time")),
        metadata=metadata,
    )
    chunks_indexed = index_document_for_workflow(workflow, document)
    return {
        "document_id": document.id,
        "chunks_indexed": chunks_indexed,
        "filename": filename,
        "content_type": content_type,
        "message": "Document indexed",
    }


def process_reindex(workflow: DeclarationQAWorkflow) -> dict[str, Any]:
    documents = (
        workflow.audit_store.list_documents()
        if hasattr(workflow.audit_store, "list_documents")
        else workflow.rag_engine.documents()
    )
    chunks_indexed = workflow.rag_engine.rebuild(documents)
    return {
        "documents_indexed": len(documents),
        "chunks_indexed": chunks_indexed,
        "message": "Knowledge base reindexed",
    }


def process_document_delete(payload: dict[str, Any], *, workflow: DeclarationQAWorkflow) -> dict[str, Any]:
    document_id = str(payload.get("document_id") or "")
    if not document_id:
        raise ValueError("document_id is required")
    workflow.rag_engine.delete_document(document_id)
    deleted = workflow.audit_store.delete_document(document_id) if hasattr(workflow.audit_store, "delete_document") else True
    return {"document_id": document_id, "deleted": deleted, "message": "Document deleted"}


def parse_optional_date(value) -> date | None:
    if isinstance(value, date):
        return value
    if not value:
        return None
    return date.fromisoformat(str(value)[:10])


FIELD_ALIASES = {
    "region_key": ["region_key", "regionCode", "Region Code", "ssqdZoneCode", "行政区划编码", "区域编码", "实施区域编码"],
    "region_name": ["region_name", "regionName", "Region Name", "ssqdZoneName", "行政区划名称", "区域名称", "实施区域", "实施区域名称"],
    "catalog_name": ["catalog_name", "catalogName", "mlqdName", "目录清单名称"],
    "catalog_subitem_name": ["catalog_subitem_name", "catalogSubitemName", "catalog_sub_item_name", "catalogSubItemName", "mlqdSubName", "目录子项名称"],
    "item_code": ["item_code", "itemCode", "ssqdCode", "事项编码", "实施清单编码"],
    "item_name": ["item_name", "itemName", "ssqdName", "事项名称"],
    "item_type": ["item_type", "itemType", "ssqdType", "事项类型"],
    "theme_code": ["theme_code", "themeCode", "infoClassPtheme", "主题编码"],
    "service_object": ["service_object", "serviceObject", "serviceObjectName", "服务对象"],
    "source_version": ["source_version", "sourceVersion", "Policy Version", "updateTime", "更新时间", "ssqdVersion", "mlqdVersion"],
    "official_source_url": ["official_source_url", "Official Source URL", "source_url", "url", "final_url"],
    "basic_code": ["baseCode", "basic_code", "基本编码"],
    "agency_name": ["ssqdDeptName", "implementOrgName", "实施主体"],
}


def extract_document_metadata(content: str, *, title: str = "", source_url: str = "") -> dict[str, Any]:
    values = extract_markdown_key_values(content)
    values.update(extract_json_key_values(content))
    metadata = normalize_metadata_values(values)
    effective_source_url = source_url or str(metadata.get("official_source_url") or "")
    for key, value in extract_source_url_metadata(effective_source_url).items():
        if key not in metadata:
            metadata[key] = value
    if effective_source_url and not metadata.get("official_source_url"):
        metadata["official_source_url"] = effective_source_url
    if not metadata.get("item_name"):
        metadata["item_name"] = normalize_item_title(title)
    return sanitize_metadata({key: value for key, value in metadata.items() if value not in ("", None, [], {})})


def extract_markdown_key_values(content: str) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for raw_line in str(content or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        bullet = re.match(r"^[-*]\s*([^:：|]+)\s*[:：]\s*(.+)$", line)
        if bullet:
            values[bullet.group(1).strip()] = bullet.group(2).strip()
            continue
        key_value = re.match(r"^([^#\s:：|][^:：|]{0,63})\s*[:：]\s*(.+)$", line)
        if key_value:
            values[key_value.group(1).strip()] = key_value.group(2).strip()
            continue
        if line.startswith("|") and line.endswith("|"):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if len(cells) >= 2 and cells[0] not in {"字段", "---"} and not set(cells[0]) <= {"-", ":"}:
                values[cells[0]] = cells[1]
    return values


def extract_json_key_values(content: str) -> dict[str, Any]:
    text = str(content or "").strip()
    if not text or text[0] not in "{[":
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}
    if isinstance(payload, dict) and isinstance(payload.get("data"), dict):
        payload = {**payload, **payload["data"]}
    if not isinstance(payload, dict):
        return {}
    return {str(key): value for key, value in payload.items() if isinstance(value, (str, int, float, bool))}


def normalize_metadata_values(values: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for target_key, aliases in FIELD_ALIASES.items():
        value = first_value(values, aliases)
        if value not in ("", None):
            normalized[target_key] = str(value).strip()
    return normalized


def first_value(values: dict[str, Any], aliases: list[str]) -> Any:
    for alias in aliases:
        if alias in values and values[alias] not in ("", None):
            return values[alias]
    lowered = {str(key).lower(): value for key, value in values.items()}
    for alias in aliases:
        value = lowered.get(alias.lower())
        if value not in ("", None):
            return value
    return None


def extract_source_url_metadata(source_url: str) -> dict[str, str]:
    if not source_url:
        return {}
    query = parse_qs(urlparse(source_url).query)
    metadata: dict[str, str] = {}
    if query.get("ssqdCode") and query["ssqdCode"][0]:
        metadata["ssqd_code"] = query["ssqdCode"][0]
        metadata.setdefault("item_code", query["ssqdCode"][0])
    if query.get("ssqdId") and query["ssqdId"][0]:
        metadata["ssqd_id"] = query["ssqdId"][0]
    return metadata


def normalize_item_title(title: str) -> str:
    text = str(title or "").strip()
    return re.sub(r"(办事指南|流程文档)$", "", text).strip()
