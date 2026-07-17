"""知识文档元数据抽取。"""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import parse_qs, urlparse

from gov_service_agent.rag.retrieval import sanitize_metadata


METADATA_FIELDS = {
    "region_key",
    "region_name",
    "catalog_name",
    "catalog_subitem_name",
    "item_code",
    "item_name",
    "item_type",
    "theme_code",
    "service_object",
    "source_version",
    "official_source_url",
    "source_url",
    "basic_code",
    "agency_name",
}


def extract_document_metadata(content: str, *, title: str = "", source_url: str = "") -> dict[str, Any]:
    """从文档正文、标题和来源地址抽取元数据。"""
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
    """从 Markdown 文本中抽取键值对。"""
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
    """从 JSON 文档中抽取顶层键值对。"""
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
    """保留标准元数据键。"""
    normalized: dict[str, Any] = {}
    for target_key in METADATA_FIELDS:
        value = values.get(target_key)
        if value not in ("", None):
            normalized[target_key] = str(value).strip()
    return normalized


def extract_source_url_metadata(source_url: str) -> dict[str, str]:
    """从来源 URL 查询参数中抽取元数据。"""
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
    """清理事项标题中的指南后缀。"""
    text = str(title or "").strip()
    return re.sub(r"(办事指南|流程文档)$", "", text).strip()

