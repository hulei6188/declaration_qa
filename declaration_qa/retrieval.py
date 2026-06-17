"""检索过滤条件、元数据清洗和链路辅助函数。"""

from __future__ import annotations

from datetime import date
from typing import Any

KNOWLEDGE_RETRIEVAL_CHAIN = "knowledge_retrieval"
TICKET_AUGMENTED_CHAIN = "ticket_augmented"


FILTER_ALIASES = {
    "regionCode": "region_code",
    "regionKey": "region_key",
    "policyVersion": "policy_version",
    "sourceVersion": "source_version",
    "catalogName": "catalog_name",
    "catalogSubitemName": "catalog_subitem_name",
    "catalogSubItemName": "catalog_subitem_name",
    "itemCode": "item_code",
    "itemName": "item_name",
    "itemType": "item_type",
    "themeCode": "theme_code",
    "serviceObject": "service_object",
    "docStatus": "doc_status",
    "auditNode": "audit_node",
    "asOf": "as_of",
    "rolloutKey": "rollout_key",
}

SUPPORTED_FILTER_KEYS = {
    "region_code",
    "region_key",
    "policy_version",
    "source_version",
    "catalog_name",
    "catalog_subitem_name",
    "item_code",
    "item_name",
    "item_type",
    "theme_code",
    "service_object",
    "doc_status",
    "audit_node",
    "as_of",
    "rollout_key",
    "status",
}

REMOVED_METADATA_KEYS = {
    "biz_type",
    "bizType",
    "source_type",
    "sourceType",
    "material_code",
    "materialCode",
    "reject_code",
    "rejectCode",
    "theme_name",
    "themeName",
    "主题分类",
    "主题名称",
}

REQUIRED_METADATA_BY_CHAIN = {
    KNOWLEDGE_RETRIEVAL_CHAIN: [],
    TICKET_AUGMENTED_CHAIN: [],
}


def normalize_retrieval_filter(filters: dict[str, Any] | None) -> dict[str, Any]:
    """标准化外部传入的检索过滤条件。"""
    normalized: dict[str, Any] = {}
    for raw_key, raw_value in (filters or {}).items():
        key = FILTER_ALIASES.get(str(raw_key), str(raw_key))
        if key not in SUPPORTED_FILTER_KEYS:
            continue
        value = normalize_filter_value(raw_value)
        if value in ("", [], None):
            continue
        normalized[key] = value
    return normalized


def normalize_filter_value(value: Any) -> Any:
    """标准化单个检索过滤值。"""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, list):
        return [normalize_filter_value(item) for item in value if normalize_filter_value(item) not in ("", None)]
    return value


def sanitize_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """移除不应参与知识检索的元数据字段。"""
    return {
        str(key): value
        for key, value in (metadata or {}).items()
        if str(key) not in REMOVED_METADATA_KEYS
    }


def metadata_coverage(filters: dict[str, Any], *, chain: str) -> dict[str, Any]:
    """计算检索链路所需元数据的覆盖情况。"""
    required = REQUIRED_METADATA_BY_CHAIN.get(chain, REQUIRED_METADATA_BY_CHAIN[KNOWLEDGE_RETRIEVAL_CHAIN])
    present = [key for key in required if bool(filters.get(key))]
    missing = [key for key in required if not filters.get(key)]
    coverage = 1.0 if not required else round(len(present) / len(required), 4)
    return {
        "required": required,
        "present": present,
        "missing": missing,
        "coverage": coverage,
    }


def is_ticket_chain(intent: str) -> bool:
    """判断意图是否属于办件增强链路。"""
    return intent in {"status", "reject", "ticket_status", "rejection_reason"}
