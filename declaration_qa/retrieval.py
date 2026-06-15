from __future__ import annotations

from datetime import date
from typing import Any

KNOWLEDGE_RETRIEVAL_CHAIN = "knowledge_retrieval"
TICKET_AUGMENTED_CHAIN = "ticket_augmented"


FILTER_ALIASES = {
    "regionCode": "region_code",
    "policyVersion": "policy_version",
    "auditNode": "audit_node",
    "asOf": "as_of",
    "rolloutKey": "rollout_key",
}

SUPPORTED_FILTER_KEYS = {
    "region_code",
    "policy_version",
    "audit_node",
    "as_of",
    "rollout_key",
    "status",
}

REQUIRED_METADATA_BY_CHAIN = {
    KNOWLEDGE_RETRIEVAL_CHAIN: ["region_code", "policy_version"],
    TICKET_AUGMENTED_CHAIN: ["region_code", "policy_version"],
}


def normalize_retrieval_filter(filters: dict[str, Any] | None) -> dict[str, Any]:
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
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, list):
        return [normalize_filter_value(item) for item in value if normalize_filter_value(item) not in ("", None)]
    return value


def metadata_coverage(filters: dict[str, Any], *, chain: str) -> dict[str, Any]:
    required = REQUIRED_METADATA_BY_CHAIN.get(chain, REQUIRED_METADATA_BY_CHAIN[KNOWLEDGE_RETRIEVAL_CHAIN])
    present = [key for key in required if bool(filters.get(key))]
    missing = [key for key in required if not filters.get(key)]
    return {
        "required": required,
        "present": present,
        "missing": missing,
        "coverage": round(len(present) / max(1, len(required)), 4),
    }


def is_ticket_chain(intent: str) -> bool:
    return intent in {"status", "reject"}
