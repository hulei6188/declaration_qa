"""线上评测候选样本的最小化脱敏。"""

from __future__ import annotations

import hashlib
import re
from typing import Any


PATTERNS = (
    (re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"), "[PHONE]"),
    (re.compile(r"(?<![0-9A-Za-z])\d{17}[0-9Xx](?![0-9A-Za-z])"), "[ID_CARD]"),
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "[EMAIL]"),
    (re.compile(r"(?<!\d)\d{16,19}(?!\d)"), "[BANK_CARD]"),
)

SENSITIVE_KEYS = {
    "user_id",
    "id_card",
    "idcard",
    "identity_number",
    "phone",
    "mobile",
    "email",
    "name",
    "applicant_name",
    "contact_name",
    "address",
    "bank_account",
    "application_id",
    "appointment_id",
}


def sanitize_text(value: str, *, salt: str) -> str:
    output = str(value or "")
    for pattern, replacement in PATTERNS:
        output = pattern.sub(replacement, output)
    return output[:20_000]


def stable_fingerprint(value: str, *, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{value}".encode("utf-8")).hexdigest()


def sanitize_value(value: Any, *, salt: str, key: str = "", depth: int = 0) -> Any:
    """递归脱敏Trace结构，并用稳定伪标识保留参数来源可比性。"""
    if depth > 8:
        return "[TRUNCATED]"
    lowered = str(key or "").strip().lower()
    if lowered in SENSITIVE_KEYS:
        return pseudonymize(value, salt=salt, label=lowered.upper())
    if isinstance(value, dict):
        return {
            str(child_key): sanitize_value(child, salt=salt, key=str(child_key), depth=depth + 1)
            for child_key, child in list(value.items())[:200]
        }
    if isinstance(value, (list, tuple)):
        return [sanitize_value(child, salt=salt, key=key, depth=depth + 1) for child in list(value)[:500]]
    if isinstance(value, str):
        return sanitize_text(value, salt=salt)
    return value


def pseudonymize(value: Any, *, salt: str, label: str) -> Any:
    if isinstance(value, (list, tuple)):
        return [pseudonymize(child, salt=salt, label=label) for child in value]
    if isinstance(value, dict):
        return {
            str(key): pseudonymize(child, salt=salt, label=label)
            for key, child in value.items()
        }
    if value in (None, ""):
        return value
    digest = stable_fingerprint(str(value), salt=salt)[:12]
    return f"[{label}:{digest}]"
