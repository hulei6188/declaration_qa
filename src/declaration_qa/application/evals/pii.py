"""线上评测候选样本的最小化脱敏。"""

from __future__ import annotations

import hashlib
import re


PATTERNS = (
    (re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"), "[PHONE]"),
    (re.compile(r"(?<![0-9A-Za-z])\d{17}[0-9Xx](?![0-9A-Za-z])"), "[ID_CARD]"),
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "[EMAIL]"),
)


def sanitize_text(value: str, *, salt: str) -> str:
    output = str(value or "")
    for pattern, replacement in PATTERNS:
        output = pattern.sub(replacement, output)
    return output[:20_000]


def stable_fingerprint(value: str, *, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{value}".encode("utf-8")).hexdigest()
