from __future__ import annotations

from typing import Any


MATERIAL_RULE_SOURCE_TYPE = "material_rule"
LEGACY_REJECT_RULE_SOURCE_TYPE = "reject_rule"


SOURCE_TYPE_ALIASES = {
    LEGACY_REJECT_RULE_SOURCE_TYPE: MATERIAL_RULE_SOURCE_TYPE,
}


def canonicalize_source_type(value: Any) -> str:
    source_type = str(value or "").strip()
    return SOURCE_TYPE_ALIASES.get(source_type, source_type)


def expand_source_type_filter(value: Any) -> list[str]:
    values = value if isinstance(value, list) else [value]
    expanded: list[str] = []
    for item in values:
        canonical = canonicalize_source_type(item)
        if not canonical:
            continue
        if canonical not in expanded:
            expanded.append(canonical)
        if canonical == MATERIAL_RULE_SOURCE_TYPE and LEGACY_REJECT_RULE_SOURCE_TYPE not in expanded:
            expanded.append(LEGACY_REJECT_RULE_SOURCE_TYPE)
    return expanded
