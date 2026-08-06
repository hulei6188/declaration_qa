"""Schema-driven multi-turn slot collection for write tools."""

from __future__ import annotations

from datetime import date
import re
from typing import Any


EMPTY_VALUES = (None, "", [], {})


def normalize_form_schema(schema: dict[str, Any] | None) -> dict[str, Any]:
    """Return a stable public form schema shape."""
    source = dict(schema or {})
    fields = []
    seen: set[str] = set()
    for raw in source.get("fields") or []:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name") or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        fields.append(
            {
                "name": name,
                "label": str(raw.get("label") or name).strip(),
                "type": str(raw.get("type") or "text").strip(),
                "required": bool(raw.get("required")),
                "description": str(raw.get("description") or "").strip(),
                "options": [str(item) for item in (raw.get("options") or [])],
                "aliases": [str(item).strip() for item in (raw.get("aliases") or []) if str(item).strip()],
                "pattern": str(raw.get("pattern") or "").strip(),
                "min_length": int(raw.get("min_length") or 0),
                "max_length": int(raw.get("max_length") or 0),
                "max_files": int(raw.get("max_files") or 0),
                "accept": [str(item).lower() for item in (raw.get("accept") or [])],
                "materials": normalize_material_definitions(raw.get("materials")),
            }
        )
    return {
        "schema_id": str(source.get("schema_id") or ""),
        "version": str(source.get("version") or "1"),
        "item_code": str(source.get("item_code") or ""),
        "item_name": str(source.get("item_name") or ""),
        "biz_type": str(source.get("biz_type") or ""),
        "fields": fields,
    }


def collect_slots(
    schema: dict[str, Any],
    current_values: dict[str, Any] | None,
    new_values: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge recognized fields and aliases without accepting undeclared data."""
    normalized = normalize_form_schema(schema)
    aliases: dict[str, str] = {}
    for field in normalized["fields"]:
        aliases[field["name"]] = field["name"]
        aliases[field["label"]] = field["name"]
        for alias in field["aliases"]:
            aliases[alias] = field["name"]
    values = {
        str(key): value
        for key, value in dict(current_values or {}).items()
        if str(key) in aliases.values()
    }
    for key, value in dict(new_values or {}).items():
        canonical = aliases.get(str(key).strip())
        if canonical:
            values[canonical] = value
    return values


def validate_slots(
    schema: dict[str, Any],
    values: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Normalize values and return missing and invalid field descriptors."""
    normalized_values: dict[str, Any] = {}
    missing: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for field in normalize_form_schema(schema)["fields"]:
        name = field["name"]
        value = values.get(name)
        if value in EMPTY_VALUES:
            if field["required"]:
                missing.append(public_field(field))
            continue
        converted, error = normalize_slot_value(field, value)
        if error:
            invalid.append({**public_field(field), "error": error})
            continue
        normalized_values[name] = converted
    return normalized_values, missing, invalid


def normalize_slot_value(field: dict[str, Any], value: Any) -> tuple[Any, str]:
    field_type = str(field.get("type") or "text")
    if field_type == "material_group":
        return normalize_material_group_value(field, value)
    if field_type in {"integer", "number"}:
        try:
            converted = int(value) if field_type == "integer" else float(value)
        except (TypeError, ValueError):
            return value, "请输入有效数字"
        return converted, ""
    if field_type == "boolean":
        if isinstance(value, bool):
            return value, ""
        mapping = {"是": True, "否": False, "true": True, "false": False, "1": True, "0": False}
        converted = mapping.get(str(value).strip().lower())
        return (converted, "") if converted is not None else (value, "请输入是或否")
    if field_type == "date":
        text = str(value).strip()
        try:
            return date.fromisoformat(text).isoformat(), ""
        except ValueError:
            return value, "请使用 YYYY-MM-DD 日期格式"
    if field_type in {"multiselect", "checkbox"}:
        selected = value if isinstance(value, list) else [item.strip() for item in str(value).split(",") if item.strip()]
        options = set(field.get("options") or [])
        if options and any(str(item) not in options for item in selected):
            return value, "请选择表单提供的有效选项"
        return [str(item) for item in selected], ""

    text = str(value).strip()
    options = set(field.get("options") or [])
    if field_type in {"select", "radio"} and options and text not in options:
        return value, "请选择表单提供的有效选项"
    if field_type == "phone" and not re.fullmatch(r"(?:\+?86[- ]?)?1\d{10}|0\d{2,3}[- ]?\d{7,8}", text):
        return value, "请输入有效联系电话"
    if field_type == "id_number" and not re.fullmatch(r"\d{17}[\dXx]|[A-Za-z0-9]{6,30}", text):
        return value, "请输入有效证件号码"
    min_length = int(field.get("min_length") or 0)
    max_length = int(field.get("max_length") or 0)
    if min_length and len(text) < min_length:
        return value, f"至少填写 {min_length} 个字符"
    if max_length and len(text) > max_length:
        return value, f"最多填写 {max_length} 个字符"
    pattern = str(field.get("pattern") or "")
    if pattern and not re.fullmatch(pattern, text):
        return value, "填写格式不符合要求"
    return text, ""


def normalize_material_definitions(value: Any) -> list[dict[str, Any]]:
    output = []
    seen: set[str] = set()
    for raw in value if isinstance(value, list) else []:
        if not isinstance(raw, dict):
            continue
        key = str(raw.get("key") or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(
            {
                "key": key,
                "label": str(raw.get("label") or key).strip(),
                "required": bool(raw.get("required")),
                "description": str(raw.get("description") or "").strip(),
                "max_files": max(1, int(raw.get("max_files") or 1)),
            }
        )
    return output


def normalize_material_group_value(field: dict[str, Any], value: Any) -> tuple[Any, str]:
    if not isinstance(value, dict):
        return value, "请上传申报材料"
    normalized: dict[str, list[dict[str, Any]]] = {}
    for material in field.get("materials") or []:
        key = str(material.get("key") or "")
        refs = value.get(key) or []
        if not isinstance(refs, list):
            return value, f"“{material.get('label') or key}”的附件格式无效"
        clean_refs = []
        for reference in refs:
            if not isinstance(reference, dict) or not re.fullmatch(
                r"[a-f0-9]{32}", str(reference.get("attachment_id") or "")
            ):
                return value, f"“{material.get('label') or key}”包含无效附件"
            clean_refs.append(
                {
                    name: reference.get(name)
                    for name in (
                        "attachment_id",
                        "filename",
                        "content_type",
                        "size_bytes",
                        "sha256",
                        "item_code",
                        "field_name",
                        "material_key",
                    )
                }
            )
        if material.get("required") and not clean_refs:
            return value, f"请上传“{material.get('label') or key}”"
        if len(clean_refs) > int(material.get("max_files") or 1):
            return value, f"“{material.get('label') or key}”最多上传 {material.get('max_files') or 1} 个文件"
        if clean_refs:
            normalized[key] = clean_refs
    return normalized, ""


def build_dialog_task(
    *,
    task_type: str,
    schema: dict[str, Any],
    values: dict[str, Any],
    missing_fields: list[dict[str, Any]],
    invalid_fields: list[dict[str, Any]],
    status: str | None = None,
) -> dict[str, Any]:
    normalized = normalize_form_schema(schema)
    requires_input = bool(missing_fields or invalid_fields)
    return {
        "task_type": task_type,
        "status": status or ("collecting" if requires_input else "ready"),
        "schema_id": normalized["schema_id"],
        "schema_version": normalized["version"],
        "item_code": normalized["item_code"],
        "item_name": normalized["item_name"],
        "values": dict(values),
        "missing_fields": list(missing_fields),
        "invalid_fields": list(invalid_fields),
        "schema": normalized,
    }


def public_field(field: dict[str, Any]) -> dict[str, Any]:
    return {
        key: field[key]
        for key in (
            "name",
            "label",
            "type",
            "required",
            "description",
            "options",
            "pattern",
            "min_length",
            "max_length",
            "max_files",
            "accept",
            "materials",
        )
        if key in field
    }


def masked_confirmation_details(task: dict[str, Any]) -> list[str]:
    """Build confirmation rows while masking common identity and contact fields."""
    fields = {item["name"]: item for item in (task.get("schema") or {}).get("fields") or [] if isinstance(item, dict)}
    details = []
    for name, value in (task.get("values") or {}).items():
        field = fields.get(name) or {}
        label = str(field.get("label") or name)
        if field.get("type") == "material_group" and isinstance(value, dict):
            material_labels = {
                str(item.get("key") or ""): str(item.get("label") or item.get("key") or "")
                for item in field.get("materials") or []
                if isinstance(item, dict)
            }
            files = [
                f"{material_labels.get(str(key), str(key))}：{len(refs or [])} 份"
                for key, refs in value.items()
                if refs
            ]
            details.append(f"{label}：{'；'.join(files) if files else '未上传'}")
            continue
        details.append(f"{label}：{mask_slot_value(name, value)}")
    return details


def mask_slot_value(name: str, value: Any) -> str:
    text = ", ".join(str(item) for item in value) if isinstance(value, list) else str(value)
    lowered = name.lower()
    if any(token in lowered for token in ("id_number", "identity", "certificate_no")) and len(text) >= 8:
        return f"{text[:4]}******{text[-4:]}"
    if any(token in lowered for token in ("phone", "mobile", "contact")) and len(text) >= 7:
        return f"{text[:3]}****{text[-4:]}"
    return text
