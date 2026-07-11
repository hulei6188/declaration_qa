"""检索计划辅助函数。"""

from __future__ import annotations

from typing import Any

ALLOWED_METADATA_NEEDS = (
    "region_code",
    "policy_version",
    "item_scope",
    "as_of",
)


def infer_retrieval_plan(
    *,
    metadata_needs: Any = None,
) -> dict[str, Any]:
    """推断并归一化检索计划。"""
    return {
        "metadata_needs": list(normalize_metadata_needs(metadata_needs)),
    }


def normalize_retrieval_plan(plan: dict[str, Any] | None) -> dict[str, Any]:
    """归一化已经生成的检索计划。"""
    raw = plan or {}
    output = {
        "metadata_needs": list(normalize_metadata_needs(raw.get("metadata_needs"))),
    }
    image_rerank_query = str(raw.get("image_rerank_query") or "").strip()
    if image_rerank_query:
        output["image_rerank_query"] = image_rerank_query
    query_modality = str(raw.get("query_modality") or "").strip()
    if query_modality:
        output["query_modality"] = query_modality
    query_image_path = str(raw.get("query_image_path") or "").strip()
    if query_image_path:
        output["query_image_path"] = query_image_path
    return output


def normalize_metadata_needs(values: Any) -> tuple[str, ...]:
    """归一化元数据需求取值。"""
    output: list[str] = []
    for value in list_values(values):
        normalized = str(value or "").strip().lower()
        if normalized in ALLOWED_METADATA_NEEDS and normalized not in output:
            output.append(normalized)
    return tuple(output)


def list_values(values: Any) -> list[Any]:
    """将类列表值返回为扁平列表。"""
    if values is None:
        return []
    if isinstance(values, (list, tuple, set)):
        return list(values)
    return [values]
