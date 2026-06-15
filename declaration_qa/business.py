from __future__ import annotations

from typing import Any, Protocol

import requests


class BusinessDataAdapter(Protocol):
    def get_apply_context(self, *, user_id: int, apply_id: str) -> dict[str, Any] | None:
        ...

    def verify_apply_access(self, *, user_id: int, apply_id: str) -> bool:
        ...

    def get_status(self, apply_context: dict[str, Any]) -> dict[str, Any]:
        ...

    def get_reject_info(self, apply_context: dict[str, Any]) -> dict[str, Any]:
        ...


class InMemoryBusinessDataAdapter:
    """Business adapter used by the isolated QA app.

    Replace this adapter with real declaration/process service calls when wiring
    the workflow to production systems.
    """

    def __init__(self) -> None:
        self._applications: dict[str, dict[str, Any]] = {
            "A1001": {
                "apply_id": "A1001",
                "user_id": 1,
                "biz_type": "health_certificate",
                "region_code": "340000",
                "status": "in_review",
                "current_node": "DEPT_REVIEW",
                "node_name": "部门审核中",
            },
            "A2002": {
                "apply_id": "A2002",
                "user_id": 1,
                "biz_type": "residence_permit",
                "region_code": "340000",
                "status": "rejected",
                "current_node": "MATERIAL_REVIEW",
                "node_name": "材料审核",
                "reject_code": "MISSING_RESIDENCE_PROOF",
                "reject_reason": "缺少有效居住证明材料",
                "material_code": "RESIDENCE_PROOF",
                "material_name": "居住证明",
                "audit_node": "材料初审",
            },
            "A3003": {
                "apply_id": "A3003",
                "user_id": 1,
                "biz_type": "id_card_replace",
                "region_code": "340000",
                "status": "rejected",
                "current_node": "PHOTO_REVIEW",
                "node_name": "照片审核",
                "reject_code": "PHOTO_BLUR",
                "reject_reason": "身份证正面照片不清晰",
                "material_code": "ID_CARD_FRONT",
                "material_name": "身份证正面照片",
                "audit_node": "人工初审",
            },
        }

    def get_apply_context(self, *, user_id: int, apply_id: str) -> dict[str, Any] | None:
        item = self._applications.get(str(apply_id or "").strip())
        if not item or int(item.get("user_id") or 0) != int(user_id):
            return None
        return dict(item)

    def verify_apply_access(self, *, user_id: int, apply_id: str) -> bool:
        return self.get_apply_context(user_id=user_id, apply_id=apply_id) is not None

    def get_status(self, apply_context: dict[str, Any]) -> dict[str, Any]:
        return {
            "apply_id": apply_context.get("apply_id"),
            "biz_type": apply_context.get("biz_type"),
            "region_code": apply_context.get("region_code"),
            "status": apply_context.get("status"),
            "current_node": apply_context.get("current_node"),
            "node_name": apply_context.get("node_name"),
        }

    def get_reject_info(self, apply_context: dict[str, Any]) -> dict[str, Any]:
        return {
            "apply_id": apply_context.get("apply_id"),
            "biz_type": apply_context.get("biz_type"),
            "region_code": apply_context.get("region_code"),
            "reject_code": apply_context.get("reject_code"),
            "reject_reason": apply_context.get("reject_reason"),
            "material_code": apply_context.get("material_code"),
            "material_name": apply_context.get("material_name"),
            "audit_node": apply_context.get("audit_node"),
        }


class HttpBusinessDataAdapter:
    """HTTP adapter for real declaration/application services.

    Expected endpoints under base_url:
    - GET /applications/{apply_id}?user_id=...
    - GET /applications/{apply_id}/status
    - GET /applications/{apply_id}/reject-info
    """

    def __init__(
        self,
        *,
        base_url: str,
        token: str = "",
        timeout_seconds: int = 5,
        application_path_template: str = "/applications/{apply_id}",
        status_path_template: str = "/applications/{apply_id}/status",
        reject_path_template: str = "/applications/{apply_id}/reject-info",
        access_path_template: str = "/applications/{apply_id}/access",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout_seconds = timeout_seconds
        self.application_path_template = application_path_template
        self.status_path_template = status_path_template
        self.reject_path_template = reject_path_template
        self.access_path_template = access_path_template

    def get_apply_context(self, *, user_id: int, apply_id: str) -> dict[str, Any] | None:
        payload = self._get(self._path(self.application_path_template, apply_id), params={"user_id": user_id})
        if not payload:
            return None
        normalized = normalize_business_payload(payload)
        if normalized.get("user_id") not in (None, "", user_id, str(user_id)):
            return None
        return normalized

    def verify_apply_access(self, *, user_id: int, apply_id: str) -> bool:
        payload = self._get(self._path(self.access_path_template, apply_id), params={"user_id": user_id})
        if not payload:
            return self.get_apply_context(user_id=user_id, apply_id=apply_id) is not None
        normalized = normalize_business_payload(payload)
        if "allowed" in normalized:
            return bool(normalized.get("allowed"))
        if "has_access" in normalized:
            return bool(normalized.get("has_access"))
        return normalized.get("user_id") in (None, "", user_id, str(user_id))

    def get_status(self, apply_context: dict[str, Any]) -> dict[str, Any]:
        apply_id = str(apply_context.get("apply_id") or "").strip()
        payload = self._get(self._path(self.status_path_template, apply_id)) if apply_id else {}
        merged = {**apply_context, **normalize_business_payload(payload or {})}
        return {
            "apply_id": merged.get("apply_id"),
            "biz_type": merged.get("biz_type"),
            "region_code": merged.get("region_code"),
            "status": merged.get("status"),
            "current_node": merged.get("current_node"),
            "node_name": merged.get("node_name"),
        }

    def get_reject_info(self, apply_context: dict[str, Any]) -> dict[str, Any]:
        apply_id = str(apply_context.get("apply_id") or "").strip()
        payload = self._get(self._path(self.reject_path_template, apply_id)) if apply_id else {}
        merged = {**apply_context, **normalize_business_payload(payload or {})}
        return {
            "apply_id": merged.get("apply_id"),
            "biz_type": merged.get("biz_type"),
            "region_code": merged.get("region_code"),
            "reject_code": merged.get("reject_code"),
            "reject_reason": merged.get("reject_reason"),
            "material_code": merged.get("material_code"),
            "material_name": merged.get("material_name"),
            "audit_node": merged.get("audit_node"),
        }

    def _get(self, path: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        last_error = None
        for _ in range(3):
            try:
                response = requests.get(
                    f"{self.base_url}{path}",
                    params=params or {},
                    headers=headers,
                    timeout=self.timeout_seconds,
                )
                break
            except requests.RequestException as exc:
                last_error = exc
        else:
            raise RuntimeError(f"Declaration business API request failed: {last_error}") from last_error
        if response.status_code == 404:
            return {}
        response.raise_for_status()
        payload = response.json()
        return payload.get("data") if isinstance(payload, dict) and isinstance(payload.get("data"), dict) else payload

    @staticmethod
    def _path(template: str, apply_id: str) -> str:
        return template.format(apply_id=apply_id)


def normalize_business_payload(payload: dict[str, Any]) -> dict[str, Any]:
    aliases = {
        "applyId": "apply_id",
        "userId": "user_id",
        "bizType": "biz_type",
        "regionCode": "region_code",
        "currentNode": "current_node",
        "nodeName": "node_name",
        "rejectCode": "reject_code",
        "rejectReason": "reject_reason",
        "materialCode": "material_code",
        "materialName": "material_name",
        "auditNode": "audit_node",
        "allowed": "allowed",
        "hasAccess": "has_access",
    }
    normalized = {}
    for key, value in (payload or {}).items():
        normalized[aliases.get(key, key)] = value
    return normalized
