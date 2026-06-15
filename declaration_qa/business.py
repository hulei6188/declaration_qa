from __future__ import annotations

from typing import Any, Protocol


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

    Production wiring uses the PostgreSQL-backed adapter on
    PostgresDeclarationQAStore; this remains for local development without a
    database.
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

