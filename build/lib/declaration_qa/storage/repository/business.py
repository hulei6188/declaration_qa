"""办件业务数据读取。"""

from __future__ import annotations

from typing import Any

from sqlalchemy import text

from declaration_qa.core.business import region_code_variants
from declaration_qa.storage.repository.base import EngineBackedRepository
from declaration_qa.storage.repository.normalizers import business_ticket_summary, normalize_business_row


class BusinessRepository(EngineBackedRepository):
    """办件上下文、进度和退回信息仓储。"""
    def get_apply_context(self, *, user_id: int, apply_id: str) -> dict[str, Any] | None:
        """查询指定办件的业务上下文。"""
        with self.engine.begin() as connection:
            row = connection.execute(
                text(
                    """
                    SELECT
                        a.apply_uid AS apply_id,
                        a.user_id,
                        a.biz_type,
                        a.region_code,
                        COALESCE(s.status, a.status) AS status,
                        COALESCE(s.current_node, a.current_node) AS current_node,
                        COALESCE(s.node_name, a.node_name) AS node_name,
                        r.reject_code,
                        r.reject_reason,
                        r.material_code,
                        r.material_name,
                        r.audit_node,
                        a.submitted_at,
                        a.completed_at,
                        GREATEST(a.updated_at, COALESCE(s.updated_at, a.updated_at), COALESCE(r.updated_at, a.updated_at)) AS updated_at,
                        COALESCE(a.payload, '{}'::jsonb) AS application_payload,
                        COALESCE(s.payload, a.payload, '{}'::jsonb) AS status_payload,
                        COALESCE(r.payload, '{}'::jsonb) AS reject_payload
                    FROM declaration_qa.business_applications a
                    LEFT JOIN declaration_qa.business_application_statuses s
                        ON s.apply_id = a.id AND s.is_current = TRUE
                    LEFT JOIN declaration_qa.business_reject_infos r
                        ON r.apply_id = a.id AND r.is_current = TRUE
                    WHERE a.apply_uid = :apply_uid
                      AND a.user_id = :user_id
                      AND a.deleted_at IS NULL
                    """
                ),
                {"apply_uid": str(apply_id or ""), "user_id": int(user_id)},
            ).mappings().first()
        return normalize_business_row(row) if row else None

    def list_applications(self, *, user_id: int, biz_type: str | None = None, region_code: str | None = None) -> list[dict[str, Any]]:
        """查询用户可访问的办件列表。"""
        region_codes = region_code_variants(region_code)
        with self.engine.begin() as connection:
            rows = connection.execute(
                text(
                    """
                    SELECT
                        a.apply_uid AS apply_id,
                        a.biz_type,
                        a.region_code,
                        COALESCE(s.status, a.status) AS status,
                        COALESCE(s.current_node, a.current_node) AS current_node,
                        COALESCE(s.node_name, a.node_name) AS node_name,
                        COALESCE(a.payload, '{}'::jsonb) AS application_payload,
                        COALESCE(s.updated_at, a.updated_at) AS updated_at
                    FROM declaration_qa.business_applications a
                    LEFT JOIN declaration_qa.business_application_statuses s
                        ON s.apply_id = a.id AND s.is_current = TRUE
                    WHERE a.user_id = :user_id
                      AND a.deleted_at IS NULL
                      AND (:biz_type = '' OR a.biz_type = :biz_type)
                      AND (:use_region_filter = FALSE OR a.region_code = ANY(:region_codes))
                    ORDER BY COALESCE(s.updated_at, a.updated_at) DESC
                    LIMIT 20
                    """
                ),
                {
                    "user_id": int(user_id),
                    "biz_type": str(biz_type or ""),
                    "use_region_filter": bool(region_codes),
                    "region_codes": region_codes or [""],
                },
            ).mappings().all()
        return [business_ticket_summary(row) for row in rows]

    def verify_apply_access(self, *, user_id: int, apply_id: str) -> bool:
        """校验用户是否有权访问指定办件。"""
        with self.engine.begin() as connection:
            row = connection.execute(
                text(
                    """
                    SELECT 1
                    FROM declaration_qa.business_applications
                    WHERE apply_uid = :apply_uid
                      AND user_id = :user_id
                      AND deleted_at IS NULL
                    """
                ),
                {"apply_uid": str(apply_id or ""), "user_id": int(user_id)},
            ).first()
        return row is not None

    def get_status(self, apply_context: dict[str, Any]) -> dict[str, Any]:
        """从办件上下文提取状态信息。"""
        apply_id = str(apply_context.get("apply_id") or "").strip()
        with self.engine.begin() as connection:
            row = connection.execute(
                text(
                    """
                    SELECT
                        a.apply_uid AS apply_id,
                        a.biz_type,
                        a.region_code,
                        COALESCE(s.status, a.status) AS status,
                        COALESCE(s.current_node, a.current_node) AS current_node,
                        COALESCE(s.node_name, a.node_name) AS node_name,
                        a.submitted_at,
                        a.completed_at,
                        COALESCE(s.updated_at, a.updated_at) AS updated_at,
                        COALESCE(a.payload, '{}'::jsonb) AS application_payload,
                        COALESCE(s.payload, a.payload, '{}'::jsonb) AS status_payload
                    FROM declaration_qa.business_applications a
                    LEFT JOIN declaration_qa.business_application_statuses s
                        ON s.apply_id = a.id AND s.is_current = TRUE
                    WHERE a.apply_uid = :apply_uid
                      AND a.deleted_at IS NULL
                    """
                ),
                {"apply_uid": apply_id},
            ).mappings().first()
        merged = {**apply_context, **(normalize_business_row(row) if row else {})}
        return {
            "apply_id": merged.get("apply_id"),
            "biz_type": merged.get("biz_type"),
            "title": merged.get("title"),
            "region_code": merged.get("region_code"),
            "status": merged.get("status"),
            "current_node": merged.get("current_node"),
            "node_name": merged.get("node_name"),
            "submitted_at": merged.get("submitted_at"),
            "completed_at": merged.get("completed_at"),
            "updated_at": merged.get("updated_at"),
            "payload": merged.get("payload") or {},
        }

    def get_reject_info(self, apply_context: dict[str, Any]) -> dict[str, Any]:
        """从办件上下文提取退回和补正信息。"""
        apply_id = str(apply_context.get("apply_id") or "").strip()
        with self.engine.begin() as connection:
            row = connection.execute(
                text(
                    """
                    SELECT
                        a.apply_uid AS apply_id,
                        a.biz_type,
                        a.region_code,
                        r.reject_code,
                        r.reject_reason,
                        r.material_code,
                        r.material_name,
                        r.audit_node,
                        COALESCE(s.status, a.status) AS status,
                        COALESCE(s.current_node, a.current_node) AS current_node,
                        COALESCE(s.node_name, a.node_name) AS node_name,
                        a.submitted_at,
                        GREATEST(a.updated_at, COALESCE(r.updated_at, a.updated_at)) AS updated_at,
                        COALESCE(a.payload, '{}'::jsonb) AS application_payload,
                        COALESCE(r.payload, '{}'::jsonb) AS reject_payload
                    FROM declaration_qa.business_applications a
                    LEFT JOIN declaration_qa.business_application_statuses s
                        ON s.apply_id = a.id AND s.is_current = TRUE
                    LEFT JOIN declaration_qa.business_reject_infos r
                        ON r.apply_id = a.id AND r.is_current = TRUE
                    WHERE a.apply_uid = :apply_uid
                      AND a.deleted_at IS NULL
                    """
                ),
                {"apply_uid": apply_id},
            ).mappings().first()
        merged = {**apply_context, **(normalize_business_row(row) if row else {})}
        return {
            "apply_id": merged.get("apply_id"),
            "biz_type": merged.get("biz_type"),
            "title": merged.get("title"),
            "region_code": merged.get("region_code"),
            "reject_code": merged.get("reject_code"),
            "reject_reason": merged.get("reject_reason"),
            "material_code": merged.get("material_code"),
            "material_name": merged.get("material_name"),
            "audit_node": merged.get("audit_node"),
            "status": merged.get("status"),
            "current_node": merged.get("current_node"),
            "node_name": merged.get("node_name"),
            "submitted_at": merged.get("submitted_at"),
            "updated_at": merged.get("updated_at"),
            "payload": merged.get("payload") or {},
        }

