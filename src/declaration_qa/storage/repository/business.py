"""办件业务数据读取。"""

from __future__ import annotations

from typing import Any

from sqlalchemy import text

from declaration_qa.core.business import region_code_variants
from declaration_qa.storage.repository.base import EngineBackedRepository


class BusinessRepository(EngineBackedRepository):
    """按可信用户和事项读取申请及状态历史。"""

    def list_ticket_items(self, *, user_id: int, region_code: str | None = None) -> list[dict[str, Any]]:
        """按事项编码汇总当前用户的申请。"""
        region_codes = region_code_variants(region_code)
        with self.engine.begin() as connection:
            rows = connection.execute(
                text(
                    """
                    SELECT
                        COALESCE(NULLIF(a.item_code, ''), a.payload->>'item_code', '') AS item_code,
                        MAX(COALESCE(NULLIF(a.item_name, ''), a.payload->>'item_name', a.payload->>'official_item_name', a.payload->>'title', '')) AS item_name,
                        COUNT(*)::INTEGER AS application_count,
                        ARRAY_AGG(DISTINCT COALESCE(s.status, a.status)) FILTER (WHERE COALESCE(s.status, a.status) <> '') AS statuses,
                        MAX(COALESCE(s.updated_at, a.updated_at)) AS updated_at
                    FROM declaration_qa.business_applications a
                    LEFT JOIN declaration_qa.business_application_statuses s
                        ON s.apply_id = a.id AND s.is_current = TRUE
                    WHERE a.user_id = :user_id
                      AND a.deleted_at IS NULL
                      AND (:use_region_filter = FALSE OR a.region_code = ANY(:region_codes))
                      AND COALESCE(NULLIF(a.item_code, ''), a.payload->>'item_code', '') <> ''
                    GROUP BY COALESCE(NULLIF(a.item_code, ''), a.payload->>'item_code', '')
                    ORDER BY MAX(COALESCE(s.updated_at, a.updated_at)) DESC
                    """
                ),
                {
                    "user_id": int(user_id),
                    "use_region_filter": bool(region_codes),
                    "region_codes": region_codes or [""],
                },
            ).mappings().all()
        return [
            {
                "item_code": str(row.get("item_code") or ""),
                "item_name": str(row.get("item_name") or ""),
                "application_count": int(row.get("application_count") or 0),
                "statuses": [str(item) for item in (row.get("statuses") or []) if str(item or "")],
                "updated_at": str(row.get("updated_at") or ""),
            }
            for row in rows
        ]

    def get_applications_by_item(
        self,
        *,
        user_id: int,
        item_code: str,
        application_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """查询同一事项的全部申请，并附带每次申请的完整状态流水。"""
        with self.engine.begin() as connection:
            applications = connection.execute(
                text(
                    """
                    SELECT
                        a.id AS application_id,
                        COALESCE(NULLIF(a.item_code, ''), a.payload->>'item_code', '') AS item_code,
                        COALESCE(NULLIF(a.item_name, ''), a.payload->>'item_name', a.payload->>'official_item_name', a.payload->>'title', '') AS item_name,
                        a.biz_type,
                        a.region_code,
                        COALESCE(s.status, a.status) AS current_status,
                        COALESCE(s.current_node, a.current_node) AS current_node,
                        COALESCE(s.node_name, a.node_name) AS node_name,
                        r.reject_code,
                        r.reject_reason,
                        r.material_code,
                        r.material_name,
                        a.submitted_at,
                        a.completed_at,
                        GREATEST(a.updated_at, COALESCE(s.updated_at, a.updated_at), COALESCE(r.updated_at, a.updated_at)) AS updated_at
                    FROM declaration_qa.business_applications a
                    LEFT JOIN declaration_qa.business_application_statuses s
                        ON s.apply_id = a.id AND s.is_current = TRUE
                    LEFT JOIN declaration_qa.business_reject_infos r
                        ON r.apply_id = a.id AND r.is_current = TRUE
                    WHERE a.user_id = :user_id
                      AND a.deleted_at IS NULL
                      AND COALESCE(NULLIF(a.item_code, ''), a.payload->>'item_code', '') = :item_code
                      AND (:application_id IS NULL OR a.id = :application_id)
                    ORDER BY GREATEST(a.updated_at, COALESCE(s.updated_at, a.updated_at)) DESC, a.id DESC
                    """
                ),
                {
                    "user_id": int(user_id),
                    "item_code": str(item_code or "").strip(),
                    "application_id": int(application_id) if application_id is not None else None,
                },
            ).mappings().all()
            application_ids = [int(row["application_id"]) for row in applications]
            if not application_ids:
                return []
            status_rows = connection.execute(
                text(
                    """
                    SELECT
                        s.id AS event_id,
                        s.apply_id AS application_id,
                        s.sequence_no,
                        s.from_status,
                        s.status,
                        s.current_node,
                        s.node_name,
                        s.reason_code,
                        s.reason,
                        s.material_code,
                        s.material_name,
                        s.is_current,
                        s.occurred_at AS changed_at
                    FROM declaration_qa.business_application_statuses s
                    WHERE s.apply_id = ANY(:application_ids)
                    ORDER BY s.apply_id, s.sequence_no, s.occurred_at, s.id
                    """
                ),
                {"application_ids": application_ids},
            ).mappings().all()
        history_by_application: dict[int, list[dict[str, Any]]] = {application_id: [] for application_id in application_ids}
        for row in status_rows:
            history_by_application[int(row["application_id"])].append(
                {
                    "event_id": int(row["event_id"]),
                    "sequence_no": int(row["sequence_no"] or 0),
                    "from_status": row["from_status"],
                    "status": str(row["status"] or ""),
                    "current_node": str(row["current_node"] or ""),
                    "node_name": str(row["node_name"] or ""),
                    "reason_code": str(row["reason_code"] or ""),
                    "reason": str(row["reason"] or ""),
                    "material_code": str(row["material_code"] or ""),
                    "material_name": str(row["material_name"] or ""),
                    "is_current": bool(row["is_current"]),
                    "changed_at": row["changed_at"].isoformat() if row["changed_at"] else "",
                }
            )
        output = []
        for row in applications:
            application = dict(row)
            application_key = int(application["application_id"])
            output.append(
                {
                    "application_id": application_key,
                    "item_code": str(application.get("item_code") or ""),
                    "item_name": str(application.get("item_name") or ""),
                    "biz_type": str(application.get("biz_type") or ""),
                    "region_code": str(application.get("region_code") or ""),
                    "current_status": str(application.get("current_status") or ""),
                    "current_node": str(application.get("current_node") or ""),
                    "node_name": str(application.get("node_name") or ""),
                    "reject_code": str(application.get("reject_code") or ""),
                    "reject_reason": str(application.get("reject_reason") or ""),
                    "material_code": str(application.get("material_code") or ""),
                    "material_name": str(application.get("material_name") or ""),
                    "submitted_at": application["submitted_at"].isoformat() if application.get("submitted_at") else "",
                    "completed_at": application["completed_at"].isoformat() if application.get("completed_at") else "",
                    "updated_at": application["updated_at"].isoformat() if application.get("updated_at") else "",
                    "status_history": history_by_application.get(application_key, []),
                }
            )
        return output

