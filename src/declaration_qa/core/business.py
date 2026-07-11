"""办件业务数据访问协议和本地内存示例实现。"""

from __future__ import annotations

from typing import Any, Protocol


class BusinessDataAdapter(Protocol):
    """办件业务数据访问适配器协议。"""
    def list_ticket_items(self, *, user_id: int, region_code: str | None = None) -> list[dict[str, Any]]:
        """按事项汇总当前用户的全部申请。"""
        ...

    def get_applications_by_item(
        self,
        *,
        user_id: int,
        item_code: str,
        application_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """返回同一事项下的申请及每次申请的完整状态历史。"""
        ...

class InMemoryBusinessDataAdapter:
    """独立问答应用使用的业务数据适配器。

    生产环境通过 PostgresDeclarationQAStore 使用 PostgreSQL 适配器；
    这里保留给没有数据库的本地开发场景。
    """

    def __init__(self) -> None:
        """初始化内存业务适配器。"""
        self._applications: dict[str, dict[str, Any]] = {
            "A1001": {
                "application_id": 1001,
                "user_id": 1,
                "item_code": "ITEM-HEALTH-CERTIFICATE",
                "item_name": "健康证办理",
                "biz_type": "health_certificate",
                "region_code": "340000",
                "status": "in_review",
                "current_node": "DEPT_REVIEW",
                "node_name": "部门审核中",
                "submitted_at": "2026-06-15T09:00:00+08:00",
                "updated_at": "2026-06-15T10:20:00+08:00",
                "payload": {"channel": "wanshitong", "sla_hours": 72},
                "status_history": [
                    {"event_id": 10011, "sequence_no": 1, "from_status": None, "status": "submitted", "node_name": "申请提交", "changed_at": "2026-06-15T09:00:00+08:00"},
                    {"event_id": 10012, "sequence_no": 2, "from_status": "submitted", "status": "in_review", "current_node": "DEPT_REVIEW", "node_name": "部门审核中", "changed_at": "2026-06-15T10:20:00+08:00"},
                ],
            },
            "A2002": {
                "application_id": 2002,
                "user_id": 1,
                "item_code": "ITEM-RESIDENCE-PERMIT",
                "item_name": "居住证办理",
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
                "submitted_at": "2026-06-14T14:00:00+08:00",
                "updated_at": "2026-06-15T09:30:00+08:00",
                "payload": {"channel": "wanshitong", "review_batch": "MATERIAL-20260615"},
                "status_history": [
                    {"event_id": 20021, "sequence_no": 1, "from_status": None, "status": "submitted", "node_name": "申请提交", "changed_at": "2026-06-14T14:00:00+08:00"},
                    {"event_id": 20022, "sequence_no": 2, "from_status": "submitted", "status": "rejected", "current_node": "MATERIAL_REVIEW", "node_name": "材料审核", "reason_code": "MISSING_RESIDENCE_PROOF", "reason": "缺少有效居住证明材料", "material_code": "RESIDENCE_PROOF", "material_name": "居住证明", "changed_at": "2026-06-15T09:30:00+08:00"},
                ],
            },
            "A3003": {
                "application_id": 3003,
                "user_id": 1,
                "item_code": "ITEM-ID-CARD-REPLACE",
                "item_name": "身份证换领",
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
                "submitted_at": "2026-06-14T16:10:00+08:00",
                "updated_at": "2026-06-15T08:40:00+08:00",
                "payload": {"channel": "wanshitong", "review_batch": "PHOTO-20260615"},
                "status_history": [
                    {"event_id": 30031, "sequence_no": 1, "from_status": None, "status": "submitted", "node_name": "申请提交", "changed_at": "2026-06-14T16:10:00+08:00"},
                    {"event_id": 30032, "sequence_no": 2, "from_status": "submitted", "status": "rejected", "current_node": "PHOTO_REVIEW", "node_name": "照片审核", "reason_code": "PHOTO_BLUR", "reason": "身份证正面照片不清晰", "material_code": "ID_CARD_FRONT", "material_name": "身份证正面照片", "changed_at": "2026-06-15T08:40:00+08:00"},
                ],
            },
        }

    def list_ticket_items(self, *, user_id: int, region_code: str | None = None) -> list[dict[str, Any]]:
        """按 item_code 汇总用户申请，不暴露外部申请单标识。"""
        groups: dict[str, dict[str, Any]] = {}
        for item in self._applications.values():
            if int(item.get("user_id") or 0) != int(user_id):
                continue
            if region_code and not region_code_matches(item.get("region_code"), region_code):
                continue
            item_code = str(item.get("item_code") or "").strip()
            if not item_code:
                continue
            group = groups.setdefault(
                item_code,
                {
                    "item_code": item_code,
                    "item_name": str(item.get("item_name") or ticket_title(item)),
                    "application_count": 0,
                    "statuses": [],
                    "updated_at": "",
                },
            )
            group["application_count"] += 1
            status = str(item.get("status") or "")
            if status and status not in group["statuses"]:
                group["statuses"].append(status)
            group["updated_at"] = max(str(group.get("updated_at") or ""), str(item.get("updated_at") or ""))
        return sorted(groups.values(), key=lambda row: str(row.get("updated_at") or ""), reverse=True)

    def get_applications_by_item(
        self,
        *,
        user_id: int,
        item_code: str,
        application_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """返回事项下全部申请和有序状态流水。"""
        output = []
        for item in self._applications.values():
            if int(item.get("user_id") or 0) != int(user_id):
                continue
            if str(item.get("item_code") or "") != str(item_code or ""):
                continue
            if application_id is not None and int(item.get("application_id") or 0) != int(application_id):
                continue
            output.append(application_detail(item))
        return sorted(output, key=lambda row: str(row.get("updated_at") or ""), reverse=True)

def application_detail(item: dict[str, Any]) -> dict[str, Any]:
    """构造 Agent 工具使用的申请详情和完整状态流水。"""
    history = sorted(
        [dict(event) for event in (item.get("status_history") or []) if isinstance(event, dict)],
        key=lambda event: (int(event.get("sequence_no") or 0), str(event.get("changed_at") or "")),
    )
    return {
        "application_id": int(item.get("application_id") or 0),
        "item_code": str(item.get("item_code") or ""),
        "item_name": str(item.get("item_name") or ticket_title(item)),
        "biz_type": str(item.get("biz_type") or ""),
        "region_code": str(item.get("region_code") or ""),
        "current_status": str(item.get("status") or ""),
        "current_node": str(item.get("current_node") or ""),
        "node_name": str(item.get("node_name") or ""),
        "submitted_at": str(item.get("submitted_at") or ""),
        "completed_at": str(item.get("completed_at") or ""),
        "updated_at": str(item.get("updated_at") or ""),
        "status_history": history,
    }


def ticket_title(item: dict[str, Any]) -> str:
    """根据业务类型生成办件标题。"""
    names = {
        "health_certificate": "健康证办理",
        "residence_permit": "居住证办理",
        "id_card_replace": "身份证换领",
    }
    return names.get(str(item.get("biz_type") or ""), str(item.get("biz_type") or "政务服务事项"))


def region_code_variants(value: Any) -> list[str]:
    """生成行政区划编码的等价匹配形式。"""
    code = str(value or "").strip()
    if not code:
        return []
    variants = {code}
    if code.isdigit():
        if len(code) == 6:
            variants.add(f"{code}000000")
        elif len(code) == 12 and code.endswith("000000"):
            variants.add(code[:6])
    return sorted(variants)


def region_code_matches(candidate: Any, requested: Any) -> bool:
    """判断候选行政区划是否匹配请求编码。"""
    requested_variants = set(region_code_variants(requested))
    if not requested_variants:
        return True
    return str(candidate or "").strip() in requested_variants

