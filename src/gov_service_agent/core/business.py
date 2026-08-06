"""办件业务数据访问协议和本地内存示例实现。"""

from __future__ import annotations

from datetime import datetime
from hashlib import sha256
from threading import Lock
from typing import Any, Protocol
from zoneinfo import ZoneInfo


class AppointmentBookingError(ValueError):
    """预约写入失败，并携带可供工具返回的稳定业务状态。"""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class AppointmentCancellationError(ValueError):
    """预约取消失败，并携带可供工具返回的稳定业务状态。"""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class BusinessDataAdapter(Protocol):
    """办件业务数据访问适配器协议。"""
    def list_ticket_items(self, *, user_id: int, region_code: str | None = None) -> list[dict[str, Any]]:
        """按事项汇总当前用户的全部申请。"""
        ...

    def get_applications_by_items(
        self,
        *,
        user_id: int,
        item_codes: list[str],
    ) -> list[dict[str, Any]]:
        """批量返回多个事项下的申请及完整状态历史。"""
        ...

    def list_appointments(
        self,
        *,
        user_id: int,
        region_code: str | None = None,
        status: str = "active",
    ) -> list[dict[str, Any]]:
        """查询当前用户的预约记录。"""
        ...

    def query_appointment_slots(
        self,
        *,
        item_code: str,
        region_code: str | None = None,
        service_date: str | None = None,
        hall_code: str | None = None,
    ) -> list[dict[str, Any]]:
        """查询事项在指定地区的可预约号源。"""
        ...

    def book_appointment(
        self,
        *,
        user_id: int,
        region_code: str | None,
        slot_id: int,
        idempotency_key: str,
    ) -> dict[str, Any]:
        """使用可信用户身份预约指定号源。"""
        ...

    def cancel_appointment(
        self,
        *,
        user_id: int,
        region_code: str | None,
        appointment_id: int,
    ) -> dict[str, Any]:
        """取消当前用户尚未签到的预约。"""
        ...

    def get_queue_status(self, *, user_id: int, appointment_id: int) -> dict[str, Any] | None:
        """查询当前用户指定预约的排队状态。"""
        ...

    def get_payment_orders_by_applications(
        self,
        *,
        user_id: int,
        application_ids: list[int],
    ) -> list[dict[str, Any]]:
        """批量查询多个申请的缴费订单。"""
        ...

    def get_payment_receipts_by_targets(
        self,
        *,
        user_id: int,
        targets: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """批量查询申请及其可选缴费订单对应的票据。"""
        ...

    def get_submission_receipts_by_applications(
        self, *, user_id: int, application_ids: list[int]
    ) -> list[dict[str, Any]]:
        """批量查询申请的提交和受理回执。"""
        ...

    def get_result_certificates_by_applications(
        self, *, user_id: int, application_ids: list[int]
    ) -> list[dict[str, Any]]:
        """批量查询申请的办结结果和电子证照。"""
        ...

    def get_delivery_statuses_by_applications(
        self, *, user_id: int, application_ids: list[int]
    ) -> list[dict[str, Any]]:
        """批量查询申请的结果送达状态。"""
        ...

    def list_consultation_records(
        self,
        *,
        user_id: int,
        status: str = "all",
        consultation_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """查询当前用户的咨询、投诉和建议记录。"""
        ...

    def get_application_form_schema(
        self,
        *,
        item_code: str,
        region_code: str | None = None,
    ) -> dict[str, Any] | None:
        """读取事项当前生效的版本化申报字段 Schema。"""
        ...

    def submit_application(
        self,
        *,
        user_id: int,
        region_code: str | None,
        item_code: str,
        item_name: str,
        biz_type: str,
        schema_version: str,
        form_data: dict[str, Any],
        idempotency_key: str,
    ) -> dict[str, Any]:
        """提交已通过 Schema 校验且经过人工确认的申报。"""
        ...

    def create_complaint(
        self,
        *,
        user_id: int,
        region_code: str | None,
        values: dict[str, Any],
        schema_version: str,
        idempotency_key: str,
    ) -> dict[str, Any]:
        """创建已通过校验且经过人工确认的投诉。"""
        ...

class InMemoryBusinessDataAdapter:
    """独立问答应用使用的业务数据适配器。

    生产环境通过 PostgresGovServiceAgentStore 使用 PostgreSQL 适配器；
    这里保留给没有数据库的本地开发场景。
    """

    def __init__(self) -> None:
        """初始化内存业务适配器。"""
        self._appointment_lock = Lock()
        self._write_lock = Lock()
        self._write_operations: dict[str, dict[str, Any]] = {}
        self._form_schemas = default_application_form_schemas()
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
            "A4004": {
                "application_id": 4004,
                "user_id": 1,
                "item_code": "ITEM-ARCHIVE-COLLECTION",
                "item_name": "流动人员档案材料收集",
                "biz_type": "archive_collection",
                "region_code": "340000",
                "status": "completed",
                "current_node": "FINISH",
                "node_name": "办结",
                "submitted_at": "2026-06-10T09:15:00+08:00",
                "completed_at": "2026-06-13T15:30:00+08:00",
                "updated_at": "2026-06-15T14:10:00+08:00",
                "payload": {"channel": "wanshitong", "delivery_mode": "mail"},
                "status_history": [
                    {"event_id": 40041, "sequence_no": 1, "from_status": None, "status": "submitted", "node_name": "申请提交", "changed_at": "2026-06-10T09:15:00+08:00"},
                    {"event_id": 40042, "sequence_no": 2, "from_status": "submitted", "status": "accepted", "node_name": "受理", "changed_at": "2026-06-10T10:05:00+08:00"},
                    {"event_id": 40043, "sequence_no": 3, "from_status": "accepted", "status": "completed", "node_name": "办结", "changed_at": "2026-06-13T15:30:00+08:00"},
                ],
            },
        }
        self._appointment_slots = [
            {
                "slot_id": 6101,
                "item_code": "ITEM-RESIDENCE-PERMIT",
                "item_name": "居住证办理",
                "region_code": "340000",
                "hall_code": "AH-ZW-01",
                "hall_name": "安徽省政务服务中心",
                "hall_address": "马鞍山路509号",
                "service_date": "2026-07-16",
                "start_time": "09:00",
                "end_time": "10:00",
                "capacity": 20,
                "reserved_count": 14,
                "status": "available",
            },
            {
                "slot_id": 6102,
                "item_code": "ITEM-RESIDENCE-PERMIT",
                "item_name": "居住证办理",
                "region_code": "340000",
                "hall_code": "AH-ZW-01",
                "hall_name": "安徽省政务服务中心",
                "hall_address": "马鞍山路509号",
                "service_date": "2026-07-16",
                "start_time": "10:00",
                "end_time": "11:00",
                "capacity": 20,
                "reserved_count": 20,
                "status": "full",
            },
            {
                "slot_id": 6103,
                "item_code": "ITEM-RESIDENCE-PERMIT",
                "item_name": "居住证办理",
                "region_code": "340000",
                "hall_code": "AH-ZW-01",
                "hall_name": "安徽省政务服务中心",
                "hall_address": "马鞍山路509号",
                "service_date": "2026-07-16",
                "start_time": "11:00",
                "end_time": "12:00",
                "capacity": 20,
                "reserved_count": 10,
                "status": "available",
            },
            {
                "slot_id": 6201,
                "source_manifest_id": "item_28_official_ahzwfw",
                "item_code": "11340000002985923L200011900100001",
                "item_name": "取水许可",
                "region_code": "340000",
                "hall_code": "AH-ZW-GCJS",
                "hall_name": "安徽省政务服务中心工程建设综合窗口",
                "hall_address": "安徽省合肥市包河区马鞍山路509号",
                "service_date": "2026-07-20",
                "start_time": "09:00",
                "end_time": "10:00",
                "capacity": 12,
                "reserved_count": 5,
                "status": "available",
            },
            {
                "slot_id": 6202,
                "source_manifest_id": "item_28_official_ahzwfw",
                "item_code": "11340000002985923L200011900100001",
                "item_name": "取水许可",
                "region_code": "340000",
                "hall_code": "AH-ZW-GCJS",
                "hall_name": "安徽省政务服务中心工程建设综合窗口",
                "hall_address": "安徽省合肥市包河区马鞍山路509号",
                "service_date": "2026-07-20",
                "start_time": "10:00",
                "end_time": "11:00",
                "capacity": 12,
                "reserved_count": 12,
                "status": "full",
            },
            {
                "slot_id": 6203,
                "source_manifest_id": "item_28_official_ahzwfw",
                "item_code": "11340000002985923L200011900100001",
                "item_name": "取水许可",
                "region_code": "340000",
                "hall_code": "AH-ZW-GCJS",
                "hall_name": "安徽省政务服务中心工程建设综合窗口",
                "hall_address": "安徽省合肥市包河区马鞍山路509号",
                "service_date": "2026-07-20",
                "start_time": "13:30",
                "end_time": "14:30",
                "capacity": 12,
                "reserved_count": 2,
                "status": "available",
            },
            {
                "slot_id": 6301,
                "source_manifest_id": "item_28_official_ahzwfw",
                "item_code": "11340000002985923L200011900100001",
                "item_name": "取水许可",
                "region_code": "340000",
                "hall_code": "AH-SLT-SLZW",
                "hall_name": "安徽省政务服务中心水利窗口",
                "hall_address": "安徽省合肥市包河区马鞍山路509号二楼水利服务区",
                "service_date": "2026-07-20",
                "start_time": "08:30",
                "end_time": "09:30",
                "capacity": 10,
                "reserved_count": 3,
                "status": "available",
            },
            {
                "slot_id": 6302,
                "source_manifest_id": "item_28_official_ahzwfw",
                "item_code": "11340000002985923L200011900100001",
                "item_name": "取水许可",
                "region_code": "340000",
                "hall_code": "AH-SLT-SLZW",
                "hall_name": "安徽省政务服务中心水利窗口",
                "hall_address": "安徽省合肥市包河区马鞍山路509号二楼水利服务区",
                "service_date": "2026-07-20",
                "start_time": "14:00",
                "end_time": "15:00",
                "capacity": 10,
                "reserved_count": 4,
                "status": "available",
            },
            {
                "slot_id": 6303,
                "source_manifest_id": "item_28_official_ahzwfw",
                "item_code": "11340000002985923L200011900100001",
                "item_name": "取水许可",
                "region_code": "340000",
                "hall_code": "AH-HF-SWZW",
                "hall_name": "合肥市政务服务中心水务窗口",
                "hall_address": "安徽省合肥市蜀山区怀宁路100号三楼水务服务区",
                "service_date": "2026-07-20",
                "start_time": "11:00",
                "end_time": "12:00",
                "capacity": 8,
                "reserved_count": 2,
                "status": "available",
            },
            {
                "slot_id": 6304,
                "source_manifest_id": "item_28_official_ahzwfw",
                "item_code": "11340000002985923L200011900100001",
                "item_name": "取水许可",
                "region_code": "340000",
                "hall_code": "AH-HF-SWZW",
                "hall_name": "合肥市政务服务中心水务窗口",
                "hall_address": "安徽省合肥市蜀山区怀宁路100号三楼水务服务区",
                "service_date": "2026-07-20",
                "start_time": "15:00",
                "end_time": "16:00",
                "capacity": 8,
                "reserved_count": 1,
                "status": "available",
            },
            {
                "slot_id": 6204,
                "source_manifest_id": "item_30_official_ahzwfw",
                "item_code": "11340000002985990J200016407900001",
                "item_name": "草原防火期内在草原上进行爆破、勘察和施工等活动审批",
                "region_code": "340000",
                "hall_code": "AH-LYJ-ZW",
                "hall_name": "安徽省人民政府政务服务中心林业窗口",
                "hall_address": "安徽省合肥市包河区马鞍山路509号",
                "service_date": "2026-07-21",
                "start_time": "09:00",
                "end_time": "10:00",
                "capacity": 8,
                "reserved_count": 3,
                "status": "available",
            },
            {
                "slot_id": 6205,
                "source_manifest_id": "item_30_official_ahzwfw",
                "item_code": "11340000002985990J200016407900001",
                "item_name": "草原防火期内在草原上进行爆破、勘察和施工等活动审批",
                "region_code": "340000",
                "hall_code": "AH-LYJ-ZW",
                "hall_name": "安徽省人民政府政务服务中心林业窗口",
                "hall_address": "安徽省合肥市包河区马鞍山路509号",
                "service_date": "2026-07-21",
                "start_time": "14:00",
                "end_time": "15:00",
                "capacity": 8,
                "reserved_count": 7,
                "status": "available",
            },
            {
                "slot_id": 6206,
                "source_manifest_id": "high_risk_sports_event_permit_official_ahzwfw",
                "item_code": "11340000002985974W234013300500001",
                "item_name": "举办高危险性体育赛事活动许可",
                "region_code": "340000",
                "hall_code": "AH-TYJ-01",
                "hall_name": "安徽省体育局政务服务窗口",
                "hall_address": "安徽省合肥市包河区芜湖路97号",
                "service_date": "2026-07-22",
                "start_time": "09:00",
                "end_time": "10:00",
                "capacity": 10,
                "reserved_count": 4,
                "status": "available",
            },
            {
                "slot_id": 6207,
                "source_manifest_id": "high_risk_sports_event_permit_official_ahzwfw",
                "item_code": "11340000002985974W234013300500001",
                "item_name": "举办高危险性体育赛事活动许可",
                "region_code": "340000",
                "hall_code": "AH-TYJ-01",
                "hall_name": "安徽省体育局政务服务窗口",
                "hall_address": "安徽省合肥市包河区芜湖路97号",
                "service_date": "2026-07-22",
                "start_time": "15:00",
                "end_time": "16:00",
                "capacity": 10,
                "reserved_count": 10,
                "status": "full",
            },
            {
                "slot_id": 6208,
                "source_manifest_id": "foreign_hmt_overseas_chinese_marriage_registration_official_ahzwfw",
                "item_code": "11340000002986424J200071100100002",
                "item_name": "涉外、涉港澳台、涉华侨结婚登记",
                "region_code": "340000",
                "hall_code": "AH-MZT-102",
                "hall_name": "省涉外婚姻收养登记服务中心102室",
                "hall_address": "安徽省合肥市包河区屯溪路435号安徽省民政厅102室",
                "service_date": "2026-07-23",
                "start_time": "08:00",
                "end_time": "08:30",
                "capacity": 6,
                "reserved_count": 4,
                "status": "available",
            },
            {
                "slot_id": 6209,
                "source_manifest_id": "foreign_hmt_overseas_chinese_marriage_registration_official_ahzwfw",
                "item_code": "11340000002986424J200071100100002",
                "item_name": "涉外、涉港澳台、涉华侨结婚登记",
                "region_code": "340000",
                "hall_code": "AH-MZT-102",
                "hall_name": "省涉外婚姻收养登记服务中心102室",
                "hall_address": "安徽省合肥市包河区屯溪路435号安徽省民政厅102室",
                "service_date": "2026-07-23",
                "start_time": "09:00",
                "end_time": "09:30",
                "capacity": 6,
                "reserved_count": 1,
                "status": "available",
            },
            {
                "slot_id": 6210,
                "source_manifest_id": "foreign_hmt_overseas_chinese_marriage_registration_official_ahzwfw",
                "item_code": "11340000002986424J200071100100002",
                "item_name": "涉外、涉港澳台、涉华侨结婚登记",
                "region_code": "340000",
                "hall_code": "AH-MZT-102",
                "hall_name": "省涉外婚姻收养登记服务中心102室",
                "hall_address": "安徽省合肥市包河区屯溪路435号安徽省民政厅102室",
                "service_date": "2026-07-23",
                "start_time": "14:30",
                "end_time": "15:00",
                "capacity": 6,
                "reserved_count": 6,
                "status": "full",
            },
            {
                "slot_id": 6211,
                "source_manifest_id": "fire_accident_identification_review_official_ahzwfw",
                "item_code": "11340000MB1C86468R234070901300001",
                "item_name": "火灾事故认定复核",
                "region_code": "340000",
                "hall_code": "AH-XFJY-C1",
                "hall_name": "安徽省消防救援总队政务窗口",
                "hall_address": "安徽省合肥市广西路与中山路交叉口中山路C1-104号",
                "service_date": "2026-07-24",
                "start_time": "08:30",
                "end_time": "09:30",
                "capacity": 8,
                "reserved_count": 2,
                "status": "available",
            },
            {
                "slot_id": 6212,
                "source_manifest_id": "fire_accident_identification_review_official_ahzwfw",
                "item_code": "11340000MB1C86468R234070901300001",
                "item_name": "火灾事故认定复核",
                "region_code": "340000",
                "hall_code": "AH-XFJY-C1",
                "hall_name": "安徽省消防救援总队政务窗口",
                "hall_address": "安徽省合肥市广西路与中山路交叉口中山路C1-104号",
                "service_date": "2026-07-24",
                "start_time": "14:30",
                "end_time": "15:30",
                "capacity": 8,
                "reserved_count": 5,
                "status": "available",
            },
        ]
        self._appointments = [
            {
                "appointment_id": 4101,
                "slot_id": 6101,
                "user_id": 1,
                "item_code": "ITEM-RESIDENCE-PERMIT",
                "item_name": "居住证办理",
                "region_code": "340000",
                "hall_code": "AH-ZW-01",
                "hall_name": "安徽省政务服务中心",
                "hall_address": "马鞍山路509号",
                "service_date": "2026-07-16",
                "start_time": "09:00",
                "end_time": "10:00",
                "status": "checked_in",
                "queue_number": "A018",
                "people_ahead": 4,
                "estimated_wait_minutes": 16,
                "counter_name": "综合窗口03",
                "checked_in_at": "2026-07-16T09:08:00+08:00",
                "updated_at": "2026-07-16T09:12:00+08:00",
            },
            {
                "appointment_id": 4201,
                "slot_id": 6102,
                "user_id": 2,
                "item_code": "ITEM-RESIDENCE-PERMIT",
                "item_name": "居住证办理",
                "region_code": "340000",
                "hall_code": "AH-ZW-01",
                "hall_name": "安徽省政务服务中心",
                "hall_address": "马鞍山路509号",
                "service_date": "2026-07-16",
                "start_time": "10:00",
                "end_time": "11:00",
                "status": "booked",
                "queue_number": "",
                "people_ahead": 0,
                "estimated_wait_minutes": 0,
                "counter_name": "",
                "checked_in_at": "",
                "updated_at": "2026-07-15T15:00:00+08:00",
            },
            {
                "appointment_id": 4301,
                "slot_id": 6201,
                "user_id": 1,
                "item_code": "11340000002985923L200011900100001",
                "item_name": "取水许可",
                "region_code": "340000",
                "hall_code": "AH-ZW-GCJS",
                "hall_name": "安徽省政务服务中心工程建设综合窗口",
                "hall_address": "安徽省合肥市包河区马鞍山路509号",
                "service_date": "2026-07-20",
                "start_time": "09:00",
                "end_time": "10:00",
                "status": "booked",
                "updated_at": "2026-07-15T10:00:00+08:00",
            },
            {
                "appointment_id": 4302,
                "slot_id": 6205,
                "user_id": 1,
                "item_code": "11340000002985990J200016407900001",
                "item_name": "草原防火期内在草原上进行爆破、勘察和施工等活动审批",
                "region_code": "340000",
                "hall_code": "AH-LYJ-ZW",
                "hall_name": "安徽省人民政府政务服务中心林业窗口",
                "hall_address": "安徽省合肥市包河区马鞍山路509号",
                "service_date": "2026-07-21",
                "start_time": "14:00",
                "end_time": "15:00",
                "status": "booked",
                "updated_at": "2026-07-15T10:10:00+08:00",
            },
            {
                "appointment_id": 4303,
                "slot_id": 6207,
                "user_id": 1,
                "item_code": "11340000002985974W234013300500001",
                "item_name": "举办高危险性体育赛事活动许可",
                "region_code": "340000",
                "hall_code": "AH-TYJ-01",
                "hall_name": "安徽省体育局政务服务窗口",
                "hall_address": "安徽省合肥市包河区芜湖路97号",
                "service_date": "2026-07-22",
                "start_time": "15:00",
                "end_time": "16:00",
                "status": "cancelled",
                "updated_at": "2026-07-15T09:20:00+08:00",
            },
            {
                "appointment_id": 4304,
                "slot_id": 6209,
                "user_id": 1,
                "item_code": "11340000002986424J200071100100002",
                "item_name": "涉外、涉港澳台、涉华侨结婚登记",
                "region_code": "340000",
                "hall_code": "AH-MZT-102",
                "hall_name": "省涉外婚姻收养登记服务中心102室",
                "hall_address": "安徽省合肥市包河区屯溪路435号安徽省民政厅102室",
                "service_date": "2026-07-23",
                "start_time": "09:00",
                "end_time": "09:30",
                "status": "booked",
                "updated_at": "2026-07-15T11:00:00+08:00",
            },
            {
                "appointment_id": 4305,
                "slot_id": 6211,
                "user_id": 1,
                "item_code": "11340000MB1C86468R234070901300001",
                "item_name": "火灾事故认定复核",
                "region_code": "340000",
                "hall_code": "AH-XFJY-C1",
                "hall_name": "安徽省消防救援总队政务窗口",
                "hall_address": "安徽省合肥市广西路与中山路交叉口中山路C1-104号",
                "service_date": "2026-07-24",
                "start_time": "08:30",
                "end_time": "09:30",
                "status": "booked",
                "updated_at": "2026-07-15T11:15:00+08:00",
            },
            {
                "appointment_id": 4306,
                "slot_id": 6208,
                "user_id": 2,
                "item_code": "11340000002986424J200071100100002",
                "item_name": "涉外、涉港澳台、涉华侨结婚登记",
                "region_code": "340000",
                "hall_code": "AH-MZT-102",
                "hall_name": "省涉外婚姻收养登记服务中心102室",
                "hall_address": "安徽省合肥市包河区屯溪路435号安徽省民政厅102室",
                "service_date": "2026-07-23",
                "start_time": "08:00",
                "end_time": "08:30",
                "status": "booked",
                "updated_at": "2026-07-15T12:00:00+08:00",
            },
        ]
        self._payment_orders = [
            {
                "payment_order_id": 5101,
                "application_id": 1001,
                "user_id": 1,
                "order_no": "PAY-20260615-1001",
                "fee_item": "健康检查服务费",
                "amount": "20.00",
                "currency": "CNY",
                "status": "paid",
                "payment_channel": "wanshitong",
                "payable_at": "2026-06-15T09:05:00+08:00",
                "expires_at": "2026-06-16T09:05:00+08:00",
                "paid_at": "2026-06-15T09:12:00+08:00",
                "updated_at": "2026-06-15T09:12:00+08:00",
            }
        ]
        self._payment_receipts = [
            {
                "payment_receipt_id": 5201,
                "payment_order_id": 5101,
                "application_id": 1001,
                "user_id": 1,
                "receipt_no": "DZPJ-20260615-1001",
                "receipt_type": "electronic_invoice",
                "title": "个人",
                "amount": "20.00",
                "status": "issued",
                "issued_at": "2026-06-15T09:20:00+08:00",
                "file_name": "健康检查服务费电子票据.pdf",
                "download_status": "available",
            }
        ]
        self._submission_receipts = [
            {
                "submission_receipt_id": 5301,
                "application_id": 2002,
                "user_id": 1,
                "receipt_no": "SLHZ-20260614-2002",
                "receipt_type": "submission",
                "status": "issued",
                "issued_at": "2026-06-14T14:01:00+08:00",
                "file_name": "居住证申请提交回执.pdf",
                "download_status": "available",
            },
            {
                "submission_receipt_id": 5302,
                "application_id": 4004,
                "user_id": 1,
                "receipt_no": "SLHZ-20260610-4004",
                "receipt_type": "acceptance",
                "status": "issued",
                "issued_at": "2026-06-10T10:06:00+08:00",
                "file_name": "档案材料收集受理回执.pdf",
                "download_status": "available",
            },
        ]
        self._result_certificates = [
            {
                "result_certificate_id": 5401,
                "application_id": 4004,
                "user_id": 1,
                "certificate_name": "档案材料接收确认单",
                "certificate_no": "DAJS-2026-4004",
                "certificate_type": "electronic_result",
                "status": "issued",
                "issued_at": "2026-06-13T15:30:00+08:00",
                "valid_until": "",
                "file_name": "档案材料接收确认单.pdf",
                "download_status": "available",
            }
        ]
        self._deliveries = [
            {
                "delivery_id": 5501,
                "application_id": 4004,
                "user_id": 1,
                "delivery_type": "mail",
                "status": "delivered",
                "courier_name": "中国邮政EMS",
                "tracking_no": "EMS202606134004",
                "pickup_address": "",
                "shipped_at": "2026-06-13T17:00:00+08:00",
                "delivered_at": "2026-06-15T14:10:00+08:00",
                "updated_at": "2026-06-15T14:10:00+08:00",
            }
        ]
        self._consultations = [
            {
                "consultation_id": 5601,
                "user_id": 1,
                "consultation_type": "consultation",
                "subject": "居住证补正材料咨询",
                "content_summary": "咨询居住证明缺失后的补正方式。",
                "status": "replied",
                "channel": "online",
                "created_at": "2026-06-15T10:00:00+08:00",
                "replied_at": "2026-06-15T11:20:00+08:00",
                "reply_summary": "请补充有效租赁合同或社区出具的居住证明后重新提交。",
            },
            {
                "consultation_id": 5602,
                "user_id": 2,
                "consultation_type": "complaint",
                "subject": "窗口等候时间投诉",
                "content_summary": "反映窗口等候时间较长。",
                "status": "processing",
                "channel": "online",
                "created_at": "2026-06-15T09:00:00+08:00",
                "replied_at": "",
                "reply_summary": "",
            },
            {
                "consultation_id": 5701,
                "user_id": 1,
                "consultation_type": "consultation",
                "subject": "取水许可现场预约咨询",
                "content_summary": "咨询取水许可是否可以预约工程建设综合窗口。",
                "status": "replied",
                "channel": "online",
                "created_at": "2026-07-15T09:00:00+08:00",
                "replied_at": "2026-07-15T09:20:00+08:00",
                "reply_summary": "可以预约马鞍山路509号工程建设综合窗口，请按预约时段携带核验材料。",
                "source_manifest_id": "item_28_official_ahzwfw",
            },
            {
                "consultation_id": 5702,
                "user_id": 1,
                "consultation_type": "consultation",
                "subject": "高危险性体育赛事材料咨询",
                "content_summary": "咨询赛事活动许可的场地和安全保障材料。",
                "status": "processing",
                "channel": "online",
                "created_at": "2026-07-15T10:30:00+08:00",
                "replied_at": "",
                "reply_summary": "",
                "source_manifest_id": "high_risk_sports_event_permit_official_ahzwfw",
            },
            {
                "consultation_id": 5703,
                "user_id": 1,
                "consultation_type": "consultation",
                "subject": "涉外婚姻登记预约咨询",
                "content_summary": "咨询双方是否需要共同到场以及可预约时段。",
                "status": "replied",
                "channel": "online",
                "created_at": "2026-07-15T11:10:00+08:00",
                "replied_at": "2026-07-15T11:40:00+08:00",
                "reply_summary": "该事项为窗口办理，请双方按预约时间共同前往省涉外婚姻收养登记服务中心。",
                "source_manifest_id": "foreign_hmt_overseas_chinese_marriage_registration_official_ahzwfw",
            },
            {
                "consultation_id": 5704,
                "user_id": 2,
                "consultation_type": "consultation",
                "subject": "火灾事故认定复核办理地点咨询",
                "content_summary": "咨询火灾事故认定复核的线下办理地点。",
                "status": "replied",
                "channel": "online",
                "created_at": "2026-07-15T14:00:00+08:00",
                "replied_at": "2026-07-15T14:20:00+08:00",
                "reply_summary": "办理地点为中山路C1-104号安徽省消防救援总队政务窗口。",
                "source_manifest_id": "fire_accident_identification_review_official_ahzwfw",
            },
        ]

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

    def get_applications_by_items(
        self,
        *,
        user_id: int,
        item_codes: list[str],
    ) -> list[dict[str, Any]]:
        """在一次内存扫描中返回多个事项的全部申请。"""
        code_set = {str(item or "").strip() for item in item_codes if str(item or "").strip()}
        output = [
            application_detail(item)
            for item in self._applications.values()
            if int(item.get("user_id") or 0) == int(user_id)
            and str(item.get("item_code") or "") in code_set
        ]
        return sorted(output, key=lambda row: str(row.get("updated_at") or ""), reverse=True)

    def list_appointments(
        self,
        *,
        user_id: int,
        region_code: str | None = None,
        status: str = "active",
    ) -> list[dict[str, Any]]:
        active_statuses = {"booked", "checked_in", "waiting", "called"}
        output = []
        for item in self._appointments:
            if int(item.get("user_id") or 0) != int(user_id):
                continue
            if region_code and not region_code_matches(item.get("region_code"), region_code):
                continue
            if status == "active" and str(item.get("status") or "") not in active_statuses:
                continue
            if status not in {"all", "active"} and str(item.get("status") or "") != status:
                continue
            output.append(
                {
                    key: value
                    for key, value in item.items()
                    if key
                    in {
                        "appointment_id",
                        "item_code",
                        "item_name",
                        "region_code",
                        "hall_code",
                        "hall_name",
                        "hall_address",
                        "service_date",
                        "start_time",
                        "end_time",
                        "status",
                        "updated_at",
                    }
                }
            )
        return sorted(output, key=lambda row: (str(row.get("service_date") or ""), str(row.get("start_time") or "")))

    def query_appointment_slots(
        self,
        *,
        item_code: str,
        region_code: str | None = None,
        service_date: str | None = None,
        hall_code: str | None = None,
    ) -> list[dict[str, Any]]:
        output = []
        for item in self._appointment_slots:
            if str(item.get("item_code") or "") != str(item_code or ""):
                continue
            if region_code and not region_code_matches(item.get("region_code"), region_code):
                continue
            if service_date and str(item.get("service_date") or "") != str(service_date):
                continue
            if hall_code and str(item.get("hall_code") or "") != str(hall_code):
                continue
            row = dict(item)
            row["remaining_count"] = max(0, int(row.get("capacity") or 0) - int(row.get("reserved_count") or 0))
            output.append(row)
        return sorted(output, key=lambda row: (str(row.get("service_date") or ""), str(row.get("start_time") or "")))

    def book_appointment(
        self,
        *,
        user_id: int,
        region_code: str | None,
        slot_id: int,
        idempotency_key: str,
    ) -> dict[str, Any]:
        """原子扣减内存号源并创建预约，重复请求不会重复占号。"""
        if not str(idempotency_key or "").strip():
            raise AppointmentBookingError("invalid_request", "预约幂等键不能为空")
        with self._appointment_lock:
            replay = next(
                (
                    item
                    for item in self._appointments
                    if int(item.get("user_id") or 0) == int(user_id)
                    and str(item.get("idempotency_key") or "") == str(idempotency_key or "")
                ),
                None,
            )
            if replay is not None:
                return public_appointment(replay, booking_status="idempotent_replay")

            slot = next(
                (item for item in self._appointment_slots if int(item.get("slot_id") or 0) == int(slot_id)),
                None,
            )
            if slot is None or (region_code and not region_code_matches(slot.get("region_code"), region_code)):
                raise AppointmentBookingError("slot_not_found", "未找到当前地区的预约号源")

            existing = next(
                (
                    item
                    for item in self._appointments
                    if int(item.get("user_id") or 0) == int(user_id)
                    and int(item.get("slot_id") or 0) == int(slot_id)
                    and str(item.get("status") or "") in {"booked", "checked_in", "waiting", "called"}
                ),
                None,
            )
            if existing is not None:
                return public_appointment(existing, booking_status="already_booked")

            remaining = int(slot.get("capacity") or 0) - int(slot.get("reserved_count") or 0)
            if str(slot.get("status") or "") != "available" or remaining <= 0:
                raise AppointmentBookingError("slot_unavailable", "该预约时段已无剩余号源")

            now = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
            appointment = {
                "appointment_id": max((int(item.get("appointment_id") or 0) for item in self._appointments), default=0) + 1,
                "slot_id": int(slot_id),
                "user_id": int(user_id),
                "idempotency_key": str(idempotency_key or ""),
                "item_code": str(slot.get("item_code") or ""),
                "item_name": str(slot.get("item_name") or ""),
                "region_code": str(slot.get("region_code") or ""),
                "hall_code": str(slot.get("hall_code") or ""),
                "hall_name": str(slot.get("hall_name") or ""),
                "hall_address": str(slot.get("hall_address") or ""),
                "service_date": str(slot.get("service_date") or ""),
                "start_time": str(slot.get("start_time") or ""),
                "end_time": str(slot.get("end_time") or ""),
                "status": "booked",
                "queue_number": "",
                "people_ahead": 0,
                "estimated_wait_minutes": 0,
                "counter_name": "",
                "checked_in_at": "",
                "created_at": now,
                "updated_at": now,
            }
            self._appointments.append(appointment)
            slot["reserved_count"] = int(slot.get("reserved_count") or 0) + 1
            if int(slot["reserved_count"]) >= int(slot.get("capacity") or 0):
                slot["status"] = "full"
            return public_appointment(appointment, booking_status="created")

    def cancel_appointment(
        self,
        *,
        user_id: int,
        region_code: str | None,
        appointment_id: int,
    ) -> dict[str, Any]:
        """原子取消内存预约并归还对应号源。"""
        with self._appointment_lock:
            appointment = next(
                (
                    item
                    for item in self._appointments
                    if int(item.get("user_id") or 0) == int(user_id)
                    and int(item.get("appointment_id") or 0) == int(appointment_id)
                    and (not region_code or region_code_matches(item.get("region_code"), region_code))
                ),
                None,
            )
            if appointment is None:
                raise AppointmentCancellationError("appointment_not_found", "未找到当前用户的预约记录")

            status = str(appointment.get("status") or "")
            if status == "cancelled":
                return public_appointment(appointment, cancellation_status="already_cancelled")
            if status != "booked":
                raise AppointmentCancellationError(
                    "cancellation_not_allowed",
                    "该预约已签到、进入排队或已经办结，当前状态不允许取消",
                )

            now = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
            appointment["status"] = "cancelled"
            appointment["cancelled_at"] = now
            appointment["updated_at"] = now
            slot = next(
                (
                    item
                    for item in self._appointment_slots
                    if int(item.get("slot_id") or 0) == int(appointment.get("slot_id") or 0)
                ),
                None,
            )
            if slot is not None:
                slot["reserved_count"] = max(0, int(slot.get("reserved_count") or 0) - 1)
                if str(slot.get("status") or "") == "full":
                    slot["status"] = "available"
                slot["updated_at"] = now
            return public_appointment(appointment, cancellation_status="cancelled")

    def get_queue_status(self, *, user_id: int, appointment_id: int) -> dict[str, Any] | None:
        for item in self._appointments:
            if int(item.get("user_id") or 0) == int(user_id) and int(item.get("appointment_id") or 0) == int(appointment_id):
                return {key: value for key, value in item.items() if key != "user_id"}
        return None

    def get_payment_orders_by_applications(
        self,
        *,
        user_id: int,
        application_ids: list[int],
    ) -> list[dict[str, Any]]:
        """在一次内存扫描中返回多个申请的缴费订单。"""
        id_set = {int(item) for item in application_ids if int(item) > 0}
        return [
            {key: value for key, value in item.items() if key != "user_id"}
            for item in self._payment_orders
            if int(item.get("user_id") or 0) == int(user_id)
            and int(item.get("application_id") or 0) in id_set
        ]

    def get_payment_receipts_by_targets(
        self,
        *,
        user_id: int,
        targets: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        all_order_applications = {
            int(target.get("application_id") or 0)
            for target in targets
            if target.get("payment_order_id") is None
        }
        target_pairs = {
            (int(target.get("application_id") or 0), int(target.get("payment_order_id") or 0))
            for target in targets
            if target.get("payment_order_id") is not None
        }
        return [
            {key: value for key, value in item.items() if key != "user_id"}
            for item in self._payment_receipts
            if int(item.get("user_id") or 0) == int(user_id)
            and (
                int(item.get("application_id") or 0) in all_order_applications
                or (
                    int(item.get("application_id") or 0),
                    int(item.get("payment_order_id") or 0),
                ) in target_pairs
            )
        ]

    def get_submission_receipts_by_applications(
        self, *, user_id: int, application_ids: list[int]
    ) -> list[dict[str, Any]]:
        return private_application_rows_by_ids(
            self._submission_receipts, user_id=user_id, application_ids=application_ids
        )

    def get_result_certificates_by_applications(
        self, *, user_id: int, application_ids: list[int]
    ) -> list[dict[str, Any]]:
        return private_application_rows_by_ids(
            self._result_certificates, user_id=user_id, application_ids=application_ids
        )

    def get_delivery_statuses_by_applications(
        self, *, user_id: int, application_ids: list[int]
    ) -> list[dict[str, Any]]:
        return private_application_rows_by_ids(
            self._deliveries, user_id=user_id, application_ids=application_ids
        )

    def list_consultation_records(
        self,
        *,
        user_id: int,
        status: str = "all",
        consultation_id: int | None = None,
    ) -> list[dict[str, Any]]:
        output = []
        for item in self._consultations:
            if int(item.get("user_id") or 0) != int(user_id):
                continue
            if status != "all" and str(item.get("status") or "") != status:
                continue
            if consultation_id is not None and int(item.get("consultation_id") or 0) != int(consultation_id):
                continue
            output.append({key: value for key, value in item.items() if key != "user_id"})
        return sorted(output, key=lambda row: str(row.get("created_at") or ""), reverse=True)

    def get_application_form_schema(
        self,
        *,
        item_code: str,
        region_code: str | None = None,
    ) -> dict[str, Any] | None:
        schema = self._form_schemas.get(str(item_code or "").strip())
        return dict(schema) if schema else None

    def submit_application(
        self,
        *,
        user_id: int,
        region_code: str | None,
        item_code: str,
        item_name: str,
        biz_type: str,
        schema_version: str,
        form_data: dict[str, Any],
        idempotency_key: str,
    ) -> dict[str, Any]:
        key = f"application:{int(user_id)}:{idempotency_key}"
        with self._write_lock:
            replay = self._write_operations.get(key)
            if replay:
                return {**replay, "submission_status": "idempotent_replay"}
            application_id = max(
                [int(item.get("application_id") or 0) for item in self._applications.values()] or [0]
            ) + 1
            now = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
            apply_uid = f"APP-{sha256(key.encode('utf-8')).hexdigest()[:24]}"
            record = {
                "application_id": application_id,
                "apply_uid": apply_uid,
                "user_id": int(user_id),
                "item_code": item_code,
                "item_name": item_name,
                "biz_type": biz_type,
                "region_code": str(region_code or "340000"),
                "status": "submitted",
                "current_node": "SUBMIT",
                "node_name": "申请提交",
                "submitted_at": now,
                "updated_at": now,
                "payload": {
                    "source": "application_submit",
                    "schema_version": schema_version,
                    "form_data": dict(form_data),
                },
                "status_history": [
                    {
                        "event_id": application_id * 100 + 1,
                        "sequence_no": 1,
                        "from_status": None,
                        "status": "submitted",
                        "current_node": "SUBMIT",
                        "node_name": "申请提交",
                        "changed_at": now,
                    }
                ],
            }
            self._applications[apply_uid] = record
            public = application_detail(record)
            self._write_operations[key] = public
            return {**public, "submission_status": "created"}

    def create_complaint(
        self,
        *,
        user_id: int,
        region_code: str | None,
        values: dict[str, Any],
        schema_version: str,
        idempotency_key: str,
    ) -> dict[str, Any]:
        key = f"complaint:{int(user_id)}:{idempotency_key}"
        with self._write_lock:
            replay = self._write_operations.get(key)
            if replay:
                return {**replay, "creation_status": "idempotent_replay"}
            complaint_id = max(
                [int(item.get("consultation_id") or 0) for item in self._consultations] or [0]
            ) + 1
            now = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
            record = {
                "consultation_id": complaint_id,
                "consultation_uid": f"COMPLAINT-{sha256(key.encode('utf-8')).hexdigest()[:24]}",
                "user_id": int(user_id),
                "consultation_type": "complaint",
                "subject": str(values.get("subject") or ""),
                "content_summary": str(values.get("content") or ""),
                "status": "submitted",
                "channel": "agent",
                "created_at": now,
                "replied_at": "",
                "reply_summary": "",
                "payload": {
                    "source": "complaint_create",
                    "schema_version": schema_version,
                    "region_code": str(region_code or ""),
                    "target": values.get("target"),
                    "occurred_at": values.get("occurred_at"),
                    "contact_phone": values.get("contact_phone"),
                },
            }
            self._consultations.append(record)
            public = {name: value for name, value in record.items() if name not in {"user_id", "payload", "content_summary"}}
            public["target"] = values.get("target")
            self._write_operations[key] = public
            return {**public, "creation_status": "created"}


def public_appointment(
    item: dict[str, Any],
    *,
    booking_status: str | None = None,
    cancellation_status: str | None = None,
) -> dict[str, Any]:
    """移除可信身份和幂等字段后返回预约结果。"""
    status_fields = {}
    if booking_status:
        status_fields["booking_status"] = booking_status
    if cancellation_status:
        status_fields["cancellation_status"] = cancellation_status
    return {
        key: value
        for key, value in {**item, **status_fields}.items()
        if key not in {"user_id", "idempotency_key"}
    }


def default_application_form_schemas() -> dict[str, dict[str, Any]]:
    """Local-development schemas demonstrating item-specific field sets."""
    common_identity = [
        {"name": "applicant_name", "label": "申请人姓名", "type": "text", "required": True, "max_length": 80, "aliases": ["姓名", "申请人"]},
        {"name": "id_number", "label": "证件号码", "type": "id_number", "required": True, "aliases": ["身份证号", "身份证号码"]},
        {"name": "contact_phone", "label": "联系电话", "type": "phone", "required": True, "aliases": ["手机号", "电话", "联系方式"]},
    ]
    schemas = [
        {
            "schema_id": "residence-permit-v1",
            "version": "1",
            "item_code": "ITEM-RESIDENCE-PERMIT",
            "item_name": "居住证办理",
            "biz_type": "residence_permit",
            "steps": [
                {
                    "id": "applicant_identity",
                    "title": "申请人信息",
                    "description": "填写申请人的身份与联系方式，证件号码仅用于本次申报核验。",
                    "fields": common_identity,
                },
                {
                    "id": "residence_details",
                    "title": "居住信息",
                    "description": "填写当前实际居住地址及对应的居住证明编号。",
                    "fields": [
                        {"name": "residence_address", "label": "现居住地址", "type": "text", "required": True, "max_length": 300, "aliases": ["住址", "居住地址"]},
                        {"name": "residence_proof_no", "label": "居住证明编号", "type": "text", "required": True, "max_length": 100, "aliases": ["证明编号"]},
                    ],
                },
            ],
        },
        {
            "schema_id": "health-certificate-v1",
            "version": "1",
            "item_code": "ITEM-HEALTH-CERTIFICATE",
            "item_name": "健康证办理",
            "biz_type": "health_certificate",
            "steps": [
                {
                    "id": "applicant_identity",
                    "title": "身份信息",
                    "description": "填写办证人员姓名和有效证件号码。",
                    "fields": common_identity[:2],
                },
                {
                    "id": "contact_employer",
                    "title": "联系与单位信息",
                    "description": "填写可接收办理通知的联系电话及当前从业单位。",
                    "fields": [
                        common_identity[2],
                        {"name": "employer_name", "label": "从业单位", "type": "text", "required": True, "max_length": 200, "aliases": ["单位", "工作单位"]},
                    ],
                },
                {
                    "id": "occupation_details",
                    "title": "从业类别",
                    "description": "选择实际从业类别，系统将据此匹配对应的健康检查要求。",
                    "fields": [
                        {"name": "occupation_type", "label": "从业类别", "type": "select", "required": True, "options": ["食品", "公共场所", "饮用水", "化妆品", "其他"], "aliases": ["职业类别", "行业"]},
                    ],
                },
            ],
        },
        {
            "schema_id": "water-permit-v2",
            "version": "2",
            "item_code": "11340000002985923L200011900100001",
            "item_name": "取水许可",
            "biz_type": "water_permit",
            "steps": [
                {
                    "id": "applicant_contact",
                    "title": "申请主体与联系人",
                    "description": "填写申请单位或个人的主体信息，以及负责本次申报的联系人信息。",
                    "fields": [
                        {"name": "applicant_name", "label": "申请单位或个人", "type": "text", "required": True, "max_length": 200, "aliases": ["申请人", "单位名称"]},
                        {"name": "credit_code", "label": "统一社会信用代码或证件号码", "type": "text", "required": True, "max_length": 40, "aliases": ["统一社会信用代码", "证件号码"]},
                        {"name": "contact_name", "label": "联系人", "type": "text", "required": True, "max_length": 80, "aliases": ["经办人"]},
                        {"name": "contact_phone", "label": "联系电话", "type": "phone", "required": True, "aliases": ["手机号", "电话"]},
                    ],
                },
                {
                    "id": "water_plan",
                    "title": "取水方案",
                    "description": "填写取水水源、预计年取水量和实际用途，数量请按立方米计。",
                    "fields": [
                        {"name": "water_source", "label": "取水水源", "type": "text", "required": True, "max_length": 200, "aliases": ["水源"]},
                        {"name": "annual_water_volume", "label": "年取水量（立方米）", "type": "number", "required": True, "aliases": ["年取水量", "取水量"]},
                        {"name": "water_use", "label": "取水用途", "type": "multiselect", "required": True, "options": ["制水供水", "原水供水", "生活用水", "工业用水", "农业用水", "其他"], "aliases": ["用途"]},
                    ],
                },
                {
                    "id": "application_materials",
                    "title": "申报材料",
                    "description": "按材料清单上传清晰、完整的电子文件；标为必传的材料缺一不可。",
                    "fields": [
                        {
                            "name": "application_materials",
                            "label": "申报材料",
                            "type": "material_group",
                            "required": True,
                            "description": "支持 PDF、Word、Excel、JPG、PNG、ZIP；单个文件不超过 20 MB。",
                            "accept": [".pdf", ".doc", ".docx", ".xls", ".xlsx", ".jpg", ".jpeg", ".png", ".zip"],
                            "materials": [
                                {"key": "water_resource_report", "label": "建设项目水资源论证报告书（表）", "required": True, "max_files": 3},
                                {"key": "water_right_transfer_report", "label": "水权转让可行性研究报告", "required": False, "description": "通过水权转让方式获得取水指标时上传", "max_files": 3},
                                {"key": "filing_materials", "label": "备案材料", "required": False, "description": "属于备案项目时上传", "max_files": 3},
                                {"key": "business_license_or_id", "label": "营业执照或身份证", "required": True, "max_files": 2},
                                {"key": "water_permit_application", "label": "取水许可申请书", "required": True, "max_files": 2},
                                {"key": "interested_party_statement", "label": "利害关系说明", "required": True, "max_files": 3},
                            ],
                        },
                    ],
                },
            ],
        },
    ]
    return {schema["item_code"]: schema for schema in schemas}


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
        "archive_collection": "流动人员档案材料收集",
    }
    return names.get(str(item.get("biz_type") or ""), str(item.get("biz_type") or "政务服务事项"))


def private_application_rows_by_ids(
    rows: list[dict[str, Any]],
    *,
    user_id: int,
    application_ids: list[int],
) -> list[dict[str, Any]]:
    """按可信用户和一组申请主键过滤内存业务数据。"""
    id_set = {int(item) for item in application_ids if int(item) > 0}
    output = []
    for item in rows:
        if int(item.get("user_id") or 0) != int(user_id):
            continue
        if int(item.get("application_id") or 0) not in id_set:
            continue
        output.append({key: value for key, value in item.items() if key != "user_id"})
    return output


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

