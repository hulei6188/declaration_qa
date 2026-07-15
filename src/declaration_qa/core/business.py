"""办件业务数据访问协议和本地内存示例实现。"""

from __future__ import annotations

from datetime import datetime
from threading import Lock
from typing import Any, Protocol
from zoneinfo import ZoneInfo


class AppointmentBookingError(ValueError):
    """预约写入失败，并携带可供工具返回的稳定业务状态。"""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


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

    def get_queue_status(self, *, user_id: int, appointment_id: int) -> dict[str, Any] | None:
        """查询当前用户指定预约的排队状态。"""
        ...

    def get_payment_orders(self, *, user_id: int, application_id: int) -> list[dict[str, Any]]:
        """查询指定申请的缴费订单。"""
        ...

    def get_payment_receipts(
        self,
        *,
        user_id: int,
        application_id: int,
        payment_order_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """查询指定申请的缴费票据。"""
        ...

    def get_submission_receipts(self, *, user_id: int, application_id: int) -> list[dict[str, Any]]:
        """查询指定申请的提交和受理回执。"""
        ...

    def get_result_certificates(self, *, user_id: int, application_id: int) -> list[dict[str, Any]]:
        """查询指定申请的办结结果和电子证照。"""
        ...

    def get_delivery_statuses(self, *, user_id: int, application_id: int) -> list[dict[str, Any]]:
        """查询指定申请的结果送达状态。"""
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

class InMemoryBusinessDataAdapter:
    """独立问答应用使用的业务数据适配器。

    生产环境通过 PostgresDeclarationQAStore 使用 PostgreSQL 适配器；
    这里保留给没有数据库的本地开发场景。
    """

    def __init__(self) -> None:
        """初始化内存业务适配器。"""
        self._appointment_lock = Lock()
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

    def get_queue_status(self, *, user_id: int, appointment_id: int) -> dict[str, Any] | None:
        for item in self._appointments:
            if int(item.get("user_id") or 0) == int(user_id) and int(item.get("appointment_id") or 0) == int(appointment_id):
                return {key: value for key, value in item.items() if key != "user_id"}
        return None

    def get_payment_orders(self, *, user_id: int, application_id: int) -> list[dict[str, Any]]:
        return private_application_rows(self._payment_orders, user_id=user_id, application_id=application_id)

    def get_payment_receipts(
        self,
        *,
        user_id: int,
        application_id: int,
        payment_order_id: int | None = None,
    ) -> list[dict[str, Any]]:
        rows = private_application_rows(self._payment_receipts, user_id=user_id, application_id=application_id)
        if payment_order_id is not None:
            rows = [row for row in rows if int(row.get("payment_order_id") or 0) == int(payment_order_id)]
        return rows

    def get_submission_receipts(self, *, user_id: int, application_id: int) -> list[dict[str, Any]]:
        return private_application_rows(self._submission_receipts, user_id=user_id, application_id=application_id)

    def get_result_certificates(self, *, user_id: int, application_id: int) -> list[dict[str, Any]]:
        return private_application_rows(self._result_certificates, user_id=user_id, application_id=application_id)

    def get_delivery_statuses(self, *, user_id: int, application_id: int) -> list[dict[str, Any]]:
        return private_application_rows(self._deliveries, user_id=user_id, application_id=application_id)

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


def public_appointment(item: dict[str, Any], *, booking_status: str) -> dict[str, Any]:
    """移除可信身份和幂等字段后返回预约结果。"""
    return {
        key: value
        for key, value in {**item, "booking_status": booking_status}.items()
        if key not in {"user_id", "idempotency_key"}
    }


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


def private_application_rows(
    rows: list[dict[str, Any]],
    *,
    user_id: int,
    application_id: int,
) -> list[dict[str, Any]]:
    """按可信用户和申请主键过滤内存业务数据。"""
    output = []
    for item in rows:
        if int(item.get("user_id") or 0) != int(user_id):
            continue
        if int(item.get("application_id") or 0) != int(application_id):
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

