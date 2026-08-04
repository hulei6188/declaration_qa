"""办件业务数据读取。"""

from __future__ import annotations

from datetime import datetime
from hashlib import sha256
import json
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import text

from gov_service_agent.core.business import (
    AppointmentBookingError,
    AppointmentCancellationError,
    region_code_variants,
)
from gov_service_agent.storage.repository.base import EngineBackedRepository


BOOKING_APPOINTMENT_COLUMNS = """
    a.id AS appointment_id,
    a.slot_id,
    a.item_code,
    a.item_name,
    a.region_code,
    a.hall_code,
    a.hall_name,
    a.hall_address,
    a.service_date,
    a.start_time,
    a.end_time,
    a.status,
    a.created_at,
    a.updated_at
"""


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

    def get_applications_by_items(
        self,
        *,
        user_id: int,
        item_codes: list[str],
    ) -> list[dict[str, Any]]:
        """查询同一事项的全部申请，并附带每次申请的完整状态流水。"""
        normalized_item_codes = list(dict.fromkeys(
            str(item or "").strip() for item in item_codes if str(item or "").strip()
        ))
        if not normalized_item_codes:
            return []
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
                      AND COALESCE(NULLIF(a.item_code, ''), a.payload->>'item_code', '') = ANY(:item_codes)
                    ORDER BY GREATEST(a.updated_at, COALESCE(s.updated_at, a.updated_at)) DESC, a.id DESC
                    """
                ),
                {
                    "user_id": int(user_id),
                    "item_codes": normalized_item_codes,
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

    def list_appointments(
        self,
        *,
        user_id: int,
        region_code: str | None = None,
        status: str = "active",
    ) -> list[dict[str, Any]]:
        """按可信用户查询预约记录。"""
        region_codes = region_code_variants(region_code)
        with self.engine.begin() as connection:
            rows = connection.execute(
                text(
                    """
                    SELECT
                        a.id AS appointment_id,
                        a.item_code,
                        a.item_name,
                        a.region_code,
                        a.hall_code,
                        a.hall_name,
                        a.hall_address,
                        a.service_date,
                        a.start_time,
                        a.end_time,
                        a.status,
                        a.updated_at
                    FROM declaration_qa.business_appointments a
                    WHERE a.user_id = :user_id
                      AND (:use_region_filter = FALSE OR a.region_code = ANY(:region_codes))
                      AND (
                          :status = 'all'
                          OR (:status = 'active' AND a.status = ANY(:active_statuses))
                          OR a.status = :status
                      )
                    ORDER BY a.service_date, a.start_time, a.id
                    """
                ),
                {
                    "user_id": int(user_id),
                    "use_region_filter": bool(region_codes),
                    "region_codes": region_codes or [""],
                    "status": str(status or "active"),
                    "active_statuses": ["booked", "checked_in", "waiting", "called"],
                },
            ).mappings().all()
        return [business_row(row) for row in rows]

    def query_appointment_slots(
        self,
        *,
        item_code: str,
        region_code: str | None = None,
        service_date: str | None = None,
        hall_code: str | None = None,
    ) -> list[dict[str, Any]]:
        """查询事项在可信地区内的公开号源。"""
        region_codes = region_code_variants(region_code)
        with self.engine.begin() as connection:
            rows = connection.execute(
                text(
                    """
                    SELECT
                        s.id AS slot_id,
                        s.item_code,
                        s.item_name,
                        s.region_code,
                        s.hall_code,
                        s.hall_name,
                        s.hall_address,
                        s.service_date,
                        s.start_time,
                        s.end_time,
                        s.capacity,
                        s.reserved_count,
                        GREATEST(s.capacity - s.reserved_count, 0) AS remaining_count,
                        s.status,
                        s.updated_at
                    FROM declaration_qa.business_appointment_slots s
                    WHERE s.item_code = :item_code
                      AND (:use_region_filter = FALSE OR s.region_code = ANY(:region_codes))
                      AND (CAST(:service_date AS DATE) IS NULL OR s.service_date = CAST(:service_date AS DATE))
                      AND (:hall_code = '' OR s.hall_code = :hall_code)
                    ORDER BY s.service_date, s.start_time, s.id
                    """
                ),
                {
                    "item_code": str(item_code or "").strip(),
                    "use_region_filter": bool(region_codes),
                    "region_codes": region_codes or [""],
                    "service_date": service_date,
                    "hall_code": str(hall_code or "").strip(),
                },
            ).mappings().all()
        return [business_row(row) for row in rows]

    def book_appointment(
        self,
        *,
        user_id: int,
        region_code: str | None,
        slot_id: int,
        idempotency_key: str,
    ) -> dict[str, Any]:
        """锁定号源后创建预约，确保并发和重试不会重复扣号。"""
        key = str(idempotency_key or "").strip()
        if not key:
            raise AppointmentBookingError("invalid_request", "预约幂等键不能为空")
        region_codes = region_code_variants(region_code)
        appointment_uid = f"APPT-{sha256(f'{user_id}:{key}'.encode('utf-8')).hexdigest()}"
        with self.engine.begin() as connection:
            replay = connection.execute(
                text(f"""
                    SELECT {BOOKING_APPOINTMENT_COLUMNS}
                    FROM declaration_qa.business_appointments a
                    WHERE a.user_id = :user_id AND a.idempotency_key = :idempotency_key
                """),
                {"user_id": int(user_id), "idempotency_key": key},
            ).mappings().first()
            if replay:
                return {**business_row(replay), "booking_status": "idempotent_replay"}

            slot = connection.execute(
                text(
                    """
                    SELECT
                        s.id AS slot_id,
                        s.item_code,
                        s.item_name,
                        s.region_code,
                        s.hall_code,
                        s.hall_name,
                        s.hall_address,
                        s.service_date,
                        s.start_time,
                        s.end_time,
                        s.capacity,
                        s.reserved_count,
                        s.status
                    FROM declaration_qa.business_appointment_slots s
                    WHERE s.id = :slot_id
                      AND (:use_region_filter = FALSE OR s.region_code = ANY(:region_codes))
                    FOR UPDATE
                    """
                ),
                {
                    "slot_id": int(slot_id),
                    "use_region_filter": bool(region_codes),
                    "region_codes": region_codes or [""],
                },
            ).mappings().first()
            if not slot:
                raise AppointmentBookingError("slot_not_found", "未找到当前地区的预约号源")

            existing = connection.execute(
                text(f"""
                    SELECT {BOOKING_APPOINTMENT_COLUMNS}
                    FROM declaration_qa.business_appointments a
                    WHERE a.user_id = :user_id
                      AND a.slot_id = :slot_id
                      AND a.status = ANY(:active_statuses)
                    ORDER BY a.created_at DESC, a.id DESC
                    LIMIT 1
                """),
                {
                    "user_id": int(user_id),
                    "slot_id": int(slot_id),
                    "active_statuses": ["booked", "checked_in", "waiting", "called"],
                },
            ).mappings().first()
            if existing:
                return {**business_row(existing), "booking_status": "already_booked"}

            remaining = int(slot.get("capacity") or 0) - int(slot.get("reserved_count") or 0)
            if str(slot.get("status") or "") != "available" or remaining <= 0:
                raise AppointmentBookingError("slot_unavailable", "该预约时段已无剩余号源")

            appointment = connection.execute(
                text(
                    """
                    INSERT INTO declaration_qa.business_appointments (
                        appointment_uid, slot_id, idempotency_key, user_id, item_code, item_name,
                        region_code, hall_code, hall_name, hall_address, service_date,
                        start_time, end_time, status, payload
                    ) VALUES (
                        :appointment_uid, :slot_id, :idempotency_key, :user_id, :item_code, :item_name,
                        :region_code, :hall_code, :hall_name, :hall_address, :service_date,
                        :start_time, :end_time, 'booked', jsonb_build_object('source', 'appointment_book')
                    )
                    RETURNING
                        id AS appointment_id, slot_id, item_code, item_name, region_code,
                        hall_code, hall_name, hall_address, service_date, start_time, end_time,
                        status, created_at, updated_at
                    """
                ),
                {
                    "appointment_uid": appointment_uid,
                    "slot_id": int(slot_id),
                    "idempotency_key": key,
                    "user_id": int(user_id),
                    "item_code": str(slot.get("item_code") or ""),
                    "item_name": str(slot.get("item_name") or ""),
                    "region_code": str(slot.get("region_code") or ""),
                    "hall_code": str(slot.get("hall_code") or ""),
                    "hall_name": str(slot.get("hall_name") or ""),
                    "hall_address": str(slot.get("hall_address") or ""),
                    "service_date": slot["service_date"],
                    "start_time": slot["start_time"],
                    "end_time": slot["end_time"],
                },
            ).mappings().one()
            next_reserved_count = int(slot.get("reserved_count") or 0) + 1
            connection.execute(
                text(
                    """
                    UPDATE declaration_qa.business_appointment_slots
                    SET reserved_count = :reserved_count,
                        status = CASE WHEN :reserved_count >= capacity THEN 'full' ELSE status END,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = :slot_id
                    """
                ),
                {"slot_id": int(slot_id), "reserved_count": next_reserved_count},
            )
        return {**business_row(appointment), "booking_status": "created"}

    def cancel_appointment(
        self,
        *,
        user_id: int,
        region_code: str | None,
        appointment_id: int,
    ) -> dict[str, Any]:
        """锁定预约并取消，确保重复请求不会重复归还号源。"""
        region_codes = region_code_variants(region_code)
        with self.engine.begin() as connection:
            appointment = connection.execute(
                text(f"""
                    SELECT {BOOKING_APPOINTMENT_COLUMNS}
                    FROM declaration_qa.business_appointments a
                    WHERE a.id = :appointment_id
                      AND a.user_id = :user_id
                      AND (:use_region_filter = FALSE OR a.region_code = ANY(:region_codes))
                    FOR UPDATE
                """),
                {
                    "appointment_id": int(appointment_id),
                    "user_id": int(user_id),
                    "use_region_filter": bool(region_codes),
                    "region_codes": region_codes or [""],
                },
            ).mappings().first()
            if not appointment:
                raise AppointmentCancellationError("appointment_not_found", "未找到当前用户的预约记录")

            status = str(appointment.get("status") or "")
            if status == "cancelled":
                return {**business_row(appointment), "cancellation_status": "already_cancelled"}
            if status != "booked":
                raise AppointmentCancellationError(
                    "cancellation_not_allowed",
                    "该预约已签到、进入排队或已经办结，当前状态不允许取消",
                )

            cancelled = connection.execute(
                text(
                    """
                    UPDATE declaration_qa.business_appointments
                    SET status = 'cancelled',
                        payload = payload || jsonb_build_object(
                            'cancelled_by', 'appointment_cancel',
                            'cancelled_at', CURRENT_TIMESTAMP
                        ),
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = :appointment_id
                    RETURNING
                        id AS appointment_id, slot_id, item_code, item_name, region_code,
                        hall_code, hall_name, hall_address, service_date, start_time, end_time,
                        status, created_at, updated_at
                    """
                ),
                {"appointment_id": int(appointment_id)},
            ).mappings().one()
            slot_id = appointment.get("slot_id")
            if slot_id is not None:
                connection.execute(
                    text(
                        """
                        UPDATE declaration_qa.business_appointment_slots
                        SET reserved_count = GREATEST(reserved_count - 1, 0),
                            status = CASE WHEN status = 'full' THEN 'available' ELSE status END,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id = :slot_id
                        """
                    ),
                    {"slot_id": int(slot_id)},
                )
        return {**business_row(cancelled), "cancellation_status": "cancelled"}

    def get_queue_status(self, *, user_id: int, appointment_id: int) -> dict[str, Any] | None:
        """按可信用户查询单条预约的排队状态。"""
        with self.engine.begin() as connection:
            row = connection.execute(
                text(
                    """
                    SELECT
                        a.id AS appointment_id,
                        a.item_code,
                        a.item_name,
                        a.region_code,
                        a.hall_code,
                        a.hall_name,
                        a.hall_address,
                        a.service_date,
                        a.start_time,
                        a.end_time,
                        a.status,
                        a.queue_number,
                        a.people_ahead,
                        a.estimated_wait_minutes,
                        a.counter_name,
                        a.checked_in_at,
                        a.updated_at
                    FROM declaration_qa.business_appointments a
                    WHERE a.id = :appointment_id AND a.user_id = :user_id
                    """
                ),
                {"appointment_id": int(appointment_id), "user_id": int(user_id)},
            ).mappings().first()
        return business_row(row) if row else None

    def get_payment_orders_by_applications(
        self,
        *,
        user_id: int,
        application_ids: list[int],
    ) -> list[dict[str, Any]]:
        """使用一条 SQL 批量查询当前用户多个申请的缴费订单。"""
        normalized_ids = list(dict.fromkeys(int(item) for item in application_ids if int(item) > 0))
        if not normalized_ids:
            return []
        with self.engine.begin() as connection:
            rows = connection.execute(
                text(
                    """
                SELECT
                    p.id AS payment_order_id,
                    p.apply_id AS application_id,
                    p.order_no,
                    p.fee_item,
                    p.amount,
                    p.currency,
                    p.status,
                    p.payment_channel,
                    p.payable_at,
                    p.expires_at,
                    p.paid_at,
                    p.updated_at
                FROM declaration_qa.business_payment_orders p
                JOIN declaration_qa.business_applications a ON a.id = p.apply_id
                WHERE p.apply_id = ANY(:application_ids)
                  AND a.user_id = :user_id
                  AND a.deleted_at IS NULL
                ORDER BY p.apply_id, p.created_at DESC, p.id DESC
                    """
                ),
                {"user_id": int(user_id), "application_ids": normalized_ids},
            ).mappings().all()
        return [business_row(row) for row in rows]

    def get_payment_receipts_by_targets(
        self,
        *,
        user_id: int,
        targets: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """使用一条 SQL 批量查询申请及可选缴费订单对应的票据。"""
        application_ids = list(dict.fromkeys(
            int(target.get("application_id") or 0)
            for target in targets
            if int(target.get("application_id") or 0) > 0
        ))
        if not application_ids:
            return []
        with self.engine.begin() as connection:
            rows = connection.execute(
                text(
                    """
                    SELECT
                        r.id AS payment_receipt_id,
                        r.payment_order_id,
                        p.apply_id AS application_id,
                        r.receipt_no,
                        r.receipt_type,
                        r.title,
                        r.amount,
                        r.status,
                        r.issued_at,
                        r.file_name,
                        r.download_status
                    FROM declaration_qa.business_payment_receipts r
                    JOIN declaration_qa.business_payment_orders p ON p.id = r.payment_order_id
                    JOIN declaration_qa.business_applications a ON a.id = p.apply_id
                    WHERE p.apply_id = ANY(:application_ids)
                      AND a.user_id = :user_id
                      AND a.deleted_at IS NULL
                    ORDER BY p.apply_id, r.issued_at DESC, r.id DESC
                    """
                ),
                {
                    "user_id": int(user_id),
                    "application_ids": application_ids,
                },
            ).mappings().all()
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
            payload
            for row in rows
            if (
                (payload := business_row(row))["application_id"] in all_order_applications
                or (payload["application_id"], payload["payment_order_id"]) in target_pairs
            )
        ]

    def get_submission_receipts_by_applications(
        self, *, user_id: int, application_ids: list[int]
    ) -> list[dict[str, Any]]:
        """批量查询当前用户多个申请的提交和受理回执。"""
        return self._application_rows_by_ids(
            user_id=user_id,
            application_ids=application_ids,
            statement="""
                SELECT
                    r.id AS submission_receipt_id,
                    r.apply_id AS application_id,
                    r.receipt_no,
                    r.receipt_type,
                    r.status,
                    r.issued_at,
                    r.file_name,
                    r.download_status
                FROM declaration_qa.business_submission_receipts r
                JOIN declaration_qa.business_applications a ON a.id = r.apply_id
                WHERE r.apply_id = ANY(:application_ids) AND a.user_id = :user_id AND a.deleted_at IS NULL
                ORDER BY r.apply_id, r.issued_at DESC, r.id DESC
            """,
        )

    def get_result_certificates_by_applications(
        self, *, user_id: int, application_ids: list[int]
    ) -> list[dict[str, Any]]:
        """批量查询当前用户多个申请的办结结果。"""
        return self._application_rows_by_ids(
            user_id=user_id,
            application_ids=application_ids,
            statement="""
                SELECT
                    r.id AS result_certificate_id,
                    r.apply_id AS application_id,
                    r.certificate_name,
                    r.certificate_no,
                    r.certificate_type,
                    r.status,
                    r.issued_at,
                    r.valid_until,
                    r.file_name,
                    r.download_status
                FROM declaration_qa.business_result_certificates r
                JOIN declaration_qa.business_applications a ON a.id = r.apply_id
                WHERE r.apply_id = ANY(:application_ids) AND a.user_id = :user_id AND a.deleted_at IS NULL
                ORDER BY r.apply_id, r.issued_at DESC, r.id DESC
            """,
        )

    def get_delivery_statuses_by_applications(
        self, *, user_id: int, application_ids: list[int]
    ) -> list[dict[str, Any]]:
        """批量查询当前用户多个申请的结果送达状态。"""
        return self._application_rows_by_ids(
            user_id=user_id,
            application_ids=application_ids,
            statement="""
                SELECT
                    d.id AS delivery_id,
                    d.apply_id AS application_id,
                    d.delivery_type,
                    d.status,
                    d.courier_name,
                    d.tracking_no,
                    d.pickup_address,
                    d.shipped_at,
                    d.delivered_at,
                    d.updated_at
                FROM declaration_qa.business_deliveries d
                JOIN declaration_qa.business_applications a ON a.id = d.apply_id
                WHERE d.apply_id = ANY(:application_ids) AND a.user_id = :user_id AND a.deleted_at IS NULL
                ORDER BY d.apply_id, d.created_at DESC, d.id DESC
            """,
        )

    def list_consultation_records(
        self,
        *,
        user_id: int,
        status: str = "all",
        consultation_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """查询当前用户的咨询、投诉和建议记录。"""
        with self.engine.begin() as connection:
            rows = connection.execute(
                text(
                    """
                    SELECT
                        c.id AS consultation_id,
                        c.consultation_type,
                        c.subject,
                        c.content_summary,
                        c.status,
                        c.channel,
                        c.created_at,
                        c.replied_at,
                        c.reply_summary,
                        c.updated_at
                    FROM declaration_qa.business_consultations c
                    WHERE c.user_id = :user_id
                      AND (:status = 'all' OR c.status = :status)
                      AND (CAST(:consultation_id AS BIGINT) IS NULL OR c.id = CAST(:consultation_id AS BIGINT))
                    ORDER BY c.created_at DESC, c.id DESC
                    """
                ),
                {
                    "user_id": int(user_id),
                    "status": str(status or "all"),
                    "consultation_id": int(consultation_id) if consultation_id is not None else None,
                },
            ).mappings().all()
        return [business_row(row) for row in rows]

    def get_application_form_schema(
        self,
        *,
        item_code: str,
        region_code: str | None = None,
    ) -> dict[str, Any] | None:
        """Load the latest published schema for an item and region."""
        region_codes = region_code_variants(region_code)
        with self.engine.begin() as connection:
            row = connection.execute(
                text(
                    """
                    SELECT schema_uid AS schema_id, version_no, item_code, item_name,
                           biz_type, fields
                    FROM declaration_qa.business_item_form_schemas
                    WHERE item_code = :item_code
                      AND status = 'published'
                      AND (:use_region_filter = FALSE OR region_code = ANY(:region_codes) OR region_code = '')
                    ORDER BY CASE WHEN region_code = ANY(:region_codes) THEN 0 ELSE 1 END,
                             version_no DESC, updated_at DESC
                    LIMIT 1
                    """
                ),
                {
                    "item_code": str(item_code or "").strip(),
                    "use_region_filter": bool(region_codes),
                    "region_codes": region_codes or [""],
                },
            ).mappings().first()
        if not row:
            return None
        return {
            "schema_id": str(row.get("schema_id") or ""),
            "version": str(row.get("version_no") or 1),
            "item_code": str(row.get("item_code") or ""),
            "item_name": str(row.get("item_name") or ""),
            "biz_type": str(row.get("biz_type") or ""),
            "fields": list(row.get("fields") or []),
        }

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
        """Create an application and initial status in one idempotent transaction."""
        key = str(idempotency_key or "").strip()
        apply_uid = f"APP-{sha256(f'{user_id}:{key}'.encode('utf-8')).hexdigest()[:40]}"
        with self.engine.begin() as connection:
            replay = connection.execute(
                text(
                    """
                    SELECT id AS application_id, apply_uid, item_code, item_name, biz_type,
                           region_code, status AS current_status, current_node, node_name,
                           submitted_at, updated_at
                    FROM declaration_qa.business_applications
                    WHERE user_id = :user_id AND idempotency_key = :idempotency_key
                    """
                ),
                {"user_id": int(user_id), "idempotency_key": key},
            ).mappings().first()
            if replay:
                return {**business_row(replay), "submission_status": "idempotent_replay"}
            application = connection.execute(
                text(
                    """
                    INSERT INTO declaration_qa.business_applications (
                        apply_uid, idempotency_key, user_id, item_code, item_name, biz_type,
                        region_code, status, current_node, node_name, submitted_at, payload
                    ) VALUES (
                        :apply_uid, :idempotency_key, :user_id, :item_code, :item_name, :biz_type,
                        :region_code, 'submitted', 'SUBMIT', '申请提交', CURRENT_TIMESTAMP,
                        jsonb_build_object(
                            'source', 'application_submit',
                            'schema_version', CAST(:schema_version AS TEXT),
                            'form_data', CAST(:form_data AS JSONB)
                        )
                    )
                    RETURNING id AS application_id, apply_uid, item_code, item_name, biz_type,
                              region_code, status AS current_status, current_node, node_name,
                              submitted_at, updated_at
                    """
                ),
                {
                    "apply_uid": apply_uid,
                    "idempotency_key": key,
                    "user_id": int(user_id),
                    "item_code": item_code,
                    "item_name": item_name,
                    "biz_type": biz_type,
                    "region_code": str(region_code or "340000"),
                    "schema_version": schema_version,
                    "form_data": json.dumps(form_data, ensure_ascii=False),
                },
            ).mappings().one()
            connection.execute(
                text(
                    """
                    INSERT INTO declaration_qa.business_application_statuses (
                        apply_id, sequence_no, from_status, status, current_node,
                        node_name, is_current, occurred_at, source_event_id
                    ) VALUES (
                        :application_id, 1, NULL, 'submitted', 'SUBMIT',
                        '申请提交', TRUE, CURRENT_TIMESTAMP, :source_event_id
                    )
                    """
                ),
                {"application_id": int(application["application_id"]), "source_event_id": f"{apply_uid}-SUBMIT"},
            )
        return {**business_row(application), "submission_status": "created"}

    def create_complaint(
        self,
        *,
        user_id: int,
        region_code: str | None,
        values: dict[str, Any],
        schema_version: str,
        idempotency_key: str,
    ) -> dict[str, Any]:
        """Create a complaint with idempotency and trusted user ownership."""
        key = str(idempotency_key or "").strip()
        complaint_uid = f"COMPLAINT-{sha256(f'{user_id}:{key}'.encode('utf-8')).hexdigest()[:40]}"
        with self.engine.begin() as connection:
            row = connection.execute(
                text(
                    """
                    INSERT INTO declaration_qa.business_consultations (
                        consultation_uid, idempotency_key, user_id, consultation_type,
                        subject, content_summary, status, channel, payload
                    ) VALUES (
                        :consultation_uid, :idempotency_key, :user_id, 'complaint',
                        :subject, :content, 'submitted', 'agent',
                        jsonb_build_object(
                            'source', 'complaint_create',
                            'schema_version', CAST(:schema_version AS TEXT),
                            'region_code', CAST(:region_code AS TEXT),
                            'target', CAST(:target AS TEXT),
                            'occurred_at', CAST(:occurred_at AS TEXT),
                            'contact_phone', CAST(:contact_phone AS TEXT)
                        )
                    )
                    ON CONFLICT (user_id, idempotency_key) WHERE idempotency_key <> ''
                    DO UPDATE SET updated_at = declaration_qa.business_consultations.updated_at
                    RETURNING id AS consultation_id, consultation_uid, consultation_type,
                              subject, status, channel, created_at, updated_at,
                              payload->>'target' AS target, (xmax = 0) AS inserted
                    """
                ),
                {
                    "consultation_uid": complaint_uid,
                    "idempotency_key": key,
                    "user_id": int(user_id),
                    "subject": str(values.get("subject") or ""),
                    "content": str(values.get("content") or ""),
                    "schema_version": schema_version,
                    "region_code": str(region_code or ""),
                    "target": str(values.get("target") or ""),
                    "occurred_at": str(values.get("occurred_at") or ""),
                    "contact_phone": str(values.get("contact_phone") or ""),
                },
            ).mappings().one()
        payload = business_row(row)
        inserted = bool(payload.pop("inserted", False))
        return {**payload, "creation_status": "created" if inserted else "idempotent_replay"}

    def _application_rows_by_ids(
        self,
        *,
        user_id: int,
        application_ids: list[int],
        statement: str,
    ) -> list[dict[str, Any]]:
        normalized_ids = list(dict.fromkeys(int(item) for item in application_ids if int(item) > 0))
        if not normalized_ids:
            return []
        with self.engine.begin() as connection:
            rows = connection.execute(
                text(statement),
                {"user_id": int(user_id), "application_ids": normalized_ids},
            ).mappings().all()
        return [business_row(row) for row in rows]


def business_row(row) -> dict[str, Any]:
    """将 SQLAlchemy 行转换为 JSON 安全的业务结果。"""
    output = {}
    for key, value in dict(row).items():
        if isinstance(value, datetime) and value.tzinfo is not None:
            value = value.astimezone(ZoneInfo("Asia/Shanghai")).isoformat()
        elif hasattr(value, "isoformat"):
            value = value.isoformat()
        elif key == "amount" and value is not None:
            value = f"{value:.2f}"
        output[str(key)] = value
    return output

