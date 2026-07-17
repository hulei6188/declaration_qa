"""独立评测数据库中的业务fixture装载和清理。"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any
from uuid import uuid4

from sqlalchemy import text


class EvaluationFixtureManager:
    """只操作独立评测库中的保留用户和eval前缀数据。"""

    MIN_EVALUATION_USER_ID = 900_000

    def __init__(self, store) -> None:
        self.store = store

    def prepare(self, case: dict[str, Any]) -> None:
        engine = getattr(self.store, "engine", None)
        if engine is None:
            return
        user_id = int(case.get("user_id") or 1)
        fixture_key = str(case.get("fixture_key") or "")
        if not fixture_key:
            return
        if user_id < self.MIN_EVALUATION_USER_ID:
            raise ValueError(
                f"fixture case must use a reserved evaluation user_id >= {self.MIN_EVALUATION_USER_ID}"
            )
        self.cleanup(case)
        if fixture_key.startswith("business_"):
            self._prepare_business_fixture(case, fixture_key)
            return
        full = "full" in fixture_key
        cancelled = "cancelled" in fixture_key
        active = "active" in fixture_key
        service_date = date.today() + timedelta(days=10)
        slot_uid = f"eval:{case.get('case_id') or uuid4().hex}:{fixture_key}"
        item_code = str((case.get("expected_outcome") or {}).get("item_code") or "000111008000")
        with engine.begin() as connection:
            slot_id = connection.execute(
                text(
                    """
                    INSERT INTO declaration_qa.business_appointment_slots (
                        slot_uid, item_code, item_name, region_code, hall_code, hall_name,
                        hall_address, service_date, start_time, end_time, capacity,
                        reserved_count, status, payload
                    ) VALUES (
                        :slot_uid, :item_code, '评测事项', :region_code, 'EVAL-HALL-001',
                        '评测政务服务大厅', '评测数据库专用地址', :service_date,
                        '09:00', '10:00', 1, :reserved_count, :status,
                        jsonb_build_object('fixture_key', CAST(:fixture_key AS TEXT))
                    )
                    ON CONFLICT (slot_uid) DO UPDATE SET reserved_count = EXCLUDED.reserved_count,
                        status = EXCLUDED.status, updated_at = CURRENT_TIMESTAMP
                    RETURNING id
                    """
                ),
                {
                    "slot_uid": slot_uid,
                    "item_code": item_code,
                    "region_code": str(case.get("region_code") or "340000"),
                    "service_date": service_date,
                    "reserved_count": 1 if full or active else 0,
                    "status": "full" if full else "available",
                    "fixture_key": fixture_key,
                },
            ).scalar_one()
            if cancelled or active:
                connection.execute(
                    text(
                        """
                        INSERT INTO declaration_qa.business_appointments (
                            appointment_uid, slot_id, user_id, item_code, item_name,
                            region_code, hall_code, hall_name, hall_address, service_date,
                            start_time, end_time, status, payload
                        ) VALUES (
                            :uid, :slot_id, :user_id, :item_code, '评测事项', :region_code,
                            'EVAL-HALL-001', '评测政务服务大厅', '评测数据库专用地址',
                            :service_date, '09:00', '10:00', :appointment_status,
                            jsonb_build_object('fixture_key', CAST(:fixture_key AS TEXT))
                        )
                        """
                    ),
                    {
                        "uid": f"eval-appointment-{uuid4().hex}",
                        "slot_id": slot_id,
                        "user_id": user_id,
                        "item_code": item_code,
                        "region_code": str(case.get("region_code") or "340000"),
                        "service_date": service_date,
                        "fixture_key": fixture_key,
                        "appointment_status": "booked" if active else "cancelled",
                    },
                )

    def cleanup(self, case: dict[str, Any]) -> None:
        engine = getattr(self.store, "engine", None)
        if engine is None:
            return
        user_id = int(case.get("user_id") or 1)
        fixture_key = str(case.get("fixture_key") or "")
        if not fixture_key:
            return
        if user_id < self.MIN_EVALUATION_USER_ID:
            raise ValueError(
                f"fixture case must use a reserved evaluation user_id >= {self.MIN_EVALUATION_USER_ID}"
            )
        with engine.begin() as connection:
            connection.execute(
                text("DELETE FROM declaration_qa.sessions WHERE user_id = :user_id"),
                {"user_id": user_id},
            )
            connection.execute(
                text("DELETE FROM declaration_qa.business_appointments WHERE user_id = :user_id"),
                {"user_id": user_id},
            )
            connection.execute(
                text("DELETE FROM declaration_qa.business_applications WHERE user_id = :user_id"),
                {"user_id": user_id},
            )
            connection.execute(
                text("DELETE FROM declaration_qa.business_consultations WHERE user_id = :user_id"),
                {"user_id": user_id},
            )
            if fixture_key:
                connection.execute(
                    text("DELETE FROM declaration_qa.business_appointment_slots WHERE payload->>'fixture_key' = :fixture_key"),
                    {"fixture_key": fixture_key},
                )

    def _prepare_business_fixture(self, case: dict[str, Any], fixture_key: str) -> None:
        engine = self.store.engine
        user_id = int(case.get("user_id") or 1)
        region_code = str(case.get("region_code") or "340000")
        suffix = uuid4().hex[:12]
        with engine.begin() as connection:
            apply_id = connection.execute(
                text(
                    """
                    INSERT INTO declaration_qa.business_applications (
                        apply_uid, user_id, item_code, item_name, biz_type, region_code,
                        status, current_node, node_name, submitted_at, payload
                    ) VALUES (
                        :apply_uid, :user_id, '000111008000', '取水许可', 'water_permit',
                        :region_code, 'completed', 'completed', '办结', CURRENT_TIMESTAMP,
                        jsonb_build_object('fixture_key', CAST(:fixture_key AS TEXT))
                    ) RETURNING id
                    """
                ),
                {
                    "apply_uid": f"eval-apply-{suffix}",
                    "user_id": user_id,
                    "region_code": region_code,
                    "fixture_key": fixture_key,
                },
            ).scalar_one()
            connection.execute(
                text(
                    """
                    INSERT INTO declaration_qa.business_application_statuses (
                        apply_id, sequence_no, status, current_node, node_name,
                        source_event_id, is_current, payload
                    ) VALUES (
                        :apply_id, 1, 'completed', 'completed', '办结', :event_id, TRUE,
                        jsonb_build_object('fixture_key', CAST(:fixture_key AS TEXT))
                    )
                    """
                ),
                {"apply_id": apply_id, "event_id": f"eval-event-{suffix}", "fixture_key": fixture_key},
            )
            order_id = connection.execute(
                text(
                    """
                    INSERT INTO declaration_qa.business_payment_orders (
                        order_uid, apply_id, order_no, fee_item, amount, status,
                        payment_channel, paid_at, payload
                    ) VALUES (
                        :uid, :apply_id, :order_no, '评测费用', 100, 'paid', 'online',
                        CURRENT_TIMESTAMP, jsonb_build_object('fixture_key', CAST(:fixture_key AS TEXT))
                    ) RETURNING id
                    """
                ),
                {"uid": f"eval-order-{suffix}", "apply_id": apply_id, "order_no": f"EVAL-{suffix}", "fixture_key": fixture_key},
            ).scalar_one()
            connection.execute(
                text(
                    """
                    INSERT INTO declaration_qa.business_payment_receipts (
                        receipt_uid, payment_order_id, receipt_no, title, amount, status,
                        issued_at, file_name, download_status, payload
                    ) VALUES (:uid, :order_id, :no, '评测电子票据', 100, 'issued',
                              CURRENT_TIMESTAMP, 'evaluation-receipt.pdf', 'available',
                              jsonb_build_object('fixture_key', CAST(:fixture_key AS TEXT)))
                    """
                ),
                {"uid": f"eval-pay-receipt-{suffix}", "order_id": order_id, "no": f"ER-{suffix}", "fixture_key": fixture_key},
            )
            connection.execute(
                text(
                    """
                    INSERT INTO declaration_qa.business_submission_receipts (
                        receipt_uid, apply_id, receipt_no, status, issued_at, file_name,
                        download_status, payload
                    ) VALUES (:uid, :apply_id, :no, 'issued', CURRENT_TIMESTAMP,
                              'submission-receipt.pdf', 'available',
                              jsonb_build_object('fixture_key', CAST(:fixture_key AS TEXT)))
                    """
                ),
                {"uid": f"eval-submit-{suffix}", "apply_id": apply_id, "no": f"SR-{suffix}", "fixture_key": fixture_key},
            )
            connection.execute(
                text(
                    """
                    INSERT INTO declaration_qa.business_result_certificates (
                        certificate_uid, apply_id, certificate_name, certificate_no,
                        status, issued_at, file_name, download_status, payload
                    ) VALUES (:uid, :apply_id, '取水许可证', :no, 'issued',
                              CURRENT_TIMESTAMP, 'water-permit.pdf', 'available',
                              jsonb_build_object('fixture_key', CAST(:fixture_key AS TEXT)))
                    """
                ),
                {"uid": f"eval-cert-{suffix}", "apply_id": apply_id, "no": f"CERT-{suffix}", "fixture_key": fixture_key},
            )
            connection.execute(
                text(
                    """
                    INSERT INTO declaration_qa.business_deliveries (
                        delivery_uid, apply_id, delivery_type, status, courier_name,
                        tracking_no, shipped_at, delivered_at, payload
                    ) VALUES (:uid, :apply_id, 'courier', 'delivered', '评测快递', :tracking,
                              CURRENT_TIMESTAMP, CURRENT_TIMESTAMP,
                              jsonb_build_object('fixture_key', CAST(:fixture_key AS TEXT)))
                    """
                ),
                {"uid": f"eval-delivery-{suffix}", "apply_id": apply_id, "tracking": f"TRACK-{suffix}", "fixture_key": fixture_key},
            )
            connection.execute(
                text(
                    """
                    INSERT INTO declaration_qa.business_consultations (
                        consultation_uid, user_id, subject, content_summary, status,
                        reply_summary, replied_at, payload
                    ) VALUES (:uid, :user_id, '取水许可咨询', '评测咨询内容', 'replied',
                              '评测回复内容', CURRENT_TIMESTAMP,
                              jsonb_build_object('fixture_key', CAST(:fixture_key AS TEXT)))
                    """
                ),
                {"uid": f"eval-consult-{suffix}", "user_id": user_id, "fixture_key": fixture_key},
            )
