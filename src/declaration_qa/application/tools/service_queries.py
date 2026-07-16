"""预约办理及缴费、回执、结果送达和咨询记录查询工具。"""

from __future__ import annotations

from datetime import date
from hashlib import sha256
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from declaration_qa.application.tools.registry import ToolExecutionContext, ToolExecutionResult
from declaration_qa.core.business import AppointmentBookingError, BusinessDataAdapter


class AppointmentListQueryInput(BaseModel):
    """用户身份和地区由服务端注入。"""

    model_config = ConfigDict(extra="forbid")

    status: Literal["all", "active", "booked", "checked_in", "waiting", "called", "completed", "cancelled"] = "active"


class AppointmentSlotQueryInput(BaseModel):
    """按事项查询公开可预约号源。"""

    model_config = ConfigDict(extra="forbid")

    item_code: str = Field(min_length=1, max_length=160)
    service_date: date | None = None
    hall_code: str | None = Field(default=None, min_length=1, max_length=80)


class AppointmentBookInput(BaseModel):
    """用户身份、地区和幂等键均由服务端控制。"""

    model_config = ConfigDict(extra="forbid")

    slot_id: int = Field(ge=1, description="需要预约的有效号源主键")
    confirmed: Literal[True] = Field(description="仅在用户明确确认预约后传 true")


class QueueStatusQueryInput(BaseModel):
    """使用预约主键查询当前用户的排队状态。"""

    model_config = ConfigDict(extra="forbid")

    appointment_id: int = Field(ge=1)


class ApplicationScopedQueryInput(BaseModel):
    """使用申请主键查询当前用户的业务数据。"""

    model_config = ConfigDict(extra="forbid")

    application_id: int = Field(ge=1)


class PaymentReceiptQueryInput(ApplicationScopedQueryInput):
    """可以按缴费订单主键进一步限定电子票据。"""

    payment_order_id: int | None = Field(default=None, ge=1)


class ConsultationRecordQueryInput(BaseModel):
    """查询当前用户的咨询、投诉和建议记录。"""

    model_config = ConfigDict(extra="forbid")

    status: Literal["all", "submitted", "processing", "replied", "closed"] = "all"
    consultation_id: int | None = Field(default=None, ge=1)


class AppointmentListQueryTool:
    name = "appointment_list_query"
    description = (
        "查询当前登录用户的预约记录，返回 appointment_id、事项、服务大厅、预约时段和状态。"
    )
    input_model = AppointmentListQueryInput

    def __init__(self, adapter: BusinessDataAdapter) -> None:
        self.adapter = adapter

    def execute(self, arguments: BaseModel, context: ToolExecutionContext) -> ToolExecutionResult:
        request = AppointmentListQueryInput.model_validate(arguments)
        appointments = self.adapter.list_appointments(
            user_id=context.user_id,
            region_code=context.region_code or None,
            status=request.status,
        )
        return business_result(self.name, "appointments", appointments)


class AppointmentSlotQueryTool:
    name = "appointment_slot_query"
    description = (
        "按事项编码查询当前地区服务大厅的可预约日期、时间段、容量和剩余号源。"
        "只查询公开号源，不查询或创建个人预约。"
    )
    input_model = AppointmentSlotQueryInput

    def __init__(self, adapter: BusinessDataAdapter) -> None:
        self.adapter = adapter

    def execute(self, arguments: BaseModel, context: ToolExecutionContext) -> ToolExecutionResult:
        request = AppointmentSlotQueryInput.model_validate(arguments)
        slots = self.adapter.query_appointment_slots(
            item_code=request.item_code,
            region_code=context.region_code or None,
            service_date=request.service_date.isoformat() if request.service_date else None,
            hall_code=request.hall_code,
        )
        return business_result(self.name, "slots", slots, item_code=request.item_code)


class AppointmentBookTool:
    name = "appointment_book"
    description = (
        "为当前登录用户预约指定的有效 slot_id。"
        "这是写操作，仅当用户明确确认具体时段后调用；事项、大厅、日期和时间由服务端号源派生。"
    )
    input_model = AppointmentBookInput

    def __init__(self, adapter: BusinessDataAdapter) -> None:
        self.adapter = adapter

    def execute(self, arguments: BaseModel, context: ToolExecutionContext) -> ToolExecutionResult:
        request = AppointmentBookInput.model_validate(arguments)
        idempotency_key = sha256(
            f"appointment_book:{context.user_id}:{context.session_id or 'direct'}:{request.slot_id}".encode("utf-8")
        ).hexdigest()
        try:
            appointment = self.adapter.book_appointment(
                user_id=context.user_id,
                region_code=context.region_code or None,
                slot_id=request.slot_id,
                idempotency_key=idempotency_key,
            )
        except AppointmentBookingError as exc:
            return ToolExecutionResult(
                tool_name=self.name,
                call_id=self.name,
                ok=False,
                status=exc.code,
                data={"slot_id": request.slot_id, "source_type": "business_tool"},
                error=str(exc),
            )
        return ToolExecutionResult(
            tool_name=self.name,
            call_id=self.name,
            ok=True,
            status="success",
            data={"appointment": appointment, "source_type": "business_tool"},
        )


class QueueStatusQueryTool:
    name = "queue_status_query"
    description = (
        "查询当前登录用户指定预约的签到、排队人数、预计等待时间、叫号窗口和预约状态。"
        "服务端会按当前登录用户校验 appointment_id。"
    )
    input_model = QueueStatusQueryInput

    def __init__(self, adapter: BusinessDataAdapter) -> None:
        self.adapter = adapter

    def execute(self, arguments: BaseModel, context: ToolExecutionContext) -> ToolExecutionResult:
        request = QueueStatusQueryInput.model_validate(arguments)
        queue_status = self.adapter.get_queue_status(
            user_id=context.user_id,
            appointment_id=request.appointment_id,
        )
        return business_result(self.name, "queue_status", queue_status, appointment_id=request.appointment_id)


class PaymentOrderQueryTool:
    name = "payment_order_query"
    description = (
        "查询当前登录用户指定申请的缴费项目、金额、支付状态、支付期限和支付时间。"
        "服务端会按当前登录用户校验 application_id。"
    )
    input_model = ApplicationScopedQueryInput

    def __init__(self, adapter: BusinessDataAdapter) -> None:
        self.adapter = adapter

    def execute(self, arguments: BaseModel, context: ToolExecutionContext) -> ToolExecutionResult:
        request = ApplicationScopedQueryInput.model_validate(arguments)
        orders = self.adapter.get_payment_orders(user_id=context.user_id, application_id=request.application_id)
        return business_result(self.name, "payment_orders", orders, application_id=request.application_id)


class PaymentReceiptQueryTool:
    name = "payment_receipt_query"
    description = (
        "查询当前登录用户指定申请的电子缴费票据和下载状态。"
        "可以使用 payment_order_id 进一步限定具体订单。"
    )
    input_model = PaymentReceiptQueryInput

    def __init__(self, adapter: BusinessDataAdapter) -> None:
        self.adapter = adapter

    def execute(self, arguments: BaseModel, context: ToolExecutionContext) -> ToolExecutionResult:
        request = PaymentReceiptQueryInput.model_validate(arguments)
        receipts = self.adapter.get_payment_receipts(
            user_id=context.user_id,
            application_id=request.application_id,
            payment_order_id=request.payment_order_id,
        )
        return business_result(self.name, "payment_receipts", receipts, application_id=request.application_id)


class SubmissionReceiptQueryTool:
    name = "submission_receipt_query"
    description = (
        "查询当前登录用户指定申请的提交回执、受理回执、签发时间和下载状态。"
        "服务端会按当前登录用户校验 application_id。"
    )
    input_model = ApplicationScopedQueryInput

    def __init__(self, adapter: BusinessDataAdapter) -> None:
        self.adapter = adapter

    def execute(self, arguments: BaseModel, context: ToolExecutionContext) -> ToolExecutionResult:
        request = ApplicationScopedQueryInput.model_validate(arguments)
        receipts = self.adapter.get_submission_receipts(user_id=context.user_id, application_id=request.application_id)
        return business_result(self.name, "submission_receipts", receipts, application_id=request.application_id)


class ResultCertificateQueryTool:
    name = "result_certificate_query"
    description = (
        "查询当前登录用户指定申请的办结结果、电子批文或电子证照是否生成及下载状态。"
        "服务端会按当前登录用户校验 application_id。"
    )
    input_model = ApplicationScopedQueryInput

    def __init__(self, adapter: BusinessDataAdapter) -> None:
        self.adapter = adapter

    def execute(self, arguments: BaseModel, context: ToolExecutionContext) -> ToolExecutionResult:
        request = ApplicationScopedQueryInput.model_validate(arguments)
        certificates = self.adapter.get_result_certificates(user_id=context.user_id, application_id=request.application_id)
        return business_result(self.name, "certificates", certificates, application_id=request.application_id)


class DeliveryStatusQueryTool:
    name = "delivery_status_query"
    description = (
        "查询当前登录用户指定申请的邮寄、自取、物流单号和送达状态。"
        "服务端会按当前登录用户校验 application_id。"
    )
    input_model = ApplicationScopedQueryInput

    def __init__(self, adapter: BusinessDataAdapter) -> None:
        self.adapter = adapter

    def execute(self, arguments: BaseModel, context: ToolExecutionContext) -> ToolExecutionResult:
        request = ApplicationScopedQueryInput.model_validate(arguments)
        deliveries = self.adapter.get_delivery_statuses(user_id=context.user_id, application_id=request.application_id)
        return business_result(self.name, "deliveries", deliveries, application_id=request.application_id)


class ConsultationRecordQueryTool:
    name = "consultation_record_query"
    description = (
        "查询当前登录用户的历史咨询、投诉、建议、处理状态和回复摘要。"
        "用户身份由服务端注入，不得询问或生成 user_id。"
    )
    input_model = ConsultationRecordQueryInput

    def __init__(self, adapter: BusinessDataAdapter) -> None:
        self.adapter = adapter

    def execute(self, arguments: BaseModel, context: ToolExecutionContext) -> ToolExecutionResult:
        request = ConsultationRecordQueryInput.model_validate(arguments)
        records = self.adapter.list_consultation_records(
            user_id=context.user_id,
            status=request.status,
            consultation_id=request.consultation_id,
        )
        return business_result(self.name, "consultations", records)


def business_result(tool_name: str, key: str, value, **context_values) -> ToolExecutionResult:
    """构造统一的业务查询结果。"""
    found = bool(value)
    return ToolExecutionResult(
        tool_name=tool_name,
        call_id=tool_name,
        ok=found,
        status="success" if found else "not_found",
        data={key: value if value is not None else ([] if key != "queue_status" else {}), **context_values, "source_type": "business_tool"},
    )
