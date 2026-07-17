"""预约办理及缴费、回执、结果送达和咨询记录查询工具。"""

from __future__ import annotations

from datetime import date
from hashlib import sha256
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from gov_service_agent.application.tools.registry import ToolExecutionContext, ToolExecutionResult
from gov_service_agent.core.business import (
    AppointmentBookingError,
    AppointmentCancellationError,
    BusinessDataAdapter,
)


class AppointmentListQueryInput(BaseModel):
    """用户身份和地区由服务端注入。"""

    model_config = ConfigDict(extra="forbid")

    status: Literal["all", "active", "booked", "checked_in", "waiting", "called", "completed", "cancelled"] = "active"


class AppointmentSlotQueryInput(BaseModel):
    """使用可信事项和大厅编码查询公开号源。"""

    model_config = ConfigDict(extra="forbid")

    item_code: str = Field(
        min_length=1,
        max_length=160,
        description=(
            "本轮可信公开信息中的真实事项编码；不得猜测、编造或使用历史对话中的 item_code"
        ),
    )
    service_date: date | None = None
    hall_codes: list[Annotated[str, Field(min_length=1, max_length=80)]] = Field(
        min_length=1,
        max_length=20,
        description=(
            "本轮可信业务结果中的数据库真实大厅编码；把所有待查询 hall_code 放入同一个数组，"
            "不得猜测、编造或使用历史对话中的 hall_code"
        ),
    )


class AppointmentHallQueryInput(BaseModel):
    """使用可信事项编码查询真实预约大厅。"""

    model_config = ConfigDict(extra="forbid")

    item_code: str = Field(
        min_length=1,
        max_length=160,
        description=(
            "本轮可信公开信息中的真实事项编码；不得猜测、编造或使用历史对话中的 item_code"
        ),
    )


class AppointmentBookInput(BaseModel):
    """使用本轮号源查询返回的 slot_id 创建预约。"""

    model_config = ConfigDict(extra="forbid")

    slot_id: int = Field(
        ge=1,
        description=(
            "本轮可信号源结果中的可用号源主键；不得猜测、编造或使用历史对话中的 slot_id"
        ),
    )
    confirmed: bool = Field(
        default=False,
        description="兼容字段；是否执行由服务端人工确认界面决定，模型不能自行确认",
    )


class AppointmentCancelInput(BaseModel):
    """使用预约列表返回的 appointment_id 取消预约。"""

    model_config = ConfigDict(extra="forbid")

    appointment_id: int = Field(
        ge=1,
        description=(
            "本轮可信预约结果中的待取消预约主键；不得猜测、编造或直接使用历史对话中的 appointment_id"
        ),
    )
    confirmed: bool = Field(
        default=False,
        description="兼容字段；是否执行由服务端人工确认界面决定，模型不能自行确认",
    )


class QueueStatusQueryInput(BaseModel):
    """使用本轮预约列表返回的 appointment_id 查询排队状态。"""

    model_config = ConfigDict(extra="forbid")

    appointment_id: int = Field(
        ge=1,
        description=(
            "本轮可信预约结果中的真实预约主键；不得猜测、编造或使用历史对话中的 appointment_id"
        ),
    )


class ApplicationBatchQueryInput(BaseModel):
    """使用本轮申请详情返回的 application_ids 批量查询业务数据。"""

    model_config = ConfigDict(extra="forbid")

    application_ids: list[Annotated[int, Field(ge=1)]] = Field(
        min_length=1,
        max_length=100,
        description=(
            "本轮可信申请结果中的申请主键数组；多个 application_id 放入同一个数组，不得猜测、"
            "编造或使用历史对话中的 application_id"
        ),
    )


class PaymentReceiptTarget(BaseModel):
    """一组申请与可选缴费订单的关联查询目标。"""

    model_config = ConfigDict(extra="forbid")

    application_id: int = Field(
        ge=1,
        description="本轮可信申请结果中的真实 application_id",
    )

    payment_order_id: int | None = Field(
        default=None,
        ge=1,
        description=(
            "可选；本轮可信缴费结果中的真实 payment_order_id，不得猜测、编造或使用历史对话中的值"
        ),
    )


class PaymentReceiptBatchQueryInput(BaseModel):
    """批量查询电子票据，并保持申请与缴费订单的对应关系。"""

    model_config = ConfigDict(extra="forbid")

    targets: list[PaymentReceiptTarget] = Field(
        min_length=1,
        max_length=100,
        description=(
            "批量查询目标；多个申请或缴费订单必须放入同一个 targets 数组，"
            "不得拆成多个工具调用"
        ),
    )


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


class AppointmentHallQueryTool:
    name = "appointment_hall_query"
    description = (
        "按事项编码查询当前可信地区号源库中的真实预约大厅，返回 data.halls[].hall_code、"
        "hall_name 和 hall_address。不得根据知识文档地址猜测或编造 hall_code。"
    )
    input_model = AppointmentHallQueryInput

    def __init__(self, adapter: BusinessDataAdapter) -> None:
        self.adapter = adapter

    def execute(self, arguments: BaseModel, context: ToolExecutionContext) -> ToolExecutionResult:
        request = AppointmentHallQueryInput.model_validate(arguments)
        slots = self.adapter.query_appointment_slots(
            item_code=request.item_code,
            region_code=context.region_code or None,
        )
        halls: list[dict[str, str]] = []
        seen: set[str] = set()
        for slot in slots:
            hall_code = str(slot.get("hall_code") or "").strip()
            if not hall_code or hall_code in seen:
                continue
            seen.add(hall_code)
            halls.append(
                {
                    "hall_code": hall_code,
                    "hall_name": str(slot.get("hall_name") or "").strip(),
                    "hall_address": str(slot.get("hall_address") or "").strip(),
                }
            )
        return business_result(self.name, "halls", halls, item_code=request.item_code)


class AppointmentSlotQueryTool:
    name = "appointment_slot_query"
    description = (
        "批量查询当前事项在多个服务大厅的可预约日期、时间段、容量和剩余号源。"
        "只查询公开号源，不查询或创建个人预约。"
    )
    input_model = AppointmentSlotQueryInput

    def __init__(self, adapter: BusinessDataAdapter) -> None:
        self.adapter = adapter

    def execute(self, arguments: BaseModel, context: ToolExecutionContext) -> ToolExecutionResult:
        request = AppointmentSlotQueryInput.model_validate(arguments)
        hall_codes = list(dict.fromkeys(request.hall_codes))
        all_slots = self.adapter.query_appointment_slots(
            item_code=request.item_code,
            region_code=context.region_code or None,
            service_date=request.service_date.isoformat() if request.service_date else None,
            hall_code=None,
        )
        allowed_halls = set(hall_codes)
        slots = [slot for slot in all_slots if str(slot.get("hall_code") or "") in allowed_halls]
        return business_result(
            self.name,
            "slots",
            slots,
            item_code=request.item_code,
            hall_codes=hall_codes,
        )


class AppointmentBookTool:
    name = "appointment_book"
    description = (
        "为当前登录用户预约指定的有效号源。这是写操作，只接受本轮可信号源结果中的可用 slot_id。"
        "工具执行前服务端会暂停并请求用户在界面中选择大厅、时段并人工确认；"
        "事项、大厅、日期和时间由服务端号源派生。"
    )
    input_model = AppointmentBookInput

    def __init__(self, adapter: BusinessDataAdapter) -> None:
        self.adapter = adapter

    def execute(self, arguments: BaseModel, context: ToolExecutionContext) -> ToolExecutionResult:
        request = AppointmentBookInput.model_validate(arguments)
        tool_call_id = str(context.metadata.get("tool_call_id") or "").strip()
        operation_id = tool_call_id or f"{context.session_id or 'direct'}:{request.slot_id}"
        idempotency_key = sha256(
            f"appointment_book:{context.user_id}:{operation_id}:{request.slot_id}".encode("utf-8")
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


class AppointmentCancelTool:
    name = "appointment_cancel"
    description = (
        "取消当前登录用户尚未签到的预约。这是写操作，只接受本轮可信预约结果中的 appointment_id。"
        "工具执行前服务端会暂停并请求用户在界面中人工确认。"
        "工具会校验预约归属；重复取消不会重复归还号源。"
    )
    input_model = AppointmentCancelInput

    def __init__(self, adapter: BusinessDataAdapter) -> None:
        self.adapter = adapter

    def execute(self, arguments: BaseModel, context: ToolExecutionContext) -> ToolExecutionResult:
        request = AppointmentCancelInput.model_validate(arguments)
        try:
            appointment = self.adapter.cancel_appointment(
                user_id=context.user_id,
                region_code=context.region_code or None,
                appointment_id=request.appointment_id,
            )
        except AppointmentCancellationError as exc:
            return ToolExecutionResult(
                tool_name=self.name,
                call_id=self.name,
                ok=False,
                status=exc.code,
                data={
                    "appointment_id": request.appointment_id,
                    "source_type": "business_tool",
                },
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
        "服务端会按当前登录用户校验该参数。"
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
        "批量查询当前登录用户多个申请的缴费项目、金额、支付状态、支付期限和支付时间。"
    )
    input_model = ApplicationBatchQueryInput

    def __init__(self, adapter: BusinessDataAdapter) -> None:
        self.adapter = adapter

    def execute(self, arguments: BaseModel, context: ToolExecutionContext) -> ToolExecutionResult:
        request = ApplicationBatchQueryInput.model_validate(arguments)
        orders = self.adapter.get_payment_orders_by_applications(
            user_id=context.user_id,
            application_ids=request.application_ids,
        )
        return business_result(self.name, "payment_orders", orders, application_ids=request.application_ids)


class PaymentReceiptQueryTool:
    name = "payment_receipt_query"
    description = (
        "批量查询当前登录用户多个申请的电子缴费票据和下载状态，可按 payment_order_id 限定具体订单。"
    )
    input_model = PaymentReceiptBatchQueryInput

    def __init__(self, adapter: BusinessDataAdapter) -> None:
        self.adapter = adapter

    def execute(self, arguments: BaseModel, context: ToolExecutionContext) -> ToolExecutionResult:
        request = PaymentReceiptBatchQueryInput.model_validate(arguments)
        targets = [target.model_dump() for target in request.targets]
        receipts = self.adapter.get_payment_receipts_by_targets(
            user_id=context.user_id,
            targets=targets,
        )
        return business_result(self.name, "payment_receipts", receipts, targets=targets)


class SubmissionReceiptQueryTool:
    name = "submission_receipt_query"
    description = (
        "批量查询当前登录用户多个申请的提交回执、受理回执、签发时间和下载状态。"
    )
    input_model = ApplicationBatchQueryInput

    def __init__(self, adapter: BusinessDataAdapter) -> None:
        self.adapter = adapter

    def execute(self, arguments: BaseModel, context: ToolExecutionContext) -> ToolExecutionResult:
        request = ApplicationBatchQueryInput.model_validate(arguments)
        receipts = self.adapter.get_submission_receipts_by_applications(
            user_id=context.user_id,
            application_ids=request.application_ids,
        )
        return business_result(self.name, "submission_receipts", receipts, application_ids=request.application_ids)


class ResultCertificateQueryTool:
    name = "result_certificate_query"
    description = (
        "批量查询当前登录用户多个申请的办结结果、电子批文或电子证照是否生成及下载状态。"
    )
    input_model = ApplicationBatchQueryInput

    def __init__(self, adapter: BusinessDataAdapter) -> None:
        self.adapter = adapter

    def execute(self, arguments: BaseModel, context: ToolExecutionContext) -> ToolExecutionResult:
        request = ApplicationBatchQueryInput.model_validate(arguments)
        certificates = self.adapter.get_result_certificates_by_applications(
            user_id=context.user_id,
            application_ids=request.application_ids,
        )
        return business_result(self.name, "certificates", certificates, application_ids=request.application_ids)


class DeliveryStatusQueryTool:
    name = "delivery_status_query"
    description = (
        "批量查询当前登录用户多个申请的邮寄、自取、物流单号和送达状态。"
    )
    input_model = ApplicationBatchQueryInput

    def __init__(self, adapter: BusinessDataAdapter) -> None:
        self.adapter = adapter

    def execute(self, arguments: BaseModel, context: ToolExecutionContext) -> ToolExecutionResult:
        request = ApplicationBatchQueryInput.model_validate(arguments)
        deliveries = self.adapter.get_delivery_statuses_by_applications(
            user_id=context.user_id,
            application_ids=request.application_ids,
        )
        return business_result(self.name, "deliveries", deliveries, application_ids=request.application_ids)


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
