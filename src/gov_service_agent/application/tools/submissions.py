"""Schema-driven application submission and complaint creation tools."""

from __future__ import annotations

from hashlib import sha256
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from gov_service_agent.application.slot_filling import (
    build_dialog_task,
    collect_slots,
    normalize_form_schema,
    validate_slots,
)
from gov_service_agent.application.application_materials import ApplicationMaterialStore
from gov_service_agent.application.tools.registry import ToolExecutionContext, ToolExecutionResult
from gov_service_agent.core.business import BusinessDataAdapter
from gov_service_agent.core.config import get_settings


COMPLAINT_SCHEMA = {
    "schema_id": "complaint-create-v1",
    "version": "1",
    "item_code": "COMPLAINT",
    "item_name": "政务服务投诉",
    "biz_type": "complaint",
    "fields": [
        {"name": "subject", "label": "投诉主题", "type": "text", "required": True, "max_length": 120, "aliases": ["主题", "标题"]},
        {"name": "content", "label": "投诉内容", "type": "textarea", "required": True, "min_length": 5, "max_length": 2000, "aliases": ["内容", "问题描述", "投诉描述"]},
        {"name": "target", "label": "被投诉对象", "type": "text", "required": True, "max_length": 200, "aliases": ["投诉对象", "部门", "窗口"]},
        {"name": "occurred_at", "label": "问题发生日期", "type": "date", "required": False, "aliases": ["发生日期", "日期"]},
        {"name": "contact_phone", "label": "联系电话", "type": "phone", "required": True, "aliases": ["手机号", "电话", "联系方式"]},
    ],
}


class ApplicationSubmitInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_code: str = Field(
        default="",
        max_length=160,
        description="首次填报使用当前轮 knowledge_search 返回的真实事项编码；继续填报时可以省略",
    )
    values: dict[str, Any] = Field(
        default_factory=dict,
        description="本轮用户明确提供的字段值；可使用 Schema 字段名或中文别名，不得猜测未提供的信息",
    )


class ComplaintCreateInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    values: dict[str, Any] = Field(
        default_factory=dict,
        description="本轮用户明确提供的投诉字段值，不得猜测未提供的信息",
    )


class ApplicationSubmitTool:
    name = "application_submit"
    description = (
        "启动或提交政务服务申报。不同 item_code 由服务端加载不同版本的字段 Schema；"
        "自然语言只传用户明确给出的 values，缺失时前端展示完整表单并一次提交全部字段。"
        "全部必填项校验通过后，工具执行前会暂停并请求用户确认。"
    )
    input_model = ApplicationSubmitInput

    def __init__(self, adapter: BusinessDataAdapter) -> None:
        self.adapter = adapter

    def confirmation_preview(self, arguments: dict[str, Any], context: ToolExecutionContext) -> dict[str, Any]:
        request = self.input_model.model_validate(arguments or {})
        task = active_task(context, "application_submission")
        item_code = str(request.item_code or task.get("item_code") or "").strip()
        if not item_code:
            return {"ready": False, "error": "item_code_required"}
        continuing_same_item = bool(task) and str(task.get("item_code") or "") == item_code
        schema = dict(task.get("schema") or {}) if continuing_same_item else self.adapter.get_application_form_schema(
            item_code=item_code,
            region_code=context.region_code or None,
        )
        if not schema:
            return {"ready": False, "error": "schema_not_found", "item_code": item_code}
        schema = normalize_form_schema(schema)
        current = task.get("values") if continuing_same_item else {}
        collected = collect_slots(schema, current, request.values)
        values, missing, invalid = validate_slots(schema, collected)
        invalid.extend(validate_uploaded_materials(schema, values, context))
        dialog_task = build_dialog_task(
            task_type="application_submission",
            schema=schema,
            values={**collected, **values},
            missing_fields=missing,
            invalid_fields=invalid,
        )
        return {"ready": not missing and not invalid, "dialog_task": dialog_task}

    def execute(self, arguments: BaseModel, context: ToolExecutionContext) -> ToolExecutionResult:
        preview = self.confirmation_preview(arguments.model_dump(), context)
        if preview.get("error"):
            return ToolExecutionResult(
                self.name,
                self.name,
                False,
                str(preview["error"]),
                data={"item_code": preview.get("item_code", ""), "source_type": "business_tool"},
            )
        task = dict(preview["dialog_task"])
        if not preview["ready"]:
            return ToolExecutionResult(
                self.name,
                self.name,
                True,
                "slot_filling_required",
                data={"dialog_task": task, "source_type": "dialog_task"},
            )
        if not bool(context.metadata.get("human_approved")):
            return ToolExecutionResult(
                self.name,
                self.name,
                False,
                "confirmation_required",
                data={"dialog_task": task, "source_type": "dialog_task"},
            )
        operation_id = str(context.metadata.get("tool_call_id") or context.session_id or "application-submit")
        idempotency_key = sha256(
            f"application_submit:{context.user_id}:{operation_id}".encode("utf-8")
        ).hexdigest()
        application = self.adapter.submit_application(
            user_id=context.user_id,
            region_code=context.region_code or None,
            item_code=str(task["item_code"]),
            item_name=str(task["item_name"]),
            biz_type=str((task.get("schema") or {}).get("biz_type") or "application"),
            schema_version=str(task["schema_version"]),
            form_data=dict(task["values"]),
            idempotency_key=idempotency_key,
        )
        task["status"] = "completed"
        return ToolExecutionResult(
            self.name,
            self.name,
            True,
            "success",
            data={"application": application, "dialog_task": task, "source_type": "business_tool"},
        )


class ComplaintCreateTool:
    name = "complaint_create"
    description = (
        "启动或提交政务服务投诉。自然语言只传用户明确提供的 values；"
        "缺失时前端展示完整表单并一次提交全部字段。全部校验通过后执行前请求人工确认。"
    )
    input_model = ComplaintCreateInput

    def __init__(self, adapter: BusinessDataAdapter) -> None:
        self.adapter = adapter

    def confirmation_preview(self, arguments: dict[str, Any], context: ToolExecutionContext) -> dict[str, Any]:
        request = self.input_model.model_validate(arguments or {})
        task = active_task(context, "complaint_create")
        collected = collect_slots(COMPLAINT_SCHEMA, task.get("values") or {}, request.values)
        values, missing, invalid = validate_slots(COMPLAINT_SCHEMA, collected)
        dialog_task = build_dialog_task(
            task_type="complaint_create",
            schema=COMPLAINT_SCHEMA,
            values={**collected, **values},
            missing_fields=missing,
            invalid_fields=invalid,
        )
        return {"ready": not missing and not invalid, "dialog_task": dialog_task}

    def execute(self, arguments: BaseModel, context: ToolExecutionContext) -> ToolExecutionResult:
        preview = self.confirmation_preview(arguments.model_dump(), context)
        task = dict(preview["dialog_task"])
        if not preview["ready"]:
            return ToolExecutionResult(
                self.name,
                self.name,
                True,
                "slot_filling_required",
                data={"dialog_task": task, "source_type": "dialog_task"},
            )
        if not bool(context.metadata.get("human_approved")):
            return ToolExecutionResult(
                self.name,
                self.name,
                False,
                "confirmation_required",
                data={"dialog_task": task, "source_type": "dialog_task"},
            )
        operation_id = str(context.metadata.get("tool_call_id") or context.session_id or "complaint-create")
        idempotency_key = sha256(
            f"complaint_create:{context.user_id}:{operation_id}".encode("utf-8")
        ).hexdigest()
        complaint = self.adapter.create_complaint(
            user_id=context.user_id,
            region_code=context.region_code or None,
            values=dict(task["values"]),
            schema_version=str(task["schema_version"]),
            idempotency_key=idempotency_key,
        )
        task["status"] = "completed"
        return ToolExecutionResult(
            self.name,
            self.name,
            True,
            "success",
            data={"complaint": complaint, "dialog_task": task, "source_type": "business_tool"},
        )


def active_task(context: ToolExecutionContext, task_type: str) -> dict[str, Any]:
    state = context.metadata.get("state") if isinstance(context.metadata, dict) else {}
    task = state.get("dialog_task") if isinstance(state, dict) else {}
    if not isinstance(task, dict) or task.get("task_type") != task_type or task.get("status") == "completed":
        return {}
    return dict(task)


def validate_uploaded_materials(
    schema: dict[str, Any],
    values: dict[str, Any],
    context: ToolExecutionContext,
) -> list[dict[str, Any]]:
    """确保附件引用确实属于当前用户、会话和材料类别。"""
    settings = get_settings()
    store = ApplicationMaterialStore(
        settings.application_material_dir,
        max_bytes=settings.application_material_max_bytes,
    )
    invalid = []
    for field in schema.get("fields") or []:
        if not isinstance(field, dict) or field.get("type") != "material_group":
            continue
        field_name = str(field.get("name") or "")
        group = values.get(field_name)
        if not isinstance(group, dict):
            continue
        for material in field.get("materials") or []:
            material_key = str(material.get("key") or "")
            for reference in group.get(material_key) or []:
                verified = store.validate_reference(
                    reference,
                    user_id=context.user_id,
                    session_id=context.session_id,
                    field_name=field_name,
                    material_key=material_key,
                )
                if verified is None:
                    invalid.append(
                        {
                            **field,
                            "error": f"“{material.get('label') or material_key}”包含已失效或不属于当前会话的附件，请重新上传",
                        }
                    )
                    break
    return invalid
