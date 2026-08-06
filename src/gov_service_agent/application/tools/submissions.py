"""Schema-driven application submission and complaint creation tools."""

from __future__ import annotations

from hashlib import sha256
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from gov_service_agent.application.slot_filling import (
    build_dialog_task,
    collect_slots,
    normalize_form_schema,
    schema_fields,
    schema_step,
    validate_slots,
    validate_step_slots,
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
    "steps": [
        {
            "id": "complaint_details",
            "title": "投诉信息",
            "description": "请说明投诉对象、问题经过和联系方式，提交后可核对全部内容。",
            "fields": [
                {"name": "subject", "label": "投诉主题", "type": "text", "required": True, "max_length": 120, "aliases": ["主题", "标题"]},
                {"name": "content", "label": "投诉内容", "type": "textarea", "required": True, "min_length": 5, "max_length": 2000, "aliases": ["内容", "问题描述", "投诉描述"]},
                {"name": "target", "label": "被投诉对象", "type": "text", "required": True, "max_length": 200, "aliases": ["投诉对象", "部门", "窗口"]},
                {"name": "occurred_at", "label": "问题发生日期", "type": "date", "required": False, "aliases": ["发生日期", "日期"]},
                {"name": "contact_phone", "label": "联系电话", "type": "phone", "required": True, "aliases": ["手机号", "电话", "联系方式"]},
            ],
        },
    ],
}


class ApplicationSubmitInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_code: str = Field(
        default="",
        max_length=160,
        description="首次填报使用当前轮 knowledge_search 返回的真实事项编码；继续填报时可以省略",
    )
    step_id: str = Field(
        default="",
        max_length=120,
        description="仅前端受控表单提交当前步骤时传入；自然语言预填不得设置",
    )
    values: dict[str, Any] = Field(
        default_factory=dict,
        description="本轮用户明确提供的字段值；可使用 Schema 字段名或中文别名，不得猜测未提供的信息",
    )


class ComplaintCreateInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: str = Field(default="", max_length=120)
    values: dict[str, Any] = Field(
        default_factory=dict,
        description="本轮用户明确提供的投诉字段值，不得猜测未提供的信息",
    )


class ApplicationSubmitTool:
    name = "application_submit"
    description = (
        "启动或分步提交政务服务申报。不同 item_code 由服务端加载不同步骤的版本化 Schema；"
        "自然语言只预填用户明确给出的 values，前端按步骤提交并携带当前 step_id。"
        "最后一步通过校验后，工具执行前会暂停并请求用户确认。"
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
        preview = step_form_preview(
            task_type="application_submission",
            schema=schema,
            task=task if continuing_same_item else {},
            new_values=request.values,
            submitted_step_id=request.step_id,
            context=context,
            validate_material_uploads=True,
        )
        return preview

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
        "启动或提交政务服务投诉。自然语言只预填用户明确提供的 values；"
        "前端按步骤提交，最后一步校验通过后请求人工确认。"
    )
    input_model = ComplaintCreateInput

    def __init__(self, adapter: BusinessDataAdapter) -> None:
        self.adapter = adapter

    def confirmation_preview(self, arguments: dict[str, Any], context: ToolExecutionContext) -> dict[str, Any]:
        request = self.input_model.model_validate(arguments or {})
        task = active_task(context, "complaint_create")
        return step_form_preview(
            task_type="complaint_create",
            schema=COMPLAINT_SCHEMA,
            task=task,
            new_values=request.values,
            submitted_step_id=request.step_id,
            context=context,
            validate_material_uploads=False,
        )

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


def step_form_preview(
    *,
    task_type: str,
    schema: dict[str, Any],
    task: dict[str, Any],
    new_values: dict[str, Any],
    submitted_step_id: str,
    context: ToolExecutionContext,
    validate_material_uploads: bool,
) -> dict[str, Any]:
    """Collect one step, reject stale submissions, and advance deterministically."""
    normalized = normalize_form_schema(schema)
    steps = normalized["steps"]
    if not steps:
        return {"ready": False, "error": "schema_steps_required"}
    current_index = int(task.get("current_step_index") or 0)
    current_index = min(max(current_index, 0), len(steps) - 1)
    current_step = schema_step(normalized, current_index)
    requested_step_id = str(submitted_step_id or "").strip()
    implicit_single_step = not requested_step_id and not task and len(steps) == 1 and task_type == "complaint_create"
    if implicit_single_step:
        requested_step_id = current_step["id"]
    if requested_step_id and requested_step_id != current_step["id"]:
        return {
            "ready": False,
            "error": "stale_or_out_of_order_step",
            "expected_step_id": current_step["id"],
        }

    collected = collect_slots(normalized, task.get("values") or {}, new_values)
    normalized_values, missing, invalid = validate_step_slots(normalized, current_index, collected)
    step_schema = {**normalized, "steps": [current_step]}
    if validate_material_uploads:
        invalid.extend(validate_uploaded_materials(step_schema, {**collected, **normalized_values}, context))
    combined_values = {**collected, **normalized_values}
    completed_step_ids = list(task.get("completed_step_ids") or [])

    # Natural-language tool calls may prefill values but never count as a step submission.
    if not requested_step_id or missing or invalid:
        dialog_task = build_dialog_task(
            task_type=task_type,
            schema=normalized,
            values=combined_values,
            missing_fields=missing,
            invalid_fields=invalid,
            current_step_index=current_index,
            completed_step_ids=completed_step_ids,
            status="collecting",
        )
        return {"ready": False, "dialog_task": dialog_task}

    if current_step["id"] not in completed_step_ids:
        completed_step_ids.append(current_step["id"])
    if current_index < len(steps) - 1:
        next_index = current_index + 1
        _, next_missing, next_invalid = validate_step_slots(normalized, next_index, combined_values)
        dialog_task = build_dialog_task(
            task_type=task_type,
            schema=normalized,
            values=combined_values,
            missing_fields=next_missing,
            invalid_fields=next_invalid,
            current_step_index=next_index,
            completed_step_ids=completed_step_ids,
            status="collecting",
        )
        return {"ready": False, "dialog_task": dialog_task}

    final_values, final_missing, final_invalid = validate_slots(normalized, combined_values)
    if validate_material_uploads:
        final_invalid.extend(validate_uploaded_materials(normalized, final_values, context))
    dialog_task = build_dialog_task(
        task_type=task_type,
        schema=normalized,
        values={**combined_values, **final_values},
        missing_fields=final_missing,
        invalid_fields=final_invalid,
        current_step_index=current_index,
        completed_step_ids=completed_step_ids,
        status="ready" if not final_missing and not final_invalid else "collecting",
    )
    return {"ready": not final_missing and not final_invalid, "dialog_task": dialog_task}


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
    for field in schema_fields(schema):
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
