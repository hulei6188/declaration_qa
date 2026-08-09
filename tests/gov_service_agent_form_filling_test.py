from pathlib import Path

import pytest
from pydantic import ValidationError

from gov_service_agent.api.schemas import ChatRequest, FormSubmissionPayload, HumanConfirmationRequest
from gov_service_agent.application.agent import AgentDecision
from gov_service_agent.application.application_materials import ApplicationMaterialStore
from gov_service_agent.application.nodes.agent_runtime import structured_form_tool_call
from gov_service_agent.application.routing import StaticIntentRouter
from gov_service_agent.application.tools import (
    ApplicationSubmitTool,
    ComplaintCreateTool,
    ToolExecutionContext,
    ToolRegistry,
)
from gov_service_agent.application.workflow import GovServiceAgentWorkflow
from gov_service_agent.core.business import InMemoryBusinessDataAdapter


def intent_router(domain, operation):
    return StaticIntentRouter(
        {
            "domain": domain,
            "operation": operation,
            "confidence": 1.0,
            "reason": "form_filling_test",
        }
    )


def tool_call(call_id, name, arguments=None):
    return {"id": call_id, "name": name, "arguments": arguments or {}}


def schema_fields(schema):
    return [field for step in schema["steps"] for field in step["fields"]]


def test_item_code_selects_a_different_versioned_form_schema():
    adapter = InMemoryBusinessDataAdapter()

    residence = adapter.get_application_form_schema(item_code="ITEM-RESIDENCE-PERMIT")
    health = adapter.get_application_form_schema(item_code="ITEM-HEALTH-CERTIFICATE")
    water = adapter.get_application_form_schema(item_code="11340000002985923L200011900100001")

    assert residence["schema_id"] == "residence-permit-v1"
    assert health["schema_id"] == "health-certificate-v1"
    assert water["schema_id"] == "water-permit-v2"
    assert {field["name"] for field in schema_fields(residence)} != {field["name"] for field in schema_fields(health)}
    assert "annual_water_volume" in {field["name"] for field in schema_fields(water)}
    material_field = next(field for field in schema_fields(water) if field["name"] == "application_materials")
    assert material_field["type"] == "material_group"
    assert sum(1 for item in material_field["materials"] if item["required"]) == 4


def test_application_start_checks_existing_user_application_before_collecting():
    adapter = InMemoryBusinessDataAdapter()
    tool = ApplicationSubmitTool(adapter)
    context = ToolExecutionContext(
        user_id=1,
        region_code="340000",
        session_id="application-guard-test",
        metadata={"state": {}, "tool_call_id": "guard-start"},
    )

    blocked = tool.confirmation_preview(
        {"item_code": "ITEM-HEALTH-CERTIFICATE", "values": {"姓名": "张三"}},
        context,
    )

    assert blocked["ready"] is False
    assert blocked["error"] == "application_in_progress"
    assert blocked["existing_applications"][0]["application_id"] == 1001
    assert blocked["existing_applications"][0]["current_status"] == "in_review"
    assert "健康证办理" in blocked["message"]
    assert "1001" in blocked["message"]


def test_application_write_rechecks_existing_application_before_persisting():
    adapter = InMemoryBusinessDataAdapter()
    tool = ApplicationSubmitTool(adapter)
    context = ToolExecutionContext(
        user_id=1,
        region_code="340000",
        session_id="application-write-guard-test",
        metadata={"state": {}, "tool_call_id": "write-guard-start"},
    )
    first = tool.confirmation_preview(
        {
            "item_code": "ITEM-RESIDENCE-PERMIT",
            "values": {
                "姓名": "张三",
                "手机号": "13800138000",
                "身份证号": "340101199001011234",
            },
        },
        context,
    )
    second = tool.confirmation_preview(
        {"step_id": "applicant_identity", "values": {}},
        ToolExecutionContext(
            user_id=1,
            region_code="340000",
            session_id=context.session_id,
            metadata={"state": {"dialog_task": first["dialog_task"]}, "tool_call_id": "write-guard-identity"},
        ),
    )
    ready = tool.confirmation_preview(
        {
            "step_id": "residence_details",
            "values": {"居住地址": "合肥市包河区示例路1号", "证明编号": "RES-2026-001"},
        },
        ToolExecutionContext(
            user_id=1,
            region_code="340000",
            session_id=context.session_id,
            metadata={"state": {"dialog_task": second["dialog_task"]}, "tool_call_id": "write-guard-ready"},
        ),
    )
    active = dict(adapter._applications["A1001"])
    active.update({"item_code": "ITEM-RESIDENCE-PERMIT", "item_name": "居住证办理"})
    adapter._applications["ACTIVE-RESIDENCE"] = active

    result = tool.execute(
        tool.input_model.model_validate({"step_id": "residence_details", "values": {}}),
        ToolExecutionContext(
            user_id=1,
            region_code="340000",
            session_id=context.session_id,
            metadata={
                "state": {"dialog_task": ready["dialog_task"]},
                "tool_call_id": "write-guard-final",
                "human_approved": True,
            },
        ),
    )

    assert result.ok is False
    assert result.status == "application_in_progress"
    assert result.data["existing_applications"][0]["application_id"] == 1001


def test_water_application_requires_uploaded_materials_and_validates_session_ownership(monkeypatch):
    from gov_service_agent.core.config import get_settings

    material_dir = Path("output/pytest-application-materials")
    material_dir.mkdir(parents=True, exist_ok=True)
    settings = get_settings()
    monkeypatch.setattr(settings, "application_material_dir", str(material_dir))
    monkeypatch.setattr(settings, "application_material_max_bytes", 1024 * 1024)
    adapter = InMemoryBusinessDataAdapter()
    tool = ApplicationSubmitTool(adapter)
    session_id = "water-material-upload"
    context = ToolExecutionContext(
        user_id=1,
        region_code="340000",
        session_id=session_id,
        metadata={"state": {}, "tool_call_id": "water-materials"},
    )
    base_values = {
        "申请单位或个人": "示例取水单位",
        "统一社会信用代码或证件号码": "91340100TEST000001",
        "联系人": "张三",
        "联系电话": "13800138000",
        "取水水源": "地表水",
        "年取水量": 25000000,
        "取水用途": ["工业用水"],
    }
    collecting = tool.confirmation_preview(
        {"item_code": "11340000002985923L200011900100001", "values": base_values},
        context,
    )
    task = collecting["dialog_task"]

    assert collecting["ready"] is False
    material_field = next(field for field in schema_fields(task["schema"]) if field["name"] == "application_materials")

    store = ApplicationMaterialStore(material_dir, max_bytes=1024 * 1024)
    group = {}
    required = [item for item in material_field["materials"] if item["required"]]
    for material in required:
        group[material["key"]] = [
            store.save(
                data=f"mock-{material['key']}".encode(),
                filename=f"{material['key']}.pdf",
                content_type="application/pdf",
                user_id=1,
                session_id=session_id,
                item_code=task["item_code"],
                field_name="application_materials",
                material_key=material["key"],
            )
        ]
    continuing_context = ToolExecutionContext(
        user_id=1,
        region_code="340000",
        session_id=session_id,
        metadata={"state": {"dialog_task": task}, "tool_call_id": "water-materials-ready"},
    )
    water_plan = tool.confirmation_preview(
        {"step_id": "applicant_contact", "values": {}},
        continuing_context,
    )
    materials_context = ToolExecutionContext(
        user_id=1,
        region_code="340000",
        session_id=session_id,
        metadata={"state": {"dialog_task": water_plan["dialog_task"]}, "tool_call_id": "water-materials-step"},
    )
    materials_step = tool.confirmation_preview(
        {"step_id": "water_plan", "values": {}},
        materials_context,
    )
    ready_context = ToolExecutionContext(
        user_id=1,
        region_code="340000",
        session_id=session_id,
        metadata={"state": {"dialog_task": materials_step["dialog_task"]}, "tool_call_id": "water-materials-ready"},
    )
    ready = tool.confirmation_preview(
        {"step_id": "application_materials", "values": {"application_materials": group}},
        ready_context,
    )
    assert ready["ready"] is True

    wrong_session_context = ToolExecutionContext(
        user_id=1,
        region_code="340000",
        session_id="another-session",
        metadata={"state": {"dialog_task": materials_step["dialog_task"]}, "tool_call_id": "water-materials-invalid"},
    )
    invalid = tool.confirmation_preview(
        {"step_id": "application_materials", "values": {"application_materials": group}},
        wrong_session_context,
    )
    assert invalid["ready"] is False
    assert "不属于当前会话" in invalid["dialog_task"]["invalid_fields"][0]["error"]


def test_structured_form_submission_keeps_attachment_references_out_of_model_parsing():
    attachment = {"attachment_id": "a" * 32, "filename": "report.pdf"}
    call, error = structured_form_tool_call(
        {
            "form_submission": {
                "task_type": "application_submission",
                "step_id": "application_materials",
                "values": {"application_materials": {"water_resource_report": [attachment]}},
            },
            "dialog_task": {
                "task_type": "application_submission",
                "status": "collecting",
                "current_step_id": "application_materials",
                "schema": {
                    "steps": [
                        {
                            "id": "application_materials",
                            "title": "申报材料",
                            "description": "上传申报材料。",
                            "fields": [{"name": "application_materials", "type": "material_group"}],
                        }
                    ]
                },
            },
        }
    )

    assert error == ""
    assert call["name"] == "application_submit"
    assert call["arguments"]["values"]["application_materials"]["water_resource_report"][0] == attachment


def test_structured_form_rejects_legacy_single_field_protocol():
    with pytest.raises(ValidationError):
        FormSubmissionPayload.model_validate(
            {
                "task_type": "application_submission",
                "field_name": "applicant_name",
                "value": "张三",
            }
        )


def test_structured_form_submission_skips_semantic_memory_and_agent_model(monkeypatch):
    class FailIfCalledAgent:
        model = None

        def decide(self, _state, _definitions):
            raise AssertionError("structured form result must not call the agent model")

    workflow = GovServiceAgentWorkflow(agent=FailIfCalledAgent())

    def fail_retrieve(**_kwargs):
        raise AssertionError("structured form submission must not call semantic memory")

    monkeypatch.setattr(workflow.memory_runtime, "retrieve", fail_retrieve)
    submission = {
        "task_type": "application_submission",
        "values": {"application_materials": {"water_resource_report": [{"attachment_id": "a" * 32}]}},
    }
    prepared = workflow.agent_prepare(
        {
            "user_id": 1,
            "query": "继续填写申报表单，已完成申报材料上传",
            "normalized_query": "继续填写申报表单，已完成申报材料上传",
            "form_submission": submission,
            "dialog_task": {
                "task_type": "application_submission",
                "status": "collecting",
                "schema": {"fields": [{"name": "application_materials", "type": "material_group"}]},
            },
        }
    )

    assert prepared["long_term_context"]["backend"] == "skipped_structured_form"

    decided = workflow.agent_decide(
        {
            **prepared,
            "form_submission": submission,
            "tool_results": [
                {
                    "tool_name": "application_submit",
                    "call_id": "structured-form-test",
                    "ok": True,
                    "status": "slot_filling_required",
                    "data": {},
                }
            ],
            "dialog_task": {
                "task_type": "application_submission",
                "status": "collecting",
                "schema": {"fields": [{"name": "next_value", "type": "text"}]},
            },
        }
    )

    assert decided["agent_next"] == "finish"
    assert decided["agent_answer_provider"] == "structured_form"
    assert decided["agent_plan"]["reason"] == "structured_form_processed"
    workflow.close()


def test_long_term_memory_failure_is_non_blocking_for_regular_turn(monkeypatch):
    workflow = GovServiceAgentWorkflow()

    def fail_retrieve(**_kwargs):
        raise ConnectionError("embedding service unavailable")

    monkeypatch.setattr(workflow.memory_runtime, "retrieve", fail_retrieve)
    prepared = workflow.agent_prepare(
        {
            "user_id": 1,
            "query": "取水许可需要哪些材料",
            "normalized_query": "取水许可需要哪些材料",
        }
    )

    assert prepared["long_term_context"]["status"] == "degraded"
    assert prepared["long_term_context"]["items"] == []
    workflow.close()


def test_application_submit_collects_slots_without_writing_until_confirmed():
    adapter = InMemoryBusinessDataAdapter()
    registry = ToolRegistry()
    registry.register(ApplicationSubmitTool(adapter))
    first_context = ToolExecutionContext(
        user_id=1,
        region_code="340000",
        session_id="form-tool-test",
        metadata={"state": {}, "tool_call_id": "submit-1", "human_approved": False},
    )

    collecting = registry.execute(
        "application_submit",
        {
            "item_code": "ITEM-RESIDENCE-PERMIT",
            "values": {"姓名": "张三", "手机号": "13800138000"},
        },
        first_context,
        call_id="submit-1",
    )

    assert collecting.ok is True
    assert collecting.status == "slot_filling_required"
    task = collecting.data["dialog_task"]
    assert task["values"]["applicant_name"] == "张三"
    assert {field["name"] for field in task["missing_fields"]} == {"id_number"}
    assert task["current_step_id"] == "applicant_identity"
    assert "next_field" not in task
    assert not any(item["application_id"] > 4004 for item in adapter._applications.values())

    # Publishing a newer schema must not change an in-flight task.
    adapter._form_schemas["ITEM-RESIDENCE-PERMIT"] = {
        **adapter._form_schemas["ITEM-RESIDENCE-PERMIT"],
        "schema_id": "residence-permit-v2",
        "version": "2",
        "steps": [
            *adapter._form_schemas["ITEM-RESIDENCE-PERMIT"]["steps"],
            {
                "id": "new_details",
                "title": "新版本字段",
                "description": "新版本字段。",
                "fields": [{"name": "new_required_field", "label": "新版本字段", "type": "text", "required": True}],
            },
        ],
    }

    ready_context = ToolExecutionContext(
        user_id=1,
        region_code="340000",
        session_id="form-tool-test",
        metadata={"state": {"dialog_task": task}, "tool_call_id": "submit-2", "human_approved": True},
    )
    identity = registry.execute(
        "application_submit",
        {
            "step_id": "applicant_identity",
            "values": {"身份证号": "340101199001011234"},
        },
        ready_context,
        call_id="submit-2",
    )
    assert identity.status == "slot_filling_required"
    submitted = registry.execute(
        "application_submit",
        {
            "step_id": "residence_details",
            "values": {
                "居住地址": "合肥市包河区示例路1号",
                "证明编号": "RES-2026-001",
            }
        },
        ToolExecutionContext(
            user_id=1,
            region_code="340000",
            session_id="form-tool-test",
            metadata={"state": {"dialog_task": identity.data["dialog_task"]}, "tool_call_id": "submit-3", "human_approved": True},
        ),
        call_id="submit-3",
    )

    assert submitted.status == "success"
    assert submitted.data["dialog_task"]["schema_version"] == "1"
    assert submitted.data["application"]["current_status"] == "submitted"
    assert submitted.data["dialog_task"]["status"] == "completed"


def test_application_submission_persists_slots_across_turns_and_confirms_once():
    class ScriptedAgent:
        def decide(self, state, _definitions):
            if state.get("tool_results"):
                task = state.get("dialog_task") or {}
                if task.get("status") == "completed":
                    return AgentDecision("finish", final_answer="申报已处理。[业务1]")
                return AgentDecision("finish", final_answer="请填写完整表单。")
            if "开始申报" in str(state.get("query") or ""):
                return AgentDecision(
                    "tools",
                    [
                        tool_call(
                            "submit-first",
                            "application_submit",
                            {
                                "item_code": "ITEM-RESIDENCE-PERMIT",
                                "values": {"姓名": "张三", "手机号": "13800138000"},
                            },
                        )
                    ],
                )
            return AgentDecision(
                "tools",
                [
                    tool_call(
                        "submit-final",
                        "application_submit",
                        {
                            "values": {
                                "身份证号": "340101199001011234",
                                "居住地址": "合肥市包河区示例路1号",
                                "证明编号": "RES-2026-001",
                            }
                        },
                    )
                ],
            )

    workflow = GovServiceAgentWorkflow(
        agent=ScriptedAgent(),
        intent_router=intent_router("application", "continue"),
    )
    session_id = "multiturn-application-submit"
    first = workflow.invoke(
        ChatRequest(query="开始申报居住证，申请人张三，电话13800138000", user_id=1, session_id=session_id)
    )

    assert first.human_confirmation == {}
    assert first.dialog_task["status"] == "collecting"
    assert first.dialog_task["values"] == {"applicant_name": "张三", "contact_phone": "13800138000"}
    assert "第 1/2 步" in first.answer
    assert "下方" in first.answer
    assert "回复" not in first.answer
    restored = workflow.get_session(user_id=1, session_id=session_id)
    assert len(restored["messages"][-1]["dialog_task"]["schema"]["steps"]) == 2

    identity_events = list(
        workflow.stream(
            ChatRequest(
                query="提交申报第一步",
                user_id=1,
                session_id=session_id,
                form_submission=FormSubmissionPayload(
                    task_type="application_submission",
                    step_id="applicant_identity",
                    values={
                        "id_number": "340101199001011234",
                    },
                ),
            )
        )
    )
    assert not any(item["event"] == "human_confirmation_required" for item in identity_events)
    events = list(
        workflow.stream(
            ChatRequest(
                query="提交申报第二步",
                user_id=1,
                session_id=session_id,
                form_submission=FormSubmissionPayload(
                    task_type="application_submission",
                    step_id="residence_details",
                    values={
                        "residence_address": "合肥市包河区示例路1号",
                        "residence_proof_no": "RES-2026-001",
                    },
                ),
            )
        )
    )
    confirmation = next(item["data"] for item in events if item["event"] == "human_confirmation_required")

    assert confirmation["title"] == "确认提交申报"
    assert confirmation["form_ready_for_confirmation"] is True
    assert confirmation["actions"][0]["tool_name"] == "application_submit"
    assert any("张三" in detail for detail in confirmation["actions"][0]["details"])
    assert any("138****8000" in detail for detail in confirmation["actions"][0]["details"])
    paused_session = workflow.get_session(user_id=1, session_id=session_id)
    assert paused_session["messages"][-1]["confirmation"]["interrupt_id"] == confirmation["interrupt_id"]
    assert paused_session["messages"][-1]["confirmation"]["status"] == "pending"
    assert paused_session["messages"][-1]["dialog_task"] == {}

    resumed = list(
        workflow.resume_stream(
            HumanConfirmationRequest(
                session_id=session_id,
                interrupt_id=confirmation["interrupt_id"],
                user_id=1,
                approved=True,
            )
        )
    )
    snapshot = workflow._checkpoint_state(user_id=1, session_id=session_id)

    assert any(item["event"] == "done" for item in resumed)
    assert snapshot["dialog_task"]["status"] == "completed"
    assert snapshot["tool_results"][0]["data"]["human_confirmation"]["completed"] is True
    assert "申报已提交，人工确认已完成" in snapshot["answer"]


def test_complaint_create_uses_the_same_slot_engine_and_confirmation_boundary():
    class ScriptedAgent:
        def decide(self, state, _definitions):
            if state.get("tool_results"):
                return AgentDecision("finish", final_answer="投诉已处理。[业务1]")
            return AgentDecision(
                "tools",
                [
                    tool_call(
                        "complaint-final",
                        "complaint_create",
                        {
                            "values": {
                                "主题": "窗口服务投诉",
                                "投诉描述": "工作人员未按规定一次性告知材料。",
                                "窗口": "综合窗口03",
                                "电话": "13900139000",
                            }
                        },
                    )
                ],
            )

    workflow = GovServiceAgentWorkflow(
        agent=ScriptedAgent(),
        intent_router=intent_router("complaint", "create"),
    )
    session_id = "multiturn-complaint-create"
    events = list(workflow.stream(ChatRequest(query="我要投诉综合窗口03", user_id=1, session_id=session_id)))
    confirmation = next(item["data"] for item in events if item["event"] == "human_confirmation_required")

    assert confirmation["title"] == "确认创建投诉"
    resumed = list(
        workflow.resume_stream(
            HumanConfirmationRequest(
                session_id=session_id,
                interrupt_id=confirmation["interrupt_id"],
                user_id=1,
                approved=True,
            )
        )
    )
    snapshot = workflow._checkpoint_state(user_id=1, session_id=session_id)

    assert any(item["event"] == "done" for item in resumed)
    assert snapshot["tool_results"][0]["data"]["complaint"]["consultation_type"] == "complaint"
    assert "投诉已提交，人工确认已完成" in snapshot["answer"]


def test_sync_invoke_keeps_session_state_when_write_action_pauses_for_confirmation():
    class ScriptedAgent:
        def decide(self, state, _definitions):
            return AgentDecision(
                "tools",
                [
                    tool_call(
                        "complaint-sync-confirm",
                        "complaint_create",
                        {
                            "values": {
                                "主题": "窗口服务投诉",
                                "投诉描述": "工作人员未按规定一次性告知材料。",
                                "窗口": "综合窗口03",
                                "电话": "13900139000",
                            }
                        },
                    )
                ],
            )

    workflow = GovServiceAgentWorkflow(
        agent=ScriptedAgent(),
        intent_router=intent_router("complaint", "create"),
    )
    session_id = "sync-invoke-human-confirmation"
    response = workflow.invoke(
        ChatRequest(query="我要创建投诉", user_id=1, session_id=session_id)
    )

    assert response.session_id == session_id
    assert response.human_confirmation["title"] == "确认创建投诉"
    assert response.answer == "操作正在等待人工确认。"


def test_rejecting_complaint_confirmation_cancels_the_entire_form_flow():
    class ScriptedAgent:
        def decide(self, state, _definitions):
            if state.get("tool_results"):
                return AgentDecision("finish", final_answer="不应继续填写。")
            return AgentDecision(
                "tools",
                [
                    tool_call(
                        "complaint-cancel-flow",
                        "complaint_create",
                        {
                            "values": {
                                "主题": "问答质量差",
                                "投诉描述": "回答内容不准确。",
                                "投诉对象": "合肥工商局",
                                "电话": "13800006292",
                            }
                        },
                    )
                ],
            )

    workflow = GovServiceAgentWorkflow(
        agent=ScriptedAgent(),
        intent_router=intent_router("complaint", "create"),
    )
    session_id = "complaint-confirmation-cancels-flow"
    events = list(
        workflow.stream(ChatRequest(query="我要投诉", user_id=1, session_id=session_id))
    )
    confirmation = next(item["data"] for item in events if item["event"] == "human_confirmation_required")

    resumed = list(
        workflow.resume_stream(
            HumanConfirmationRequest(
                session_id=session_id,
                interrupt_id=confirmation["interrupt_id"],
                user_id=1,
                approved=False,
            )
        )
    )
    snapshot = workflow._checkpoint_state(user_id=1, session_id=session_id)
    workflow.close()

    assert any(item["event"] == "dialog_flow_cancelled" for item in resumed)
    done = next(item["data"] for item in resumed if item["event"] == "done")
    assert done["dialog_task"]["status"] == "cancelled"
    assert snapshot["dialog_task"]["status"] == "cancelled"
    assert "next_field" not in snapshot["dialog_task"]
    assert snapshot["workflow_gate"]["mode"] == "dialog_cancelled"
    assert snapshot["tool_results"][0]["status"] == "human_rejected"
    assert snapshot["answer"] == "已取消当前表单填写流程。"


def test_migration_defines_schema_versions_and_idempotent_writes():
    migration = Path("sql/postgresql/014_multiturn_form_filling.sql").read_text(encoding="utf-8")
    materials_migration = Path("sql/postgresql/015_water_permit_application_materials.sql").read_text(encoding="utf-8")
    document_schema_migration = Path(
        "sql/postgresql/016_document_derived_application_form_schemas.sql"
    ).read_text(encoding="utf-8")
    combined = Path("sql/postgresql/declaration_qa_postgresql_all.sql").read_text(encoding="utf-8")

    for token in ["business_item_form_schemas", "version_no", "fields JSONB", "idempotency_key"]:
        assert token in migration
        assert token in combined
    for token in ["water-permit-v2", '"type":"material_group"', "version_no < 2"]:
        assert token in materials_migration
        assert token in combined
    for token in [
        "disabled-veterans-centralized-support-v1",
        "fire-accident-review-v1",
        "foreign-marriage-registration-v1",
        "foreign-adoption-registration-v1",
        "high-risk-sports-event-permit-v1",
        "source_document_uid",
    ]:
        assert token in document_schema_migration
        assert token in combined
