"""政务服务智能体工作流的回答和审计节点。"""

from __future__ import annotations

import re
from uuid import uuid4

from gov_service_agent.application.agent import AGENT_SYSTEM_PROMPT
from gov_service_agent.application.evidence_registry import build_evidence_registry
from gov_service_agent.application.state import GovServiceAgentState
from gov_service_agent.application.helpers.actions import suggested_actions
from gov_service_agent.application.helpers.events import add_event
from gov_service_agent.memory.evidence import HISTORICAL_ASSISTANT_PREFIX
from gov_service_agent.rag.prompting import application_history_answer, ticket_applications


class ResponseNodes:
    """拒答、回答生成、后处理和审计节点。"""

    def refuse_or_human(self, state: GovServiceAgentState) -> GovServiceAgentState:
        """在证据不足时生成拒答和人工建议。"""
        next_state = dict(state)
        next_state["refused"] = True
        reason = next_state.get("refuse_reason") or "out_of_scope"
        next_state["refuse_reason"] = reason
        if reason == "clarification_required":
            answer = "请补充你想咨询的事项名称、地区或具体问题，例如材料清单、办理流程、办理进度或驳回原因。"
        elif reason == "unsafe_or_out_of_scope":
            answer = "该问题不属于申报辅助范围，或涉及不合规操作，暂不能提供处理建议。"
        elif reason == "agent_tool_not_found":
            answer = "暂未查询到与该事项匹配的个人申请或公开办理依据。"
        elif reason == "agent_tool_invalid_arguments":
            answer = "当前问题中的事项信息不足，暂时无法完成业务查询。"
        elif reason == "agent_tool_error":
            answer = "业务查询服务暂时不可用，请稍后重试。"
        else:
            answer = "暂未查询到与当前地区和业务类型完全匹配的办理依据，建议查看办事指南或联系人工客服确认。"
        next_state["answer"] = answer
        add_event(next_state, "refused", {"reason": reason})
        return next_state

    def commit_agent_answer(self, state: GovServiceAgentState) -> GovServiceAgentState:
        """直接使用单 Agent 最后一轮无 Tool Call 的文本响应。"""
        next_state = dict(state)
        answer = strip_historical_assistant_prefix(next_state.get("agent_final_answer"))
        prompt = str(next_state.get("agent_final_prompt") or "")
        provider = str(next_state.get("agent_answer_provider") or "")
        next_state["evidence_registry"] = build_evidence_registry(
            next_state.get("retrieved_sources") or [],
            next_state.get("tool_results") or [],
        )
        transaction_answer = authoritative_appointment_write_answer(next_state)
        if transaction_answer:
            answer = transaction_answer
        if not answer:
            next_state["refused"] = True
            next_state["refuse_reason"] = next_state.get("refuse_reason") or "agent_empty_answer"
            answer = "当前无法生成可靠回答，请稍后重试。"
        answer, history_fallback = ensure_complete_application_history(answer, next_state)
        effective_provider = (
            "deterministic_transaction_result"
            if transaction_answer
            else provider or "native_tool_calling"
        )
        generation_plan = {
            "mode": next_state.get("answer_mode") or "direct",
            "provider": effective_provider,
            "history_integrity_fallback": history_fallback,
            "transaction_result_override": bool(transaction_answer),
        }
        next_state["answer"] = answer
        next_state["agent_answer_provider"] = effective_provider
        next_state["answer_generation_plan"] = generation_plan
        audit_payload = dict(next_state.get("audit_payload") or {})
        audit_payload["final_prompt"] = prompt
        audit_payload["answer_generation_plan"] = generation_plan
        next_state["audit_payload"] = audit_payload
        add_event(next_state, "answer_generated", {"intent": next_state.get("intent"), **generation_plan})
        return next_state

    def answer_postprocess(self, state: GovServiceAgentState) -> GovServiceAgentState:
        """补充默认回答并整理输出。"""
        next_state = dict(state)
        next_state["suggested_actions"] = suggested_actions(next_state)
        add_event(next_state, "suggested_actions", {"items": next_state["suggested_actions"]})
        return next_state

    def audit_log(self, state: GovServiceAgentState) -> GovServiceAgentState:
        """写入问答审计日志。"""
        next_state = dict(state)
        agent_messages = list(next_state.get("agent_messages") or [])
        answer = str(next_state.get("answer") or "")
        if answer:
            agent_messages.append(
                {
                    "id": uuid4().hex,
                    "role": "assistant",
                    "content": answer,
                }
            )
        next_state["agent_messages"] = agent_messages
        payload = {
            "user_id": next_state.get("user_id"),
            "session_id": next_state.get("session_id"),
            "query": next_state.get("query"),
            "query_image": next_state.get("query_image") or None,
            "query_image_ocr_text": next_state.get("query_image_ocr_text") or "",
            "normalized_query": next_state.get("normalized_query"),
            "intent": next_state.get("intent"),
            "sub_intent": next_state.get("sub_intent"),
            "route": next_state.get("route"),
            "route_decision": next_state.get("route_decision") or {},
            "retrieval_chain": next_state.get("retrieval_chain"),
            "region_code": next_state.get("region_code"),
            "biz_type": next_state.get("biz_type"),
            "business_snapshot": next_state.get("business_snapshot"),
            "agent_plan": next_state.get("agent_plan") or {},
            "agent_query_plan": next_state.get("agent_query_plan") or {},
            "agent_iteration": int(next_state.get("agent_iteration") or 0),
            "agent_answer_provider": next_state.get("agent_answer_provider") or "",
            "answer_mode": next_state.get("answer_mode") or "",
            "agent_final_model": next_state.get("agent_final_model") or "",
            "agent_final_token_usage": next_state.get("agent_final_token_usage") or {},
            "agent_final_latency_ms": next_state.get("agent_final_latency_ms"),
            "agent_system_prompt": AGENT_SYSTEM_PROMPT,
            "agent_prompt_version": "single-agent-tool-loop-v1",
            "tool_registry_version": "tool-registry-v1",
            "agent_model_config": agent_model_config(getattr(self, "agent", None)),
            "agent_selected_item_code": next_state.get("agent_selected_item_code") or "",
            "agent_selected_application_id": next_state.get("agent_selected_application_id"),
            "tool_results": next_state.get("tool_results") or [],
            "agent_messages": agent_messages,
            "retrieval_filter": next_state.get("retrieval_filter"),
            "retrieval_query": next_state.get("retrieval_query") or "",
            "rag_query": next_state.get("rag_query") or "",
            "retrieval_queries": next_state.get("retrieval_queries") or [],
            "retrieved_sources": next_state.get("retrieved_sources") or [],
            "rag_status": next_state.get("rag_status"),
            "evidence_registry": next_state.get("evidence_registry") or {"knowledge": [], "business": []},
            "refused": next_state.get("refused"),
            "refuse_reason": next_state.get("refuse_reason"),
            "answer": next_state.get("answer"),
            "suggested_actions": next_state.get("suggested_actions") or [],
            "events": next_state.get("events") or [],
            "langfuse_trace_id": next_state.get("langfuse_trace_id") or "",
            "long_term_context": next_state.get("long_term_context") or {},
            "conversation_memory": next_state.get("conversation_memory") or {},
            "working_memory": next_state.get("working_memory") or {},
            **(next_state.get("audit_payload") or {}),
        }
        audit_id = self.audit_store.append(payload)
        next_state["audit_id"] = audit_id
        memory_status = self.memory_runtime.schedule_extraction(
            user_id=int(next_state.get("user_id") or 0),
            query=str(next_state.get("query") or ""),
            answer=answer,
        )
        add_event(next_state, "long_term_memory", memory_status)
        add_event(next_state, "audit_logged", {"audit_id": audit_id})
        return next_state


def strip_historical_assistant_prefix(value: object) -> str:
    """移除模型误抄到最终回答开头的历史消息信任标记。"""
    answer = str(value or "").strip()
    while answer.startswith(HISTORICAL_ASSISTANT_PREFIX):
        answer = answer[len(HISTORICAL_ASSISTANT_PREFIX) :].lstrip()
    return answer


def authoritative_appointment_write_answer(state: dict) -> str:
    """根据已提交的预约写工具结果生成不可歧义的最终状态。"""
    results = list(state.get("tool_results") or [])
    registry = (state.get("evidence_registry") or {}).get("business") or []
    for result in reversed(results):
        if not isinstance(result, dict) or not result.get("ok"):
            continue
        tool_name = str(result.get("tool_name") or "")
        if tool_name not in {"appointment_book", "appointment_cancel"}:
            continue
        data = result.get("data") if isinstance(result.get("data"), dict) else {}
        appointment = data.get("appointment") if isinstance(data.get("appointment"), dict) else {}
        if not appointment:
            continue
        citation = next(
            (
                f"[{entry['evidence_id']}]"
                for entry in registry
                if entry.get("tool_name") == tool_name
                and str(entry.get("call_id") or "") == str(result.get("call_id") or "")
            ),
            "",
        )
        if not citation:
            continue
        if tool_name == "appointment_book":
            return booking_result_answer(appointment, citation)
        return cancellation_result_answer(appointment, citation)
    return ""


def booking_result_answer(appointment: dict, citation: str) -> str:
    """格式化已经经过人工确认的预约落库结果。"""
    booking_status = str(appointment.get("booking_status") or "")
    if booking_status == "created":
        headline = "预约成功，人工确认已完成，预约已经正式生效。"
    elif booking_status in {"already_booked", "idempotent_replay"}:
        headline = "该时段已有有效预约，本次没有重复创建预约。"
    else:
        headline = "预约操作已成功完成。"
    return "\n".join(
        [
            f"{headline}{citation}",
            "",
            f"- 预约编号：{appointment.get('appointment_id')} {citation}",
            f"- 事项：{appointment.get('item_name') or '-'} {citation}",
            f"- 时间：{appointment_time_text(appointment)} {citation}",
            f"- 大厅：{appointment.get('hall_name') or '-'} {citation}",
            f"- 地址：{appointment.get('hall_address') or '-'} {citation}",
            f"- 当前状态：{appointment_status_text(appointment.get('status') or 'booked')} {citation}",
        ]
    )


def cancellation_result_answer(appointment: dict, citation: str) -> str:
    """格式化已经经过人工确认的预约取消结果。"""
    cancellation_status = str(appointment.get("cancellation_status") or "")
    headline = (
        "预约此前已经取消，本次没有重复释放号源。"
        if cancellation_status == "already_cancelled"
        else "预约已成功取消，人工确认已完成，取消结果已经正式生效。"
    )
    return "\n".join(
        [
            f"{headline}{citation}",
            "",
            f"- 预约编号：{appointment.get('appointment_id')} {citation}",
            f"- 事项：{appointment.get('item_name') or '-'} {citation}",
            f"- 原预约时间：{appointment_time_text(appointment)} {citation}",
            f"- 大厅：{appointment.get('hall_name') or '-'} {citation}",
            f"- 当前状态：{appointment_status_text(appointment.get('status') or 'cancelled')} {citation}",
        ]
    )


def appointment_time_text(appointment: dict) -> str:
    """组合预约日期和起止时间。"""
    date = str(appointment.get("service_date") or "-")
    start = display_clock(appointment.get("start_time"))
    end = display_clock(appointment.get("end_time"))
    time = "-".join(value for value in (start, end) if value)
    return " ".join(value for value in (date, time) if value)


def display_clock(value: object) -> str:
    """把数据库时间统一显示为时分。"""
    raw = str(value or "")
    return raw[:5] if len(raw) >= 5 and raw[2:3] == ":" else raw


def appointment_status_text(value: object) -> str:
    """把预约机器状态转换成中文。"""
    return {
        "booked": "已预约",
        "checked_in": "已签到",
        "waiting": "排队中",
        "called": "已叫号",
        "completed": "已完成",
        "cancelled": "已取消",
    }.get(str(value or ""), str(value or "-"))


def agent_model_config(agent: object | None) -> dict[str, object]:
    """返回用于复现实验的非敏感模型标识字段。"""
    model = getattr(agent, "model", None)
    if model is None:
        return {}
    multimodal_model = getattr(agent, "multimodal_model", None)
    return {
        "class": type(model).__name__,
        "model": str(getattr(model, "model_name", None) or getattr(model, "model", None) or ""),
        "text_model": str(getattr(model, "model_name", None) or getattr(model, "model", None) or ""),
        "multimodal_model": str(
            getattr(multimodal_model, "model_name", None)
            or getattr(multimodal_model, "model", None)
            or ""
        ),
        "multimodal_enabled": bool(getattr(agent, "multimodal_enabled", False)),
        "max_evidence_images": int(getattr(agent, "max_evidence_images", 0) or 0),
    }


def ensure_complete_application_history(answer: str, state: dict[str, object]) -> tuple[str, bool]:
    """保留 Agent 回答，仅在遗漏整次申请时追加对应历史。"""
    query = str(state.get("normalized_query") or state.get("query") or "")
    history_intents = ["进度", "状态", "节点", "退回", "驳回", "补正", "审核", "受理", "办结", "是否通过", "申请记录", "历史申请"]
    if not any(token in query for token in history_intents):
        return answer, False
    applications = ticket_applications(state)
    if not applications:
        return answer, False
    missing_applications = [
        application
        for application in applications
        if not answer_mentions_application(answer, application.get("application_id"))
    ]
    if not missing_applications:
        return answer, False
    history = application_history_answer(state, applications=missing_applications)
    if not answer:
        return history, True
    return f"{answer}\n\n为确保申请记录完整，补充模型遗漏的申请历史：\n\n{history}", True


def answer_mentions_application(answer: str, application_id: object) -> bool:
    """仅在明确标注为申请时匹配申请标识。"""
    identifier = str(application_id or "").strip()
    if not identifier:
        return False
    pattern = rf"(?:application[\s_-]*id|申请(?:编号|ID)?)[：:\s#-]*{re.escape(identifier)}(?!\d)"
    return re.search(pattern, answer, flags=re.IGNORECASE) is not None
