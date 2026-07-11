"""申报问答工作流的回答和审计节点。"""

from __future__ import annotations

from declaration_qa.application.state import DeclarationQAState
from declaration_qa.application.workflow_helpers.actions import suggested_actions
from declaration_qa.application.workflow_helpers.events import add_event
from declaration_qa.rag.prompting import application_history_answer, ticket_applications


class WorkflowResponseNodes:
    """拒答、回答生成、后处理和审计节点。"""
    def refuse_or_human(self, state: DeclarationQAState) -> DeclarationQAState:
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

    def agent_finalize_answer(self, state: DeclarationQAState) -> DeclarationQAState:
        """使用 Agent 的最终响应；仅在 Agent 无法作答时走本地兼容兜底。"""
        next_state = dict(state)
        answer = str(next_state.get("agent_final_answer") or "").strip()
        prompt = str(next_state.get("agent_final_prompt") or "")
        provider = str(next_state.get("agent_answer_provider") or "")
        fallback = False
        if not answer:
            answer, prompt = self.answer_generator.generate(next_state)
            provider = f"fallback:{getattr(self.answer_generator, 'name', type(self.answer_generator).__name__)}"
            fallback = True
        answer, history_fallback = ensure_complete_application_history(answer, next_state)
        fallback = fallback or history_fallback
        generation_plan = {
            "mode": "agent",
            "provider": provider or "native_tool_calling",
            "fallback": fallback,
            "history_integrity_fallback": history_fallback,
        }
        next_state["answer"] = answer
        next_state["answer_generation_plan"] = generation_plan
        audit_payload = dict(next_state.get("audit_payload") or {})
        audit_payload["final_prompt"] = prompt
        audit_payload["answer_generation_plan"] = generation_plan
        next_state["audit_payload"] = audit_payload
        add_event(next_state, "answer_generated", {"intent": next_state.get("intent"), **generation_plan})
        return next_state

    def answer_postprocess(self, state: DeclarationQAState) -> DeclarationQAState:
        """补充默认回答并整理输出。"""
        next_state = dict(state)
        next_state["suggested_actions"] = suggested_actions(next_state)
        add_event(next_state, "suggested_actions", {"items": next_state["suggested_actions"]})
        return next_state

    def audit_log(self, state: DeclarationQAState) -> DeclarationQAState:
        """写入问答审计日志。"""
        next_state = dict(state)
        payload = {
            "user_id": next_state.get("user_id"),
            "session_id": next_state.get("session_id"),
            "query": next_state.get("query"),
            "query_image": next_state.get("query_image") or None,
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
            "agent_selected_item_code": next_state.get("agent_selected_item_code") or "",
            "agent_selected_application_id": next_state.get("agent_selected_application_id"),
            "tool_results": next_state.get("tool_results") or [],
            "retrieval_filter": next_state.get("retrieval_filter"),
            "retrieval_query": next_state.get("retrieval_query") or "",
            "rag_query": next_state.get("rag_query") or "",
            "retrieval_queries": next_state.get("retrieval_queries") or [],
            "retrieved_sources": next_state.get("retrieved_sources") or [],
            "rag_status": next_state.get("rag_status"),
            "evidence_ok": next_state.get("evidence_ok"),
            "evidence_decision": next_state.get("evidence_decision") or {},
            "refused": next_state.get("refused"),
            "refuse_reason": next_state.get("refuse_reason"),
            "answer": next_state.get("answer"),
            "suggested_actions": next_state.get("suggested_actions") or [],
            "events": next_state.get("events") or [],
            "langfuse_trace_id": next_state.get("langfuse_trace_id") or "",
            "short_term_context": next_state.get("short_term_context") or {},
            "long_term_context": next_state.get("long_term_context") or {},
            **(next_state.get("audit_payload") or {}),
        }
        audit_id = self.audit_store.append(payload)
        next_state["audit_id"] = audit_id
        memory_manager = getattr(self, "context_memory_manager", None)
        if memory_manager is not None:
            memory_status = memory_manager.after_interaction(payload, audit_id=audit_id)
            add_event(next_state, "context_memory", memory_status)
        add_event(next_state, "audit_logged", {"audit_id": audit_id})
        return next_state


def ensure_complete_application_history(answer: str, state: dict[str, object]) -> tuple[str, bool]:
    """防止 Agent 漏掉同一事项的申请或状态流水。"""
    intent = str(state.get("intent") or "")
    if intent not in {"status", "ticket_status", "reject", "rejection_reason"}:
        return answer, False
    applications = ticket_applications(state)
    if not applications:
        return answer, False
    required_values: list[str] = []
    for application in applications:
        required_values.append(str(application.get("application_id") or ""))
        for event in application.get("status_history") or []:
            if not isinstance(event, dict):
                continue
            required_values.extend(
                str(value)
                for value in (event.get("status"), event.get("changed_at"), event.get("reason"))
                if str(value or "").strip()
            )
    if all(value in answer for value in required_values if value):
        return answer, False
    history = application_history_answer(state)
    if "knowledge" in set(state.get("agent_required_evidence") or []) and answer:
        history = f"{history}\n\n公开办理依据补充：\n{answer}"
    return history, True
