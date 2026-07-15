"""申报问答工作流的回答和审计节点。"""

from __future__ import annotations

import re
from uuid import uuid4

from declaration_qa.application.agent import AGENT_SYSTEM_PROMPT
from declaration_qa.application.state import DeclarationQAState
from declaration_qa.application.helpers.actions import suggested_actions
from declaration_qa.application.helpers.events import add_event
from declaration_qa.rag.prompting import application_history_answer, ticket_applications


class ResponseNodes:
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
        elif reason == "evidence_coverage_incomplete":
            answer = "已进行多轮检索，但公开证据仍未完整覆盖当前问题的全部方面，建议补充更具体的事项名称或联系办理部门确认。"
        else:
            answer = "暂未查询到与当前地区和业务类型完全匹配的办理依据，建议查看办事指南或联系人工客服确认。"
        next_state["answer"] = answer
        add_event(next_state, "refused", {"reason": reason})
        return next_state

    def commit_agent_answer(self, state: DeclarationQAState) -> DeclarationQAState:
        """直接使用单 Agent 最后一轮无 Tool Call 的文本响应。"""
        next_state = dict(state)
        answer = str(next_state.get("agent_final_answer") or "").strip()
        prompt = str(next_state.get("agent_final_prompt") or "")
        provider = str(next_state.get("agent_answer_provider") or "")
        if not answer:
            next_state["refused"] = True
            next_state["refuse_reason"] = next_state.get("refuse_reason") or "agent_empty_answer"
            answer = "当前无法生成可靠回答，请稍后重试。"
        answer, history_fallback = ensure_complete_application_history(answer, next_state)
        generation_plan = {
            "mode": next_state.get("answer_mode") or "direct",
            "provider": provider or "native_tool_calling",
            "history_integrity_fallback": history_fallback,
        }
        next_state["answer"] = answer
        next_state["agent_answer_provider"] = provider
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
            "evidence_ok": bool(next_state.get("agent_final_answer")),
            "evidence_decision": {},
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
        memory_status = self.memory_runtime.schedule_extraction(
            user_id=int(next_state.get("user_id") or 0),
            query=str(next_state.get("query") or ""),
            answer=answer,
        )
        add_event(next_state, "long_term_memory", memory_status)
        add_event(next_state, "audit_logged", {"audit_id": audit_id})
        return next_state


def agent_model_config(agent: object | None) -> dict[str, object]:
    """Return non-secret model identity fields for run reproducibility."""
    model = getattr(agent, "model", None)
    if model is None:
        return {}
    return {
        "class": type(model).__name__,
        "model": str(getattr(model, "model_name", None) or getattr(model, "model", None) or ""),
        "multimodal_enabled": bool(getattr(agent, "multimodal_enabled", False)),
        "max_evidence_images": int(getattr(agent, "max_evidence_images", 0) or 0),
    }


def ensure_complete_application_history(answer: str, state: dict[str, object]) -> tuple[str, bool]:
    """保留 Agent 回答，仅在遗漏整次申请时追加对应历史。"""
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
    """Match an application identifier only when it is labeled as an application."""
    identifier = str(application_id or "").strip()
    if not identifier:
        return False
    pattern = rf"(?:application[\s_-]*id|申请(?:编号|ID)?)[：:\s#-]*{re.escape(identifier)}(?!\d)"
    return re.search(pattern, answer, flags=re.IGNORECASE) is not None
