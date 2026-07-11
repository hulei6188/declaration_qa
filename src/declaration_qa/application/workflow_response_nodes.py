"""申报问答工作流的回答和审计节点。"""

from __future__ import annotations

from langgraph.config import get_stream_writer

from declaration_qa.application.state import DeclarationQAState
from declaration_qa.application.workflow_helpers.actions import suggested_actions
from declaration_qa.application.workflow_helpers.events import add_event
from declaration_qa.application.workflow_helpers.streaming import is_declaration_qa_stream_run
from declaration_qa.observability.langfuse import observation, safe_value
from declaration_qa.memory.context import build_short_term_context


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

    def llm_generate(self, state: DeclarationQAState) -> DeclarationQAState:
        """调用回答生成器生成最终回答。"""
        next_state = dict(state)
        memory_manager = getattr(self, "context_memory_manager", None)
        if memory_manager is not None:
            next_state["long_term_context"] = memory_manager.retrieve_long_term_context(next_state)
        next_state["short_term_context"] = build_short_term_context(next_state)
        short_term_metrics = (next_state.get("short_term_context") or {}).get("metrics") or {}
        long_term_metrics = long_term_context_metrics(next_state.get("long_term_context") or {})
        generation_plan = answer_generation_plan(self.answer_generator, next_state)
        next_state["answer_generation_plan"] = generation_plan
        add_event(next_state, "short_term_context", short_term_metrics)
        add_event(next_state, "long_term_context", long_term_metrics)
        add_event(next_state, "answer_generation_routed", generation_plan)
        with observation(
            "workflow.answer_generation",
            input={
                "query": next_state.get("query") or "",
                "intent": next_state.get("intent") or "",
                "source_count": len(next_state.get("retrieved_sources") or []),
                "short_term_context": short_term_metrics,
                "long_term_context": long_term_metrics,
                "generation_plan": generation_plan,
            },
            metadata=response_observation_metadata(next_state),
        ) as span:
            if is_declaration_qa_stream_run():
                stream_generate = getattr(self.answer_generator, "stream_generate", None)
            else:
                stream_generate = None
            if callable(stream_generate):
                token_stream, prompt = stream_generate(dict(next_state))
                writer = get_stream_writer()
                answer_parts = []
                for token in token_stream:
                    text = str(token or "")
                    if not text:
                        continue
                    answer_parts.append(text)
                    writer({"event": "token", "data": {"content": text}})
                answer = "".join(answer_parts)
                add_event(next_state, "answer_generated", {"intent": next_state.get("intent"), "streamed": True})
                streamed = True
            else:
                answer, prompt = self.answer_generator.generate(next_state)
                add_event(next_state, "answer_generated", {"intent": next_state.get("intent")})
                streamed = False
            next_state["answer"] = answer
            audit_payload = dict(next_state.get("audit_payload") or {})
            audit_payload["final_prompt"] = prompt
            audit_payload["short_term_context"] = next_state.get("short_term_context") or {}
            audit_payload["long_term_context"] = next_state.get("long_term_context") or {}
            audit_payload["answer_generation_plan"] = generation_plan
            next_state["audit_payload"] = audit_payload
            span.update(
                output=safe_value(
                    {
                        "answer": answer,
                        "prompt_length": len(prompt),
                        "streamed": streamed,
                        "short_term_context": short_term_metrics,
                        "long_term_context": long_term_metrics,
                        "generation_plan": generation_plan,
                    }
                )
            )
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
            "query_rewrite.stage1": next_state.get("query_rewrite.stage1") or {},
            "retrieval_chain": next_state.get("retrieval_chain"),
            "region_code": next_state.get("region_code"),
            "biz_type": next_state.get("biz_type"),
            "business_snapshot": next_state.get("business_snapshot"),
            "agent_plan": next_state.get("agent_plan") or {},
            "agent_iteration": int(next_state.get("agent_iteration") or 0),
            "agent_selected_item_code": next_state.get("agent_selected_item_code") or "",
            "agent_selected_application_id": next_state.get("agent_selected_application_id"),
            "tool_results": next_state.get("tool_results") or [],
            "retrieval_filter": next_state.get("retrieval_filter"),
            "retrieval_query": next_state.get("retrieval_query") or "",
            "rag_query": next_state.get("rag_query") or "",
            "query_rewrite.stage2": next_state.get("query_rewrite") or {},
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


def response_observation_metadata(state: dict[str, object]) -> dict[str, object]:
    """构建响应节点观测的紧凑元数据。"""
    return safe_value(
        {
            "langfuse_trace_id": state.get("langfuse_trace_id") or "",
            "user_id": state.get("user_id"),
            "session_id": state.get("session_id") or "",
            "route": state.get("route") or "",
            "intent": state.get("intent") or "",
            "retrieval_chain": state.get("retrieval_chain") or "",
            "source_count": len(state.get("retrieved_sources") or []),
            "short_term_context": ((state.get("short_term_context") or {}).get("metrics") or {}),
            "long_term_context": long_term_context_metrics(state.get("long_term_context") or {}),
            "answer_generation_plan": state.get("answer_generation_plan") or {},
        }
    )


def answer_generation_plan(answer_generator: object, state: dict[str, object]) -> dict[str, object]:
    """获取回答生成器的文本或多模态路由计划。"""
    planner = getattr(answer_generator, "generation_plan", None)
    if not callable(planner):
        return {"mode": "text", "model": getattr(answer_generator, "name", type(answer_generator).__name__)}
    try:
        plan = planner(state)
    except Exception as exc:
        return {
            "mode": "text",
            "model": getattr(answer_generator, "name", type(answer_generator).__name__),
            "route_error": f"{type(exc).__name__}: {str(exc)[:160]}",
        }
    return plan if isinstance(plan, dict) else {"mode": "text"}


def long_term_context_metrics(context: dict[str, object]) -> dict[str, object]:
    """提取长期记忆上下文指标。"""
    items = context.get("items") if isinstance(context, dict) else []
    status = context.get("status") if isinstance(context, dict) else {}
    cache = context.get("cache") if isinstance(context, dict) else {}
    return {
        "memory_count": len(items or []) if isinstance(items, list) else 0,
        "retrieval_mode": (status or {}).get("retrieval_mode") if isinstance(status, dict) else "",
        "distance_threshold": (status or {}).get("distance_threshold") if isinstance(status, dict) else None,
        "cache_hit": bool((cache or {}).get("hit")) if isinstance(cache, dict) else False,
    }
