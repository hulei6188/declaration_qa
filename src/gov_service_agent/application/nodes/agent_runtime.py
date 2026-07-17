"""单 Agent 的规划、工具执行和观察节点。"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from datetime import datetime, timezone
from time import perf_counter
from typing import Any
from uuid import uuid4

from langgraph.types import interrupt

from gov_service_agent.application.agent import accumulated_knowledge_queries, choose_application_id
from gov_service_agent.application.evidence_registry import build_evidence_registry
from gov_service_agent.application.state import GovServiceAgentState
from gov_service_agent.application.tools import ToolExecutionContext, ToolExecutionResult
from gov_service_agent.application.tools.knowledge import KnowledgeSearchInput
from gov_service_agent.application.helpers.events import add_event
from gov_service_agent.application.helpers.retrieval_sources import merge_accumulated_source_payloads
from gov_service_agent.application.helpers.text_safety import is_unsafe_query, mask_sensitive_text
from gov_service_agent.observability.performance import log_performance, performance_stage
from gov_service_agent.observability.langfuse import observation, record_error, safe_value
from gov_service_agent.memory.evidence import (
    RECENT_CONVERSATION_USER_TURNS,
    build_conversation_memory,
    project_historical_messages,
)
from gov_service_agent.rag.engine.text import normalize_text
from gov_service_agent.rag.retrieval import KNOWLEDGE_RETRIEVAL_CHAIN, normalize_retrieval_filter
from gov_service_agent.rag.router import ROUTE_KNOWLEDGE_RETRIEVAL


class AgentRuntimeNodes:
    """有界 ReAct 循环使用的 LangGraph 节点。"""

    def agent_prepare(self, state: GovServiceAgentState) -> GovServiceAgentState:
        """冻结可信知识输入、加载非业务上下文并初始化 Agent。"""
        next_state = dict(state)
        query_text = str(next_state.get("normalized_query") or "")
        if is_unsafe_query(query_text):
            next_state["refuse_reason"] = "unsafe_or_out_of_scope"
        elif not query_text and not next_state.get("query_image"):
            next_state["refuse_reason"] = "clarification_required"
        frozen_query = normalize_text(mask_sensitive_text(next_state.get("normalized_query") or next_state.get("query") or ""))
        next_state["frozen_knowledge_query"] = frozen_query
        next_state["long_term_context"] = self.memory_runtime.retrieve(
            user_id=int(next_state.get("user_id") or 0),
            query=str(next_state.get("normalized_query") or next_state.get("query") or ""),
        )
        next_state["agent_iteration"] = 0
        next_state["agent_pending_tool_calls"] = []
        next_state["agent_call_signatures"] = []
        messages = project_historical_messages(
            list(next_state.get("agent_messages") or []),
            keep_recent_user_turns=max(0, RECENT_CONVERSATION_USER_TURNS - 1),
            memory_context=dict(next_state.get("memory_context") or {}),
        )
        messages.append(
            {
                "id": uuid4().hex,
                "role": "user",
                "content": str(next_state.get("query") or ""),
            }
        )
        next_state["agent_messages"] = messages
        next_state["retrieved_sources"] = []
        next_state["tool_results"] = []
        next_state["conversation_memory"] = build_conversation_memory(
            next_state.get("memory_context") or {},
        )
        next_state["working_memory"] = {
            "active_knowledge_chunk_ids": [
                str(item.get("chunk_id") or "") for item in next_state["retrieved_sources"]
            ],
            "active_business_call_ids": [
                str(item.get("call_id") or "") for item in next_state["tool_results"]
            ],
            "memory_policy": "current_turn_evidence_only",
            "sufficiency": "agent_decides",
        }
        next_state["active_evidence_registry"] = build_evidence_registry(
            next_state["retrieved_sources"],
            next_state["tool_results"],
        )
        next_state["answer_mode"] = ""
        next_state["agent_plan"] = {}
        next_state["agent_query_plan"] = {}
        next_state["agent_final_answer"] = ""
        next_state["agent_final_prompt"] = ""
        next_state["agent_answer_provider"] = ""
        next_state["agent_final_model"] = ""
        next_state["agent_final_token_usage"] = {}
        next_state["agent_final_latency_ms"] = None
        next_state["agent_selected_item_code"] = ""
        next_state["agent_selected_application_id"] = None
        next_state["route"] = "agentic_rag"
        next_state["retrieval_chain"] = "agentic_rag"
        add_event(next_state, "agent_prepared", {"knowledge_query_frozen": True, "query_planner": "agent_tool_schema"})
        return next_state

    def agent_summarize(self, state: GovServiceAgentState) -> GovServiceAgentState:
        """根据滚动摘要和最近的完整消息构建模型输入。"""
        next_state = dict(state)
        summarized, memory_context = self.memory_runtime.summarize(
            list(next_state.get("agent_messages") or []),
            dict(next_state.get("memory_context") or {}),
        )
        next_state["agent_input_messages"] = summarized
        next_state["memory_context"] = memory_context
        next_state["conversation_memory"] = build_conversation_memory(memory_context)
        running_summary = memory_context.get("running_summary")
        add_event(
            next_state,
            "conversation_summarized",
            {
                "active_message_count": len(summarized),
                "has_running_summary": bool(getattr(running_summary, "summary", "")),
            },
        )
        return next_state

    def agent_refresh_input(self, state: GovServiceAgentState) -> GovServiceAgentState:
        """工具执行后应用已有摘要，不在当前用户轮次触发新摘要。"""
        next_state = dict(state)
        prepared, memory_context = self.memory_runtime.prepare_input(
            list(next_state.get("agent_messages") or []),
            dict(next_state.get("memory_context") or {}),
        )
        next_state["agent_input_messages"] = prepared
        next_state["memory_context"] = memory_context
        add_event(
            next_state,
            "agent_input_refreshed",
            {"active_message_count": len(prepared)},
        )
        return next_state

    def agent_decide(self, state: GovServiceAgentState) -> GovServiceAgentState:
        """让 Agent 决定下一批工具调用或结束循环。"""
        next_state = dict(state)
        iteration = int(next_state.get("agent_iteration") or 0) + 1
        next_state["agent_iteration"] = iteration
        next_state["answer_mode"] = ""
        if next_state.get("refuse_reason"):
            decision_action = "finish"
            calls = []
            decision_payload = {"action": "finish", "reason": next_state["refuse_reason"], "provider": "guardrail"}
        else:
            final_only = iteration > self.agent_max_iterations
            decision = self.agent.decide(
                next_state,
                [] if final_only else self.tool_registry.definitions(),
            )
            planned_calls = [
                {**call, "origin": "agent"}
                for call in decision.tool_calls
            ] if not final_only else []
            calls = unique_new_tool_calls(
                planned_calls,
                existing_signatures=set(next_state.get("agent_call_signatures") or []),
            )
            no_progress = bool(planned_calls) and not calls
            if no_progress:
                decision = self.agent.decide(next_state, [])
            decision_action = "tools" if calls else "finish"
            decision_payload = {
                "action": decision_action,
                "reason": (
                    "max_iterations_final"
                    if final_only
                    else "duplicate_tool_call_final"
                    if no_progress
                    else decision.reason
                ),
                "provider": decision.provider,
                "intent": decision.intent,
                "tool_calls": [audit_tool_call(call) for call in calls],
                "iteration": iteration,
                "assistant_content": decision.assistant_content,
                "model": decision.model,
                "token_usage": decision.token_usage,
                "latency_ms": decision.latency_ms,
            }
            next_state["intent"] = decision.intent
            next_state["sub_intent"] = decision.intent
            if decision_action == "finish":
                next_state["agent_final_answer"] = decision.final_answer or decision.assistant_content
                next_state["agent_final_prompt"] = decision.final_prompt
                next_state["agent_answer_provider"] = decision.provider
                next_state["agent_final_model"] = decision.model
                next_state["agent_final_token_usage"] = dict(decision.token_usage or {})
                next_state["agent_final_latency_ms"] = decision.latency_ms
            if calls:
                messages = list(next_state.get("agent_messages") or [])
                messages.append(
                    {
                        "id": uuid4().hex,
                        "role": "assistant",
                        "content": decision.assistant_content,
                        "tool_calls": [
                            {
                                "id": str(call.get("id") or call.get("name") or ""),
                                "name": str(call.get("name") or ""),
                                "args": dict(call.get("arguments") or {}),
                                "type": "tool_call",
                            }
                            for call in calls
                        ],
                    }
                )
                next_state["agent_messages"] = messages
        has_evidence_flow = bool(next_state.get("tool_results") or next_state.get("retrieved_sources"))
        final_answer = str(next_state.get("agent_final_answer") or "").strip()
        if decision_action == "finish" and final_answer:
            next_state["answer_mode"] = "tool_grounded" if has_evidence_flow else "direct"
        elif has_evidence_flow:
            next_state["answer_mode"] = "tool_grounded"
        elif decision_action == "finish" and not next_state.get("refuse_reason"):
            next_state["refuse_reason"] = "agent_empty_answer"
        next_state["agent_pending_tool_calls"] = calls
        next_state["agent_plan"] = decision_payload
        next_state["agent_next"] = decision_action
        add_event(next_state, "agent_decision", decision_payload)
        return next_state

    def agent_execute_tools(self, state: GovServiceAgentState) -> GovServiceAgentState:
        """经人工确认后并发执行当前 Agent 规划的工具调用。"""
        next_state = dict(state)
        results = list(next_state.get("tool_results") or [])
        signatures = list(next_state.get("agent_call_signatures") or [])
        calls = list(next_state.get("agent_pending_tool_calls") or [])
        protected_calls = [call for call in calls if requires_human_confirmation(str(call.get("name") or ""))]
        approved_call_ids: set[str] = set()
        if protected_calls:
            confirmation = interrupt(build_human_confirmation_payload(protected_calls, next_state))
            approved = bool((confirmation or {}).get("approved")) if isinstance(confirmation, dict) else False
            if approved:
                calls = apply_human_confirmation_selections(
                    calls,
                    next_state,
                    (confirmation or {}).get("selected_slot_ids") or {},
                )
                protected_calls = [
                    call for call in calls
                    if requires_human_confirmation(str(call.get("name") or ""))
                ]
            approved_call_ids = {
                str(call.get("id") or call.get("name") or "")
                for call in protected_calls
            } if approved else set()
            next_state["human_confirmation"] = {
                "approved": approved,
                "actions": build_human_confirmation_actions(protected_calls, next_state),
            }
            add_event(
                next_state,
                "human_confirmation_resolved",
                {
                    "approved": approved,
                    "call_ids": [str(call.get("id") or call.get("name") or "") for call in protected_calls],
                },
            )

        with observation(
            "agent.tool_wave",
            input={
                "call_count": len(calls),
                "tools": [str(call.get("name") or "") for call in calls],
                "parallel": len(calls) > 1,
            },
            metadata={
                "langfuse_trace_id": str(next_state.get("langfuse_trace_id") or ""),
                "session_id": str(next_state.get("session_id") or ""),
                "user_id": int(next_state.get("user_id") or 0),
                "agent_iteration": int(next_state.get("agent_iteration") or 0),
            },
            trace_id=str(next_state.get("langfuse_trace_id") or ""),
        ) as wave_span:
            execution_outputs = self._execute_tool_wave(
                calls,
                next_state,
                approved_call_ids=approved_call_ids,
            )
            wave_span.update(
                output={
                    "call_count": len(execution_outputs),
                    "results": [
                        {
                            "tool_name": str(call.get("name") or ""),
                            "call_id": str(payload.get("call_id") or ""),
                            "ok": bool(payload.get("ok")),
                            "status": str(payload.get("status") or ""),
                            "latency_ms": int(payload.get("latency_ms") or 0),
                        }
                        for call, payload, _ in execution_outputs
                    ],
                }
            )
        for call, payload, signature in execution_outputs:
            results.append(payload)
            append_tool_message(next_state, call, payload)
            if payload.get("status") not in {"policy_denied", "invalid_arguments"}:
                signatures.append(signature)
            add_event(
                next_state,
                "tool_executed",
                {
                    "tool_name": call.get("name") or "",
                    "call_id": payload["call_id"],
                    "ok": payload["ok"],
                    "status": payload["status"],
                    "latency_ms": payload.get("latency_ms"),
                    "parallel": len(calls) > 1,
                },
            )
        next_state["tool_results"] = results
        next_state["agent_call_signatures"] = signatures
        next_state["agent_pending_tool_calls"] = []
        return next_state

    def _execute_tool_wave(
        self,
        calls: list[dict[str, Any]],
        state: GovServiceAgentState,
        *,
        approved_call_ids: set[str] | None = None,
    ) -> list[tuple[dict[str, Any], dict[str, Any], str]]:
        """并发执行一组彼此独立的工具调用，并保持 Agent 给出的顺序。"""
        if not calls:
            return []
        approved_ids = approved_call_ids or set()
        executable_calls = []
        rejected_outputs: dict[str, tuple[dict[str, Any], str]] = {}
        for call in calls:
            call_id = str(call.get("id") or call.get("name") or "")
            if requires_human_confirmation(str(call.get("name") or "")) and call_id not in approved_ids:
                rejected_outputs[call_id] = self._observe_rejected_tool_call(call, state)
            else:
                executable_calls.append(call)
        executed_outputs: dict[str, tuple[dict[str, Any], str]] = {}
        if executable_calls:
            if len(executable_calls) == 1:
                call = executable_calls[0]
                call_id = str(call.get("id") or call.get("name") or "")
                executed_outputs[call_id] = self._execute_agent_tool_call(
                    call,
                    state,
                    human_approved=call_id in approved_ids,
                )
            else:
                with ThreadPoolExecutor(max_workers=min(4, len(executable_calls)), thread_name_prefix="gov-service-agent-tool") as executor:
                    futures = [
                        executor.submit(
                            copy_context().run,
                            self._execute_agent_tool_call,
                            call,
                            state,
                            human_approved=(str(call.get("id") or call.get("name") or "") in approved_ids),
                        )
                        for call in executable_calls
                    ]
                    for call, future in zip(executable_calls, futures):
                        executed_outputs[str(call.get("id") or call.get("name") or "")] = future.result()
        outputs = {**rejected_outputs, **executed_outputs}
        return [
            (call, *outputs[str(call.get("id") or call.get("name") or "")])
            for call in calls
        ]

    def _execute_agent_tool_call(
        self,
        call: dict[str, Any],
        state: GovServiceAgentState,
        *,
        human_approved: bool = False,
    ) -> tuple[dict[str, Any], str]:
        """执行单个已规划工具调用。"""
        started_at = datetime.now(timezone.utc)
        started_clock = perf_counter()
        name = str(call.get("name") or "")
        arguments = dict(call.get("arguments") or {})
        signature = tool_call_signature(name, arguments)
        context = ToolExecutionContext(
            user_id=int(state.get("user_id") or 0),
            region_code=str(state.get("region_code") or ""),
            session_id=str(state.get("session_id") or ""),
            original_query=str(state.get("frozen_knowledge_query") or state.get("normalized_query") or ""),
            metadata={
                "state": state,
                "tool_call_origin": str(call.get("origin") or "agent"),
                "tool_call_id": str(call.get("id") or name),
            },
        )
        call_id = str(call.get("id") or name)
        trace_id = str(state.get("langfuse_trace_id") or "")
        with observation(
            f"agent.tool.{name or 'unknown'}",
            as_type="tool",
            input={
                "tool_name": name,
                "call_id": call_id,
                "arguments": audit_tool_arguments(name, arguments),
            },
            metadata={
                "langfuse_trace_id": trace_id,
                "session_id": str(state.get("session_id") or ""),
                "user_id": int(state.get("user_id") or 0),
                "agent_iteration": int(state.get("agent_iteration") or 0),
                "tool_call_origin": str(call.get("origin") or "agent"),
                "human_approved": human_approved,
            },
            trace_id=trace_id,
        ) as tool_span:
            denial_reason = "" if human_approved else agent_tool_call_denial_reason(
                name,
                str(state.get("normalized_query") or state.get("query") or ""),
                arguments=arguments,
            )
            if denial_reason:
                result = ToolExecutionResult(
                    tool_name=name,
                    call_id=call_id,
                    ok=False,
                    status="policy_denied",
                    error=denial_reason,
                )
            else:
                result = self.tool_registry.execute(
                    name,
                    arguments,
                    context,
                    call_id=call_id,
                )
            result_payload = result.payload()
            if human_approved and requires_human_confirmation(name):
                result_payload["data"] = {
                    **(result_payload.get("data") or {}),
                    "human_confirmation": {
                        "required": True,
                        "approved": True,
                        "completed": True,
                    },
                }
            completed_at = datetime.now(timezone.utc)
            payload = {
                **result_payload,
                "arguments": audit_tool_arguments(name, arguments),
                "started_at": started_at.isoformat(),
                "completed_at": completed_at.isoformat(),
                "latency_ms": max(0, round((perf_counter() - started_clock) * 1000)),
            }
            observation_output = safe_value(
                {
                    "tool_name": name,
                    "call_id": call_id,
                    "ok": result.ok,
                    "status": result.status,
                    "data": result_payload.get("data") or {},
                    "error": result.error,
                    "latency_ms": payload["latency_ms"],
                }
            )
            if result.status in {"tool_not_found", "invalid_arguments", "policy_denied", "tool_error"}:
                record_error(
                    tool_span,
                    stage=f"agent.tool.{name or 'unknown'}",
                    reason=result.status,
                    extra=observation_output,
                )
            else:
                tool_span.update(
                    output=observation_output,
                    metadata={"tool_status": result.status, "tool_ok": result.ok},
                )
        log_performance(
            "agent.tool_call",
            started_clock,
            tool_name=name,
            call_id=call_id,
            tool_status=result.status,
            ok=result.ok,
        )
        return payload, signature

    def _observe_rejected_tool_call(
        self,
        call: dict[str, Any],
        state: GovServiceAgentState,
    ) -> tuple[dict[str, Any], str]:
        """记录因人工拒绝而未执行的工具调用。"""
        name = str(call.get("name") or "")
        call_id = str(call.get("id") or name)
        arguments = dict(call.get("arguments") or {})
        payload, signature = human_rejected_tool_result(call)
        trace_id = str(state.get("langfuse_trace_id") or "")
        with observation(
            f"agent.tool.{name or 'unknown'}",
            as_type="tool",
            input={
                "tool_name": name,
                "call_id": call_id,
                "arguments": audit_tool_arguments(name, arguments),
            },
            metadata={
                "langfuse_trace_id": trace_id,
                "session_id": str(state.get("session_id") or ""),
                "user_id": int(state.get("user_id") or 0),
                "agent_iteration": int(state.get("agent_iteration") or 0),
                "tool_call_origin": str(call.get("origin") or "agent"),
                "human_approved": False,
                "tool_status": "human_rejected",
            },
            trace_id=trace_id,
        ) as tool_span:
            tool_span.update(
                output={
                    "tool_name": name,
                    "call_id": call_id,
                    "ok": False,
                    "status": "human_rejected",
                    "latency_ms": 0,
                }
            )
        return payload, signature

    def agent_observe(self, state: GovServiceAgentState) -> GovServiceAgentState:
        """把知识证据和业务工具结果写入彼此隔离的状态槽。"""
        next_state = dict(state)
        for result in next_state.get("tool_results") or []:
            if not isinstance(result, dict) or not result.get("ok"):
                continue
            data = result.get("data") if isinstance(result.get("data"), dict) else {}
            if result.get("tool_name") == "knowledge_search":
                next_state["retrieved_sources"] = merge_accumulated_source_payloads(
                    [*(next_state.get("retrieved_sources") or []), *(data.get("sources") or [])],
                    min_per_query=int(getattr(self, "min_sources_per_query", 2) or 0),
                    max_sources=int(getattr(self, "max_accumulated_sources", 48) or 48),
                )
                next_state["rag_status"] = dict(data.get("rag_status") or {})
                next_state["retrieval_filter"] = dict(data.get("retrieval_filter") or {})
                next_state["retrieval_queries"] = accumulated_knowledge_queries(next_state.get("tool_results") or [])
                next_state["agent_query_plan"] = {
                    "requested_queries": list(data.get("requested_queries") or []),
                    "executed_queries": next_state["retrieval_queries"],
                    "search_round": int(data.get("search_round") or 1),
                    "accumulated_query_count": len(next_state["retrieval_queries"]),
                    "query_isolation": data.get("query_isolation") or "",
                }
            elif result.get("tool_name") == "ticket_detail_query":
                applications = list(data.get("applications") or [])
                selected_application_id = choose_application_id(
                    str(next_state.get("normalized_query") or next_state.get("query") or ""),
                    applications,
                )
                selected_application = next(
                    (
                        item
                        for item in applications
                        if isinstance(item, dict)
                        and int(item.get("application_id") or 0) == int(selected_application_id or 0)
                    ),
                    {},
                )
                selected_item_code = str(selected_application.get("item_code") or "")
                next_state["agent_selected_item_code"] = selected_item_code
                next_state["agent_selected_application_id"] = selected_application_id
                next_state["business_snapshot"] = {
                    "source": "ticket_detail_query",
                    "item_code": selected_item_code,
                    "item_codes": list(data.get("item_codes") or []),
                    "selected_application_id": next_state.get("agent_selected_application_id"),
                    "applications": applications,
                }
                next_state["answer_mode"] = "tool_grounded"
        next_state["active_evidence_registry"] = build_evidence_registry(
            next_state.get("retrieved_sources") or [],
            next_state.get("tool_results") or [],
        )
        working_memory = dict(next_state.get("working_memory") or {})
        working_memory.update(
            {
                "active_knowledge_chunk_ids": [
                    str(item.get("chunk_id") or "")
                    for item in (next_state.get("retrieved_sources") or [])
                    if isinstance(item, dict)
                ],
                "active_business_call_ids": [
                    str(item.get("call_id") or "")
                    for item in (next_state.get("tool_results") or [])
                    if isinstance(item, dict) and item.get("tool_name") != "knowledge_search" and item.get("ok")
                ],
                "attempted_call_signatures": list(next_state.get("agent_call_signatures") or []),
                "sufficiency": "evidence_available"
                if next_state.get("retrieved_sources") or next_state.get("tool_results")
                else "insufficient",
            }
        )
        next_state["working_memory"] = working_memory
        add_event(
            next_state,
            "agent_observation",
            {
                "tool_result_count": len(next_state.get("tool_results") or []),
                "knowledge_source_count": len(next_state.get("retrieved_sources") or []),
                "selected_item_code": next_state.get("agent_selected_item_code") or "",
                "selected_application_id": next_state.get("agent_selected_application_id"),
            },
        )
        return next_state

    def execute_knowledge_search(
        self,
        request: KnowledgeSearchInput,
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        """执行 Agent 本轮生成的查询计划并统一重排本轮候选。"""
        state = dict(context.metadata.get("state") or {})
        search_round = 1 + sum(
            1
            for item in (state.get("tool_results") or [])
            if isinstance(item, dict) and item.get("tool_name") == "knowledge_search"
        )
        retrieval_queries: list[dict[str, Any]] = []
        seen: set[str] = set()
        for index, query in enumerate(request.queries, start=1):
            text = normalize_text(mask_sensitive_text(query.text))
            if not text or text in seen:
                continue
            seen.add(text)
            retrieval_queries.append(
                {
                    "id": f"agent_q{index}",
                    "text": text,
                    "strategy": query.strategy,
                    "top_k": int(getattr(self.rag_engine, "rerank_top_n_per_query", 12) or 12),
                }
            )
        rerank_query = "\n".join(item["text"] for item in retrieval_queries)
        isolated_state = {
            **state,
            "query": rerank_query,
            "normalized_query": rerank_query,
            "retrieval_query": rerank_query,
            "frozen_knowledge_query": rerank_query,
            "route": ROUTE_KNOWLEDGE_RETRIEVAL,
            "business_snapshot": None,
            "tool_results": [],
            "retrieval_chain": KNOWLEDGE_RETRIEVAL_CHAIN,
            "retrieval_queries": retrieval_queries,
            "knowledge_search_round": search_round,
        }
        filters = normalize_retrieval_filter(state.get("metadata_filter") or {})
        with performance_stage(
            "knowledge_search.retrieve",
            query_count=len(retrieval_queries),
            search_round=search_round,
        ):
            result = self.multi_retrieve(isolated_state, filters)
        current_payloads = [source.payload() for source in result.sources]
        final_sources = merge_accumulated_source_payloads(
            current_payloads,
            min_per_query=int(getattr(self, "min_sources_per_query", 2) or 0),
            max_sources=int(getattr(self, "max_accumulated_sources", 48) or 48),
        )
        rag_status = {
            **(result.status or {}),
            "final_source_count": len(final_sources),
            "final_rerank_applied": False,
            "historical_merge_mode": "current_search_only",
        }
        return {
            "sources": final_sources,
            "rag_status": rag_status,
            "retrieval_filter": result.filter,
            "executed_queries": retrieval_queries,
            "search_round": search_round,
            "accumulated_query_count": len(accumulated_knowledge_queries(state.get("tool_results") or [])) + len(retrieval_queries),
        }


def tool_call_signature(name: str, arguments: dict[str, Any]) -> str:
    if name == "knowledge_search":
        queries = []
        for query in arguments.get("queries") or []:
            if not isinstance(query, dict):
                continue
            queries.append(
                {
                    "text": normalize_text(str(query.get("text") or "")),
                    "strategy": str(query.get("strategy") or "hybrid"),
                }
            )
        queries.sort(key=lambda item: (item["text"], item["strategy"]))
        return f"{name}:{json.dumps(queries, ensure_ascii=False, sort_keys=True)}"
    return f"{name}:{json.dumps(arguments, ensure_ascii=False, sort_keys=True, default=str)}"


def unique_new_tool_calls(calls: list[dict[str, Any]], *, existing_signatures: set[str]) -> list[dict[str, Any]]:
    output = []
    seen = set(existing_signatures)
    normalized_calls = coalesce_batch_business_calls(coalesce_knowledge_search_calls(calls))
    for call in normalized_calls:
        name = str(call.get("name") or "").strip()
        arguments = dict(call.get("arguments") or {})
        signature = tool_call_signature(name, arguments)
        if not name or signature in seen:
            continue
        seen.add(signature)
        output.append(
            {
                "id": str(call.get("id") or name),
                "name": name,
                "arguments": arguments,
                "origin": str(call.get("origin") or "agent"),
            }
        )
    return output


def coalesce_knowledge_search_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """将同一 Agent 决策中的多次知识检索合并为一个受控查询批次。"""
    output: list[dict[str, Any]] = []
    knowledge_call: dict[str, Any] | None = None
    knowledge_index = -1
    seen_queries: set[tuple[str, str]] = set()
    merged_queries: list[dict[str, Any]] = []
    for call in calls:
        if str(call.get("name") or "").strip() != "knowledge_search":
            output.append(call)
            continue
        if knowledge_call is None:
            knowledge_call = dict(call)
            knowledge_index = len(output)
            output.append(knowledge_call)
        for query in dict(call.get("arguments") or {}).get("queries") or []:
            if len(merged_queries) >= 6:
                break
            if not isinstance(query, dict):
                continue
            normalized = {
                "text": normalize_text(str(query.get("text") or "")),
                "strategy": str(query.get("strategy") or "hybrid"),
            }
            signature = (normalized["text"], normalized["strategy"])
            if not normalized["text"] or signature in seen_queries:
                continue
            seen_queries.add(signature)
            merged_queries.append(normalized)
    if knowledge_call is not None:
        if merged_queries:
            output[knowledge_index] = {
                **knowledge_call,
                "arguments": {"queries": merged_queries},
            }
        else:
            output.pop(knowledge_index)
    return output


def coalesce_batch_business_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把申请详情和缴费订单的多次单条调用合并为批量调用。"""
    output = coalesce_ticket_detail_calls(calls)
    output = coalesce_payment_order_calls(output)
    output = coalesce_payment_receipt_calls(output)
    for tool_name in (
        "submission_receipt_query",
        "result_certificate_query",
        "delivery_status_query",
    ):
        output = coalesce_application_batch_calls(output, tool_name=tool_name)
    return output


def coalesce_ticket_detail_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """合并同一轮内模型生成的多个批量申请详情调用。"""
    ticket_calls = [call for call in calls if str(call.get("name") or "").strip() == "ticket_detail_query"]
    if any(not isinstance(dict(call.get("arguments") or {}).get("item_codes"), list) for call in ticket_calls):
        return calls
    output: list[dict[str, Any]] = []
    merged_call: dict[str, Any] | None = None
    merged_index = -1
    item_codes: list[str] = []
    seen_codes: set[str] = set()
    detail_types: set[str] = set()
    for call in calls:
        if str(call.get("name") or "").strip() != "ticket_detail_query":
            output.append(call)
            continue
        if merged_call is None:
            merged_call = dict(call)
            merged_index = len(output)
            output.append(merged_call)
        arguments = dict(call.get("arguments") or {})
        raw_item_codes = arguments.get("item_codes") or []
        candidates = list(raw_item_codes) if isinstance(raw_item_codes, list) else [raw_item_codes]
        for candidate in candidates:
            value = str(candidate or "").strip()
            if value and value not in seen_codes and len(item_codes) < 50:
                seen_codes.add(value)
                item_codes.append(value)
        detail_types.add(str(arguments.get("detail_type") or "full"))
    if merged_call is not None:
        if item_codes:
            detail_type = next(iter(detail_types)) if len(detail_types) == 1 else "full"
            output[merged_index] = {
                **merged_call,
                "arguments": {"item_codes": item_codes, "detail_type": detail_type},
            }
        else:
            output.pop(merged_index)
    return output


def coalesce_payment_order_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """合并同一轮内模型生成的多个批量缴费订单调用。"""
    payment_calls = [call for call in calls if str(call.get("name") or "").strip() == "payment_order_query"]
    if any(not isinstance(dict(call.get("arguments") or {}).get("application_ids"), list) for call in payment_calls):
        return calls
    output: list[dict[str, Any]] = []
    merged_call: dict[str, Any] | None = None
    merged_index = -1
    application_ids: list[int] = []
    seen_ids: set[int] = set()
    for call in calls:
        if str(call.get("name") or "").strip() != "payment_order_query":
            output.append(call)
            continue
        if merged_call is None:
            merged_call = dict(call)
            merged_index = len(output)
            output.append(merged_call)
        arguments = dict(call.get("arguments") or {})
        raw_application_ids = arguments.get("application_ids") or []
        candidates = list(raw_application_ids) if isinstance(raw_application_ids, list) else [raw_application_ids]
        for candidate in candidates:
            try:
                value = int(candidate)
            except (TypeError, ValueError):
                continue
            if value > 0 and value not in seen_ids and len(application_ids) < 100:
                seen_ids.add(value)
                application_ids.append(value)
    if merged_call is not None:
        if application_ids:
            output[merged_index] = {
                **merged_call,
                "arguments": {"application_ids": application_ids},
            }
        else:
            output.pop(merged_index)
    return output


def coalesce_payment_receipt_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """合并同一轮内模型生成的多个电子票据批量调用。"""
    receipt_calls = [call for call in calls if str(call.get("name") or "").strip() == "payment_receipt_query"]
    if any(not isinstance(dict(call.get("arguments") or {}).get("targets"), list) for call in receipt_calls):
        return calls
    output: list[dict[str, Any]] = []
    merged_call: dict[str, Any] | None = None
    merged_index = -1
    targets: list[dict[str, int | None]] = []
    seen_targets: set[tuple[int, int | None]] = set()
    for call in calls:
        if str(call.get("name") or "").strip() != "payment_receipt_query":
            output.append(call)
            continue
        if merged_call is None:
            merged_call = dict(call)
            merged_index = len(output)
            output.append(merged_call)
        for target in dict(call.get("arguments") or {}).get("targets") or []:
            if not isinstance(target, dict):
                continue
            try:
                application_id = int(target.get("application_id") or 0)
                payment_order_id = (
                    int(target["payment_order_id"])
                    if target.get("payment_order_id") is not None
                    else None
                )
            except (TypeError, ValueError):
                continue
            signature = (application_id, payment_order_id)
            if application_id > 0 and signature not in seen_targets and len(targets) < 100:
                seen_targets.add(signature)
                targets.append(
                    {
                        "application_id": application_id,
                        "payment_order_id": payment_order_id,
                    }
                )
    if merged_call is not None:
        if targets:
            output[merged_index] = {**merged_call, "arguments": {"targets": targets}}
        else:
            output.pop(merged_index)
    return output


def coalesce_application_batch_calls(
    calls: list[dict[str, Any]],
    *,
    tool_name: str,
) -> list[dict[str, Any]]:
    """合并同一轮内指定工具的多个 application_ids 批量调用。"""
    matched_calls = [call for call in calls if str(call.get("name") or "").strip() == tool_name]
    if any(not isinstance(dict(call.get("arguments") or {}).get("application_ids"), list) for call in matched_calls):
        return calls
    output: list[dict[str, Any]] = []
    merged_call: dict[str, Any] | None = None
    merged_index = -1
    application_ids: list[int] = []
    seen_ids: set[int] = set()
    for call in calls:
        if str(call.get("name") or "").strip() != tool_name:
            output.append(call)
            continue
        if merged_call is None:
            merged_call = dict(call)
            merged_index = len(output)
            output.append(merged_call)
        for candidate in dict(call.get("arguments") or {}).get("application_ids") or []:
            try:
                value = int(candidate)
            except (TypeError, ValueError):
                continue
            if value > 0 and value not in seen_ids and len(application_ids) < 100:
                seen_ids.add(value)
                application_ids.append(value)
    if merged_call is not None:
        if application_ids:
            output[merged_index] = {**merged_call, "arguments": {"application_ids": application_ids}}
        else:
            output.pop(merged_index)
    return output


def agent_failure_reason(results: list[dict[str, Any]]) -> str:
    statuses = [str(item.get("status") or "") for item in results if isinstance(item, dict)]
    if "not_found" in statuses:
        return "agent_tool_not_found"
    if "invalid_arguments" in statuses:
        return "agent_tool_invalid_arguments"
    if "tool_error" in statuses:
        return "agent_tool_error"
    return "no_matched_source"


def agent_tool_call_allowed(tool_name: str, query: str) -> bool:
    """检查与关键词意图无关的工具级安全约束。"""
    return not agent_tool_call_denial_reason(tool_name, query)


def agent_tool_call_denial_reason(
    tool_name: str,
    query: str,
    *,
    arguments: dict[str, Any] | None = None,
) -> str:
    """返回模型外策略拒绝原因；空字符串表示允许执行。"""
    return ""


HUMAN_CONFIRMATION_TOOL_NAMES = {"appointment_book", "appointment_cancel"}


def requires_human_confirmation(tool_name: str) -> bool:
    """判断工具是否必须在执行前经过人工确认。"""
    return str(tool_name or "") in HUMAN_CONFIRMATION_TOOL_NAMES


def build_human_confirmation_payload(
    calls: list[dict[str, Any]],
    state: dict[str, Any],
) -> dict[str, Any]:
    """构造可由前端直接展示的人工确认载荷。"""
    actions = build_human_confirmation_actions(calls, state)
    tool_names = {str(action.get("tool_name") or "") for action in actions}
    if tool_names == {"appointment_cancel"}:
        title = f"确认取消 {len(actions)} 条预约" if len(actions) > 1 else "确认取消预约"
        warning = "确认后将取消所选预约并释放对应号源。"
    elif tool_names == {"appointment_book"}:
        title = f"确认提交 {len(actions)} 条预约" if len(actions) > 1 else "确认提交预约"
        warning = "确认后将占用对应号源并创建预约记录。"
    else:
        title = "确认执行预约操作"
        warning = "请核对以下操作，确认后系统才会执行。"
    return {
        "type": "tool_confirmation",
        "title": title,
        "warning": warning,
        "actions": actions,
    }


def build_human_confirmation_actions(
    calls: list[dict[str, Any]],
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    """把写工具参数关联到本轮可信查询结果，生成确认明细。"""
    actions: list[dict[str, Any]] = []
    for call in calls:
        tool_name = str(call.get("name") or "")
        arguments = dict(call.get("arguments") or {})
        call_id = str(call.get("id") or tool_name)
        if tool_name == "appointment_cancel":
            appointment_id = int(arguments.get("appointment_id") or 0)
            record = find_business_record(
                state.get("tool_results") or [],
                tool_name="appointment_list_query",
                collection="appointments",
                key="appointment_id",
                value=appointment_id,
            )
            actions.append(
                {
                    "call_id": call_id,
                    "tool_name": tool_name,
                    "label": "取消预约",
                    "subject": str(record.get("item_name") or f"预约 {appointment_id}"),
                    "details": appointment_record_details(record),
                    "arguments": {"appointment_id": appointment_id},
                }
            )
            continue
        slot_id = int(arguments.get("slot_id") or 0)
        record = find_business_record(
            state.get("tool_results") or [],
            tool_name="appointment_slot_query",
            collection="slots",
            key="slot_id",
            value=slot_id,
        )
        actions.append(
            {
                "call_id": call_id,
                "tool_name": tool_name,
                "label": "提交预约",
                "subject": str(record.get("item_name") or f"预约号源 {slot_id}"),
                "details": appointment_record_details(record),
                "arguments": {"slot_id": slot_id},
                "slot_options": appointment_slot_options(state.get("tool_results") or [], record),
            }
        )
    return actions


def appointment_slot_options(
    results: list[dict[str, Any]],
    selected_record: dict[str, Any],
) -> list[dict[str, Any]]:
    """提取本轮同一事项下仍可预约的大厅和时段供人工选择。"""
    item_code = str(selected_record.get("item_code") or "").strip()
    options: list[dict[str, Any]] = []
    seen: set[int] = set()
    for result in results:
        if str(result.get("tool_name") or "") != "appointment_slot_query" or not result.get("ok"):
            continue
        data = result.get("data") if isinstance(result.get("data"), dict) else {}
        for record in data.get("slots") or []:
            if not isinstance(record, dict):
                continue
            slot_id = int(record.get("slot_id") or 0)
            if slot_id <= 0 or slot_id in seen:
                continue
            if item_code and str(record.get("item_code") or "") != item_code:
                continue
            if str(record.get("status") or "") != "available":
                continue
            if int(record.get("remaining_count") or 0) <= 0:
                continue
            seen.add(slot_id)
            options.append(
                {
                    "slot_id": slot_id,
                    "hall_code": str(record.get("hall_code") or ""),
                    "hall_name": str(record.get("hall_name") or ""),
                    "hall_address": str(record.get("hall_address") or ""),
                    "service_date": str(record.get("service_date") or ""),
                    "start_time": str(record.get("start_time") or ""),
                    "end_time": str(record.get("end_time") or ""),
                    "remaining_count": int(record.get("remaining_count") or 0),
                }
            )
    return sorted(
        options,
        key=lambda item: (
            item["hall_name"],
            item["service_date"],
            item["start_time"],
            item["slot_id"],
        ),
    )


def apply_human_confirmation_selections(
    calls: list[dict[str, Any]],
    state: dict[str, Any],
    selected_slot_ids: dict[str, int] | None,
) -> list[dict[str, Any]]:
    """校验人工选择并替换预约写调用中的号源主键。"""
    selections = {str(key): int(value) for key, value in (selected_slot_ids or {}).items()}
    book_call_ids = {
        str(call.get("id") or call.get("name") or "")
        for call in calls
        if str(call.get("name") or "") == "appointment_book"
    }
    unknown = set(selections) - book_call_ids
    if unknown:
        raise ValueError("预约选项已失效，请重新查询可预约大厅和时段")

    updated: list[dict[str, Any]] = []
    for call in calls:
        if str(call.get("name") or "") != "appointment_book":
            updated.append(dict(call))
            continue
        call_id = str(call.get("id") or call.get("name") or "")
        arguments = dict(call.get("arguments") or {})
        original_slot_id = int(arguments.get("slot_id") or 0)
        selected_slot_id = int(selections.get(call_id) or original_slot_id)
        original = find_business_record(
            state.get("tool_results") or [],
            tool_name="appointment_slot_query",
            collection="slots",
            key="slot_id",
            value=original_slot_id,
        )
        selected = find_business_record(
            state.get("tool_results") or [],
            tool_name="appointment_slot_query",
            collection="slots",
            key="slot_id",
            value=selected_slot_id,
        )
        if not selected:
            raise ValueError("所选预约时段不属于本轮查询结果，请重新查询")
        if original and str(original.get("item_code") or "") != str(selected.get("item_code") or ""):
            raise ValueError("所选预约时段与当前事项不一致，请重新选择")
        if str(selected.get("status") or "") != "available" or int(selected.get("remaining_count") or 0) <= 0:
            raise ValueError("所选预约时段已无剩余号源，请重新查询")
        updated.append({**call, "arguments": {**arguments, "slot_id": selected_slot_id}})
    return updated


def find_business_record(
    results: list[dict[str, Any]],
    *,
    tool_name: str,
    collection: str,
    key: str,
    value: int,
) -> dict[str, Any]:
    """从本轮可信工具结果中查找人工确认所需的业务实体。"""
    for result in reversed(results):
        if str(result.get("tool_name") or "") != tool_name or not result.get("ok"):
            continue
        data = result.get("data") if isinstance(result.get("data"), dict) else {}
        for record in data.get(collection) or []:
            if isinstance(record, dict) and int(record.get(key) or 0) == int(value):
                return dict(record)
    return {}


def appointment_record_details(record: dict[str, Any]) -> list[str]:
    """提取适合人工核对的预约时间和办理地点。"""
    date = str(record.get("service_date") or "").strip()
    start = str(record.get("start_time") or "").strip()
    end = str(record.get("end_time") or "").strip()
    time_text = " ".join(part for part in [date, f"{start}-{end}" if start or end else ""] if part)
    hall = str(record.get("hall_name") or "").strip()
    address = str(record.get("hall_address") or "").strip()
    return [part for part in [time_text, hall, address] if part]


def human_rejected_tool_result(call: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """构造用户拒绝人工确认后的稳定工具结果。"""
    now = datetime.now(timezone.utc).isoformat()
    name = str(call.get("name") or "")
    arguments = dict(call.get("arguments") or {})
    call_id = str(call.get("id") or name)
    return (
        {
            "tool_name": name,
            "call_id": call_id,
            "ok": False,
            "status": "human_rejected",
            "data": {"source_type": "human_confirmation"},
            "error": "用户未批准该写操作",
            "arguments": audit_tool_arguments(name, arguments),
            "started_at": now,
            "completed_at": now,
            "latency_ms": 0,
        },
        tool_call_signature(name, arguments),
    )




def append_tool_message(
    state: dict[str, Any],
    call: dict[str, Any],
    payload: dict[str, Any],
) -> None:
    """按工具类型及其生命周期保存可重放的完整工具观察。"""
    messages = list(state.get("agent_messages") or [])
    messages.append(
        {
            "id": uuid4().hex,
            "role": "tool",
            "tool_call_id": str(call.get("id") or payload.get("call_id") or call.get("name") or ""),
            "name": str(call.get("name") or payload.get("tool_name") or ""),
            "content": dict(payload),
        }
    )
    state["agent_messages"] = messages


def audit_tool_arguments(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """遮盖知识查询中的敏感文本，业务工具参数保持结构化原值。"""
    if name != "knowledge_search":
        return dict(arguments)
    output = dict(arguments)
    queries = []
    for query in output.get("queries") or []:
        if not isinstance(query, dict):
            continue
        queries.append({**query, "text": mask_sensitive_text(str(query.get("text") or ""))})
    output["queries"] = queries
    return output


def audit_tool_call(call: dict[str, Any]) -> dict[str, Any]:
    """返回适合事件和审计记录的工具调用。"""
    name = str(call.get("name") or "")
    payload = {
        "id": str(call.get("id") or name),
        "name": name,
        "arguments": audit_tool_arguments(name, dict(call.get("arguments") or {})),
    }
    if call.get("origin"):
        payload["origin"] = str(call.get("origin"))
    return payload
