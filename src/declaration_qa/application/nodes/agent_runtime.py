"""单 Agent 的规划、工具执行和观察节点。"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from datetime import datetime, timezone
from time import perf_counter
from typing import Any
from uuid import uuid4

from declaration_qa.application.agent import accumulated_knowledge_queries, choose_application_id
from declaration_qa.application.state import DeclarationQAState
from declaration_qa.application.tools import ToolExecutionContext, ToolExecutionResult
from declaration_qa.application.tools.knowledge import KnowledgeSearchInput
from declaration_qa.application.helpers.events import add_event
from declaration_qa.application.helpers.retrieval_sources import merge_accumulated_source_payloads
from declaration_qa.application.helpers.text_safety import is_unsafe_query, mask_sensitive_text
from declaration_qa.memory.context import build_short_term_context
from declaration_qa.rag.engine.text import normalize_text
from declaration_qa.rag.retrieval import KNOWLEDGE_RETRIEVAL_CHAIN, normalize_retrieval_filter
from declaration_qa.rag.router import ROUTE_KNOWLEDGE_RETRIEVAL


class AgentRuntimeNodes:
    """有界 ReAct 循环使用的 LangGraph 节点。"""

    def agent_prepare(self, state: DeclarationQAState) -> DeclarationQAState:
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
        next_state["short_term_context"] = build_short_term_context(next_state)
        next_state["agent_iteration"] = 0
        next_state["agent_pending_tool_calls"] = []
        next_state["agent_call_signatures"] = []
        messages = list(next_state.get("agent_messages") or [])
        messages.append(
            {
                "id": uuid4().hex,
                "role": "user",
                "content": str(next_state.get("query") or ""),
            }
        )
        next_state["agent_messages"] = messages
        next_state["answer_mode"] = ""
        next_state["agent_plan"] = {}
        next_state["agent_query_plan"] = {}
        next_state["agent_final_answer"] = ""
        next_state["agent_final_prompt"] = ""
        next_state["agent_answer_provider"] = ""
        next_state["agent_final_model"] = ""
        next_state["agent_final_token_usage"] = {}
        next_state["agent_final_latency_ms"] = None
        next_state["tool_results"] = []
        next_state["agent_selected_item_code"] = ""
        next_state["agent_selected_application_id"] = None
        next_state["route"] = "agentic_rag"
        next_state["retrieval_chain"] = "agentic_rag"
        add_event(next_state, "agent_prepared", {"knowledge_query_frozen": True, "query_planner": "agent_tool_schema"})
        return next_state

    def agent_summarize(self, state: DeclarationQAState) -> DeclarationQAState:
        """Build the model input from the running summary and recent complete messages."""
        next_state = dict(state)
        summarized, memory_context = self.memory_runtime.summarize(
            list(next_state.get("agent_messages") or []),
            dict(next_state.get("memory_context") or {}),
        )
        next_state["agent_input_messages"] = summarized
        next_state["memory_context"] = memory_context
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

    def agent_decide(self, state: DeclarationQAState) -> DeclarationQAState:
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

    def agent_execute_tools(self, state: DeclarationQAState) -> DeclarationQAState:
        """并发执行独立工具，并在列表结果就绪后执行依赖详情。"""
        next_state = dict(state)
        results = list(next_state.get("tool_results") or [])
        signatures = list(next_state.get("agent_call_signatures") or [])
        pending_calls = list(next_state.get("agent_pending_tool_calls") or [])
        batch_has_list = any(str(call.get("name") or "") == "ticket_list_query" for call in pending_calls)
        has_prior_list = bool(listed_item_codes(results))
        deferred = [
            call
            for call in pending_calls
            if str(call.get("name") or "") == "ticket_detail_query" and batch_has_list and not has_prior_list
        ]
        primary = [call for call in pending_calls if call not in deferred]

        for call, payload, signature in self._execute_tool_wave(primary, next_state, results):
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
                    "parallel": len(primary) > 1,
                },
            )
        for call, payload, signature in self._execute_tool_wave(deferred, next_state, results):
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
                    "parallel": len(deferred) > 1,
                    "dependency": "ticket_list_query",
                },
            )
        next_state["tool_results"] = results
        next_state["agent_call_signatures"] = signatures
        next_state["agent_pending_tool_calls"] = []
        return next_state

    def _execute_tool_wave(
        self,
        calls: list[dict[str, Any]],
        state: DeclarationQAState,
        prior_results: list[dict[str, Any]],
    ) -> list[tuple[dict[str, Any], dict[str, Any], str]]:
        """并发执行一组彼此独立的工具调用，并保持 Agent 给出的顺序。"""
        if not calls:
            return []
        snapshot = list(prior_results)
        if len(calls) == 1:
            payload, signature = self._execute_agent_tool_call(calls[0], state, snapshot)
            return [(calls[0], payload, signature)]
        with ThreadPoolExecutor(max_workers=min(4, len(calls)), thread_name_prefix="declaration-qa-tool") as executor:
            futures = [
                executor.submit(copy_context().run, self._execute_agent_tool_call, call, state, snapshot)
                for call in calls
            ]
            outputs = [future.result() for future in futures]
        return [(call, payload, signature) for call, (payload, signature) in zip(calls, outputs)]

    def _execute_agent_tool_call(
        self,
        call: dict[str, Any],
        state: DeclarationQAState,
        prior_results: list[dict[str, Any]],
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
            },
        )
        denial_reason = agent_tool_call_denial_reason(
            name,
            str(state.get("normalized_query") or state.get("query") or ""),
            arguments=arguments,
            prior_results=prior_results,
        )
        if denial_reason:
            result = ToolExecutionResult(
                tool_name=name,
                call_id=str(call.get("id") or name),
                ok=False,
                status="policy_denied",
                error=denial_reason,
            )
        else:
            result = self.tool_registry.execute(
                name,
                arguments,
                context,
                call_id=str(call.get("id") or name),
            )
        completed_at = datetime.now(timezone.utc)
        return {
            **result.payload(),
            "arguments": audit_tool_arguments(name, arguments),
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "latency_ms": max(0, round((perf_counter() - started_clock) * 1000)),
        }, signature

    def agent_observe(self, state: DeclarationQAState) -> DeclarationQAState:
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
                next_state["agent_selected_item_code"] = str(data.get("item_code") or "")
                next_state["agent_selected_application_id"] = choose_application_id(
                    str(next_state.get("normalized_query") or next_state.get("query") or ""),
                    applications,
                )
                next_state["business_snapshot"] = {
                    "source": "ticket_detail_query",
                    "item_code": data.get("item_code") or "",
                    "selected_application_id": next_state.get("agent_selected_application_id"),
                    "applications": applications,
                }
                next_state["answer_mode"] = "tool_grounded"
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
    for call in coalesce_knowledge_search_calls(calls):
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
    prior_results: list[dict[str, Any]] | None = None,
) -> str:
    """返回模型外策略拒绝原因；空字符串表示允许执行。"""
    if tool_name == "knowledge_search":
        return ""
    if tool_name in {"ticket_list_query", "ticket_detail_query"}:
        if tool_name == "ticket_detail_query" and prior_results is not None:
            listed_codes = listed_item_codes(prior_results)
            if not listed_codes:
                return "ticket_detail_query requires a successful ticket_list_query observation"
            requested_code = str((arguments or {}).get("item_code") or "").strip()
            if requested_code not in listed_codes:
                return "item_code was not returned by ticket_list_query for the authenticated user"
        return ""
    return ""


def listed_item_codes(results: list[dict[str, Any]]) -> set[str]:
    """读取当前 Agent 已成功观察到的用户事项编码。"""
    codes: set[str] = set()
    for result in results:
        if not isinstance(result, dict) or result.get("tool_name") != "ticket_list_query" or not result.get("ok"):
            continue
        data = result.get("data") if isinstance(result.get("data"), dict) else {}
        for item in data.get("items") or []:
            if isinstance(item, dict) and str(item.get("item_code") or "").strip():
                codes.add(str(item["item_code"]).strip())
    return codes


def append_tool_message(
    state: dict[str, Any],
    call: dict[str, Any],
    payload: dict[str, Any],
) -> None:
    """按 tool_call_id 保存可重放的工具观察，知识正文由证据槽单独提供。"""
    observation = dict(payload)
    messages = list(state.get("agent_messages") or [])
    messages.append(
        {
            "id": uuid4().hex,
            "role": "tool",
            "tool_call_id": str(call.get("id") or payload.get("call_id") or call.get("name") or ""),
            "name": str(call.get("name") or payload.get("tool_name") or ""),
            "content": observation,
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
