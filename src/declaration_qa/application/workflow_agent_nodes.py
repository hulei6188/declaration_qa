"""Agentic RAG 的规划、工具执行、观察和证据门控节点。"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from datetime import datetime, timezone
from time import perf_counter
from typing import Any

from declaration_qa.application.agent import choose_application_id, needs_public_knowledge, rule_knowledge_queries
from declaration_qa.application.state import DeclarationQAState
from declaration_qa.application.tools import ToolExecutionContext, ToolExecutionResult
from declaration_qa.application.tools.knowledge import KnowledgeSearchInput
from declaration_qa.application.workflow_helpers.events import add_event
from declaration_qa.application.workflow_helpers.intent import is_unsafe_query, mask_sensitive_text
from declaration_qa.memory.context import build_short_term_context
from declaration_qa.rag.evidence import decide_evidence
from declaration_qa.rag.engine.text import normalize_text
from declaration_qa.rag.retrieval import KNOWLEDGE_RETRIEVAL_CHAIN, normalize_retrieval_filter
from declaration_qa.rag.router import ROUTE_KNOWLEDGE_RETRIEVAL
from declaration_qa.rag.router import INTENT_REJECTION_REASON, INTENT_TICKET_LIST, INTENT_TICKET_STATUS, classify_ticket_intent


class WorkflowAgentNodes:
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
        memory_manager = getattr(self, "context_memory_manager", None)
        if memory_manager is not None:
            next_state["long_term_context"] = memory_manager.retrieve_long_term_context(next_state)
        next_state["short_term_context"] = build_short_term_context(next_state)
        next_state["agent_iteration"] = 0
        next_state["agent_pending_tool_calls"] = []
        next_state["agent_call_signatures"] = []
        next_state["agent_plan"] = {}
        next_state["agent_query_plan"] = {}
        next_state["agent_final_answer"] = ""
        next_state["agent_final_prompt"] = ""
        next_state["agent_answer_provider"] = ""
        next_state["agent_final_model"] = ""
        next_state["agent_final_token_usage"] = {}
        next_state["agent_final_latency_ms"] = None
        ticket_intent = classify_ticket_intent(query_text)
        if ticket_intent == INTENT_TICKET_LIST:
            required_evidence = ["ticket_list"]
        elif ticket_intent:
            required_evidence = ["ticket_detail"]
            if needs_public_knowledge(query_text):
                required_evidence.append("knowledge")
        else:
            required_evidence = ["knowledge"]
        next_state["agent_required_evidence"] = required_evidence
        next_state["tool_results"] = []
        next_state["agent_selected_item_code"] = ""
        next_state["agent_selected_application_id"] = None
        next_state["route"] = "agentic_rag"
        next_state["retrieval_chain"] = "agentic_rag"
        add_event(next_state, "agent_prepared", {"knowledge_query_frozen": True, "query_planner": "agent_tool_schema"})
        return next_state

    def agent_decide(self, state: DeclarationQAState) -> DeclarationQAState:
        """让 Agent 决定下一批工具调用或结束循环。"""
        next_state = dict(state)
        iteration = int(next_state.get("agent_iteration") or 0) + 1
        next_state["agent_iteration"] = iteration
        if next_state.get("refuse_reason"):
            decision_action = "finish"
            calls = []
            decision_payload = {"action": "finish", "reason": next_state["refuse_reason"], "provider": "guardrail"}
        elif iteration > self.agent_max_iterations:
            decision_action = "finish"
            calls: list[dict[str, Any]] = []
            decision_payload = {"action": "finish", "reason": "max_iterations", "provider": "guardrail"}
        else:
            decision = self.agent.decide(next_state, self.tool_registry.definitions())
            calls = unique_new_tool_calls(
                decision.tool_calls,
                existing_signatures=set(next_state.get("agent_call_signatures") or []),
            )
            decision_action = "tools" if calls else "finish"
            decision_payload = {
                "action": decision_action,
                "reason": decision.reason,
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
                next_state["agent_final_answer"] = decision.final_answer
                next_state["agent_final_prompt"] = decision.final_prompt
                next_state["agent_answer_provider"] = decision.provider
        requirements = set(next_state.get("agent_required_evidence") or [])
        requirements.update(evidence_requirements(calls, intent=str(next_state.get("intent") or "")))
        next_state["agent_required_evidence"] = sorted(requirements)
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
            metadata={"state": state},
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
            if name == "knowledge_search" and result.status == "invalid_arguments":
                fallback_arguments = {"queries": rule_knowledge_queries(state)}
                result = self.tool_registry.execute(
                    name,
                    fallback_arguments,
                    context,
                    call_id=str(call.get("id") or name),
                )
                if result.ok:
                    result = ToolExecutionResult(
                        tool_name=result.tool_name,
                        call_id=result.call_id,
                        ok=True,
                        status=result.status,
                        data={**result.data, "agent_query_schema_fallback": True},
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
                next_state["retrieved_sources"] = list(data.get("sources") or [])
                next_state["rag_status"] = dict(data.get("rag_status") or {})
                next_state["retrieval_filter"] = dict(data.get("retrieval_filter") or {})
                next_state["retrieval_queries"] = list(data.get("executed_queries") or [])
                next_state["agent_query_plan"] = {
                    "requested_queries": list(data.get("requested_queries") or []),
                    "executed_queries": list(data.get("executed_queries") or []),
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

    def agent_evidence_check(self, state: DeclarationQAState) -> DeclarationQAState:
        """分别校验知识证据和业务工具结果，再决定是否允许回答。"""
        next_state = dict(state)
        sources = list(next_state.get("retrieved_sources") or [])
        knowledge_decision = decide_evidence(
            sources,
            next_state.get("retrieval_filter") or {},
            intent=str(next_state.get("intent") or "knowledge_qa"),
            chain=KNOWLEDGE_RETRIEVAL_CHAIN,
        ) if sources else None
        detail_results = [
            result
            for result in (next_state.get("tool_results") or [])
            if isinstance(result, dict) and result.get("tool_name") == "ticket_detail_query" and result.get("ok")
        ]
        list_results = [
            result
            for result in (next_state.get("tool_results") or [])
            if isinstance(result, dict) and result.get("tool_name") == "ticket_list_query" and result.get("ok")
        ]
        list_ok = bool(list_results)
        detail_ok = bool(detail_results)
        business_ok = list_ok if next_state.get("intent") == INTENT_TICKET_LIST else detail_ok
        knowledge_ok = bool(knowledge_decision and knowledge_decision.ok)
        required = set(next_state.get("agent_required_evidence") or [])
        if not required:
            ticket_intent = classify_ticket_intent(str(next_state.get("normalized_query") or next_state.get("query") or ""))
            if ticket_intent == INTENT_TICKET_LIST:
                required.add("ticket_list")
            elif ticket_intent:
                required.add("ticket_detail")
            else:
                required.add("knowledge")
        checks = {
            "knowledge": knowledge_ok,
            "ticket_list": list_ok,
            "ticket_detail": detail_ok,
        }
        next_state["evidence_ok"] = all(checks.get(item, False) for item in required)
        next_state["evidence_decision"] = {
            "ok": next_state["evidence_ok"],
            "business_tool_ok": business_ok,
            "ticket_list_ok": list_ok,
            "ticket_detail_ok": detail_ok,
            "knowledge_ok": knowledge_ok,
            "required_evidence": sorted(required),
            "checks": checks,
            "knowledge": knowledge_decision.payload() if knowledge_decision else {},
            "tool_result_count": len(next_state.get("tool_results") or []),
            "source_count": len(sources),
            "iteration": int(next_state.get("agent_iteration") or 0),
            "isolation": "business_and_knowledge_separate",
        }
        if (
            not next_state["evidence_ok"]
            and (
                next_state.get("refuse_reason")
                or int(next_state.get("agent_iteration") or 0) >= self.agent_max_iterations
            )
        ):
            next_state["refuse_reason"] = next_state.get("refuse_reason") or agent_failure_reason(next_state.get("tool_results") or [])
        add_event(next_state, "evidence", next_state["evidence_decision"])
        return next_state

    def execute_knowledge_search(
        self,
        request: KnowledgeSearchInput,
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        """执行 Agent 查询计划，并强制追加冻结的脱敏原始问题。"""
        state = dict(context.metadata.get("state") or {})
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
                    "aspect": query.aspect,
                    "top_k": 6,
                }
            )
        original_query = normalize_text(mask_sensitive_text(context.original_query))
        if original_query and original_query not in seen:
            retrieval_queries.append(
                {
                    "id": "original_query",
                    "text": original_query,
                    "strategy": "hybrid",
                    "aspect": "original",
                    "top_k": 6,
                }
            )
        isolated_state = {
            **state,
            "query": original_query,
            "normalized_query": original_query,
            "retrieval_query": original_query,
            "frozen_knowledge_query": original_query,
            "route": ROUTE_KNOWLEDGE_RETRIEVAL,
            "business_snapshot": None,
            "tool_results": [],
            "retrieval_chain": KNOWLEDGE_RETRIEVAL_CHAIN,
            "retrieval_queries": retrieval_queries,
        }
        filters = normalize_retrieval_filter(state.get("metadata_filter") or {})
        result = self.multi_retrieve(isolated_state, filters)
        return {
            "sources": [source.payload() for source in result.sources],
            "rag_status": result.status,
            "retrieval_filter": result.filter,
            "executed_queries": retrieval_queries,
        }


def tool_call_signature(name: str, arguments: dict[str, Any]) -> str:
    if name == "knowledge_search":
        return name
    return f"{name}:{json.dumps(arguments, ensure_ascii=False, sort_keys=True, default=str)}"


def unique_new_tool_calls(calls: list[dict[str, Any]], *, existing_signatures: set[str]) -> list[dict[str, Any]]:
    output = []
    seen = set(existing_signatures)
    for call in calls:
        name = str(call.get("name") or "").strip()
        arguments = dict(call.get("arguments") or {})
        signature = tool_call_signature(name, arguments)
        if not name or signature in seen:
            continue
        seen.add(signature)
        output.append({"id": str(call.get("id") or name), "name": name, "arguments": arguments})
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
    """在模型之外强制执行工单工具授权策略。"""
    return not agent_tool_call_denial_reason(tool_name, query)


def agent_tool_call_denial_reason(
    tool_name: str,
    query: str,
    *,
    arguments: dict[str, Any] | None = None,
    prior_results: list[dict[str, Any]] | None = None,
) -> str:
    """返回模型外策略拒绝原因；空字符串表示允许执行。"""
    ticket_intent = classify_ticket_intent(query)
    if tool_name == "knowledge_search" and prior_results is not None:
        observed_business = any(
            isinstance(result, dict)
            and str(result.get("tool_name") or "").startswith("ticket_")
            and result.get("status") != "policy_denied"
            for result in prior_results
        )
        if observed_business:
            return "knowledge_search must be planned before observing business tool results"
        return ""
    if tool_name in {"ticket_list_query", "ticket_detail_query"}:
        if not ticket_intent:
            return "ticket tools require a personal ticket intent"
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


def evidence_requirements(calls: list[dict[str, Any]], *, intent: str) -> set[str]:
    """根据 Agent 的工具计划累计最终回答必须满足的证据类型。"""
    names = {str(call.get("name") or "") for call in calls}
    required: set[str] = set()
    if "knowledge_search" in names:
        required.add("knowledge")
    if intent == INTENT_TICKET_LIST and "ticket_list_query" in names:
        required.add("ticket_list")
    elif intent in {INTENT_TICKET_STATUS, INTENT_REJECTION_REASON} and names.intersection(
        {"ticket_list_query", "ticket_detail_query"}
    ):
        required.add("ticket_detail")
    if "ticket_detail_query" in names:
        required.add("ticket_detail")
    return required


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
    return {
        "id": str(call.get("id") or name),
        "name": name,
        "arguments": audit_tool_arguments(name, dict(call.get("arguments") or {})),
    }
