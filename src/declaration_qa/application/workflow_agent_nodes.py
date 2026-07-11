"""Agentic RAG 的规划、工具执行、观察和证据门控节点。"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from typing import Any

from declaration_qa.application.agent_planner import choose_application_id
from declaration_qa.application.state import DeclarationQAState
from declaration_qa.application.tools import ToolExecutionContext, ToolExecutionResult
from declaration_qa.application.workflow_helpers.events import add_event
from declaration_qa.application.workflow_helpers.intent import is_unsafe_query
from declaration_qa.rag.evidence import decide_evidence
from declaration_qa.rag.retrieval import KNOWLEDGE_RETRIEVAL_CHAIN, normalize_retrieval_filter
from declaration_qa.rag.router import ROUTE_KNOWLEDGE_RETRIEVAL
from declaration_qa.rag.router import INTENT_REJECTION_REASON, INTENT_TICKET_LIST, INTENT_TICKET_STATUS, classify_ticket_intent


class WorkflowAgentNodes:
    """有界 ReAct 循环使用的 LangGraph 节点。"""

    def agent_prepare(self, state: DeclarationQAState) -> DeclarationQAState:
        """冻结知识检索输入，并初始化 Agent 运行状态。"""
        next_state = dict(state)
        query_text = str(next_state.get("normalized_query") or "")
        if is_unsafe_query(query_text):
            next_state["refuse_reason"] = "unsafe_or_out_of_scope"
        elif not query_text and not next_state.get("query_image"):
            next_state["refuse_reason"] = "clarification_required"
        sanitized = {**next_state, "business_snapshot": None, "tool_results": []}
        planned_queries: list[dict[str, Any]] = []
        stage1_payload: dict[str, Any] = {}
        if str(next_state.get("normalized_query") or "").strip() and not next_state.get("refuse_reason"):
            resolution = self.query_rewrite_service.resolve(sanitized)
            stage1_payload = resolution.payload()
            for question in resolution.questions:
                for query in question.get("retrieval_queries") or []:
                    if isinstance(query, dict) and str(query.get("text") or "").strip():
                        planned_queries.append(dict(query))
        next_state["query_rewrite.stage1"] = stage1_payload
        next_state["agent_knowledge_query"] = str(next_state.get("normalized_query") or next_state.get("query") or "")
        next_state["agent_knowledge_queries"] = planned_queries
        next_state["agent_iteration"] = 0
        next_state["agent_pending_tool_calls"] = []
        next_state["agent_call_signatures"] = []
        next_state["agent_plan"] = {}
        next_state["tool_results"] = []
        next_state["agent_selected_item_code"] = ""
        next_state["agent_selected_application_id"] = None
        next_state["route"] = "agentic_rag"
        next_state["retrieval_chain"] = "agentic_rag"
        add_event(next_state, "agent_prepared", {"knowledge_query_frozen": True, "query_count": len(planned_queries)})
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
            decision = self.agent_planner.decide(next_state, self.tool_registry.definitions())
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
                "tool_calls": calls,
                "iteration": iteration,
            }
            next_state["intent"] = decision.intent
            next_state["sub_intent"] = decision.intent
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
            if payload.get("status") != "policy_denied":
                signatures.append(signature)
            add_event(
                next_state,
                "tool_executed",
                {
                    "tool_name": call.get("name") or "",
                    "call_id": payload["call_id"],
                    "ok": payload["ok"],
                    "status": payload["status"],
                    "parallel": len(primary) > 1,
                },
            )
        for call, payload, signature in self._execute_tool_wave(deferred, next_state, results):
            results.append(payload)
            if payload.get("status") != "policy_denied":
                signatures.append(signature)
            add_event(
                next_state,
                "tool_executed",
                {
                    "tool_name": call.get("name") or "",
                    "call_id": payload["call_id"],
                    "ok": payload["ok"],
                    "status": payload["status"],
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
        name = str(call.get("name") or "")
        arguments = dict(call.get("arguments") or {})
        signature = tool_call_signature(name, arguments)
        context = ToolExecutionContext(
            user_id=int(state.get("user_id") or 0),
            region_code=str(state.get("region_code") or ""),
            session_id=str(state.get("session_id") or ""),
            original_query=str(state.get("agent_knowledge_query") or state.get("normalized_query") or ""),
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
        return {**result.payload(), "arguments": arguments}, signature

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
        business_ok = bool(list_results) if next_state.get("intent") == INTENT_TICKET_LIST else bool(detail_results)
        knowledge_ok = bool(knowledge_decision and knowledge_decision.ok)
        ticket_intent = classify_ticket_intent(str(next_state.get("normalized_query") or next_state.get("query") or ""))
        ticket_intent = ticket_intent or (
            str(next_state.get("intent") or "")
            if next_state.get("intent") in {INTENT_TICKET_LIST, INTENT_TICKET_STATUS, INTENT_REJECTION_REASON}
            else ""
        )
        next_state["evidence_ok"] = business_ok if ticket_intent else knowledge_ok
        next_state["evidence_decision"] = {
            "ok": next_state["evidence_ok"],
            "business_tool_ok": business_ok,
            "knowledge_ok": knowledge_ok,
            "required_evidence": "business_tool" if ticket_intent else "knowledge",
            "knowledge": knowledge_decision.payload() if knowledge_decision else {},
            "tool_result_count": len(next_state.get("tool_results") or []),
            "source_count": len(sources),
            "isolation": "business_and_knowledge_separate",
        }
        if not next_state["evidence_ok"]:
            next_state["refuse_reason"] = next_state.get("refuse_reason") or agent_failure_reason(next_state.get("tool_results") or [])
        add_event(next_state, "evidence", next_state["evidence_decision"])
        return next_state

    def execute_knowledge_search(self, context: ToolExecutionContext) -> dict[str, Any]:
        """使用冻结状态调用现有混合检索，不读取任何业务工具结果。"""
        state = dict(context.metadata.get("state") or {})
        isolated_state = {
            **state,
            "query": context.original_query,
            "normalized_query": context.original_query,
            "retrieval_query": context.original_query,
            "route": ROUTE_KNOWLEDGE_RETRIEVAL,
            "business_snapshot": None,
            "tool_results": [],
            "retrieval_chain": KNOWLEDGE_RETRIEVAL_CHAIN,
            "retrieval_queries": list(state.get("agent_knowledge_queries") or []),
        }
        filters = normalize_retrieval_filter(state.get("metadata_filter") or {})
        result = self.multi_retrieve(isolated_state, filters)
        return {
            "sources": [source.payload() for source in result.sources],
            "rag_status": result.status,
            "retrieval_filter": result.filter,
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
