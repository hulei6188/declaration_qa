"""受控 Agent Tool-Calling 决策器。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from declaration_qa.observability.langfuse import invoke_with_langfuse
from declaration_qa.rag.router import INTENT_REJECTION_REASON, INTENT_TICKET_LIST, INTENT_TICKET_STATUS, classify_knowledge_intent, classify_ticket_intent


AGENT_SYSTEM_PROMPT = """你是政务申报系统的工具调度 Agent，只负责选择工具，不直接回答用户。
必须使用模型原生工具调用，不要输出 Thought、推理过程或最终业务答案。

规则：
1. 查询个人申请列表时调用 ticket_list_query；查询进度、状态、退回原因时先调用 ticket_list_query，再从返回事项中自主确定 item_code，调用 ticket_detail_query。
2. ticket_detail_query 会返回同一 item_code 下的全部 applications 和完整 status_history；你应根据用户问题自主识别相关 application_id。
3. 政策、材料、流程、条件、费用、时限、模板：调用 knowledge_search。
4. knowledge_search 可与工单工具并行调用，不设置意图或关键词限制；是否调用由你根据问题自主决定。
5. user_id 由服务端注入，任何工具参数中都不得生成 user_id。
6. 不要求用户选择工单，不调用未注册工具，不重复相同调用。
7. 工单结果不得写入 knowledge_search 的 query_focus；knowledge_search 只能围绕用户原始问题。
8. 已有充分工具结果时停止调用工具，返回简短的完成标记即可。"""


@dataclass(frozen=True)
class AgentDecision:
    """单轮 Agent 决策。"""

    action: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    intent: str = "knowledge_qa"
    reason: str = ""
    provider: str = "rule"


class ToolCallingAgentPlanner:
    """优先使用模型原生 Tool Calling，失败时使用规则兜底。"""

    def __init__(self, model: Any | None = None) -> None:
        self.model = model

    def decide(self, state: dict[str, Any], tool_definitions: list[dict[str, Any]]) -> AgentDecision:
        if self.model is not None:
            try:
                return self._model_decision(state, tool_definitions)
            except Exception:
                pass
        return rule_agent_decision(state)

    def _model_decision(self, state: dict[str, Any], tool_definitions: list[dict[str, Any]]) -> AgentDecision:
        from langchain_core.messages import HumanMessage, SystemMessage

        bound = self.model.bind_tools(tool_definitions)
        observations = compact_tool_observations(state.get("tool_results") or [])
        prompt = {
            "query": state.get("normalized_query") or state.get("query") or "",
            "iteration": int(state.get("agent_iteration") or 0),
            "observations": observations,
            "selected_item_code": state.get("agent_selected_item_code") or "",
            "selected_application_id": state.get("agent_selected_application_id"),
            "knowledge_evidence_count": len(state.get("retrieved_sources") or []),
        }
        response = invoke_with_langfuse(
            bound,
            [SystemMessage(content=AGENT_SYSTEM_PROMPT), HumanMessage(content=json.dumps(prompt, ensure_ascii=False))],
            state,
            run_name="agent.decide",
        )
        raw_calls = list(getattr(response, "tool_calls", None) or [])
        calls = []
        for index, item in enumerate(raw_calls, start=1):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            calls.append(
                {
                    "id": str(item.get("id") or f"call_{index}"),
                    "name": name,
                    "arguments": dict(item.get("args") or {}),
                }
            )
        intent = infer_agent_intent(str(state.get("normalized_query") or state.get("query") or ""))
        if calls:
            return AgentDecision("tools", calls, intent=intent, reason="model_tool_calls", provider="native_tool_calling")
        if not state.get("tool_results") and not state.get("retrieved_sources"):
            return rule_agent_decision(state)
        return AgentDecision("finish", intent=intent, reason="model_finished", provider="native_tool_calling")


def rule_agent_decision(state: dict[str, Any]) -> AgentDecision:
    """无模型或模型失败时的确定性工具调度。"""
    query = str(state.get("normalized_query") or state.get("query") or "")
    ticket_intent = classify_ticket_intent(query)
    intent = ticket_intent or classify_knowledge_intent(query)
    results = [item for item in (state.get("tool_results") or []) if isinstance(item, dict)]
    completed = {str(item.get("tool_name") or "") for item in results if item.get("ok")}
    calls: list[dict[str, Any]] = []
    if ticket_intent and "ticket_list_query" not in completed:
        calls.append({"id": "ticket_list", "name": "ticket_list_query", "arguments": {}})
        if "knowledge_search" not in completed:
            calls.append({"id": "knowledge", "name": "knowledge_search", "arguments": {"query_focus": query}})
        return AgentDecision("tools", calls, intent=intent, reason="ticket_intent_requires_item_resolution")
    if ticket_intent == INTENT_TICKET_LIST:
        return AgentDecision("finish", intent=intent, reason="ticket_list_is_final")
    if ticket_intent and "ticket_detail_query" not in completed:
        item_code = choose_item_code(query, latest_tool_data(results, "ticket_list_query").get("items") or [])
        if item_code:
            detail_type = "status" if ticket_intent == INTENT_TICKET_STATUS else "rejection"
            calls.append(
                {
                    "id": "ticket_detail",
                    "name": "ticket_detail_query",
                    "arguments": {"item_code": item_code, "detail_type": detail_type},
                }
            )
            return AgentDecision("tools", calls, intent=intent, reason="agent_selected_item_code")
    if not ticket_intent and "knowledge_search" not in completed:
        return AgentDecision(
            "tools",
            [{"id": "knowledge", "name": "knowledge_search", "arguments": {"query_focus": query}}],
            intent=intent,
            reason="knowledge_intent",
        )
    return AgentDecision("finish", intent=intent, reason="available_evidence_is_final")


def choose_item_code(query: str, items: list[dict[str, Any]]) -> str:
    """规则兜底只做语义词面匹配；主路径由模型自主选择。"""
    candidates = [item for item in items if isinstance(item, dict) and str(item.get("item_code") or "")]
    if not candidates:
        return ""
    scored = []
    for item in candidates:
        name = str(item.get("item_name") or "")
        compact_name = name.replace("办理", "").replace("登记", "")
        score = 0
        if name and name in query:
            score += 100
        score += sum(1 for token in compact_name if token.strip() and token in query)
        scored.append((score, str(item.get("updated_at") or ""), str(item.get("item_code") or "")))
    scored.sort(reverse=True)
    if scored[0][0] > 0 or len(scored) == 1:
        return scored[0][2]
    return ""


def choose_application_id(query: str, applications: list[dict[str, Any]]) -> int | None:
    """根据问题和完整状态历史选择相关申请，供回答和审计使用。"""
    candidates = [item for item in applications if isinstance(item, dict) and int(item.get("application_id") or 0) > 0]
    if not candidates:
        return None
    ordered = sorted(candidates, key=lambda item: str(item.get("updated_at") or item.get("submitted_at") or ""), reverse=True)
    years = set(re.findall(r"20\d{2}", query))
    if years:
        matched = [
            item
            for item in ordered
            if any(year in str(item.get("submitted_at") or item.get("updated_at") or "") for year in years)
        ]
        if matched:
            return int(matched[0]["application_id"])
    if any(token in query for token in ("最早", "第一次", "首笔")):
        return int(ordered[-1]["application_id"])
    if any(token in query for token in ("上一次", "上一笔", "之前那次", "前一次")) and len(ordered) > 1:
        return int(ordered[1]["application_id"])
    rejection_query = bool(classify_ticket_intent(query) == INTENT_REJECTION_REASON)
    scored = []
    for item in candidates:
        history = [event for event in (item.get("status_history") or []) if isinstance(event, dict)]
        has_rejection = any(
            event.get("reason")
            or str(event.get("status") or "").lower() in {"rejected", "returned", "correction"}
            for event in history
        )
        status = str(item.get("current_status") or "").lower()
        active = status not in {"completed", "cancelled", "withdrawn", "已办结", "已撤回"}
        semantic_score = 100 if rejection_query and has_rejection else 50 if not rejection_query and active else 0
        scored.append((semantic_score, str(item.get("updated_at") or ""), int(item["application_id"])))
    scored.sort(reverse=True)
    return scored[0][2]


def latest_tool_data(results: list[dict[str, Any]], tool_name: str) -> dict[str, Any]:
    for result in reversed(results):
        if result.get("tool_name") == tool_name and isinstance(result.get("data"), dict):
            return result["data"]
    return {}


def compact_tool_observations(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """限制传回规划模型的工具结果体积。"""
    output = []
    for item in results[-8:]:
        if not isinstance(item, dict):
            continue
        data = item.get("data") if isinstance(item.get("data"), dict) else {}
        if item.get("tool_name") == "knowledge_search":
            data = {
                "source_count": len(data.get("sources") or []),
                "query_isolation": data.get("query_isolation") or "",
            }
        output.append(
            {
                "tool_name": item.get("tool_name"),
                "ok": bool(item.get("ok")),
                "status": item.get("status"),
                "data": data,
            }
        )
    return output


def infer_agent_intent(query: str) -> str:
    return classify_ticket_intent(query) or classify_knowledge_intent(query)
