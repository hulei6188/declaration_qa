"""受控 Agent Tool-Calling 决策器。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from declaration_qa.observability.langfuse import invoke_with_langfuse
from declaration_qa.rag.router import INTENT_REJECTION_REASON, INTENT_TICKET_LIST, INTENT_TICKET_STATUS, classify_knowledge_intent, classify_ticket_intent


AGENT_SYSTEM_PROMPT = """你是政务申报系统的端到端 Agent。你负责选择工具、观察工具结果，并在证据充分后直接回答用户。
必须使用模型原生工具调用。不要输出 Thought 或内部推理过程。

规则：
1. 查询个人申请列表时调用 ticket_list_query；查询进度、状态、退回原因时先调用 ticket_list_query，再从返回事项中自主确定 item_code，调用 ticket_detail_query。
2. ticket_detail_query 会返回同一 item_code 下的全部 applications 和完整 status_history；你应根据用户问题自主识别相关 application_id。
3. 政策、材料、流程、条件、费用、时限、模板和公开规范调用 knowledge_search。你必须通过 queries 数组完成消歧、拆分和改写；简单问题生成一路，复杂问题按原子证据需求生成多路。
4. 混合问题应在第一轮并行调用 knowledge_search 和 ticket_list_query。knowledge_search 的 queries 只能根据用户原始问题、会话上下文和用户上传图片生成，绝不能使用任何工单工具结果。
5. knowledge_search 的 keyword、semantic、hybrid 只是检索偏好。保留材料名称、证明名称、法规文号等精确原词，不得捏造用户未提及的具体事实。
6. user_id、region_code、过滤条件、召回数量和本地图片路径由服务端注入，任何工具参数中都不得生成这些字段。
7. 不要求用户选择工单，不调用未注册工具，不重复相同调用。
8. 工具结果是唯一事实来源：业务工具只证明个人申请事实；knowledge_search 只证明公开政策和材料规范，两者不得互相替代。
9. 对申请进度、状态、退回或补正问题，最终回答必须逐个输出同一事项下全部 application_id，并完整保留每个申请的 status_history，不得只回答选中的一次申请。
10. 证据不足时继续调用必要工具；证据充分时不再调用工具，直接输出清晰、可操作的最终答案。"""


@dataclass(frozen=True)
class AgentDecision:
    """单轮 Agent 决策。"""

    action: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    intent: str = "knowledge_qa"
    reason: str = ""
    provider: str = "rule"
    final_answer: str = ""
    final_prompt: str = ""


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
            "session_messages": list(state.get("session_messages") or [])[-8:],
            "short_term_context": state.get("short_term_context") or {},
            "long_term_context": state.get("long_term_context") or {},
            "query_image_ocr_text": str(state.get("query_image_ocr_text") or "")[:4000],
            "required_evidence": list(state.get("agent_required_evidence") or []),
        }
        prompt_text = json.dumps(prompt, ensure_ascii=False, default=str)
        response = invoke_with_langfuse(
            bound,
            [SystemMessage(content=AGENT_SYSTEM_PROMPT), HumanMessage(content=prompt_text)],
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
        content = normalize_model_content(getattr(response, "content", response))
        return AgentDecision(
            "finish",
            intent=intent,
            reason="model_finished",
            provider="native_tool_calling",
            final_answer=content,
            final_prompt=prompt_text,
        )


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
        if needs_public_knowledge(query) and "knowledge_search" not in completed:
            calls.append(
                {
                    "id": "knowledge",
                    "name": "knowledge_search",
                    "arguments": {"queries": rule_knowledge_queries(state)},
                }
            )
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
            [{"id": "knowledge", "name": "knowledge_search", "arguments": {"queries": rule_knowledge_queries(state)}}],
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
    """限制工具结果体积，同时保留 Agent 最终回答所需证据。"""
    output = []
    for item in results[-8:]:
        if not isinstance(item, dict):
            continue
        data = item.get("data") if isinstance(item.get("data"), dict) else {}
        if item.get("tool_name") == "knowledge_search":
            data = {
                "source_count": len(data.get("sources") or []),
                "query_isolation": data.get("query_isolation") or "",
                "executed_queries": data.get("executed_queries") or [],
                "sources": compact_knowledge_sources(data.get("sources") or []),
            }
        output.append(
            {
                "tool_name": item.get("tool_name"),
                "ok": bool(item.get("ok")),
                "status": item.get("status"),
                "error": item.get("error") or "",
                "data": data,
            }
        )
    return output


def infer_agent_intent(query: str) -> str:
    return classify_ticket_intent(query) or classify_knowledge_intent(query)


def rule_knowledge_queries(state: dict[str, Any]) -> list[dict[str, str]]:
    """无模型时使用冻结问题或图片 OCR 构造单路混合检索。"""
    text = str(
        state.get("normalized_query")
        or state.get("query")
        or state.get("query_image_ocr_text")
        or "用户上传图片内容"
    ).strip()
    return [{"text": text[:300], "strategy": "hybrid", "aspect": "general"}]


def needs_public_knowledge(query: str) -> bool:
    """判断工单问题是否同时明确要求公开办理依据或补正指导。"""
    return any(
        token in str(query or "")
        for token in (
            "材料",
            "证明",
            "委托书",
            "照片",
            "规定",
            "要求",
            "依据",
            "政策",
            "法规",
            "流程",
            "条件",
            "费用",
            "时限",
            "模板",
            "样表",
            "怎么补",
            "如何补",
            "补什么",
        )
    )


def compact_knowledge_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """返回适合 Agent 回答的知识证据，不暴露本地资产路径。"""
    output = []
    for source in sources[:12]:
        if not isinstance(source, dict):
            continue
        metadata = source.get("metadata") if isinstance(source.get("metadata"), dict) else {}
        evidence = str(
            source.get("evidence_text")
            or source.get("content")
            or source.get("snippet")
            or ""
        )[:3500]
        output.append(
            {
                "chunk_id": source.get("chunk_id") or "",
                "title": source.get("title") or "",
                "evidence": evidence,
                "score": source.get("rerank_score") or source.get("score") or 0,
                "source_url": source.get("source_url") or metadata.get("source_url") or "",
                "page_no": metadata.get("attachment_page_no") or metadata.get("page_no"),
                "chunk_type": metadata.get("chunk_type") or metadata.get("semantic_type") or "",
            }
        )
    return output


def normalize_model_content(value: Any) -> str:
    """兼容文本和多内容块模型响应。"""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts).strip()
    return str(value or "").strip()
