"""端到端 Tool-Calling Agent 及其确定性兜底策略。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

from gov_service_agent.application.evidence_registry import build_evidence_registry
from gov_service_agent.application.prompts import build_agent_system_prompt
from gov_service_agent.observability.langfuse import invoke_with_langfuse
from gov_service_agent.observability.performance import log_performance, performance_stage
from gov_service_agent.rag.model_providers.answers import (
    MULTIMODAL_EVIDENCE_RULES,
    image_content_block,
    select_evidence_images,
    source_metadata,
    valid_query_image,
    visual_evidence_label,
)
from gov_service_agent.rag.prompting import format_source_payload, format_tool_result_payload, knowledge_answer


@dataclass(frozen=True)
class AgentDecision:
    """单轮 Agent 决策。"""

    action: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    intent: str = "domain_agent"
    reason: str = ""
    provider: str = "rule"
    final_answer: str = ""
    final_prompt: str = ""
    assistant_content: str = ""
    model: str = ""
    token_usage: dict[str, int] = field(default_factory=dict)
    latency_ms: int | None = None


class ToolCallingAgent:
    """使用同一模型循环规划工具并直接生成最终答案。"""

    def __init__(
        self,
        model: Any | None = None,
        *,
        multimodal_model: Any | None = None,
        multimodal_enabled: bool = False,
        allowed_image_dirs: list[str] | None = None,
        max_evidence_images: int = 3,
        min_image_score: float = 0.7,
        max_image_bytes: int = 10 * 1024 * 1024,
    ) -> None:
        self.model = model
        self.multimodal_enabled = bool(multimodal_enabled)
        self.multimodal_model = multimodal_model or (model if self.multimodal_enabled else None)
        self.allowed_image_dirs = [
            Path(item).resolve()
            for item in (allowed_image_dirs or [])
            if str(item or "").strip()
        ]
        self.max_evidence_images = max(0, int(max_evidence_images or 0))
        self.min_image_score = max(0.0, float(min_image_score or 0.0))
        self.max_image_bytes = max(1, int(max_image_bytes or 1))

    def decide(self, state: dict[str, Any], tool_definitions: list[dict[str, Any]]) -> AgentDecision:
        if self.model is not None:
            try:
                return self._model_decision(state, tool_definitions)
            except Exception as exc:
                return rule_agent_decision(state, model_error=exc)
        return rule_agent_decision(state)

    def _model_decision(self, state: dict[str, Any], tool_definitions: list[dict[str, Any]]) -> AgentDecision:
        from langchain_core.messages import SystemMessage

        use_multimodal_model = bool(state.get("query_image")) and self.multimodal_model is not None
        active_model = self.multimodal_model if use_multimodal_model else self.model
        bound = active_model.bind_tools(tool_definitions) if tool_definitions else active_model
        evidence_registry = build_evidence_registry(
            state.get("retrieved_sources") or [],
            state.get("tool_results") or [],
        )
        prompt = {
            "iteration": int(state.get("agent_iteration") or 0),
            "route_decision": state.get("route_decision") or {},
            "agent_tool_scope": list(state.get("agent_tool_scope") or []),
            "long_term_context": state.get("long_term_context") or {},
            "conversation_memory": state.get("conversation_memory") or {},
            "working_memory": state.get("working_memory") or {},
            "dialog_task": state.get("dialog_task") or {},
            "query_image_ocr_text": str(state.get("query_image_ocr_text") or "")[:4000],
            "evidence_registry": evidence_registry,
            "trusted_business_evidence": format_tool_result_payload(state.get("tool_results") or []),
            "trusted_knowledge_evidence": format_source_payload(state.get("retrieved_sources") or []),
            "citation_requirement": "所有外部事实必须逐行标注[证据N]或[业务N]。",
        }
        prompt_text = json.dumps(prompt, ensure_ascii=False, default=str)
        messages: list[Any] = [SystemMessage(content=build_agent_system_prompt(state))]
        messages.extend(list(state.get("agent_input_messages") or []))
        messages.append(
            build_agent_human_message(
                self,
                state,
                prompt_text,
                include_images=use_multimodal_model,
            )
        )
        started_at = perf_counter()
        with performance_stage(
            "agent.model_call",
            iteration=int(state.get("agent_iteration") or 0),
            model=str(getattr(active_model, "model_name", "") or getattr(active_model, "model", "")),
            tool_definition_count=len(tool_definitions),
        ):
            response = invoke_with_langfuse(
                bound,
                messages,
                state,
                run_name="agent.decide",
            )
        latency_ms = max(0, round((perf_counter() - started_at) * 1000))
        model_name, usage = response_model_metadata(response, active_model)
        content = normalize_model_content(getattr(response, "content", response))
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
        intent = "domain_agent"
        if calls:
            log_performance(
                "agent.decision",
                started_at,
                action="tools",
                iteration=int(state.get("agent_iteration") or 0),
                tool_call_count=len(calls),
                input_tokens=usage.get("input_tokens"),
                output_tokens=usage.get("output_tokens"),
            )
            return AgentDecision(
                "tools",
                calls,
                intent=intent,
                reason="model_tool_calls",
                provider="native_tool_calling",
                assistant_content=content,
                model=model_name,
                token_usage=usage,
                latency_ms=latency_ms,
            )
        log_performance(
            "agent.decision",
            started_at,
            action="finish",
            iteration=int(state.get("agent_iteration") or 0),
            tool_call_count=0,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
        )
        return AgentDecision(
            "finish",
            intent=intent,
            reason="model_finished",
            provider="native_tool_calling",
            final_answer=content,
            final_prompt=prompt_text,
            assistant_content=content,
            model=model_name,
            token_usage=usage,
            latency_ms=latency_ms,
        )


def rule_agent_decision(state: dict[str, Any], model_error: Exception | None = None) -> AgentDecision:
    """模型不可用时停止自主调度；已有证据时用确定性答案兜底。"""
    reason = (
        f"agent_model_error:{type(model_error).__name__}"
        if model_error is not None
        else "agent_model_unavailable"
    )
    if model_error is not None:
        answer = "已检索到相关依据，但回答生成服务暂时不可用，请稍后重试。"
        return AgentDecision(
            "finish",
            intent="domain_agent",
            reason=reason,
            final_answer=answer,
            assistant_content=answer,
        )
    sources = select_generation_sources(state.get("retrieved_sources") or [])
    if sources:
        answer = knowledge_answer(state, sources)
        return AgentDecision(
            "finish",
            intent="domain_agent",
            reason=reason,
            final_answer=answer,
            assistant_content=answer,
        )
    return AgentDecision(
        "finish",
        intent="domain_agent",
        reason=reason,
    )


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
    scored = []
    for item in candidates:
        status = str(item.get("current_status") or "").lower()
        active = status not in {"completed", "cancelled", "withdrawn", "已办结", "已撤回"}
        semantic_score = 50 if active else 0
        scored.append((semantic_score, str(item.get("updated_at") or ""), int(item["application_id"])))
    scored.sort(reverse=True)
    return scored[0][2]


def build_agent_human_message(
    agent: ToolCallingAgent,
    state: dict[str, Any],
    text: str,
    *,
    include_images: bool = True,
) -> Any:
    """把用户图片和召回图片直接加入当前 Agent 轮。"""
    from langchain_core.messages import HumanMessage

    use_images = include_images and agent.multimodal_enabled and bool(agent.allowed_image_dirs)
    if not use_images:
        return HumanMessage(content=text)
    query_image = valid_query_image(
        state.get("query_image"),
        allowed_dirs=agent.allowed_image_dirs,
        max_bytes=agent.max_image_bytes,
    )
    evidence_images = select_evidence_images(
        state,
        limit=agent.max_evidence_images,
        min_score=agent.min_image_score,
        allowed_dirs=agent.allowed_image_dirs,
        max_bytes=agent.max_image_bytes,
    )
    if not query_image and not evidence_images:
        return HumanMessage(content=text)
    content: list[dict[str, Any]] = [
        {"type": "text", "text": f"{text}\n\n{MULTIMODAL_EVIDENCE_RULES}"}
    ]
    if query_image:
        content.extend(
            [
                {"type": "text", "text": "【用户查询图片】仅用于理解用户问题。"},
                image_content_block(query_image["path"], query_image.get("content_type") or ""),
            ]
        )
    for index, item in enumerate(evidence_images, start=1):
        source = item["source"]
        metadata = source_metadata(source)
        content.extend(
            [
                {"type": "text", "text": visual_evidence_label(source, metadata, index=index)},
                image_content_block(item["path"], metadata.get("asset_content_type") or ""),
            ]
        )
    return HumanMessage(content=content)


def accumulated_knowledge_queries(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按首次出现顺序汇总历次知识检索实际执行的查询。"""
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for result in results:
        if not isinstance(result, dict) or result.get("tool_name") != "knowledge_search" or not result.get("ok"):
            continue
        data = result.get("data") if isinstance(result.get("data"), dict) else {}
        for query in data.get("executed_queries") or []:
            if not isinstance(query, dict):
                continue
            text = " ".join(str(query.get("text") or "").split())
            if not text or text in seen:
                continue
            seen.add(text)
            output.append(
                {
                    "text": text,
                    "strategy": str(query.get("strategy") or "hybrid"),
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


def select_generation_sources(
    sources: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """按重排顺序去重最终生成证据，不设置第二个数量上限。"""
    normalized = [source for source in sources if isinstance(source, dict)]
    if not normalized:
        return []
    selected: list[dict[str, Any]] = []
    seen_chunk_ids: set[str] = set()
    seen_texts: set[str] = set()
    seen_table_nodes: set[str] = set()
    for source in normalized:
        chunk_id = str(source.get("chunk_id") or "").strip()
        evidence = source_evidence_for_dedup(source)
        text_key = re.sub(r"\s+", "", evidence).lower()
        if chunk_id and chunk_id in seen_chunk_ids:
            continue
        if text_key and text_key in seen_texts:
            continue
        if source_chunk_type(source) == "table_kv":
            table_key = table_node_key(source)
            if table_key and table_key in seen_table_nodes:
                continue
            if table_key:
                seen_table_nodes.add(table_key)
        if chunk_id:
            seen_chunk_ids.add(chunk_id)
        if text_key:
            seen_texts.add(text_key)
        selected.append(source)
    return selected


def source_chunk_type(source: dict[str, Any]) -> str:
    metadata = source.get("metadata") if isinstance(source.get("metadata"), dict) else {}
    return str(metadata.get("chunk_type") or metadata.get("semantic_type") or "")


def source_evidence_for_dedup(source: dict[str, Any]) -> str:
    return str(source.get("evidence_text") or source.get("content") or source.get("snippet") or "")


def table_node_key(source: dict[str, Any]) -> str:
    metadata = source.get("metadata") if isinstance(source.get("metadata"), dict) else {}
    node_title = str(metadata.get("node_title") or "").strip()
    if not node_title:
        return ""
    return "|".join(
        (
            str(source.get("doc_id") or metadata.get("document_id") or source.get("title") or ""),
            str(metadata.get("section_title") or ""),
            node_title,
        )
    )


def response_model_metadata(response: Any, model: Any) -> tuple[str, dict[str, int]]:
    """从模型响应中读取非敏感的模型标识和令牌用量。"""
    response_metadata = getattr(response, "response_metadata", None) or {}
    usage = normalize_token_usage(
        getattr(response, "usage_metadata", None)
        or (response_metadata.get("token_usage") if isinstance(response_metadata, dict) else None)
        or {}
    )
    model_name = str(
        (response_metadata.get("model_name") if isinstance(response_metadata, dict) else "")
        or getattr(model, "model_name", None)
        or getattr(model, "model", None)
        or ""
    )
    return model_name, usage


def normalize_token_usage(value: Any) -> dict[str, int]:
    """统一 LangChain 和 OpenAI 兼容接口的用量元数据。"""
    payload = value if isinstance(value, dict) else {}
    aliases = {
        "input_tokens": ("input_tokens", "prompt_tokens"),
        "output_tokens": ("output_tokens", "completion_tokens"),
        "total_tokens": ("total_tokens",),
    }
    output: dict[str, int] = {}
    for target, keys in aliases.items():
        for key in keys:
            try:
                if payload.get(key) is not None:
                    output[target] = max(0, int(payload[key]))
                    break
            except (TypeError, ValueError):
                continue
    if "total_tokens" not in output and ("input_tokens" in output or "output_tokens" in output):
        output["total_tokens"] = output.get("input_tokens", 0) + output.get("output_tokens", 0)
    return output
