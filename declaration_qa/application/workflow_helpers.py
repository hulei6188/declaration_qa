"""辅助函数：检索、办件摘要、流式事件和意图规则。"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import Any

from langgraph.config import get_config

from declaration_qa.application.state import DeclarationQAState
from declaration_qa.core.domain import RetrievalResult, RetrievedSource
from declaration_qa.rag.engine import (
    normalize_text,
    safe_int,
    structured_backfill_chunk_type,
    structured_source_matches,
)
from declaration_qa.rag.retrieval import normalize_retrieval_filter
from declaration_qa.rag.router import (
    INTENT_CORRECTION_GUIDANCE,
    INTENT_REJECTION_REASON,
    INTENT_TICKET_STATUS,
)


def ticket_summary_from_context(context: dict[str, Any]) -> dict[str, Any]:
    """从办件上下文生成摘要。"""
    return {
        "ticket_id": str(context.get("apply_id") or ""),
        "biz_type": str(context.get("biz_type") or ""),
        "region_code": str(context.get("region_code") or ""),
        "status": str(context.get("status") or ""),
        "current_node": str(context.get("current_node") or ""),
        "node_name": str(context.get("node_name") or ""),
        "title": business_title(context),
        "updated_at": str(context.get("updated_at") or ""),
    }


def ticket_title_from_biz_type(biz_type: str) -> str:
    """根据业务类型生成办件名称。"""
    names = {
        "health_certificate": "健康证办理",
        "residence_permit": "居住证办理",
        "id_card_replace": "身份证换领",
    }
    return names.get(str(biz_type or ""), str(biz_type or "政务服务事项"))


def ticket_rag_query(state: DeclarationQAState) -> str:
    """构造办件增强检索查询文本。"""
    business = state.get("business_status") or state.get("reject_info") or {}
    apply_context = state.get("apply_context") or {}
    mapping = state.get("material_mapping") or {}
    parts = [
        business_title(business, apply_context, state),
        business.get("node_name"),
        business.get("current_node"),
        business.get("reject_reason"),
        business.get("material_name"),
        business.get("reject_code"),
        mapping.get("material_code"),
        state.get("normalized_query"),
    ]
    return normalize_text(" ".join(str(item) for item in parts if item))


def knowledge_rag_query(state: DeclarationQAState) -> str:
    """构造普通知识检索查询，已选办件时继承事项范围。"""
    query = normalize_text(str(state.get("normalized_query") or state.get("query") or ""))
    if metadata_filter_has_item_scope(state.get("retrieval_filter") or {}):
        return query
    scope = selected_ticket_scope_text(state)
    if not scope:
        return query
    return normalize_text(f"{scope} {query}")


def selected_ticket_scope_text(state: DeclarationQAState) -> str:
    """从当前办件提取事项范围，不包含节点、退回原因等工单事实。"""
    apply_context = state.get("apply_context") or {}
    if not isinstance(apply_context, dict) or not apply_context:
        return ""
    payload = apply_context.get("payload") if isinstance(apply_context.get("payload"), dict) else {}
    parts = [
        business_title(apply_context, state),
        payload.get("official_item_name"),
        payload.get("item_name"),
        payload.get("title"),
        payload.get("catalog_name"),
        payload.get("source_file"),
        payload.get("item_code"),
        apply_context.get("biz_type"),
    ]
    return normalize_text(" ".join(unique_strings([str(item) for item in parts if item])))


def metadata_filter_has_item_scope(filters: dict[str, Any]) -> bool:
    """判断检索条件是否已具备事项级约束。"""
    return any(filters.get(key) for key in ("item_code", "ssqd_code", "ssqd_id", "item_name"))


def business_title(*sources: dict[str, Any]) -> str:
    """生成业务类型展示名称。"""
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in ("title", "item_name", "itemName", "catalog_name", "catalogName"):
            title = str(source.get(key) or "").strip()
            if title:
                return title
        payload = source.get("payload") if isinstance(source.get("payload"), dict) else {}
        for key in ("item_name", "title", "itemName", "catalog_name", "catalogName"):
            title = str(payload.get(key) or "").strip()
            if title:
                return title
    for source in sources:
        if isinstance(source, dict):
            title = ticket_title_from_biz_type(str(source.get("biz_type") or ""))
            if title:
                return title
    return "政务服务事项"


def base_policy_filter(state: DeclarationQAState, *, allow_request_filter: bool = True) -> dict[str, Any]:
    """构造基础政策检索过滤条件。"""
    filters: dict[str, Any] = {}
    if allow_request_filter:
        filters.update(normalize_retrieval_filter(state.get("metadata_filter") or {}))
    filters.update(ticket_metadata_filter(state))
    return normalize_retrieval_filter(filters)


def ticket_metadata_filter(state: DeclarationQAState) -> dict[str, Any]:
    """从当前办件上下文生成知识库可执行的事项过滤条件。"""
    apply_context = state.get("apply_context") or {}
    if not isinstance(apply_context, dict) or not apply_context:
        return {}
    payload = apply_context.get("payload") if isinstance(apply_context.get("payload"), dict) else {}
    for key in ("item_code", "ssqd_code", "ssqd_id"):
        value = first_non_empty(apply_context.get(key), payload.get(key), payload.get(camelize_metadata_key(key)))
        if value:
            return {key: value}
    item_name = first_non_empty(
        apply_context.get("item_name"),
        payload.get("official_item_name"),
        payload.get("item_name"),
        payload.get("itemName"),
        apply_context.get("title"),
        payload.get("title"),
    )
    if item_name:
        return {"item_name": item_name}
    return {}


def first_non_empty(*values: Any) -> str:
    """返回第一个非空字符串值。"""
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def camelize_metadata_key(key: str) -> str:
    """将 snake_case 元数据键转为常见 camelCase 别名。"""
    parts = key.split("_")
    return parts[0] + "".join(part.capitalize() for part in parts[1:])


def build_business_snapshot(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    """构造可用于回答的办件快照。"""
    cleaned = {key: value for key, value in (payload or {}).items() if value not in ("", None, [], {})}
    if not cleaned:
        return {}
    return {
        "type": kind,
        "source": "business_adapter",
        "apply_id": cleaned.get("apply_id"),
        "biz_type": cleaned.get("biz_type"),
        "title": business_title(cleaned),
        "region_code": cleaned.get("region_code"),
        "status": cleaned.get("status"),
        "current_node": cleaned.get("current_node"),
        "node_name": cleaned.get("node_name"),
        "material_code": cleaned.get("material_code"),
        "material_name": cleaned.get("material_name"),
        "reject_code": cleaned.get("reject_code"),
        "reject_reason": cleaned.get("reject_reason"),
        "audit_node": cleaned.get("audit_node"),
        "submitted_at": cleaned.get("submitted_at"),
        "completed_at": cleaned.get("completed_at"),
        "updated_at": cleaned.get("updated_at"),
        "payload": redact_business_payload(cleaned.get("payload") or {}),
    }


def redact_business_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """脱敏业务载荷中的敏感字段。"""
    redacted = {}
    for key, value in (payload or {}).items():
        lowered = str(key).lower()
        if (
            any(token in lowered for token in ["id_card", "idcard", "phone", "mobile"])
            or lowered in {"name", "applicant_name", "contact_name", "person_name"}
            or lowered.endswith("_address")
        ):
            redacted[key] = "[redacted]"
        else:
            redacted[key] = value
    return redacted


def add_event(state: dict[str, Any], event: str, data: dict[str, Any]) -> None:
    """向工作流事件列表追加事件。"""
    events = list(state.get("events") or [])
    events.append({"event": event, "data": data})
    state["events"] = events


def apply_planned_question(state: dict[str, Any], question: dict[str, Any]) -> None:
    """将 planner 理解后的单问题写回状态，保留原始问题在 question_plan 中。"""
    planned_query = normalize_text(str(question.get("original_query") or question.get("query") or ""))
    if not planned_query:
        return
    state["query"] = planned_query
    retrieval_query = normalize_text(str(question.get("retrieval_query") or planned_query))
    state["normalized_query"] = retrieval_query
    state["retrieval_query"] = retrieval_query
    state["requires_ticket_context"] = bool(question.get("requires_ticket_context"))
    state["ticket_fact_type"] = question.get("ticket_fact_type") or "none"
    state["retrieval_focus"] = question.get("retrieval_focus") or classify_intent(planned_query)
    state["biz_type"] = state.get("biz_type") or infer_biz_type(retrieval_query) or infer_biz_type(planned_query)


def unique_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按 chunk/doc 去重来源。"""
    seen = set()
    output = []
    for source in sources:
        key = (source.get("chunk_id") or "", source.get("doc_id") or "", source.get("title") or "")
        if key in seen:
            continue
        seen.add(key)
        output.append(source)
    return output


def dedupe_retrieval_queries(retrieval_queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按检索文本去重，保留 rewrite 原始顺序。"""
    output: list[dict[str, Any]] = []
    seen = set()
    for query in retrieval_queries:
        text = normalize_text(str(query.get("text") or ""))
        if not text or text in seen:
            continue
        seen.add(text)
        item = dict(query)
        item["text"] = text
        output.append(item)
    return output


def query_status_from_result(retrieval_query: dict[str, Any], result: RetrievalResult) -> dict[str, Any]:
    """构造单条 rewrite query 的检索状态。"""
    return {
        "id": retrieval_query.get("id") or "",
        "strategy": retrieval_query.get("strategy") or "",
        "query": str(retrieval_query.get("text") or "").strip(),
        "top_k": int(retrieval_query.get("top_k") or 3),
        "source_count": len(result.sources),
        "cache": (result.status or {}).get("cache") or {},
        "channels": (result.status or {}).get("channels") or {},
        "batch_mode": bool((result.status or {}).get("batch_mode")),
    }


def merge_retrieval_sources(results: list[RetrievalResult]) -> dict[str, RetrievedSource]:
    """合并多路检索来源，同一 chunk 保留最高分命中。"""
    merged: dict[str, RetrievedSource] = {}
    for result in results:
        for source in result.sources:
            existing = merged.get(source.chunk.id)
            if existing is None or float(source.score or 0.0) > float(existing.score or 0.0):
                merged[source.chunk.id] = source
    return merged


def select_answer_sources(merged: dict[str, RetrievedSource], query: str, *, limit: int = 6) -> list[RetrievedSource]:
    """选择进入答案生成的来源，优先保全结构化章节节点。"""
    ranked = sorted(
        merged.values(),
        key=lambda source: (float(source.rerank_score or 0.0), float(source.score or 0.0)),
        reverse=True,
    )
    target_chunk_type = structured_backfill_chunk_type(query)
    primary_section_id = primary_structured_section_id(ranked, target_chunk_type)
    if not primary_section_id:
        return ranked[:limit]

    selected: list[RetrievedSource] = []
    seen_chunk_ids: set[str] = set()
    section_intro = [
        source
        for source in ranked
        if source_parent_section_id(source) == primary_section_id
        and source_chunk_type(source) == "section_leaf"
        and "说明" in source_node_title(source)
    ]
    for source in section_intro[:1]:
        append_source_once(selected, source, seen_chunk_ids)

    structured_sources = [
        source
        for source in ranked
        if source_parent_section_id(source) == primary_section_id
        and source_chunk_type(source) == target_chunk_type
    ]
    for source in sorted(structured_sources, key=structured_source_sort_key):
        append_source_once(selected, source, seen_chunk_ids)

    target_limit = max(limit, len(selected))
    for source in ranked:
        if len(selected) >= target_limit:
            break
        append_source_once(selected, source, seen_chunk_ids)
    return selected


def primary_structured_section_id(sources: list[RetrievedSource], target_chunk_type: str) -> str:
    """按排序后的命中结果选择最相关的结构化章节。"""
    if not target_chunk_type:
        return ""
    for source in sources:
        parent_section_id = source_parent_section_id(source)
        if parent_section_id and structured_source_matches(source, target_chunk_type):
            return parent_section_id
    return ""


def append_source_once(output: list[RetrievedSource], source: RetrievedSource, seen_chunk_ids: set[str]) -> bool:
    """按 chunk id 追加来源。"""
    if source.chunk.id in seen_chunk_ids:
        return False
    seen_chunk_ids.add(source.chunk.id)
    output.append(source)
    return True


def structured_source_sort_key(source: RetrievedSource) -> tuple[int, int, int, str]:
    """按结构化节点在原指南中的顺序排序。"""
    metadata = source.chunk.metadata or {}
    return (
        safe_int(metadata.get("node_order")),
        safe_int(metadata.get("parent_window_index")),
        safe_int(metadata.get("child_window_index")),
        source.chunk.id,
    )


def source_parent_section_id(source: RetrievedSource) -> str:
    """读取来源所属结构化章节 ID。"""
    return str((source.chunk.metadata or {}).get("parent_section_id") or "")


def source_chunk_type(source: RetrievedSource) -> str:
    """读取来源的结构化片段类型。"""
    return str((source.chunk.metadata or {}).get("chunk_type") or "")


def source_node_title(source: RetrievedSource) -> str:
    """读取来源的结构化节点标题。"""
    return str((source.chunk.metadata or {}).get("node_title") or "")


def unique_tickets(tickets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按 ticket_id 去重办件摘要。"""
    seen = set()
    output = []
    for ticket in tickets:
        ticket_id = str(ticket.get("ticket_id") or "")
        if not ticket_id or ticket_id in seen:
            continue
        seen.add(ticket_id)
        output.append(ticket)
    return output


def unique_strings(values: list[str]) -> list[str]:
    """按出现顺序去重字符串。"""
    seen = set()
    output = []
    for value in values:
        item = str(value or "")
        if not item or item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


STREAM_TOKEN_SIZE = 3
STREAM_TOKEN_DELAY_SECONDS = 0.02


def token_events(
    text: str,
    *,
    size: int = STREAM_TOKEN_SIZE,
    delay_seconds: float = STREAM_TOKEN_DELAY_SECONDS,
    sleep: Callable[[float], None] | None = None,
) -> Iterable[dict[str, Any]]:
    """将文本切分为 token 事件。"""
    if sleep is None:
        from time import sleep as sleep_func

        sleep = sleep_func
    value = str(text or "")
    for index in range(0, len(value), size):
        if index > 0 and delay_seconds > 0:
            sleep(delay_seconds)
        yield {"event": "token", "data": {"content": value[index : index + size]}}


def graph_stream_part(chunk: Any) -> tuple[str, Any]:
    """归一化 LangGraph 多模式 stream chunk。"""
    if isinstance(chunk, tuple) and len(chunk) == 2:
        return str(chunk[0]), chunk[1]
    return "updates", chunk


def normalize_stream_event(payload: Any) -> dict[str, Any] | None:
    """将 LangGraph custom stream payload 转为前端 SSE 事件。"""
    if not isinstance(payload, dict):
        return None
    return {"event": str(payload.get("event") or "event"), "data": payload.get("data") or {}}


def should_emit_sources(node_name: str, state: DeclarationQAState) -> bool:
    """判断节点完成后是否需要补发检索来源事件。"""
    if not state.get("retrieved_sources"):
        return False
    if node_name in {"ticket_knowledge_rag", "hybrid_retrieve"}:
        return True
    return node_name == "multi_question_collect" and not state.get("need_ticket_selection")


def done_event(state: DeclarationQAState) -> dict[str, Any]:
    """构造流式响应结束事件。"""
    return {
        "event": "done",
        "data": {
            "session_id": state.get("session_id"),
            "audit_id": state.get("audit_id"),
            "intent": state.get("intent"),
            "route": state.get("route"),
            "sub_intent": state.get("sub_intent"),
            "multi_question": bool(state.get("multi_question")),
            "question_plan": state.get("question_plan") or {},
            "sub_tasks": state.get("sub_tasks") or [],
            "need_ticket_selection": bool(state.get("need_ticket_selection")),
            "ticket_selection_reason": state.get("ticket_selection_reason"),
            "tickets": state.get("tickets") or [],
            "ticket_context": state.get("ticket_context") or {},
            "retrieval_chain": state.get("retrieval_chain"),
            "refused": bool(state.get("refused")),
            "refuse_reason": state.get("refuse_reason"),
            "suggested_actions": state.get("suggested_actions") or [],
        },
    }


def is_declaration_qa_stream_run() -> bool:
    """判断当前节点是否由 DeclarationQAWorkflow.stream 启动。"""
    try:
        config = get_config()
    except RuntimeError:
        return False
    configurable = config.get("configurable") or {}
    return bool(configurable.get("declaration_qa_stream"))


def mask_sensitive_text(value: str) -> str:
    """遮盖文本中的敏感证件和手机号信息。"""
    text = str(value or "")
    text = re.sub(r"\b\d{17}[\dXx]\b", "[id_card]", text)
    text = re.sub(r"\b1[3-9]\d{9}\b", "[phone]", text)
    return text


def is_unsafe_query(query: str) -> bool:
    """判断是否为不应回答的高风险问题。"""
    return any(keyword in str(query or "") for keyword in ["伪造", "造假", "假材料", "绕过审核"])


def task_intent(retrieval_focus: str, ticket_fact_type: str, requires_ticket_context: bool, query: str = "") -> str:
    """从 Planner 任务字段派生兼容的 intent 展示值。"""
    if requires_ticket_context:
        if ticket_fact_type == "status":
            return INTENT_TICKET_STATUS
        if ticket_fact_type == "rejection":
            return INTENT_REJECTION_REASON
        if ticket_fact_type == "correction":
            return INTENT_CORRECTION_GUIDANCE
    if retrieval_focus == "other":
        return classify_intent(query) if query else "knowledge_qa"
    return str(retrieval_focus or "knowledge_qa")


def classify_intent(query: str) -> str:
    """按查询文本粗分类业务意图。"""
    text = str(query or "")
    if any(keyword in text for keyword in ["伪造", "造假", "假材料", "绕过审核"]):
        return "unsafe"
    if len(text.strip()) < 3:
        return "clarification"
    if any(keyword in text for keyword in ["退回", "驳回", "补正", "不通过", "被拒", "被退"]):
        return INTENT_REJECTION_REASON
    if any(keyword in text for keyword in ["进度", "到哪", "状态", "审核到", "办到哪", "多久能办完", "什么情况", "怎么样"]):
        return INTENT_TICKET_STATUS
    if any(keyword in text for keyword in ["材料", "要带", "提交什么", "需要什么", "清单"]):
        return "material_requirements"
    if any(keyword in text for keyword in ["流程", "步骤", "怎么办", "如何办理", "怎么申请"]):
        return "process_steps"
    if any(keyword in text for keyword in ["条件", "资格", "能不能办", "是否可以"]):
        return "eligibility"
    if any(keyword in text for keyword in ["费用", "收费", "多少钱", "时限", "多久", "几天", "多少天", "拿结果", "办结"]):
        return "fee_time_limit"
    if any(keyword in text for keyword in ["地点", "窗口", "地址", "哪里办"]):
        return "office_location"
    if any(keyword in text for keyword in ["电话", "咨询", "投诉"]):
        return "contact"
    if any(keyword in text for keyword in ["网上办", "入口", "链接", "在线"]):
        return "online_apply"
    if any(keyword in text for keyword in ["依据", "政策", "法规", "文件"]):
        return "policy_basis"
    if any(keyword in text for keyword in ["表格", "模板", "样表", "申请表"]):
        return "form_template"
    return "knowledge_qa"


def infer_biz_type(query: str) -> str | None:
    """根据文本推断业务类型。"""
    text = str(query or "")
    if "健康证" in text:
        return "health_certificate"
    if "居住证" in text or "居住证明" in text:
        return "residence_permit"
    if "身份证" in text:
        return "id_card_replace"
    return None


def material_tags(reject: dict[str, Any]) -> list[str]:
    """根据退回信息生成材料标签。"""
    material_code = str(reject.get("material_code") or "")
    reject_code = str(reject.get("reject_code") or "")
    tags = []
    if "PHOTO" in reject_code or "PHOTO" in material_code:
        tags.extend(["照片规范", "图片清晰度", "上传规范"])
    if material_code:
        tags.append(material_code)
    if reject_code:
        tags.append(reject_code)
    return tags


def suggested_actions(state: DeclarationQAState) -> list[dict[str, Any]]:
    """根据状态生成后续建议动作。"""
    app_version = state.get("app_version") or ""
    apply_context = state.get("apply_context") or {}
    if state.get("refused"):
        return [{"type": "human_service", "label": "联系人工客服", "payload": {"appVersion": app_version}}]
    if state.get("intent") in {INTENT_TICKET_STATUS}:
        return [
            {
                "type": "open_apply_detail",
                "label": "查看申请单详情",
                "payload": {"applyId": apply_context.get("apply_id"), "appVersion": app_version},
            }
        ]
    if state.get("intent") in {INTENT_REJECTION_REASON}:
        reject_info = state.get("reject_info") or {}
        return [
            {
                "type": "open_material_reupload",
                "label": "重新上传补正材料",
                "payload": {
                    "applyId": apply_context.get("apply_id"),
                    "materialCode": reject_info.get("material_code"),
                    "appVersion": app_version,
                },
            }
        ]
    return [
        {
            "type": "open_guide",
            "label": "查看办事指南",
            "payload": {"bizType": state.get("biz_type"), "appVersion": app_version},
        }
    ]
