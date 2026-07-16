"""多轮 Agentic RAG 的会话证据记忆与激活策略。"""

from __future__ import annotations

import re
from datetime import date
from typing import Any

from declaration_qa.application.evidence_registry import extract_business_key
from declaration_qa.core.business import region_code_matches
from declaration_qa.core.domain import RetrievedSource
from declaration_qa.rag.evidence import source_effective


MAX_KNOWLEDGE_REFS = 64
MAX_BUSINESS_REFS = 64
MAX_RECALLED_KNOWLEDGE = 12
HISTORICAL_ASSISTANT_PREFIX = "【历史助手回答，仅用于理解指代，不得作为事实证据】"
RETROSPECTIVE_TERMS = ("刚才", "刚刚", "之前", "先前", "上一轮", "上次", "当时", "前面显示")
CURRENT_STATE_TERMS = ("现在", "当前", "目前", "最新", "实时", "是否已经", "有没有变化")
FOLLOW_UP_TERMS = (
    "这个",
    "那个",
    "它",
    "上述",
    "上面",
    "前面",
    "其中",
    "第一个",
    "第二个",
    "第三个",
    "第一项",
    "第二项",
    "第三项",
    "还有呢",
)

TOOL_QUERY_TERMS: dict[str, tuple[str, ...]] = {
    "ticket_list_query": ("申请列表", "历史申请", "哪些申请", "办过"),
    "ticket_detail_query": ("申请", "进度", "状态", "退回", "补正", "受理", "办结"),
    "appointment_list_query": ("预约列表", "预约记录", "预约号"),
    "appointment_slot_query": ("号源", "时段", "预约时间"),
    "queue_status_query": ("排队", "等待", "窗口", "叫号"),
    "payment_order_query": ("缴费", "支付", "订单"),
    "payment_receipt_query": ("票据", "缴费凭证"),
    "submission_receipt_query": ("提交回执", "受理回执", "回执"),
    "result_certificate_query": ("办结结果", "结果证书", "证书"),
    "delivery_status_query": ("送达", "邮寄", "物流"),
    "consultation_record_query": ("咨询记录", "投诉", "建议记录"),
}


def project_historical_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """构造历史对话投影，同时隔离助手事实和压缩历史工具载荷。"""
    output: list[dict[str, Any]] = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        copied = dict(item)
        role = str(copied.get("role") or "")
        if role == "assistant":
            content = str(copied.get("content") or "").strip()
            if copied.get("tool_calls"):
                copied["content"] = ""
            elif content:
                if content.startswith(HISTORICAL_ASSISTANT_PREFIX):
                    content = content[len(HISTORICAL_ASSISTANT_PREFIX) :].strip()
                copied["content"] = f"{HISTORICAL_ASSISTANT_PREFIX}{content[:1200]}"
        elif role == "tool":
            copied["content"] = compact_historical_tool_observation(
                copied.get("content"),
                tool_name=str(copied.get("name") or ""),
                call_id=str(copied.get("tool_call_id") or ""),
            )
        output.append(copied)
    return output


def compact_historical_tool_observation(value: Any, *, tool_name: str, call_id: str) -> dict[str, Any]:
    """把历史工具结果压缩成可追溯引用，避免旧载荷进入模型上下文。"""
    payload = dict(value) if isinstance(value, dict) else {}
    resolved_tool = str(payload.get("tool_name") or tool_name or "")
    resolved_call = str(payload.get("call_id") or call_id or "")
    if resolved_tool == "knowledge_search":
        data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
        refs = [dict(item) for item in data.get("source_refs") or [] if isinstance(item, dict)]
        if not refs:
            refs = [
                {
                    "chunk_id": item.get("chunk_id") or "",
                    "doc_id": item.get("doc_id") or item.get("document_id") or "",
                    "title": item.get("title") or "",
                }
                for item in data.get("sources") or []
                if isinstance(item, dict)
            ]
        return {
            "tool_name": resolved_tool,
            "call_id": resolved_call,
            "ok": bool(payload.get("ok")),
            "status": payload.get("status") or "",
            "source_refs": refs,
            "memory_trust": "reference_only",
        }
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    business_key = extract_business_key(resolved_tool, data)
    return {
        "tool_name": resolved_tool,
        "call_id": resolved_call,
        "ok": bool(payload.get("ok")),
        "status": payload.get("status") or "",
        "business_key": business_key,
        "observed_at": payload.get("completed_at") or payload.get("observed_at") or "",
        "memory_trust": "reference_only",
    }


def update_session_evidence_refs(
    existing: dict[str, Any] | None,
    *,
    sources: list[dict[str, Any]],
    tool_results: list[dict[str, Any]],
    query: str,
) -> dict[str, list[dict[str, Any]]]:
    """把本轮成功观察更新为轻量会话证据引用。"""
    current = dict(existing or {})
    knowledge = [dict(item) for item in current.get("knowledge") or [] if isinstance(item, dict)]
    business = [dict(item) for item in current.get("business") or [] if isinstance(item, dict)]

    for source in sources:
        if not isinstance(source, dict):
            continue
        chunk_id = str(source.get("chunk_id") or "").strip()
        if not chunk_id:
            continue
        metadata = source.get("metadata") if isinstance(source.get("metadata"), dict) else {}
        ref = {
            "chunk_id": chunk_id,
            "document_id": str(source.get("doc_id") or source.get("document_id") or ""),
            "title": str(source.get("title") or ""),
            "region_code": str(source.get("region_code") or metadata.get("region_code") or ""),
            "biz_type": str(metadata.get("biz_type") or ""),
            "policy_version": str(source.get("policy_version") or metadata.get("policy_version") or ""),
            "source_version": str(source.get("source_version") or metadata.get("source_version") or ""),
            "item_code": str(source.get("item_code") or metadata.get("item_code") or ""),
            "item_name": str(source.get("item_name") or metadata.get("item_name") or ""),
            "score": float(source.get("score") or 0.0),
            "rerank_score": float(source.get("rerank_score") or 0.0),
            "query_text": str(query or "")[:500],
            "source_turn_id": str(source.get("source_turn_id") or ""),
            "pending_turn": True,
        }
        knowledge = upsert_ref(knowledge, ref, key="chunk_id")[-MAX_KNOWLEDGE_REFS:]

    for result in tool_results:
        if not isinstance(result, dict) or not result.get("ok"):
            continue
        tool_name = str(result.get("tool_name") or "")
        if not tool_name or tool_name == "knowledge_search" or tool_name == "appointment_book":
            continue
        data = result.get("data") if isinstance(result.get("data"), dict) else {}
        business_key = extract_business_key(tool_name, data)
        call_id = str(result.get("call_id") or "").strip()
        if not call_id or not business_key:
            continue
        ref = {
            "call_id": call_id,
            "tool_name": tool_name,
            "business_key": business_key,
            "observed_at": str(result.get("completed_at") or result.get("observed_at") or ""),
            "source_turn_id": str(result.get("source_turn_id") or ""),
            "result_ref": dict(result.get("result_ref") or {}),
            "pending_turn": True,
        }
        business = upsert_ref(business, ref, key="call_id")[-MAX_BUSINESS_REFS:]
    return {"knowledge": knowledge, "business": business}


def finalize_session_evidence_refs(refs: dict[str, Any] | None, *, audit_id: str) -> dict[str, list[dict[str, Any]]]:
    """在审计写入后为本轮引用绑定稳定的运行标识。"""
    output: dict[str, list[dict[str, Any]]] = {"knowledge": [], "business": []}
    for kind in output:
        for value in (refs or {}).get(kind) or []:
            if not isinstance(value, dict):
                continue
            item = dict(value)
            if item.pop("pending_turn", False):
                item["source_turn_id"] = str(audit_id or "")
                if kind == "business":
                    item["result_ref"] = {
                        "table": "audit_logs",
                        "audit_uid": str(audit_id or ""),
                        "field": "tool_results",
                        "call_id": str(item.get("call_id") or ""),
                    }
            output[kind].append(item)
    return output


def build_conversation_memory(
    refs: dict[str, Any] | None,
    memory_context: dict[str, Any] | None,
) -> dict[str, Any]:
    """构造只用于指代解析的会话上下文摘要。"""
    anchors: dict[str, Any] = {}
    for ref in reversed(list((refs or {}).get("business") or [])):
        if not isinstance(ref, dict):
            continue
        for field, values in dict(ref.get("business_key") or {}).items():
            if field not in anchors and isinstance(values, list) and values:
                anchors[field] = values[0]
    running_summary = (memory_context or {}).get("running_summary")
    summary_text = getattr(running_summary, "summary", "")
    if isinstance(running_summary, dict):
        summary_text = str(running_summary.get("summary") or "")
    return {
        "entity_anchors": anchors,
        "running_summary": str(summary_text or ""),
        "assistant_history_trust": "conversation_only",
    }


def select_knowledge_refs(query: str, refs: dict[str, Any] | None) -> list[dict[str, Any]]:
    """根据当前问题选择可能相关的历史知识引用。"""
    candidates = [dict(item) for item in (refs or {}).get("knowledge") or [] if isinstance(item, dict)]
    if not candidates:
        return []
    normalized = normalize_for_match(query)
    contextual = is_contextual_follow_up(query)
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for index, ref in enumerate(candidates):
        reference_text = " ".join(
            str(ref.get(field) or "")
            for field in ("title", "item_name", "item_code", "query_text")
        )
        score = text_overlap_score(normalized, normalize_for_match(reference_text))
        if score > 0:
            scored.append((score, index, ref))
    if scored:
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [item[2] for item in scored[:MAX_RECALLED_KNOWLEDGE]]
    if not contextual:
        return []
    latest_turn = str(candidates[-1].get("source_turn_id") or "")
    recent = [item for item in candidates if str(item.get("source_turn_id") or "") == latest_turn]
    return recent[-MAX_RECALLED_KNOWLEDGE:]


def rehydrate_knowledge_sources(
    rag_engine: Any,
    selected_refs: list[dict[str, Any]],
    state: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """按可信 chunk_id 回载正文，并校验地区、版本和有效期。"""
    if not selected_refs:
        return [], []
    chunks = rag_engine.get_chunks_by_ids([str(item.get("chunk_id") or "") for item in selected_refs])
    by_id = {str(chunk.id): chunk for chunk in chunks}
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for ref in selected_refs:
        chunk_id = str(ref.get("chunk_id") or "")
        chunk = by_id.get(chunk_id)
        reason = knowledge_rejection_reason(chunk, state, ref)
        if reason:
            rejected.append({"chunk_id": chunk_id, "reason": reason})
            continue
        payload = RetrievedSource(
            chunk=chunk,
            score=max(0.18, float(ref.get("score") or 0.0)),
            rerank_score=float(ref.get("rerank_score") or 0.0),
            retrieval_channel="memory_rehydrate",
        ).payload()
        payload["memory_origin"] = "rehydrated_history"
        payload["source_turn_id"] = str(ref.get("source_turn_id") or "")
        accepted.append(payload)
    return accepted, rejected


def recall_historical_business_results(
    *,
    query: str,
    refs: dict[str, Any] | None,
    audit_store: Any,
    user_id: int,
    session_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """仅为明确的回顾性问题加载历史业务快照。"""
    if not is_retrospective_query(query):
        return [], []
    candidates = [dict(item) for item in (refs or {}).get("business") or [] if isinstance(item, dict)]
    matching_tools = tools_for_query(query)
    if matching_tools:
        candidates = [item for item in candidates if item.get("tool_name") in matching_tools]
    loaded: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for ref in reversed(candidates[-8:]):
        result_ref = ref.get("result_ref") if isinstance(ref.get("result_ref"), dict) else {}
        audit_id = str(result_ref.get("audit_uid") or ref.get("source_turn_id") or "")
        audit = audit_store.get(audit_id) if audit_id else None
        if not isinstance(audit, dict) or int(audit.get("user_id") or 0) != int(user_id):
            rejected.append({"call_id": ref.get("call_id") or "", "reason": "audit_not_available"})
            continue
        audit_session = audit.get("session_id")
        if isinstance(audit_session, str) and audit_session and audit_session != session_id:
            rejected.append({"call_id": ref.get("call_id") or "", "reason": "session_mismatch"})
            continue
        result = find_tool_result(
            audit.get("tool_results") or [],
            call_id=str(ref.get("call_id") or ""),
            tool_name=str(ref.get("tool_name") or ""),
        )
        if result is None:
            rejected.append({"call_id": ref.get("call_id") or "", "reason": "result_not_available"})
            continue
        recalled = dict(result)
        recalled["memory_origin"] = "historical_snapshot"
        recalled["observed_at"] = str(ref.get("observed_at") or recalled.get("completed_at") or "")
        recalled["source_turn_id"] = str(ref.get("source_turn_id") or "")
        loaded.append(recalled)
        break
    return loaded, rejected


def build_working_memory(
    *,
    query: str,
    conversation_memory: dict[str, Any],
    knowledge_sources: list[dict[str, Any]],
    business_results: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
) -> dict[str, Any]:
    """构造本轮可审计的工作记忆，不保存模型内部推理。"""
    return {
        "standalone_query": str(query or ""),
        "resolved_context": dict(conversation_memory.get("entity_anchors") or {}),
        "active_knowledge_chunk_ids": [str(item.get("chunk_id") or "") for item in knowledge_sources],
        "active_business_call_ids": [str(item.get("call_id") or "") for item in business_results],
        "rejected_memory_refs": rejected[:20],
        "sufficiency": "agent_decides",
    }


def is_contextual_follow_up(query: str) -> bool:
    """识别需要利用上一轮指代信息的问题。"""
    normalized = str(query or "").strip()
    return any(term in normalized for term in FOLLOW_UP_TERMS) or (
        len(normalized) <= 14 and normalized.endswith(("呢", "吗", "？", "?"))
    )


def is_retrospective_query(query: str) -> bool:
    """判断用户是否明确询问历史展示结果。"""
    text = str(query or "")
    if any(term in text for term in CURRENT_STATE_TERMS):
        return False
    return any(term in text for term in RETROSPECTIVE_TERMS)


def tools_for_query(query: str) -> set[str]:
    """根据业务词选择历史快照候选工具。"""
    return {
        tool_name
        for tool_name, terms in TOOL_QUERY_TERMS.items()
        if any(term in str(query or "") for term in terms)
    }


def knowledge_rejection_reason(chunk: Any, state: dict[str, Any], ref: dict[str, Any] | None = None) -> str:
    """返回历史知识片段不能在本轮激活的原因。"""
    if chunk is None:
        return "chunk_not_found"
    if str(getattr(chunk, "status", "") or "published") not in {"published", "active"}:
        return "chunk_not_published"
    requested_region = str(state.get("region_code") or "")
    chunk_region = str(getattr(chunk, "region_code", "") or "")
    if requested_region and chunk_region and not region_code_matches(chunk_region, requested_region):
        return "region_mismatch"
    requested_policy = str(state.get("policy_version") or "")
    chunk_policy = str(getattr(chunk, "policy_version", "") or "")
    if requested_policy and chunk_policy and requested_policy != chunk_policy:
        return "policy_version_mismatch"
    remembered_source_version = str((ref or {}).get("source_version") or "")
    current_source_version = str((getattr(chunk, "metadata", {}) or {}).get("source_version") or "")
    if remembered_source_version and current_source_version and remembered_source_version != current_source_version:
        return "source_version_mismatch"
    requested_biz = str(state.get("biz_type") or "")
    chunk_biz = str((getattr(chunk, "metadata", {}) or {}).get("biz_type") or "")
    if requested_biz and chunk_biz and requested_biz != chunk_biz:
        return "biz_type_mismatch"
    payload = RetrievedSource(chunk=chunk, score=1.0).payload()
    if not source_effective(payload, date.today()):
        return "outside_effective_period"
    return ""


def find_tool_result(results: Any, *, call_id: str, tool_name: str) -> dict[str, Any] | None:
    """从审计载荷中定位指定业务调用结果。"""
    for result in results if isinstance(results, list) else []:
        if not isinstance(result, dict):
            continue
        if str(result.get("call_id") or "") == call_id and str(result.get("tool_name") or "") == tool_name:
            return result if result.get("ok") else None
    return None


def upsert_ref(items: list[dict[str, Any]], ref: dict[str, Any], *, key: str) -> list[dict[str, Any]]:
    """更新同标识引用并把最近观察移动到末尾。"""
    value = str(ref.get(key) or "")
    output = [dict(item) for item in items if str(item.get(key) or "") != value]
    output.append(ref)
    return output


def normalize_for_match(value: str) -> str:
    """保留中英文和数字，用于轻量相关性比较。"""
    normalized = "".join(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", str(value or ""))).lower()
    for generic in ("怎么办理", "如何办理", "办理", "申请", "需要", "哪些", "什么", "怎么", "材料", "流程", "要求", "规定", "指南"):
        normalized = normalized.replace(generic, "")
    return normalized


def text_overlap_score(left: str, right: str) -> float:
    """使用字符二元组计算无需模型的历史主题重合度。"""
    if len(left) < 2 or len(right) < 2:
        return 0.0
    left_pairs = {left[index : index + 2] for index in range(len(left) - 1)}
    right_pairs = {right[index : index + 2] for index in range(len(right) - 1)}
    overlap = left_pairs & right_pairs
    if len(overlap) < 2 and not (len(left) >= 4 and (left in right or right in left)):
        return 0.0
    return len(overlap) / max(1, min(len(left_pairs), len(right_pairs)))
