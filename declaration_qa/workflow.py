from __future__ import annotations

import re
from datetime import date
from typing import Any
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from collections.abc import Iterable

from declaration_qa.audit import InMemoryAuditStore
from declaration_qa.business import BusinessDataAdapter, InMemoryBusinessDataAdapter
from declaration_qa.evidence import decide_evidence
from declaration_qa.prompting import DeterministicAnswerGenerator
from declaration_qa.rag_engine import InMemoryRAGEngine, build_seed_rag_engine, normalize_text
from declaration_qa.schemas import ChatRequest, ChatResponse
from declaration_qa.source_types import MATERIAL_RULE_SOURCE_TYPE
from declaration_qa.state import DeclarationQAState


class DeclarationQAWorkflow:
    def __init__(
        self,
        *,
        rag_engine: InMemoryRAGEngine | None = None,
        business_adapter: BusinessDataAdapter | None = None,
        audit_store: InMemoryAuditStore | None = None,
        answer_generator: DeterministicAnswerGenerator | None = None,
    ) -> None:
        self.rag_engine = rag_engine or build_seed_rag_engine()
        self.business_adapter = business_adapter or InMemoryBusinessDataAdapter()
        self.audit_store = audit_store or InMemoryAuditStore()
        self.answer_generator = answer_generator or DeterministicAnswerGenerator()
        self.graph = self._build_graph()

    def invoke(self, request: ChatRequest) -> ChatResponse:
        initial_state = self._initial_state(request)
        final_state = self.graph.invoke(initial_state)
        return ChatResponse(
            session_id=str(final_state["session_id"]),
            audit_id=str(final_state["audit_id"]),
            intent=str(final_state.get("intent") or "unknown"),
            answer=str(final_state.get("answer") or ""),
            refused=bool(final_state.get("refused")),
            refuse_reason=final_state.get("refuse_reason"),
            sources=final_state.get("retrieved_sources") or [],
            suggested_actions=final_state.get("suggested_actions") or [],
            events=final_state.get("events") or [],
        )

    def stream(self, request: ChatRequest) -> Iterable[dict[str, Any]]:
        state: DeclarationQAState = self._initial_state(request)
        yield {"event": "run_started", "data": {"session_id": state["session_id"]}}

        def run_node(name: str, handler) -> None:
            nonlocal state
            before = len(state.get("events") or [])
            state = handler(state)
            for event in (state.get("events") or [])[before:]:
                yield_event = {"event": event.get("event") or name, "data": event.get("data") or {}}
                emitted.append(yield_event)

        emitted: list[dict[str, Any]] = []

        def drain() -> Iterable[dict[str, Any]]:
            while emitted:
                yield emitted.pop(0)

        for name, handler in [
            ("context_build", self.context_build),
            ("query_preprocess", self.query_preprocess),
            ("intent_route", self.intent_route),
        ]:
            run_node(name, handler)
            yield from drain()

        route = state.get("route") or "refuse"
        if route == "status":
            run_node("status_apply_context_check", self.status_apply_context_check)
            yield from drain()
            if state.get("apply_context"):
                for name, handler in [
                    ("business_status_query", self.business_status_query),
                    ("status_explain_rag", self.status_explain_rag),
                    ("llm_generate", self.llm_generate),
                ]:
                    run_node(name, handler)
                    yield from drain()
                    if name == "status_explain_rag" and state.get("retrieved_sources"):
                        yield {"event": "sources", "data": {"items": state.get("retrieved_sources") or []}}
                    if name == "llm_generate":
                        yield from token_events(str(state.get("answer") or ""))
            else:
                run_node("refuse_or_human", self.refuse_or_human)
                yield from drain()
                yield from token_events(str(state.get("answer") or ""))
        elif route == "reject":
            run_node("reject_apply_context_check", self.reject_apply_context_check)
            yield from drain()
            if state.get("apply_context"):
                for name, handler in [
                    ("business_reject_query", self.business_reject_query),
                    ("material_mapping", self.material_mapping),
                    ("material_rule_rag", self.material_rule_rag),
                    ("evidence_check", self.evidence_check),
                ]:
                    run_node(name, handler)
                    yield from drain()
                    if name == "material_rule_rag" and state.get("retrieved_sources"):
                        yield {"event": "sources", "data": {"items": state.get("retrieved_sources") or []}}
                if state.get("evidence_ok"):
                    run_node("llm_generate", self.llm_generate)
                else:
                    run_node("refuse_or_human", self.refuse_or_human)
                yield from drain()
                yield from token_events(str(state.get("answer") or ""))
            else:
                run_node("refuse_or_human", self.refuse_or_human)
                yield from drain()
                yield from token_events(str(state.get("answer") or ""))
        elif route == "knowledge":
            for name, handler in [
                ("knowledge_filter_build", self.knowledge_filter_build),
                ("hybrid_retrieve", self.hybrid_retrieve),
                ("evidence_check", self.evidence_check),
            ]:
                run_node(name, handler)
                yield from drain()
                if name == "hybrid_retrieve" and state.get("retrieved_sources"):
                    yield {"event": "sources", "data": {"items": state.get("retrieved_sources") or []}}
            if state.get("evidence_ok"):
                run_node("llm_generate", self.llm_generate)
            else:
                run_node("refuse_or_human", self.refuse_or_human)
            yield from drain()
            yield from token_events(str(state.get("answer") or ""))
        else:
            run_node("refuse_or_human", self.refuse_or_human)
            yield from drain()
            yield from token_events(str(state.get("answer") or ""))

        for name, handler in [
            ("answer_postprocess", self.answer_postprocess),
            ("audit_log", self.audit_log),
        ]:
            run_node(name, handler)
            yield from drain()

        yield {
            "event": "done",
            "data": {
                "session_id": state.get("session_id"),
                "audit_id": state.get("audit_id"),
                "intent": state.get("intent"),
                "refused": bool(state.get("refused")),
                "refuse_reason": state.get("refuse_reason"),
                "suggested_actions": state.get("suggested_actions") or [],
            },
        }

    def _initial_state(self, request: ChatRequest) -> DeclarationQAState:
        apply_context = request.apply_context.model_dump(exclude_none=True) if request.apply_context else None
        return {
            "user_id": request.user_id,
            "session_id": request.session_id or uuid4().hex,
            "query": request.query,
            "normalized_query": "",
            "region_code": request.region_code,
            "app_version": request.app_version,
            "policy_version": request.policy_version,
            "biz_type": request.biz_type,
            "apply_id": request.apply_id,
            "apply_context": apply_context,
            "intent": "unknown",
            "route": "",
            "retrieval_filter": {},
            "retrieved_sources": [],
            "rag_status": {},
            "evidence_ok": False,
            "refuse_reason": None,
            "answer": "",
            "refused": False,
            "suggested_actions": [],
            "events": [],
            "audit_payload": {},
        }

    def _build_graph(self):
        builder = StateGraph(DeclarationQAState)
        builder.add_node("context_build", self.context_build)
        builder.add_node("query_preprocess", self.query_preprocess)
        builder.add_node("intent_route", self.intent_route)
        builder.add_node("status_apply_context_check", self.status_apply_context_check)
        builder.add_node("reject_apply_context_check", self.reject_apply_context_check)
        builder.add_node("business_status_query", self.business_status_query)
        builder.add_node("status_explain_rag", self.status_explain_rag)
        builder.add_node("business_reject_query", self.business_reject_query)
        builder.add_node("material_mapping", self.material_mapping)
        builder.add_node("material_rule_rag", self.material_rule_rag)
        builder.add_node("knowledge_filter_build", self.knowledge_filter_build)
        builder.add_node("hybrid_retrieve", self.hybrid_retrieve)
        builder.add_node("evidence_check", self.evidence_check)
        builder.add_node("refuse_or_human", self.refuse_or_human)
        builder.add_node("llm_generate", self.llm_generate)
        builder.add_node("answer_postprocess", self.answer_postprocess)
        builder.add_node("audit_log", self.audit_log)

        builder.add_edge(START, "context_build")
        builder.add_edge("context_build", "query_preprocess")
        builder.add_edge("query_preprocess", "intent_route")
        builder.add_conditional_edges(
            "intent_route",
            lambda state: state.get("route") or "refuse",
            {
                "status": "status_apply_context_check",
                "reject": "reject_apply_context_check",
                "knowledge": "knowledge_filter_build",
                "refuse": "refuse_or_human",
            },
        )
        builder.add_conditional_edges(
            "status_apply_context_check",
            lambda state: "ok" if state.get("apply_context") else "missing",
            {"ok": "business_status_query", "missing": "refuse_or_human"},
        )
        builder.add_edge("business_status_query", "status_explain_rag")
        builder.add_edge("status_explain_rag", "llm_generate")
        builder.add_conditional_edges(
            "reject_apply_context_check",
            lambda state: "ok" if state.get("apply_context") else "missing",
            {"ok": "business_reject_query", "missing": "refuse_or_human"},
        )
        builder.add_edge("business_reject_query", "material_mapping")
        builder.add_edge("material_mapping", "material_rule_rag")
        builder.add_edge("material_rule_rag", "evidence_check")
        builder.add_edge("knowledge_filter_build", "hybrid_retrieve")
        builder.add_edge("hybrid_retrieve", "evidence_check")
        builder.add_conditional_edges(
            "evidence_check",
            lambda state: "ok" if state.get("evidence_ok") else "refuse",
            {"ok": "llm_generate", "refuse": "refuse_or_human"},
        )
        builder.add_edge("llm_generate", "answer_postprocess")
        builder.add_edge("refuse_or_human", "answer_postprocess")
        builder.add_edge("answer_postprocess", "audit_log")
        builder.add_edge("audit_log", END)
        return builder.compile()

    def context_build(self, state: DeclarationQAState) -> DeclarationQAState:
        next_state = dict(state)
        if not next_state.get("apply_context") and next_state.get("apply_id"):
            next_state["apply_context"] = self.business_adapter.get_apply_context(
                user_id=int(next_state["user_id"]),
                apply_id=str(next_state["apply_id"]),
            )
        if next_state.get("apply_context"):
            context_apply_id = str((next_state.get("apply_context") or {}).get("apply_id") or next_state.get("apply_id") or "")
            if context_apply_id and not self.business_adapter.verify_apply_access(
                user_id=int(next_state["user_id"]),
                apply_id=context_apply_id,
            ):
                next_state["apply_context"] = None
                next_state["refuse_reason"] = "apply_access_denied"
                add_event(next_state, "apply_access_denied", {"apply_id": context_apply_id})
        apply_context = next_state.get("apply_context") or {}
        next_state["biz_type"] = next_state.get("biz_type") or apply_context.get("biz_type")
        next_state["region_code"] = apply_context.get("region_code") or next_state.get("region_code") or "340000"
        add_event(next_state, "context_built", {"has_apply_context": bool(next_state.get("apply_context"))})
        return next_state

    def query_preprocess(self, state: DeclarationQAState) -> DeclarationQAState:
        next_state = dict(state)
        query = normalize_text(mask_sensitive_text(next_state.get("query") or ""))
        next_state["normalized_query"] = query
        next_state["biz_type"] = next_state.get("biz_type") or infer_biz_type(query)
        add_event(next_state, "query_preprocessed", {"normalized_query": query, "biz_type": next_state.get("biz_type")})
        return next_state

    def intent_route(self, state: DeclarationQAState) -> DeclarationQAState:
        next_state = dict(state)
        intent = classify_intent(next_state.get("normalized_query") or next_state.get("query") or "")
        next_state["intent"] = intent
        next_state["route"] = intent if intent in {"status", "reject", "knowledge"} else "refuse"
        if intent == "unsafe":
            next_state["refuse_reason"] = "unsafe_or_out_of_scope"
        add_event(next_state, "intent", {"intent": intent})
        return next_state

    def status_apply_context_check(self, state: DeclarationQAState) -> DeclarationQAState:
        next_state = dict(state)
        if not next_state.get("apply_context"):
            next_state["refuse_reason"] = "apply_context_required_for_status"
        return next_state

    def reject_apply_context_check(self, state: DeclarationQAState) -> DeclarationQAState:
        next_state = dict(state)
        if not next_state.get("apply_context"):
            next_state["refuse_reason"] = "apply_context_required_for_reject"
        return next_state

    def business_status_query(self, state: DeclarationQAState) -> DeclarationQAState:
        next_state = dict(state)
        status = self.business_adapter.get_status(next_state.get("apply_context") or {})
        next_state["business_status"] = status
        next_state["biz_type"] = next_state.get("biz_type") or status.get("biz_type")
        add_event(next_state, "business_status", status)
        return next_state

    def status_explain_rag(self, state: DeclarationQAState) -> DeclarationQAState:
        next_state = dict(state)
        status = next_state.get("business_status") or {}
        filters = base_policy_filter(next_state, source_type="process")
        next_state["retrieval_filter"] = filters
        query = f"{status.get('node_name') or status.get('current_node') or ''} {next_state.get('normalized_query') or ''}"
        result = self.rag_engine.retrieve(query, filters, top_k=3)
        next_state["retrieved_sources"] = [source.payload() for source in result.sources]
        next_state["rag_status"] = result.status
        add_event(next_state, "rag_status", result.status)
        return next_state

    def business_reject_query(self, state: DeclarationQAState) -> DeclarationQAState:
        next_state = dict(state)
        reject_info = self.business_adapter.get_reject_info(next_state.get("apply_context") or {})
        next_state["reject_info"] = reject_info
        next_state["biz_type"] = next_state.get("biz_type") or reject_info.get("biz_type")
        next_state["region_code"] = reject_info.get("region_code") or next_state.get("region_code")
        add_event(next_state, "business_reject", reject_info)
        return next_state

    def material_mapping(self, state: DeclarationQAState) -> DeclarationQAState:
        next_state = dict(state)
        reject = next_state.get("reject_info") or {}
        mapping = {
            "material_code": reject.get("material_code") or "",
            "material_name": reject.get("material_name") or "",
            "reject_code": reject.get("reject_code") or "",
            "audit_node": reject.get("audit_node") or "",
            "knowledge_tags": material_tags(reject),
        }
        next_state["material_mapping"] = mapping
        add_event(next_state, "material_mapping", mapping)
        return next_state

    def material_rule_rag(self, state: DeclarationQAState) -> DeclarationQAState:
        next_state = dict(state)
        reject = next_state.get("reject_info") or {}
        filters = base_policy_filter(next_state, source_type=MATERIAL_RULE_SOURCE_TYPE)
        filters.update(
            {
                "material_code": reject.get("material_code") or "",
                "reject_code": reject.get("reject_code") or "",
            }
        )
        if reject.get("audit_node"):
            filters["audit_node"] = reject.get("audit_node")
        query = " ".join(
            str(item)
            for item in [
                reject.get("reject_reason"),
                reject.get("material_name"),
                reject.get("reject_code"),
                next_state.get("normalized_query"),
            ]
            if item
        )
        result = self.rag_engine.retrieve(query, filters, top_k=4)
        if not result.sources and filters.get("material_code") and (filters.get("reject_code") or filters.get("audit_node")):
            relaxed_filters = dict(filters)
            relaxed_filters.pop("reject_code", None)
            relaxed_filters.pop("audit_node", None)
            relaxed_result = self.rag_engine.retrieve(query, relaxed_filters, top_k=4)
            if relaxed_result.sources:
                filters = relaxed_filters
                result = relaxed_result
                add_event(
                    next_state,
                    "material_rule_filter_relaxed",
                    {"material_code": filters.get("material_code"), "source_type": filters.get("source_type")},
                )
        next_state["retrieval_filter"] = filters
        next_state["retrieved_sources"] = [source.payload() for source in result.sources]
        next_state["rag_status"] = result.status
        add_event(next_state, "rag_status", result.status)
        return next_state

    def knowledge_filter_build(self, state: DeclarationQAState) -> DeclarationQAState:
        next_state = dict(state)
        if not next_state.get("biz_type"):
            next_state["biz_type"] = infer_biz_type(next_state.get("normalized_query") or "")
        filters = base_policy_filter(next_state, source_type=["guide", "process", "faq"])
        next_state["retrieval_filter"] = filters
        add_event(next_state, "knowledge_filter", filters)
        return next_state

    def hybrid_retrieve(self, state: DeclarationQAState) -> DeclarationQAState:
        next_state = dict(state)
        result = self.rag_engine.retrieve(
            next_state.get("normalized_query") or next_state.get("query") or "",
            next_state.get("retrieval_filter") or {},
            top_k=4,
        )
        next_state["retrieved_sources"] = [source.payload() for source in result.sources]
        next_state["rag_status"] = result.status
        add_event(next_state, "rag_status", result.status)
        return next_state

    def evidence_check(self, state: DeclarationQAState) -> DeclarationQAState:
        next_state = dict(state)
        decision = decide_evidence(
            next_state.get("retrieved_sources") or [],
            next_state.get("retrieval_filter") or {},
            intent=str(next_state.get("intent") or ""),
        )
        next_state["evidence_ok"] = decision.ok
        if not decision.ok:
            next_state["refuse_reason"] = decision.reason
        add_event(next_state, "evidence", {"ok": decision.ok, "reason": decision.reason, "top_score": decision.top_score})
        return next_state

    def refuse_or_human(self, state: DeclarationQAState) -> DeclarationQAState:
        next_state = dict(state)
        next_state["refused"] = True
        reason = next_state.get("refuse_reason") or "out_of_scope"
        next_state["refuse_reason"] = reason
        if reason == "apply_context_required_for_status":
            answer = "该问题需要绑定具体申请单。请先从办件列表选择对应申请单后再查询进度。"
        elif reason == "apply_context_required_for_reject":
            answer = "该问题需要绑定具体申请单。请先选择被退回的申请单后再查看驳回原因。"
        elif reason == "unsafe_or_out_of_scope":
            answer = "该问题不属于申报辅助范围，或涉及不合规操作，暂不能提供处理建议。"
        elif reason == "apply_access_denied":
            answer = "当前登录用户无权查看该申请单，请确认申请单是否属于本人或重新登录后再试。"
        else:
            answer = "暂未查询到与当前地区和业务类型完全匹配的办理依据，建议查看办事指南或联系人工客服确认。"
        next_state["answer"] = answer
        add_event(next_state, "refused", {"reason": reason})
        return next_state

    def llm_generate(self, state: DeclarationQAState) -> DeclarationQAState:
        next_state = dict(state)
        answer, prompt = self.answer_generator.generate(next_state)
        next_state["answer"] = answer
        audit_payload = dict(next_state.get("audit_payload") or {})
        audit_payload["final_prompt"] = prompt
        next_state["audit_payload"] = audit_payload
        add_event(next_state, "answer_generated", {"intent": next_state.get("intent")})
        return next_state

    def answer_postprocess(self, state: DeclarationQAState) -> DeclarationQAState:
        next_state = dict(state)
        next_state["suggested_actions"] = suggested_actions(next_state)
        add_event(next_state, "suggested_actions", {"items": next_state["suggested_actions"]})
        return next_state

    def audit_log(self, state: DeclarationQAState) -> DeclarationQAState:
        next_state = dict(state)
        payload = {
            "user_id": next_state.get("user_id"),
            "session_id": next_state.get("session_id"),
            "query": next_state.get("query"),
            "normalized_query": next_state.get("normalized_query"),
            "intent": next_state.get("intent"),
            "region_code": next_state.get("region_code"),
            "biz_type": next_state.get("biz_type"),
            "apply_id": (next_state.get("apply_context") or {}).get("apply_id") if next_state.get("apply_context") else None,
            "business_status": next_state.get("business_status"),
            "reject_info": next_state.get("reject_info"),
            "material_mapping": next_state.get("material_mapping"),
            "retrieval_filter": next_state.get("retrieval_filter"),
            "retrieved_sources": next_state.get("retrieved_sources") or [],
            "rag_status": next_state.get("rag_status"),
            "evidence_ok": next_state.get("evidence_ok"),
            "refused": next_state.get("refused"),
            "refuse_reason": next_state.get("refuse_reason"),
            "answer": next_state.get("answer"),
            "suggested_actions": next_state.get("suggested_actions") or [],
            "events": next_state.get("events") or [],
            **(next_state.get("audit_payload") or {}),
        }
        audit_id = self.audit_store.append(payload)
        next_state["audit_id"] = audit_id
        add_event(next_state, "audit_logged", {"audit_id": audit_id})
        return next_state


def base_policy_filter(state: DeclarationQAState, *, source_type: str | list[str]) -> dict[str, Any]:
    return {
        "biz_type": state.get("biz_type") or "",
        "region_code": state.get("region_code") or "340000",
        "policy_version": state.get("policy_version") or "2025",
        "as_of": date.today().isoformat(),
        "rollout_key": str(state.get("user_id") or state.get("session_id") or ""),
        "status": "published",
        "source_type": source_type,
    }


def add_event(state: dict[str, Any], event: str, data: dict[str, Any]) -> None:
    events = list(state.get("events") or [])
    events.append({"event": event, "data": data})
    state["events"] = events


def token_events(text: str, *, size: int = 24) -> Iterable[dict[str, Any]]:
    value = str(text or "")
    for index in range(0, len(value), size):
        yield {"event": "token", "data": {"content": value[index : index + size]}}


def mask_sensitive_text(value: str) -> str:
    text = str(value or "")
    text = re.sub(r"\b\d{17}[\dXx]\b", "[id_card]", text)
    text = re.sub(r"\b1[3-9]\d{9}\b", "[phone]", text)
    return text


def classify_intent(query: str) -> str:
    text = str(query or "")
    if any(keyword in text for keyword in ["伪造", "造假", "假材料", "绕过审核"]):
        return "unsafe"
    if any(keyword in text for keyword in ["退回", "驳回", "补正", "不通过", "被拒"]):
        return "reject"
    if any(keyword in text for keyword in ["进度", "到哪", "状态", "审核到", "办到哪"]):
        return "status"
    return "knowledge"


def infer_biz_type(query: str) -> str | None:
    text = str(query or "")
    if "健康证" in text:
        return "health_certificate"
    if "居住证" in text or "居住证明" in text:
        return "residence_permit"
    if "身份证" in text:
        return "id_card_replace"
    return None


def material_tags(reject: dict[str, Any]) -> list[str]:
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
    app_version = state.get("app_version") or ""
    apply_context = state.get("apply_context") or {}
    if state.get("refused"):
        return [{"type": "human_service", "label": "联系人工客服", "payload": {"appVersion": app_version}}]
    if state.get("intent") == "status":
        return [
            {
                "type": "open_apply_detail",
                "label": "查看申请单详情",
                "payload": {"applyId": apply_context.get("apply_id"), "appVersion": app_version},
            }
        ]
    if state.get("intent") == "reject":
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
