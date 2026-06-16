from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


ROUTE_TICKET_CONTEXT_UNRELATED = "ticket_context_unrelated"
ROUTE_TICKET_CONTEXT_RELATED = "ticket_context_related"
ROUTE_CLARIFICATION = "clarification"
ROUTE_REFUSE = "refuse"

INTENT_UNSAFE = "unsafe"
INTENT_TICKET_STATUS = "ticket_status"
INTENT_REJECTION_REASON = "rejection_reason"
INTENT_CORRECTION_GUIDANCE = "correction_guidance"
INTENT_MATERIAL_REQUIREMENTS = "material_requirements"
INTENT_PROCESS_STEPS = "process_steps"
INTENT_ELIGIBILITY = "eligibility"
INTENT_FEE_TIME_LIMIT = "fee_time_limit"
INTENT_OFFICE_LOCATION = "office_location"
INTENT_CONTACT = "contact"
INTENT_ONLINE_APPLY = "online_apply"
INTENT_POLICY_BASIS = "policy_basis"
INTENT_FORM_TEMPLATE = "form_template"
INTENT_KNOWLEDGE_QA = "knowledge_qa"
INTENT_CLARIFICATION = "clarification"

ALLOWED_ROUTES = {
    ROUTE_TICKET_CONTEXT_UNRELATED,
    ROUTE_TICKET_CONTEXT_RELATED,
    ROUTE_CLARIFICATION,
    ROUTE_REFUSE,
}

ROUTER_SYSTEM_PROMPT = """你是政务服务智能问答系统的请求路由器，只能返回一个 JSON 对象。
你的任务是判断用户问题是否和当前/个人工单上下文有关，不生成业务答案。

可选 route:
- ticket_context_related: 问题需要结合当前/个人工单上下文理解或回答，包括进度、状态、驳回、补正，也包括“这个、该事项、当前申请、办这个”等指向当前工单事项的材料、流程、条件、费用等问题。
- ticket_context_unrelated: 问题不依赖当前/个人工单上下文，可作为通用办事咨询处理。
- clarification: 问题过短、上下文不足，无法判断是否和当前/个人工单上下文有关。
- refuse: 违法违规、伪造材料、绕过审核、冒用他人信息等风险问题。

决策要求:
1. 只判断问题是否和当前/个人工单上下文有关，不判断材料、流程、状态、驳回等业务意图。
2. 当前有工单不代表一定相关；用户明确咨询其他事项且不依赖当前工单事实时，route=ticket_context_unrelated。
3. 请求上下文中的 current_ticket 是当前已绑定工单的非敏感摘要；用户使用“这个、该事项、当前申请、这单、办这个”等指代时，优先结合 current_ticket 判断是否相关。
4. 用户没有绑定工单但询问“我的申请、这单、进度、驳回、补正”等个人办件问题时，route=ticket_context_related。
5. 只输出下方 JSON 字段，不要添加其他分类字段；系统会在路由后内部推导处理类型。

JSON 字段:
{
  "route": "ticket_context_related|ticket_context_unrelated|clarification|refuse",
  "confidence": 0.0,
  "requires_ticket_context": false,
  "can_answer_from_knowledge": true,
  "reason": "简短英文原因"
}"""


@dataclass(frozen=True)
class RouteDecision:
    route: str
    intent: str
    sub_intent: str = ""
    confidence: float = 0.0
    requires_ticket_context: bool = False
    can_answer_from_knowledge: bool = True
    reason: str = ""
    risk_level: str = "pass"
    entities: dict[str, Any] = field(default_factory=dict)

    def payload(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "intent": self.intent,
            "sub_intent": self.sub_intent,
            "confidence": round(float(self.confidence), 4),
            "requires_ticket_context": self.requires_ticket_context,
            "can_answer_from_knowledge": self.can_answer_from_knowledge,
            "reason": self.reason,
            "risk_level": self.risk_level,
            "entities": dict(self.entities or {}),
        }


class IntentRouter:
    name = "rule"

    def route(self, state: dict[str, Any]) -> RouteDecision:
        query = str(state.get("normalized_query") or state.get("query") or "")
        entities = extract_entities(query, state)

        risk = risk_check(query)
        if risk["level"] == "block":
            return RouteDecision(
                route=ROUTE_REFUSE,
                intent=INTENT_UNSAFE,
                sub_intent=INTENT_UNSAFE,
                confidence=0.99,
                requires_ticket_context=False,
                can_answer_from_knowledge=False,
                reason=risk["reason"],
                risk_level=risk["level"],
                entities=entities,
            )

        ticket_intent = classify_ticket_intent(query)
        knowledge_intent = classify_knowledge_intent(query)
        if ticket_context_related(query, state):
            intent = ticket_intent or knowledge_intent
            return RouteDecision(
                route=ROUTE_TICKET_CONTEXT_RELATED,
                intent=intent,
                sub_intent=intent,
                confidence=0.9 if state.get("apply_id") or state.get("apply_context") else 0.78,
                requires_ticket_context=True,
                can_answer_from_knowledge=not bool(ticket_intent),
                reason="ticket_context_related",
                risk_level=risk["level"],
                entities=entities,
            )

        if knowledge_intent == INTENT_CLARIFICATION:
            return RouteDecision(
                route=ROUTE_CLARIFICATION,
                intent=INTENT_CLARIFICATION,
                sub_intent=INTENT_CLARIFICATION,
                confidence=0.38,
                requires_ticket_context=False,
                can_answer_from_knowledge=False,
                reason="low_route_confidence",
                risk_level=risk["level"],
                entities=entities,
            )

        return RouteDecision(
            route=ROUTE_TICKET_CONTEXT_UNRELATED,
            intent=knowledge_intent,
            sub_intent=knowledge_intent,
            confidence=0.72 if knowledge_intent != INTENT_KNOWLEDGE_QA else 0.58,
            requires_ticket_context=False,
            can_answer_from_knowledge=True,
            reason="knowledge_question",
            risk_level=risk["level"],
            entities=entities,
        )


class LLMIntentRouter:
    name = "llm"

    def __init__(self, llm: Any, *, fallback_router: IntentRouter | None = None) -> None:
        self.llm = llm
        self.fallback_router = fallback_router or IntentRouter()

    def route(self, state: dict[str, Any]) -> RouteDecision:
        query = str(state.get("normalized_query") or state.get("query") or "")
        entities = extract_entities(query, state)
        risk = risk_check(query)
        if risk["level"] == "block":
            return RouteDecision(
                route=ROUTE_REFUSE,
                intent=INTENT_UNSAFE,
                sub_intent=INTENT_UNSAFE,
                confidence=0.99,
                requires_ticket_context=False,
                can_answer_from_knowledge=False,
                reason=risk["reason"],
                risk_level=risk["level"],
                entities={**entities, "router_provider": "risk_guard"},
            )

        try:
            payload = self._invoke_llm(state, entities)
            return self._decision_from_payload(payload, state=state, fallback_entities=entities, risk=risk)
        except Exception as exc:
            fallback = self.fallback_router.route(state)
            return RouteDecision(
                route=fallback.route,
                intent=fallback.intent,
                sub_intent=fallback.sub_intent,
                confidence=min(float(fallback.confidence), 0.55),
                requires_ticket_context=fallback.requires_ticket_context,
                can_answer_from_knowledge=fallback.can_answer_from_knowledge,
                reason=f"llm_router_fallback:{type(exc).__name__}:{fallback.reason}",
                risk_level=fallback.risk_level,
                entities={**fallback.entities, "router_provider": "rule_fallback"},
            )

    def _invoke_llm(self, state: dict[str, Any], entities: dict[str, Any]) -> dict[str, Any]:
        prompt = build_router_prompt(state, entities)
        response = self.llm.invoke(prompt)
        content = getattr(response, "content", response)
        return parse_json_object(str(content))

    def _decision_from_payload(
        self,
        payload: dict[str, Any],
        *,
        state: dict[str, Any],
        fallback_entities: dict[str, Any],
        risk: dict[str, str],
    ) -> RouteDecision:
        route = normalize_route(payload.get("route"))
        intent = derive_internal_intent(route, state)
        sub_intent = intent
        confidence = clamp_float(payload.get("confidence"), default=0.0)
        requires_ticket_context = bool(payload.get("requires_ticket_context"))
        can_answer_from_knowledge = bool(payload.get("can_answer_from_knowledge", route == ROUTE_TICKET_CONTEXT_UNRELATED))
        reason = str(payload.get("reason") or "llm_router").strip()[:120]
        entities = normalize_entities(fallback_entities, state)
        route, intent, sub_intent, requires_ticket_context, can_answer_from_knowledge, reason = enforce_route_contract(
            route=route,
            intent=intent,
            sub_intent=sub_intent,
            requires_ticket_context=requires_ticket_context,
            can_answer_from_knowledge=can_answer_from_knowledge,
            reason=reason,
        )
        return RouteDecision(
            route=route,
            intent=intent,
            sub_intent=sub_intent,
            confidence=confidence,
            requires_ticket_context=requires_ticket_context,
            can_answer_from_knowledge=can_answer_from_knowledge,
            reason=reason,
            risk_level=risk["level"],
            entities={**entities, "router_provider": "llm"},
        )


def risk_check(query: str) -> dict[str, str]:
    if contains_any(query, ["伪造", "造假", "假材料", "绕过审核", "买卖证件", "冒名"]):
        return {"level": "block", "reason": "unsafe_or_out_of_scope"}
    if contains_any(query, ["身份证号", "手机号", "个人隐私", "他人申请"]):
        return {"level": "caution", "reason": "privacy_sensitive"}
    return {"level": "pass", "reason": ""}


def ticket_context_related(query: str, state: dict[str, Any]) -> bool:
    if personal_ticket_signal(query):
        return True
    if not (state.get("apply_id") or state.get("apply_context")):
        return False
    if explicit_other_business_query(query, state):
        return False
    return contains_any(query, ["这个", "该事项", "当前", "这项", "办这个", "申请", "工单", "办件", "材料", "流程", "条件", "费用", "入口", "模板", "进度", "状态", "退回", "驳回", "补正"])


def personal_ticket_signal(query: str) -> bool:
    return contains_any(query, ["我的", "我这个", "这单", "申请单", "办件", "订单", "提交的", "被退回了", "被驳回了"])


def explicit_other_business_query(query: str, state: dict[str, Any]) -> bool:
    current_biz_type = str((state.get("apply_context") or {}).get("biz_type") or state.get("biz_type") or "")
    requested_biz_type = infer_biz_type(query)
    return bool(current_biz_type and requested_biz_type and requested_biz_type != current_biz_type)


def classify_ticket_intent(query: str) -> str:
    if contains_any(query, ["退回", "驳回", "补正", "不通过", "被拒", "被退"]):
        return INTENT_REJECTION_REASON
    if contains_any(query, ["进度", "到哪", "状态", "审核到", "办到哪", "多久能办完", "什么情况", "怎么样"]):
        return INTENT_TICKET_STATUS
    return ""


def classify_knowledge_intent(query: str) -> str:
    if contains_any(query, ["材料", "要带", "提交什么", "需要什么", "清单"]):
        return INTENT_MATERIAL_REQUIREMENTS
    if contains_any(query, ["流程", "步骤", "怎么办", "如何办理", "怎么申请"]):
        return INTENT_PROCESS_STEPS
    if contains_any(query, ["条件", "资格", "能不能办", "是否可以"]):
        return INTENT_ELIGIBILITY
    if contains_any(query, ["费用", "收费", "多少钱", "时限", "多久"]):
        return INTENT_FEE_TIME_LIMIT
    if contains_any(query, ["地点", "窗口", "地址", "哪里办"]):
        return INTENT_OFFICE_LOCATION
    if contains_any(query, ["电话", "咨询", "投诉"]):
        return INTENT_CONTACT
    if contains_any(query, ["网上办", "入口", "链接", "在线"]):
        return INTENT_ONLINE_APPLY
    if contains_any(query, ["依据", "政策", "法规", "文件"]):
        return INTENT_POLICY_BASIS
    if contains_any(query, ["表格", "模板", "样表", "申请表"]):
        return INTENT_FORM_TEMPLATE
    if len(query.strip()) < 3:
        return INTENT_CLARIFICATION
    return INTENT_KNOWLEDGE_QA


def extract_entities(query: str, state: dict[str, Any]) -> dict[str, Any]:
    entities = {
        "apply_id": state.get("apply_id") or "",
        "biz_type": state.get("biz_type") or infer_biz_type(query) or "",
        "region_code": state.get("region_code") or "",
    }
    match = re.search(r"\b[A-Za-z]\d{3,}\b", query)
    if match and not entities["apply_id"]:
        entities["apply_id"] = match.group(0)
    return {key: value for key, value in entities.items() if value not in ("", None)}


def infer_biz_type(query: str) -> str | None:
    if "健康证" in query:
        return "health_certificate"
    if "居住证" in query or "居住证明" in query:
        return "residence_permit"
    if "身份证" in query:
        return "id_card_replace"
    return None


def ticket_title_from_biz_type(biz_type: str) -> str:
    names = {
        "health_certificate": "健康证办理",
        "residence_permit": "居住证办理",
        "id_card_replace": "身份证换领",
    }
    return names.get(str(biz_type or ""), str(biz_type or "政务服务事项"))


def contains_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in str(text or "") for keyword in keywords)


def build_router_prompt(state: dict[str, Any], entities: dict[str, Any]) -> str:
    current_ticket = build_current_ticket_context(state)
    request_context = {
        "query": state.get("query") or "",
        "normalized_query": state.get("normalized_query") or "",
        "region_code": state.get("region_code") or "",
        "request_biz_type": state.get("biz_type") or "",
        "pre_extracted_entities": entities,
        "current_ticket": current_ticket,
    }
    return f"{ROUTER_SYSTEM_PROMPT}\n\n请求上下文 JSON:\n{json.dumps(request_context, ensure_ascii=False)}"


def build_current_ticket_context(state: dict[str, Any]) -> dict[str, Any]:
    apply_context = state.get("apply_context") or {}
    biz_type = apply_context.get("biz_type") or state.get("biz_type") or ""
    return compact_dict(
        {
            "has_bound_ticket": bool(state.get("apply_id") or apply_context.get("apply_id")),
            "apply_id": state.get("apply_id") or apply_context.get("apply_id") or "",
            "biz_type": biz_type,
            "title": apply_context.get("title") or ticket_title_from_biz_type(str(biz_type or "")),
            "region_code": apply_context.get("region_code") or state.get("region_code") or "",
            "status": apply_context.get("status") or "",
            "current_node": apply_context.get("current_node") or "",
            "node_name": apply_context.get("node_name") or "",
            "reject_reason": apply_context.get("reject_reason") or "",
            "material_name": apply_context.get("material_name") or "",
            "audit_node": apply_context.get("audit_node") or "",
        }
    )


def compact_dict(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value not in ("", None, [], {})}


def parse_json_object(value: str) -> dict[str, Any]:
    text = value.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("LLM router response must be a JSON object")
    return parsed


def normalize_route(value: Any) -> str:
    route = str(value or "").strip()
    if route not in ALLOWED_ROUTES:
        raise ValueError(f"Unsupported route: {route}")
    return route


def derive_internal_intent(route: str, state: dict[str, Any]) -> str:
    query = str(state.get("normalized_query") or state.get("query") or "")
    if route == ROUTE_REFUSE:
        return INTENT_UNSAFE
    if route == ROUTE_CLARIFICATION:
        return INTENT_CLARIFICATION
    ticket_intent = classify_ticket_intent(query)
    if route == ROUTE_TICKET_CONTEXT_RELATED:
        return ticket_intent or classify_knowledge_intent(query)
    if ticket_intent == INTENT_REJECTION_REASON:
        return INTENT_CORRECTION_GUIDANCE
    return classify_knowledge_intent(query)


def normalize_entities(values: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    query = str(state.get("normalized_query") or state.get("query") or "")
    entities = {
        "apply_id": values.get("apply_id") or state.get("apply_id") or "",
        "biz_type": values.get("biz_type") or state.get("biz_type") or infer_biz_type(query) or "",
        "region_code": values.get("region_code") or state.get("region_code") or "",
    }
    return {key: value for key, value in entities.items() if value not in ("", None)}


def enforce_route_contract(
    *,
    route: str,
    intent: str,
    sub_intent: str,
    requires_ticket_context: bool,
    can_answer_from_knowledge: bool,
    reason: str,
) -> tuple[str, str, str, bool, bool, str]:
    if route == ROUTE_TICKET_CONTEXT_RELATED:
        requires_ticket_context = True
        can_answer_from_knowledge = intent not in {INTENT_TICKET_STATUS, INTENT_REJECTION_REASON, INTENT_UNSAFE, INTENT_CLARIFICATION}
    elif route == ROUTE_TICKET_CONTEXT_UNRELATED:
        requires_ticket_context = False
        can_answer_from_knowledge = True
        if intent in {INTENT_TICKET_STATUS, INTENT_REJECTION_REASON, INTENT_UNSAFE, INTENT_CLARIFICATION}:
            intent = INTENT_KNOWLEDGE_QA
            sub_intent = intent
            reason = append_reason(reason, "normalized_knowledge_intent")
    elif route == ROUTE_REFUSE:
        intent = INTENT_UNSAFE
        sub_intent = INTENT_UNSAFE
        requires_ticket_context = False
        can_answer_from_knowledge = False
    elif route == ROUTE_CLARIFICATION:
        intent = INTENT_CLARIFICATION
        sub_intent = INTENT_CLARIFICATION
        requires_ticket_context = False
        can_answer_from_knowledge = False
    return route, intent, sub_intent, requires_ticket_context, can_answer_from_knowledge, reason


def append_reason(reason: str, suffix: str) -> str:
    base = reason or "llm_router"
    return f"{base}|{suffix}"[:120]


def clamp_float(value: Any, *, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, number))
