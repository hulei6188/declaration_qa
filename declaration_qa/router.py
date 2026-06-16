from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


ROUTE_KNOWLEDGE = "knowledge_only"
ROUTE_TICKET = "ticket_aware"
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

ALLOWED_ROUTES = {ROUTE_KNOWLEDGE, ROUTE_TICKET, ROUTE_CLARIFICATION, ROUTE_REFUSE}
TICKET_INTENTS = {INTENT_TICKET_STATUS, INTENT_REJECTION_REASON}

ROUTER_SYSTEM_PROMPT = """你是政务服务智能问答系统的请求路由器，只能返回一个 JSON 对象。
你的任务是判断用户问题应该走知识检索线还是工单增强线，不生成业务答案。

可选 route:
- knowledge_only: 通用政策、材料、流程、FAQ、入口、费用、地点、模板等知识问题。
- ticket_aware: 需要查询用户真实工单状态、驳回原因、补正要求的问题。
- clarification: 问题过短、上下文不足、无法判断，或工单问题与当前工单无关。
- refuse: 违法违规、伪造材料、绕过审核、冒用他人信息等风险问题。

决策要求:
1. 当前有工单不代表一定使用工单。只有问题依赖真实工单事实时才 route=ticket_aware。
2. ticket_aware 仅用于用户真实工单状态、驳回原因或补正要求问题。
3. 只输出下方 JSON 字段，不要添加其他分类字段；系统会在路由后内部推导处理类型。

JSON 字段:
{
  "route": "knowledge_only|ticket_aware|clarification|refuse",
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

        ticket_signal = ticket_context_signal(query, state)
        ticket_intent = classify_ticket_intent(query)
        if ticket_intent and ticket_signal:
            return RouteDecision(
                route=ROUTE_TICKET,
                intent=ticket_intent,
                sub_intent=ticket_intent,
                confidence=0.9 if state.get("apply_id") or state.get("apply_context") else 0.78,
                requires_ticket_context=True,
                can_answer_from_knowledge=False,
                reason="personal_ticket_question",
                risk_level=risk["level"],
                entities=entities,
            )

        knowledge_intent = classify_knowledge_intent(query)
        if ticket_intent and not ticket_signal:
            knowledge_intent = INTENT_CORRECTION_GUIDANCE if ticket_intent == INTENT_REJECTION_REASON else knowledge_intent
            return RouteDecision(
                route=ROUTE_KNOWLEDGE,
                intent=knowledge_intent,
                sub_intent=knowledge_intent,
                confidence=0.68,
                requires_ticket_context=False,
                can_answer_from_knowledge=True,
                reason="generic_ticket_keyword_without_personal_context",
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
            route=ROUTE_KNOWLEDGE,
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
        can_answer_from_knowledge = bool(payload.get("can_answer_from_knowledge", route == ROUTE_KNOWLEDGE))
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


def ticket_context_signal(query: str, state: dict[str, Any]) -> bool:
    if state.get("apply_id") or state.get("apply_context"):
        return True
    return contains_any(query, ["我的", "我这个", "申请单", "办件", "订单", "提交的", "被退回了", "被驳回了"])


def classify_ticket_intent(query: str) -> str:
    if contains_any(query, ["退回", "驳回", "补正", "不通过", "被拒", "被退"]):
        return INTENT_REJECTION_REASON
    if contains_any(query, ["进度", "到哪", "状态", "审核到", "办到哪", "多久能办完"]):
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


def contains_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in str(text or "") for keyword in keywords)


def build_router_prompt(state: dict[str, Any], entities: dict[str, Any]) -> str:
    apply_context = state.get("apply_context") or {}
    current_ticket = {
        "has_apply_id": bool(state.get("apply_id") or apply_context.get("apply_id")),
        "apply_id": state.get("apply_id") or apply_context.get("apply_id") or "",
        "biz_type": apply_context.get("biz_type") or state.get("biz_type") or "",
        "current_node": apply_context.get("current_node") or "",
        "status": apply_context.get("status") or "",
    }
    request_context = {
        "query": state.get("query") or "",
        "normalized_query": state.get("normalized_query") or "",
        "region_code": state.get("region_code") or "",
        "request_biz_type": state.get("biz_type") or "",
        "pre_extracted_entities": entities,
        "current_ticket": current_ticket,
    }
    return f"{ROUTER_SYSTEM_PROMPT}\n\n请求上下文 JSON:\n{json.dumps(request_context, ensure_ascii=False)}"


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
    if route == ROUTE_TICKET:
        return ticket_intent or INTENT_TICKET_STATUS
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
    if route == ROUTE_TICKET and intent not in TICKET_INTENTS:
        intent = INTENT_REJECTION_REASON if intent == INTENT_CORRECTION_GUIDANCE else INTENT_TICKET_STATUS
        sub_intent = intent
        reason = append_reason(reason, "normalized_ticket_intent")
    if route == ROUTE_TICKET:
        requires_ticket_context = True
        can_answer_from_knowledge = False
    elif route == ROUTE_KNOWLEDGE:
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
