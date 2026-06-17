"""申报问答共享路由常量和轻量规则工具。"""

from __future__ import annotations

import json
import re
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

def risk_check(query: str) -> dict[str, str]:
    """识别请求中的高风险表达。"""
    if contains_any(query, ["伪造", "造假", "假材料", "绕过审核", "买卖证件", "冒名"]):
        return {"level": "block", "reason": "unsafe_or_out_of_scope"}
    if contains_any(query, ["身份证号", "手机号", "个人隐私", "他人申请"]):
        return {"level": "caution", "reason": "privacy_sensitive"}
    return {"level": "pass", "reason": ""}


def ticket_context_related(query: str, state: dict[str, Any]) -> bool:
    """判断问题是否与当前办件上下文相关。"""
    if personal_ticket_signal(query):
        return True
    if not (state.get("apply_id") or state.get("apply_context")):
        return False
    if explicit_other_business_query(query, state):
        return False
    return contains_any(query, ["这个", "该事项", "当前", "这项", "办这个", "申请", "工单", "办件", "材料", "流程", "条件", "费用", "入口", "模板", "进度", "状态", "退回", "驳回", "补正"])


def personal_ticket_signal(query: str) -> bool:
    """判断问题是否包含个人办件信号。"""
    return contains_any(query, ["我的", "我这个", "这单", "申请单", "办件", "订单", "提交的", "被退回了", "被驳回了"])


def explicit_other_business_query(query: str, state: dict[str, Any]) -> bool:
    """判断用户是否显式询问其他业务。"""
    current_biz_type = str((state.get("apply_context") or {}).get("biz_type") or state.get("biz_type") or "")
    requested_biz_type = infer_biz_type(query)
    return bool(current_biz_type and requested_biz_type and requested_biz_type != current_biz_type)


def classify_ticket_intent(query: str) -> str:
    """按规则识别办件类意图。"""
    if contains_any(query, ["退回", "驳回", "补正", "不通过", "被拒", "被退"]):
        return INTENT_REJECTION_REASON
    if contains_any(query, ["进度", "到哪", "状态", "审核到", "办到哪", "多久能办完", "什么情况", "怎么样"]):
        return INTENT_TICKET_STATUS
    return ""


def classify_knowledge_intent(query: str) -> str:
    """按规则识别知识问答类意图。"""
    if contains_any(query, ["材料", "要带", "提交什么", "需要什么", "清单"]):
        return INTENT_MATERIAL_REQUIREMENTS
    if contains_any(query, ["流程", "步骤", "怎么办", "如何办理", "怎么申请"]):
        return INTENT_PROCESS_STEPS
    if contains_any(query, ["条件", "资格", "能不能办", "是否可以"]):
        return INTENT_ELIGIBILITY
    if contains_any(query, ["费用", "收费", "多少钱", "时限", "多久", "几天", "多少天", "拿结果", "办结"]):
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
    """从问题和上下文中抽取路由实体。"""
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
    """根据文本推断业务类型。"""
    if "健康证" in query:
        return "health_certificate"
    if "居住证" in query or "居住证明" in query:
        return "residence_permit"
    if "身份证" in query:
        return "id_card_replace"
    return None


def ticket_title_from_biz_type(biz_type: str) -> str:
    """根据业务类型生成办件名称。"""
    names = {
        "health_certificate": "健康证办理",
        "residence_permit": "居住证办理",
        "id_card_replace": "身份证换领",
    }
    return names.get(str(biz_type or ""), str(biz_type or "政务服务事项"))


def contains_any(text: str, keywords: list[str]) -> bool:
    """判断文本是否包含任一关键词。"""
    return any(keyword in str(text or "") for keyword in keywords)


def parse_json_object(value: str) -> dict[str, Any]:
    """从模型输出中解析 JSON 对象。"""
    text = value.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("LLM JSON response must be an object")
    return parsed
