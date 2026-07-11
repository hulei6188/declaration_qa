"""申报问答共享路由常量和轻量规则工具。"""

from __future__ import annotations

import json
import re
from typing import Any


ROUTE_KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"

INTENT_UNSAFE = "unsafe"
INTENT_TICKET_LIST = "ticket_list"
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

def classify_ticket_intent(query: str) -> str:
    """按规则识别办件类意图。"""
    if contains_any(query, ["退回", "驳回", "补正", "不通过", "被拒", "被退"]):
        return INTENT_REJECTION_REASON
    if contains_any(query, ["我的办件", "我的申请记录", "历史申请", "申请列表", "我办过哪些", "我提交过哪些"]):
        return INTENT_TICKET_LIST
    if contains_any(query, ["进度", "到哪", "状态", "审核到", "办到哪", "多久能办完", "什么情况", "怎么样"]):
        return INTENT_TICKET_STATUS
    if "我的申请" in query and not contains_any(query, ["材料", "流程", "条件", "费用", "收费", "时限", "模板", "怎么填"]):
        return INTENT_TICKET_LIST
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


def infer_biz_type(query: str) -> str | None:
    """根据文本推断业务类型。"""
    if "健康证" in query:
        return "health_certificate"
    if "居住证" in query or "居住证明" in query:
        return "residence_permit"
    if "身份证" in query:
        return "id_card_replace"
    return None


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
