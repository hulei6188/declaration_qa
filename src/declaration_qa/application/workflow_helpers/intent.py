"""工作流意图规则和文本安全辅助函数。"""

from __future__ import annotations

import re

from declaration_qa.rag.router import (
    INTENT_CORRECTION_GUIDANCE,
    INTENT_REJECTION_REASON,
    INTENT_TICKET_STATUS,
)


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
    """从 QuestionResolver 任务字段派生 intent 展示值。"""
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
