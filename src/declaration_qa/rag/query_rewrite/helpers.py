"""两阶段 Query Rewrite 辅助函数。"""

from __future__ import annotations

import re
from typing import Any

from declaration_qa.rag.query_rewrite.schema import DEFAULT_RETRIEVAL_TOP_K
from declaration_qa.rag.retrieval_plan import infer_retrieval_plan
from declaration_qa.rag.router import ticket_context_related


KNOWN_TOPIC_TERMS = (
    "身份证",
    "居住证",
    "健康证",
    "取水许可",
    "草原防火许可",
    "草原防火期内在草原上进行爆破、勘察和施工等活动审批",
)


INTENT_REWRITE_TERMS: dict[str, list[str]] = {
    "material_requirements": [
        "申请材料 材料清单 必要材料 提交材料",
        "原件 复印件 电子材料 材料要求",
    ],
    "process_steps": [
        "办理流程 办理环节 申请流程 办事指南",
        "受理 审查 决定 办结",
    ],
    "eligibility": [
        "办理条件 申请条件 受理条件",
        "申请人资格 办理对象",
    ],
    "fee_time_limit": [
        "法定办结时限 承诺办结时限 办理时限",
        "多久办结 多长时间 几天拿结果 审批结果",
    ],
    "office_location": [
        "办理地点 服务窗口 地址",
        "窗口地址 办事大厅",
    ],
    "contact": [
        "咨询电话 投诉电话 监督电话 联系方式",
        "咨询方式 投诉监督",
    ],
    "online_apply": [
        "网上办理 在线办理 申报入口",
        "安徽政务服务网 线上提交",
    ],
    "policy_basis": [
        "政策依据 法律依据 法规依据",
        "设定依据 办理依据",
    ],
    "form_template": [
        "申请表 表格 模板 样表",
        "空白表格 示例样表",
    ],
    "knowledge_qa": [
        "办事指南 申请要求",
    ],
    "other": [],
}


QUERY_TERM_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("material_requirements", ("材料", "要带", "提交什么", "需要什么", "清单", "补什么", "上传什么")),
    ("process_steps", ("流程", "步骤", "怎么办", "如何办理", "怎么申请", "办理环节")),
    ("eligibility", ("条件", "资格", "能不能办", "是否可以")),
    ("fee_time_limit", ("费用", "收费", "多少钱", "时限", "多久", "多长时间", "几天", "多少天", "拿结果", "办结")),
    ("office_location", ("地点", "窗口", "地址", "哪里办")),
    ("contact", ("电话", "咨询", "投诉", "联系方式")),
    ("online_apply", ("网上办", "入口", "链接", "在线")),
    ("policy_basis", ("依据", "政策", "法规", "文件")),
    ("form_template", ("表格", "模板", "样表", "申请表")),
)


def question_item(query: str, state: dict[str, Any]) -> dict[str, Any]:
    """构建子问题条目。"""
    normalized = normalize_text(query).strip("。；;？?")
    retrieval_query = contextualized_retrieval_query(normalized, state)
    is_expanded = retrieval_query != normalized
    requires_ticket_context = ticket_context_related(normalized, {**state, "query": normalized, "normalized_query": normalized})
    ticket_fact_type = normalize_ticket_fact_type("", normalized, requires_ticket_context=requires_ticket_context)
    return {
        "id": "",
        "original_query": normalized,
        "query": normalized,
        "retrieval_query": retrieval_query,
        "requires_ticket_context": requires_ticket_context,
        "ticket_fact_type": ticket_fact_type,
        "retrieval_plan": infer_retrieval_plan(ticket_fact_type=ticket_fact_type),
        "retrieval_queries": [],
        "is_expanded": is_expanded,
        "expansion_reason": "session_context_rewrite" if is_expanded else "",
    }


def split_questions(query: str) -> list[str]:
    """保守拆分复合中文问题。"""
    text = normalize_text(query).strip("。；;？?")
    if not text:
        return []
    primary = [part for part in re.split(r"[？?；;]\s*", text) if part.strip()]
    if len(primary) > 1:
        return numbered_questions(primary)
    secondary = re.split(r"\s*(?:，|,)?(?:另外|还有|以及|并且|同时|顺便|再问下|再问一下)\s*", text)
    parts = [part for part in secondary if part.strip()]
    if len(parts) > 1 and has_distinct_question_signals(parts):
        return numbered_questions(parts)
    comma_parts = [part for part in re.split(r"[，,]\s*", text) if part.strip()]
    if len(comma_parts) > 1 and has_distinct_question_signals(comma_parts):
        return numbered_questions(comma_parts)
    return [text]


def has_distinct_question_signals(parts: list[str]) -> bool:
    """返回拆分片段是否像不同问题。"""
    signals = ["材料", "流程", "条件", "费用", "收费", "多久", "时限", "进度", "状态", "驳回", "补正", "干嘛", "是什么"]
    return sum(1 for part in parts if any(signal in part for signal in signals)) >= 2


def numbered_questions(parts: list[str]) -> list[str]:
    """清理并去重拆分后问题。"""
    output = []
    seen = set()
    for part in parts:
        item = normalize_text(part).strip("。；;？?")
        item = re.sub(r"^(?:问题)?\s*\d+[.、:：]\s*", "", item)
        if not item or item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def normalize_ticket_fact_type(value: str, query: str, *, requires_ticket_context: bool) -> str:
    """归一化办件事实类型。"""
    allowed = {"status", "rejection", "correction", "none"}
    normalized = str(value or "").strip().lower()
    if normalized in allowed:
        return normalized if requires_ticket_context else "none"
    if not requires_ticket_context:
        return "none"
    text = str(query or "")
    if any(keyword in text for keyword in ["退回", "驳回", "不通过", "被拒", "被退", "为什么"]):
        return "rejection"
    if any(keyword in text for keyword in ["补正", "补什么", "还要补", "修改", "重新提交"]):
        return "correction"
    if any(keyword in text for keyword in ["进度", "到哪", "状态", "审核到", "办到哪", "什么情况", "怎么样"]):
        return "status"
    return "none"


def normalize_text(value: Any) -> str:
    """将文本空白归一化为紧凑格式。"""
    return re.sub(r"\s+", " ", str(value or "").strip())


def contextualized_retrieval_query(query: str, state: dict[str, Any]) -> str:
    """用会话短期上下文补全省略式检索问题。"""
    normalized = normalize_text(query).strip("。；;？?")
    if not needs_context_topic(normalized) or extract_topic(normalized):
        return normalized
    topic = current_context_topic(state)
    if not topic or topic in normalized:
        return normalized
    return join_query_parts(topic, normalized)


def needs_context_topic(query: str) -> bool:
    """判断当前问题是否像省略事项的追问。"""
    text = str(query or "")
    if not text:
        return False
    if any(keyword in text for keyword in ["这个", "该事项", "这项", "办这个", "申请", "办理"]):
        return True
    return any(
        keyword in text
        for keyword in ["多久", "多长时间", "几天", "收费", "材料", "流程", "条件", "电话", "地点", "入口", "怎么弄", "怎么办"]
    )


def current_context_topic(state: dict[str, Any]) -> str:
    """按优先级从当前工单、最近会话和短期摘要中提取事项主题。"""
    apply_context = state.get("apply_context") if isinstance(state.get("apply_context"), dict) else {}
    payload = apply_context.get("payload") if isinstance(apply_context.get("payload"), dict) else {}
    for value in [
        apply_context.get("title"),
        apply_context.get("item_name"),
        payload.get("official_item_name"),
        payload.get("item_name"),
        payload.get("title"),
        payload.get("catalog_name"),
    ]:
        topic = extract_topic(str(value or ""))
        if topic:
            return topic
    current_query = normalize_text(state.get("normalized_query") or state.get("query") or "")
    for message in reversed(list(state.get("session_messages") or [])):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = normalize_text(message.get("content") or "")
        if not content or content == current_query:
            continue
        topic = extract_topic(content)
        if topic:
            return topic
    short_memory = state.get("short_memory") if isinstance(state.get("short_memory"), dict) else {}
    topic = extract_topic(short_memory.get("summary") or "")
    return topic


def extract_topic(text: str) -> str:
    """从中文问句中提取适合作为检索范围的事项主题。"""
    normalized = normalize_text(text)
    if not normalized:
        return ""
    for term in KNOWN_TOPIC_TERMS:
        if term in normalized:
            return term
    patterns = [
        r"([\u4e00-\u9fa5A-Za-z0-9]{2,40}?(?:许可|审批|备案|登记|认定|采集|检验|申请|证明|证书|证))",
        r"([\u4e00-\u9fa5A-Za-z0-9]{2,30}?)(?:如何办理|怎么办理|怎么办|怎么申请|办理要|需要什么|要什么|多久|收费)",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            topic = match.group(1).strip("，,。；;：: 的")
            if len(topic) >= 2 and not any(word in topic for word in ["办理", "申请材料", "流程"]):
                return topic
    return ""


def normalize_strategy_name(value: Any) -> str:
    """归一化策略名称。"""
    text = re.sub(r"[^a-zA-Z0-9_:-]+", "_", str(value or "stage2_rewrite").strip().lower())
    return text[:40] or "stage2_rewrite"


def normalize_top_k(value: Any) -> int:
    """限制单条查询的 top_k。"""
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = DEFAULT_RETRIEVAL_TOP_K
    return max(1, min(6, number))


def join_query_parts(*parts: Any) -> str:
    """拼接非空查询片段。"""
    return normalize_text(" ".join(str(part) for part in parts if str(part or "").strip()))


def selected_ticket_scope_text(state: dict[str, Any]) -> str:
    """从当前办件上下文提取事项范围。"""
    apply_context = state.get("apply_context") or {}
    if not isinstance(apply_context, dict) or not apply_context:
        return ""
    payload = apply_context.get("payload") if isinstance(apply_context.get("payload"), dict) else {}
    parts = [
        ticket_title(apply_context, state),
        payload.get("official_item_name"),
        apply_context.get("item_name"),
        payload.get("item_name"),
        apply_context.get("title"),
        payload.get("title"),
        payload.get("catalog_name"),
        apply_context.get("source_file"),
        payload.get("source_file"),
        apply_context.get("item_code"),
        payload.get("item_code"),
        apply_context.get("biz_type"),
    ]
    return normalize_text(" ".join(unique_strings([str(item) for item in parts if item])))


def has_item_scope_filter(filters: dict[str, Any]) -> bool:
    """返回元数据过滤条件是否已包含事项范围。"""
    return any(filters.get(key) for key in ("item_code", "ssqd_code", "ssqd_id", "item_name"))


def selected_ticket_fact_text(state: dict[str, Any]) -> str:
    """提取允许用于检索的办件事实。"""
    ticket_fact_type = str(state.get("ticket_fact_type") or "none")
    if ticket_fact_type == "status":
        business = state.get("business_status") or state.get("business_snapshot") or {}
        parts = [business.get("node_name"), business.get("current_node"), business.get("status")]
    elif ticket_fact_type in {"rejection", "correction"}:
        business = state.get("reject_info") or state.get("business_snapshot") or {}
        parts = [
            business.get("reject_reason"),
            business.get("material_name"),
            business.get("reject_code"),
            business.get("material_code"),
            business.get("audit_node"),
        ]
    else:
        parts = []
    return normalize_text(" ".join(str(item) for item in parts if item))


def ticket_title(*sources: dict[str, Any]) -> str:
    """提取当前事项标题。"""
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in ("title", "item_name", "catalog_name"):
            title = str(source.get(key) or "").strip()
            if title:
                return title
        payload = source.get("payload") if isinstance(source.get("payload"), dict) else {}
        for key in ("official_item_name", "item_name", "title", "catalog_name"):
            title = str(payload.get(key) or "").strip()
            if title:
                return title
    for source in sources:
        if isinstance(source, dict):
            title = str(source.get("biz_type") or "").strip()
            if title:
                return title
    return ""


def unique_strings(values: list[str]) -> list[str]:
    """按首次出现顺序去重字符串。"""
    seen = set()
    output = []
    for value in values:
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output
