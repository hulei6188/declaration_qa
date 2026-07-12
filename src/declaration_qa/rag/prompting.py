"""问答提示词和本地确定性回答生成器。"""

from __future__ import annotations

import json
from typing import Any

from declaration_qa.memory.context import format_long_term_context, format_short_term_context
from declaration_qa.rag.router import INTENT_TICKET_LIST

try:
    from langchain_core.prompts import PromptTemplate
except ModuleNotFoundError:
    class PromptTemplate:
        """未安装 langchain_core 时使用的最小兜底实现。"""

        def __init__(self, template: str) -> None:
            self.template = template

        @classmethod
        def from_template(cls, template: str) -> "PromptTemplate":
            return cls(template)

        def format(self, **kwargs: Any) -> str:
            return self.template.format(**kwargs)


ANSWER_PROMPT = PromptTemplate.from_template(
    """你是一证办申报辅助问答模块，只能基于业务工具结果和知识检索证据回答。
不能编造政策，不能承诺审核结果，不能替代人工审核结论。

问题：{query}
意图：{intent}
链路：{retrieval_chain}
短期上下文（仅用于理解多轮省略和本轮附件，不得替代工具结果或检索证据）：{short_term_context}
长期上下文记忆（仅用于理解历史咨询主题，不含实时个人工单事实；如与检索证据冲突，以本轮检索证据为准）：{long_term_context}
业务工具结果（只用于个人申请事实，不能作为公开政策依据）：{tool_payload}
知识检索证据（只用于公开政策、材料、流程和规范，不能作为实时工单事实）：{source_payload}

业务工具结果与知识检索证据必须分别判断和引用，不得把任何工单字段补写进知识依据。
对于申请进度、状态、退回或补正问题，必须按 application_id 输出该事项下全部 applications，并完整保留每个申请的 status_history，不得只展示 Agent 选中的一条申请。

请用清晰、可操作的语言回答。"""
)

class DeterministicAnswerGenerator:
    """作为安全本地 LLM 边界使用的确定性生成器。

    提示词仍通过 LangChain 构建，因此后续可替换为真实聊天模型，
    而无需改变工作流契约。
    """

    def generate(self, state: dict[str, Any]) -> tuple[str, str]:
        """生成非流式回答和提示词。"""
        sources = state.get("retrieved_sources") or []
        source_text = format_source_payload(sources)
        prompt = ANSWER_PROMPT.format(
            query=state.get("query") or "",
            intent=state.get("intent") or "",
            retrieval_chain=state.get("retrieval_chain") or "",
            short_term_context=format_short_term_context(state),
            long_term_context=format_long_term_context(state),
            tool_payload=format_tool_result_payload(state.get("tool_results") or []),
            source_payload=source_text,
        )
        intent = state.get("intent")
        if intent == INTENT_TICKET_LIST:
            answer = ticket_list_answer(state)
        elif intent in {"status", "ticket_status", "reject", "rejection_reason"}:
            answer = application_history_answer(state)
        else:
            answer = knowledge_answer(state, sources)
        return answer, prompt


def ticket_list_answer(state: dict[str, Any]) -> str:
    """列出当前用户办理过的事项和各事项申请次数。"""
    items = []
    for result in reversed(state.get("tool_results") or []):
        if not isinstance(result, dict) or result.get("tool_name") != "ticket_list_query" or not result.get("ok"):
            continue
        data = result.get("data") if isinstance(result.get("data"), dict) else {}
        items = [item for item in data.get("items") or [] if isinstance(item, dict)]
        break
    if not items:
        return "当前未查询到可查看的申请记录。"
    lines = [
        f"- {item.get('item_name') or item.get('item_code')}：{int(item.get('application_count') or 0)} 次申请"
        for item in items
    ]
    return "当前账号下查询到以下申请事项：\n" + "\n".join(lines)


def application_history_answer(state: dict[str, Any]) -> str:
    """逐个输出同一事项下每次申请的完整状态变化。"""
    applications = ticket_applications(state)
    if not applications:
        return "当前未查询到该事项的申请状态历史。"
    sections = []
    for application in applications:
        application_id = int(application.get("application_id") or 0)
        submitted_at = str(application.get("submitted_at") or "")
        current_status = str(application.get("current_status") or "")
        completed_at = str(application.get("completed_at") or "")
        header_fields = [f"application_id：{application_id}"]
        if submitted_at:
            header_fields.append(f"提交时间：{submitted_at}")
        if current_status:
            header_fields.append(f"当前状态：{current_status}")
        if completed_at:
            header_fields.append(f"办结时间：{completed_at}")
        header = "；".join(header_fields)
        history = [event for event in application.get("status_history") or [] if isinstance(event, dict)]
        lines = []
        for event in history:
            sequence_no = int(event.get("sequence_no") or 0)
            from_status = str(event.get("from_status") or "起始")
            status = str(event.get("status") or "未知")
            changed_at = str(event.get("changed_at") or "")
            node = str(event.get("node_name") or event.get("current_node") or "")
            reason_code = str(event.get("reason_code") or "")
            reason = str(event.get("reason") or "")
            material_name = str(event.get("material_name") or "")
            material_code = str(event.get("material_code") or "")
            detail_fields = [f"#{sequence_no}", changed_at, f"{from_status} -> {status}"]
            if node:
                detail_fields.append(f"节点：{node}")
            if reason:
                detail_fields.append(f"原因：{reason}")
            if reason_code:
                detail_fields.append(f"原因编码：{reason_code}")
            if material_name or material_code:
                material = material_name or material_code
                if material_name and material_code:
                    material = f"{material_name}（{material_code}）"
                detail_fields.append(f"关联材料：{material}")
            lines.append("- " + "；".join(field for field in detail_fields if field))
        sections.append(header + "\n" + ("\n".join(lines) if lines else "- 暂无状态流水"))
    return "该事项下的申请状态历史如下：\n\n" + "\n\n".join(sections)


def format_tool_result_payload(results: list[dict[str, Any]]) -> str:
    """格式化业务工具结果，不混入知识证据。"""
    safe_results = []
    for result in results:
        if not isinstance(result, dict) or not str(result.get("tool_name") or "").startswith("ticket_"):
            continue
        safe_results.append(
            {
                "tool_name": result.get("tool_name"),
                "ok": bool(result.get("ok")),
                "status": result.get("status"),
                "data": result.get("data") if isinstance(result.get("data"), dict) else {},
            }
        )
    return json.dumps(safe_results, ensure_ascii=False, default=str)


def ticket_applications(state: dict[str, Any]) -> list[dict[str, Any]]:
    """读取 ticket_detail_query 返回的全部申请。"""
    applications: dict[int, dict[str, Any]] = {}
    for result in state.get("tool_results") or []:
        if not isinstance(result, dict) or result.get("tool_name") != "ticket_detail_query" or not result.get("ok"):
            continue
        data = result.get("data") if isinstance(result.get("data"), dict) else {}
        for item in data.get("applications") or []:
            if isinstance(item, dict) and int(item.get("application_id") or 0) > 0:
                applications[int(item["application_id"])] = item
    return list(applications.values())


def knowledge_answer(state: dict[str, Any], sources: list[dict[str, Any]]) -> str:
    """生成知识检索类问题的确定性回答。"""
    if not sources:
        return "暂未查询到与当前地区和业务类型完全匹配的办理依据，建议查看办事指南或联系人工客服确认。"
    title = sources[0].get("title") or "办理指南"
    snippet = source_evidence_text(sources[0])
    return f"根据《{title}》，{snippet} 请以页面展示的材料清单和提交入口为准。"


def format_source_payload(sources: list[dict[str, Any]]) -> str:
    """用完整证据文本格式化回答生成输入。"""
    return "\n".join(
        f"- {source_task_label(item)}{item.get('title')}: {source_evidence_text(item)}"
        for item in sources
    )


def source_task_label(source: dict[str, Any]) -> str:
    """为并行 Worker 证据标注其所属子问题。"""
    metadata = source.get("metadata") if isinstance(source.get("metadata"), dict) else {}
    sub_question = str(metadata.get("sub_question") or "").strip()
    return f"[对应子问题：{sub_question}] " if sub_question else ""


def source_evidence_text(source: dict[str, Any]) -> str:
    """返回完整证据文本，缺失时退回展示摘要。"""
    text = str(source.get("evidence_text") or source.get("content") or source.get("snippet") or "")
    table_context = table_context_for_llm(source)
    if table_context:
        return f"{text}\n\n表格结构（精简）：{table_context}".strip()
    return text


def table_context_for_llm(source: dict[str, Any]) -> str:
    """返回表格证据的精简结构上下文。"""
    metadata = source.get("metadata") if isinstance(source.get("metadata"), dict) else {}
    semantic_type = str(metadata.get("chunk_type") or "")
    if semantic_type not in {"attachment_table", "docling_table", "table_parent", "table_summary", "table_row", "table_kv"}:
        return ""
    context = metadata.get("llm_table_context")
    if not context:
        return ""
    try:
        return json.dumps(context, ensure_ascii=False, default=str)[:3000]
    except (TypeError, ValueError):
        return str(context)[:3000]
