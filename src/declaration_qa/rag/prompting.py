"""问答提示词和本地确定性回答生成器。"""

from __future__ import annotations

import hashlib
import json
from html import escape
from typing import Any

from declaration_qa.memory.context import format_long_term_context, format_short_term_context

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
    """你是一证办申报辅助问答模块，只能基于业务查询结果和检索证据回答。
不能编造政策，不能承诺审核结果，不能替代人工审核结论。

问题：{query}
意图：{intent}
链路：{retrieval_chain}
短期上下文（仅用于理解多轮省略、当前工单和附件，不得替代检索证据作为政策依据）：{short_term_context}
长期上下文记忆（仅用于理解用户历史偏好、历史咨询事实和事项延续；如与检索证据冲突，以检索证据和业务数据为准）：{long_term_context}
业务快照：{business_snapshot}
业务数据：{business_payload}
检索证据：{source_payload}

请用清晰、可操作的语言回答。"""
)

MULTI_QUESTION_ANSWER_PROMPT = PromptTemplate.from_template(
    """你是一个极其严格的一证办申报辅助问答模块，只能基于按子问题分组的业务查询结果和检索证据回答。
不能编造政策，不能承诺审核结果，不能替代人工审核结论。
请严格基于以下按问题划分的证据回答，禁止跨越标签范围套用数据。
不得跨子问题挪用证据；不得新增证据中没有的材料、时限、费用、条件或审核结论。
如果某个子问题需要用户选择工单，只能提示先选择工单，不要推测进度或驳回原因。
如果某个子问题证据不足，只能说明暂未查询到可靠依据。

【重要规则：指针解码】
下方的每个子问题（SubQuery）中仅包含证据的引用标签（如 <Ref>chunk_a7b1c2d3</Ref>）。
在回答该子问题时，你必须前往顶部的 <SharedEvidencePool> 中，找到 id 与 Ref 文本完全一致的 <Evidence> 标签，将其内部的具体文本作为你回答该问题的唯一合法依据。
严禁跨越该子问题指定的 <EvidenceRefs> 去偷看、引用或套用其他未授权的证据 id。
不要向用户输出 Ref、Evidence id 或 XML 标签本身，只输出自然语言答案。

用户原始问题：{query}
短期上下文（仅用于理解多轮省略、当前工单和附件，不得替代各子问题证据）：{short_term_context}
长期上下文记忆（仅用于理解用户历史偏好、历史咨询事实和事项延续；如与各子问题证据冲突，以检索证据和业务数据为准）：{long_term_context}
{sub_task_payload}

请按用户自然提问方式组织最终答案。
如果多个子问题属于同一事项或同一办事情境，应合并成连贯段落回答，不要机械输出“1、2”编号。
只有多个问题彼此明显无关时，才使用编号分点。
对于证据不足的部分，用一句自然提醒说明，避免生硬堆叠“暂未查询到可靠依据”。"""
)


class DeterministicAnswerGenerator:
    """作为安全本地 LLM 边界使用的确定性生成器。

    提示词仍通过 LangChain 构建，因此后续可替换为真实聊天模型，
    而无需改变工作流契约。
    """

    def generate(self, state: dict[str, Any]) -> tuple[str, str]:
        """生成非流式回答和提示词。"""
        if state.get("multi_question"):
            return multi_question_answer(state)
        sources = state.get("retrieved_sources") or []
        source_text = format_source_payload(sources)
        business_payload = state.get("business_status") or state.get("reject_info") or {}
        prompt = ANSWER_PROMPT.format(
            query=state.get("query") or "",
            intent=state.get("intent") or "",
            retrieval_chain=state.get("retrieval_chain") or "",
            short_term_context=format_short_term_context(state),
            long_term_context=format_long_term_context(state),
            business_snapshot=state.get("business_snapshot") or {},
            business_payload=business_payload,
            source_payload=source_text,
        )
        intent = state.get("intent")
        if intent in {"status", "ticket_status"}:
            answer = status_answer(state, sources)
        elif intent in {"reject", "rejection_reason"}:
            answer = reject_answer(state, sources)
        else:
            answer = knowledge_answer(state, sources)
        return answer, prompt


def status_answer(state: dict[str, Any], sources: list[dict[str, Any]]) -> str:
    """生成办件状态类问题的确定性回答。"""
    status = state.get("business_status") or {}
    node_name = status.get("node_name") or status.get("current_node") or "当前节点"
    answer = f"您的申请当前处于“{node_name}”。该状态来自办件流程服务，RAG 仅用于解释节点含义。"
    if sources:
        answer += f" 根据《{sources[0].get('title')}》，该节点通常表示材料已进入对应部门处理，请关注后续补正或办结通知。"
    return answer


def reject_answer(state: dict[str, Any], sources: list[dict[str, Any]]) -> str:
    """生成退回补正类问题的确定性回答。"""
    reject = state.get("reject_info") or {}
    material_name = reject.get("material_name") or reject.get("material_code") or "相关材料"
    reason = reject.get("reject_reason") or "材料不符合要求"
    answer = f"您的申请被退回主要与“{material_name}”有关，业务系统记录的原因是：{reason}。"
    if sources:
        answer += f" 结合《{sources[0].get('title')}》，建议按规范重新准备或上传该材料。"
    answer += " 修改后可回到申请单详情页重新提交补正材料。"
    return answer


def knowledge_answer(state: dict[str, Any], sources: list[dict[str, Any]]) -> str:
    """生成知识检索类问题的确定性回答。"""
    if not sources:
        return "暂未查询到与当前地区和业务类型完全匹配的办理依据，建议查看办事指南或联系人工客服确认。"
    title = sources[0].get("title") or "办理指南"
    snippet = source_evidence_text(sources[0])
    return f"根据《{title}》，{snippet} 请以页面展示的材料清单和提交入口为准。"


def multi_question_answer(state: dict[str, Any]) -> tuple[str, str]:
    """生成多问题确定性兜底回答和对应 prompt。"""
    sub_tasks = state.get("sub_tasks") or []
    prompt = MULTI_QUESTION_ANSWER_PROMPT.format(
        query=state.get("query") or "",
        short_term_context=format_short_term_context(state),
        long_term_context=format_long_term_context(state),
        sub_task_payload=format_sub_task_payload(sub_tasks),
    )
    lines = []
    for task in sub_tasks:
        if task.get("status") == "need_ticket_selection":
            lines.append("这个问题需要先确定具体申请单，请先选择要咨询的工单后继续。")
        elif task.get("status") == "refused":
            lines.append(str(task.get("answer") or "这个问题暂时无法回答，请补充信息或联系人工客服。"))
        else:
            lines.append(deterministic_sub_task_answer(task))
    return "\n\n".join(dedupe_adjacent_text(lines)), prompt


def deterministic_sub_task_answer(task: dict[str, Any]) -> str:
    """基于子任务证据生成确定性兜底片段。"""
    if task.get("intent") in {"status", "ticket_status"}:
        snapshot = task.get("business_snapshot") or {}
        node_name = snapshot.get("node_name") or snapshot.get("current_node") or "当前节点"
        return f"您的申请当前处于“{node_name}”。该状态来自办件流程服务。"
    if task.get("intent") in {"reject", "rejection_reason"}:
        snapshot = task.get("business_snapshot") or {}
        material_name = snapshot.get("material_name") or "相关材料"
        reason = snapshot.get("reject_reason") or "材料不符合要求"
        return f"该申请退回主要与“{material_name}”有关，业务系统记录的原因是：{reason}。"
    sources = task.get("sources") or []
    if not sources:
        return "暂未查询到可靠依据，建议查看办事指南或联系人工客服确认。"
    title = sources[0].get("title") or "办理指南"
    snippet = source_evidence_text(sources[0])
    return f"根据《{title}》，{snippet} 请以页面展示的信息为准。"


def format_source_payload(sources: list[dict[str, Any]]) -> str:
    """用完整证据文本格式化回答生成输入。"""
    return "\n".join(
        f"- {item.get('title')}: {source_evidence_text(item)}"
        for item in sources
    )


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
    if semantic_type not in {"attachment_table", "docling_table"}:
        return ""
    context = metadata.get("llm_table_context")
    if not context:
        return ""
    try:
        return json.dumps(context, ensure_ascii=False, default=str)[:3000]
    except (TypeError, ValueError):
        return str(context)[:3000]


def format_sub_task_payload(sub_tasks: list[dict[str, Any]]) -> str:
    """把子问题状态格式化为最终回答模型输入。"""
    evidence_pool = build_shared_evidence_pool(sub_tasks)
    evidence_lookup = {item["key"]: item["id"] for item in evidence_pool}
    pool_text = "\n".join(
        "\n".join(
            [
                f"  <Evidence id=\"{item['id']}\">",
                f"    <Title>{xml_text(item['title'])}</Title>",
                f"    <Snippet>{xml_text(item['snippet'])}</Snippet>",
                "  </Evidence>",
            ]
        )
        for item in evidence_pool
    )
    sections = []
    for index, task in enumerate(sub_tasks, start=1):
        sources = task.get("sources") or []
        ticket_context = (
            [f"    <TicketContext>{xml_text(task.get('ticket_context') or {})}</TicketContext>"]
            if task.get("route") == "ticket_context_related"
            else []
        )
        ref_text = "\n".join(
            f"      <Ref>{evidence_lookup[evidence_key(source)]}</Ref>"
            for source in sources
            if evidence_key(source) in evidence_lookup
        )
        sections.append(
            "\n".join(
                [
                    f"  <SubQuery_{index}>",
                    f"    <Question>{xml_text(task.get('query') or '')}</Question>",
                    f"    <Status>{xml_text(task.get('status') or '')}</Status>",
                    f"    <Route>{xml_text(task.get('route') or '')}</Route>",
                    f"    <Intent>{xml_text(task.get('intent') or '')}</Intent>",
                    f"    <NeedTicketSelection>{str(bool(task.get('need_ticket_selection'))).lower()}</NeedTicketSelection>",
                    f"    <RefuseReason>{xml_text(task.get('refuse_reason') or '')}</RefuseReason>",
                    *ticket_context,
                    f"    <BusinessContext>{xml_text(task.get('business_snapshot') or {})}</BusinessContext>",
                    "    <EvidenceRefs>",
                    ref_text or "      <Ref>none</Ref>",
                    "    </EvidenceRefs>",
                    f"  </SubQuery_{index}>",
                ]
            )
        )
    return "\n".join(
        [
            "<SharedEvidencePool>",
            pool_text or "  <Evidence id=\"none\">暂无</Evidence>",
            "</SharedEvidencePool>",
            "",
            "<SubQueries>",
            "\n\n".join(sections),
            "</SubQueries>",
        ]
    )


def build_shared_evidence_pool(sub_tasks: list[dict[str, Any]]) -> list[dict[str, str]]:
    """构造全局去重证据池。"""
    pool = []
    seen = set()
    for task in sub_tasks:
        for source in task.get("sources") or []:
            key = evidence_key(source)
            if not key or key in seen:
                continue
            seen.add(key)
            pool.append(
                {
                    "id": evidence_id(source),
                    "key": key,
                    "title": str(source.get("title") or ""),
                    "snippet": source_evidence_text(source),
                }
            )
    return pool


def evidence_key(source: dict[str, Any]) -> str:
    """生成证据去重键。"""
    return "|".join(
        [
            str(source.get("chunk_id") or ""),
            str(source.get("doc_id") or ""),
            str(source.get("title") or ""),
        ]
    )


def evidence_id(source: dict[str, Any]) -> str:
    """生成不易和回答编号混淆的证据 ID。"""
    raw = evidence_key(source) or str(source)
    return f"chunk_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:10]}"


def xml_text(value: Any) -> str:
    """转义 XML-like 标签中的文本值。"""
    return escape(str(value or ""), quote=False)


def dedupe_adjacent_text(values: list[str]) -> list[str]:
    """去掉相邻重复的空白和文本片段。"""
    output = []
    previous = ""
    for value in values:
        text = str(value or "").strip()
        if not text or text == previous:
            continue
        output.append(text)
        previous = text
    return output
