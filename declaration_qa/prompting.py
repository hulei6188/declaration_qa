"""问答提示词和本地确定性回答生成器。"""

from __future__ import annotations

from html import escape
from typing import Any

try:
    from langchain_core.prompts import PromptTemplate
except ModuleNotFoundError:
    class PromptTemplate:
        """Minimal fallback used when langchain_core is not installed."""

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
业务快照：{business_snapshot}
业务数据：{business_payload}
检索证据：{source_payload}

请用清晰、可操作的语言回答。"""
)

MULTI_QUESTION_ANSWER_PROMPT = PromptTemplate.from_template(
    """你是一证办申报辅助问答模块，只能基于按子问题分组的业务查询结果和检索证据回答。
不能编造政策，不能承诺审核结果，不能替代人工审核结论。
请严格基于以下按问题划分的证据回答，禁止跨越标签范围套用数据。
不得跨子问题挪用证据；不得新增证据中没有的材料、时限、费用、条件或审核结论。
如果某个子问题需要用户选择工单，只能提示先选择工单，不要推测进度或驳回原因。
如果某个子问题证据不足，只能说明暂未查询到可靠依据。

用户原始问题：{query}
<SubQueries>
{sub_task_payload}
</SubQueries>

请按子问题原顺序分点回答，语言自然但必须可追溯。"""
)


class DeterministicAnswerGenerator:
    """Deterministic generator used as a safe local LLM boundary.

    The prompt is still built through LangChain so the integration point can be
    replaced by a real chat model without changing the workflow contract.
    """

    def generate(self, state: dict[str, Any]) -> tuple[str, str]:
        """生成非流式回答和提示词。"""
        if state.get("multi_question"):
            return multi_question_answer(state)
        sources = state.get("retrieved_sources") or []
        source_text = "\n".join(f"- {item.get('title')}: {item.get('snippet')}" for item in sources[:3])
        business_payload = state.get("business_status") or state.get("reject_info") or {}
        prompt = ANSWER_PROMPT.format(
            query=state.get("query") or "",
            intent=state.get("intent") or "",
            retrieval_chain=state.get("retrieval_chain") or "",
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
    snippet = sources[0].get("snippet") or ""
    return f"根据《{title}》，{snippet} 请以页面展示的材料清单和提交入口为准。"


def multi_question_answer(state: dict[str, Any]) -> tuple[str, str]:
    """生成多问题确定性兜底回答和对应 prompt。"""
    sub_tasks = state.get("sub_tasks") or []
    prompt = MULTI_QUESTION_ANSWER_PROMPT.format(
        query=state.get("query") or "",
        sub_task_payload=format_sub_task_payload(sub_tasks),
    )
    lines = [f"我把你的问题分成 {len(sub_tasks)} 部分："]
    for index, task in enumerate(sub_tasks, start=1):
        lines.append(f"{index}. {task.get('query') or '子问题'}")
        if task.get("status") == "need_ticket_selection":
            lines.append("这个问题需要先确定具体申请单，请先选择要咨询的工单后继续。")
        elif task.get("status") == "refused":
            lines.append(str(task.get("answer") or "这个问题暂时无法回答，请补充信息或联系人工客服。"))
        else:
            lines.append(deterministic_sub_task_answer(task))
    return "\n\n".join(lines), prompt


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
    snippet = sources[0].get("snippet") or ""
    return f"根据《{title}》，{snippet} 请以页面展示的信息为准。"


def format_sub_task_payload(sub_tasks: list[dict[str, Any]]) -> str:
    """把子问题状态格式化为最终回答模型输入。"""
    sections = []
    for index, task in enumerate(sub_tasks, start=1):
        sources = task.get("sources") or []
        ticket_context = (
            [f"    <TicketContext>{xml_text(task.get('ticket_context') or {})}</TicketContext>"]
            if task.get("route") == "ticket_context_related"
            else []
        )
        source_text = "\n".join(
            "\n".join(
                [
                    f"    <Evidence index=\"{source_index}\">",
                    f"      <Title>{xml_text(source.get('title') or '')}</Title>",
                    f"      <Snippet>{xml_text(source.get('snippet') or '')}</Snippet>",
                    "    </Evidence>",
                ]
            )
            for source_index, source in enumerate(sources[:6], start=1)
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
                    "    <EvidenceList>",
                    source_text or "    <Evidence>暂无</Evidence>",
                    "    </EvidenceList>",
                    f"  </SubQuery_{index}>",
                ]
            )
        )
    return "\n\n".join(sections)


def xml_text(value: Any) -> str:
    """转义 XML-like 标签中的文本值。"""
    return escape(str(value or ""), quote=False)
