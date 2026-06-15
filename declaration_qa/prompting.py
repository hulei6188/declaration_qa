from __future__ import annotations

from typing import Any

from langchain_core.prompts import PromptTemplate


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


class DeterministicAnswerGenerator:
    """Deterministic generator used as a safe local LLM boundary.

    The prompt is still built through LangChain so the integration point can be
    replaced by a real chat model without changing the workflow contract.
    """

    def generate(self, state: dict[str, Any]) -> tuple[str, str]:
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
        if intent == "status":
            answer = status_answer(state, sources)
        elif intent == "reject":
            answer = reject_answer(state, sources)
        else:
            answer = knowledge_answer(state, sources)
        return answer, prompt


def status_answer(state: dict[str, Any], sources: list[dict[str, Any]]) -> str:
    status = state.get("business_status") or {}
    node_name = status.get("node_name") or status.get("current_node") or "当前节点"
    answer = f"您的申请当前处于“{node_name}”。该状态来自办件流程服务，RAG 仅用于解释节点含义。"
    if sources:
        answer += f" 根据《{sources[0].get('title')}》，该节点通常表示材料已进入对应部门处理，请关注后续补正或办结通知。"
    return answer


def reject_answer(state: dict[str, Any], sources: list[dict[str, Any]]) -> str:
    reject = state.get("reject_info") or {}
    material_name = reject.get("material_name") or reject.get("material_code") or "相关材料"
    reason = reject.get("reject_reason") or "材料不符合要求"
    answer = f"您的申请被退回主要与“{material_name}”有关，业务系统记录的原因是：{reason}。"
    if sources:
        answer += f" 结合《{sources[0].get('title')}》，建议按规范重新准备或上传该材料。"
    answer += " 修改后可回到申请单详情页重新提交补正材料。"
    return answer


def knowledge_answer(state: dict[str, Any], sources: list[dict[str, Any]]) -> str:
    if not sources:
        return "暂未查询到与当前地区和业务类型完全匹配的办理依据，建议查看办事指南或联系人工客服确认。"
    title = sources[0].get("title") or "办理指南"
    snippet = sources[0].get("snippet") or ""
    return f"根据《{title}》，{snippet} 请以页面展示的材料清单和提交入口为准。"
