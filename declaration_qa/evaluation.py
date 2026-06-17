"""离线评测数据集运行与指标计算工具。"""

from __future__ import annotations

from typing import Any

from declaration_qa.schemas import ChatRequest
from declaration_qa.workflow import DeclarationQAWorkflow


def run_evaluation(
    *,
    workflow: DeclarationQAWorkflow,
    store,
    dataset_id: str,
    note: str = "",
    top_k: int = 4,
) -> dict[str, Any]:
    """运行指定评测数据集并写入评测结果。"""
    cases = store.list_eval_cases(dataset_id)
    run = store.create_eval_run(dataset_id=dataset_id, note=note, top_k=top_k)
    metric_inputs = []
    try:
        for case in cases:
            response = workflow.invoke(chat_request_from_case(case))
            retrieved_doc_ids = unique_doc_ids(source.doc_id for source in response.sources)[:top_k]
            item = build_run_item(case, response, retrieved_doc_ids)
            metric_inputs.append(item)
            store.append_eval_run_item(run_id=run["run_id"], item=item)
        metrics = calculate_metrics(metric_inputs)
        return store.finish_eval_run(run_id=run["run_id"], status="succeeded", metrics=metrics)
    except Exception as exc:
        return store.finish_eval_run(run_id=run["run_id"], status="failed", metrics={}, error_message=str(exc))


def chat_request_from_case(case: dict[str, Any]) -> ChatRequest:
    """将评测用例转换为问答请求。"""
    payload = {
        "query": str(case.get("query") or ""),
        "user_id": int(case.get("user_id") or 1),
        "region_code": str(case.get("region_code") or "340000"),
        "policy_version": str(case.get("policy_version") or "2025"),
        "biz_type": case.get("biz_type"),
        "apply_id": case.get("apply_id"),
        "apply_context": case.get("apply_context"),
    }
    return ChatRequest(**{key: value for key, value in payload.items() if value is not None})


def build_run_item(case: dict[str, Any], response, retrieved_doc_ids: list[str]) -> dict[str, Any]:
    """构造单条评测运行明细。"""
    expected_doc_ids = [str(item) for item in case.get("expected_doc_ids") or []]
    expected_keywords = [str(item) for item in case.get("expected_answer_keywords") or []]
    expected_intent = str(case.get("expected_intent") or "")
    expected_refused = case.get("expected_refused")
    return {
        "case_id": str(case.get("case_id") or ""),
        "query": str(case.get("query") or ""),
        "expected_intent": expected_intent,
        "actual_intent": response.intent,
        "expected_refused": expected_refused,
        "actual_refused": response.refused,
        "expected_doc_ids": expected_doc_ids,
        "retrieved_doc_ids": retrieved_doc_ids,
        "expected_answer_keywords": expected_keywords,
        "answer": response.answer,
        "audit_id": response.audit_id,
        "recall_hit": bool(set(expected_doc_ids).intersection(retrieved_doc_ids)) if expected_doc_ids else None,
        "refusal_match": (bool(expected_refused) == bool(response.refused)) if expected_refused is not None else None,
        "intent_match": (expected_intent == response.intent) if expected_intent else None,
        "keyword_hit": all(keyword in response.answer for keyword in expected_keywords) if expected_keywords else None,
    }


def calculate_metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
    """根据评测明细计算整体指标。"""
    return {
        "case_count": len(items),
        "recall_at_k": ratio(item.get("recall_hit") for item in items),
        "refusal_accuracy": ratio(item.get("refusal_match") for item in items),
        "intent_accuracy": ratio(item.get("intent_match") for item in items),
        "answer_keyword_hit_rate": ratio(item.get("keyword_hit") for item in items),
    }


def ratio(values) -> float | None:
    """计算布尔结果序列的命中比例。"""
    effective = [value for value in values if value is not None]
    if not effective:
        return None
    return round(sum(1 for value in effective if value) / len(effective), 4)


def unique_doc_ids(values) -> list[str]:
    """按出现顺序去重文档 ID。"""
    seen = set()
    output = []
    for value in values:
        doc_id = str(value or "")
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        output.append(doc_id)
    return output
