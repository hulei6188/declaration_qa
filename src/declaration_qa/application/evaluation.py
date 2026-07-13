"""离线评测数据集运行与指标计算工具。"""

from __future__ import annotations

from typing import Any

from declaration_qa.api.schemas import ChatRequest
from declaration_qa.application.workflow import DeclarationQAWorkflow


def run_evaluation(
    *,
    workflow: DeclarationQAWorkflow,
    store,
    dataset_id: str,
    note: str = "",
    top_k: int = 4,
    ragas_evaluator=None,
    ragas_requested: bool = False,
    ragas_unavailable_reason: str = "",
) -> dict[str, Any]:
    """运行指定评测数据集并写入评测结果。"""
    cases = store.list_eval_cases(dataset_id)
    run = store.create_eval_run(dataset_id=dataset_id, note=note, top_k=top_k)
    metric_inputs = []
    try:
        for case in cases:
            response = workflow.invoke(chat_request_from_case(case))
            retrieved_doc_ids = unique_doc_ids(source.doc_id for source in response.sources)[:top_k]
            retrieved_contexts = unique_contexts(
                (source.evidence_text or source.snippet)
                for source in response.sources[:top_k]
            )
            ragas_result = evaluate_with_ragas(
                ragas_evaluator,
                query=str(case.get("query") or ""),
                answer=response.answer,
                retrieved_contexts=retrieved_contexts,
                reference_answer=str(case.get("reference_answer") or ""),
                reference_contexts=list(case.get("reference_contexts") or []),
                refused=bool(response.refused),
                requested=ragas_requested,
                unavailable_reason=ragas_unavailable_reason,
            )
            item = build_run_item(case, response, retrieved_doc_ids, retrieved_contexts, ragas_result)
            metric_inputs.append(item)
            store.append_eval_run_item(run_id=run["run_id"], item=item)
        metrics = calculate_metrics(metric_inputs, ragas_requested=ragas_requested, unavailable_reason=ragas_unavailable_reason)
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
    }
    return ChatRequest(**{key: value for key, value in payload.items() if value is not None})


def build_run_item(
    case: dict[str, Any],
    response,
    retrieved_doc_ids: list[str],
    retrieved_contexts: list[str],
    ragas_result: dict[str, Any],
) -> dict[str, Any]:
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
        "reference_answer": str(case.get("reference_answer") or ""),
        "reference_contexts": list(case.get("reference_contexts") or []),
        "retrieved_contexts": retrieved_contexts,
        "answer": response.answer,
        "audit_id": response.audit_id,
        "recall_hit": bool(set(expected_doc_ids).intersection(retrieved_doc_ids)) if expected_doc_ids else None,
        "refusal_match": (bool(expected_refused) == bool(response.refused)) if expected_refused is not None else None,
        "intent_match": (expected_intent == response.intent) if expected_intent else None,
        "keyword_hit": all(keyword in response.answer for keyword in expected_keywords) if expected_keywords else None,
        "ragas_scores": dict(ragas_result.get("scores") or {}),
        "ragas_status": dict(ragas_result.get("status") or {}),
    }


def calculate_metrics(
    items: list[dict[str, Any]],
    *,
    ragas_requested: bool = False,
    unavailable_reason: str = "",
) -> dict[str, Any]:
    """根据评测明细计算整体指标。"""
    metrics = {
        "case_count": len(items),
        "recall_at_k": ratio(item.get("recall_hit") for item in items),
        "refusal_accuracy": ratio(item.get("refusal_match") for item in items),
        "intent_accuracy": ratio(item.get("intent_match") for item in items),
        "answer_keyword_hit_rate": ratio(item.get("keyword_hit") for item in items),
    }
    metrics["ragas"] = summarize_ragas(
        items,
        requested=ragas_requested,
        unavailable_reason=unavailable_reason,
    )
    return metrics


def evaluate_with_ragas(
    evaluator,
    *,
    query: str,
    answer: str,
    retrieved_contexts: list[str],
    reference_answer: str,
    reference_contexts: list[str],
    refused: bool,
    requested: bool,
    unavailable_reason: str,
) -> dict[str, Any]:
    if not requested:
        return {"scores": {}, "status": {"provider": "ragas", "status": "disabled"}}
    if refused:
        return {"scores": {}, "status": {"provider": "ragas", "status": "skipped", "reason": "refused_answer"}}
    if evaluator is None:
        return {
            "scores": {},
            "status": {"provider": "ragas", "status": "unavailable", "reason": unavailable_reason or "not_configured"},
        }
    if not answer.strip() or not retrieved_contexts:
        return {
            "scores": {},
            "status": {"provider": "ragas", "status": "skipped", "reason": "answer_or_context_missing"},
        }
    try:
        result = evaluator.evaluate(
            query=query,
            answer=answer,
            retrieved_contexts=retrieved_contexts,
            reference_answer=reference_answer,
            reference_contexts=reference_contexts,
        )
        return {"scores": dict(result.scores), "status": dict(result.status)}
    except Exception as exc:
        return {
            "scores": {},
            "status": {"provider": "ragas", "status": "failed", "reason": f"{type(exc).__name__}: {str(exc)[:240]}"},
        }


def summarize_ragas(items: list[dict[str, Any]], *, requested: bool, unavailable_reason: str) -> dict[str, Any]:
    if not requested:
        return {"enabled": False, "status": "disabled", "evaluated_case_count": 0, "scores": {}}
    values: dict[str, list[float]] = {}
    statuses: dict[str, int] = {}
    for item in items:
        status = str((item.get("ragas_status") or {}).get("status") or "unknown")
        statuses[status] = statuses.get(status, 0) + 1
        for name, value in (item.get("ragas_scores") or {}).items():
            values.setdefault(str(name), []).append(float(value))
    scores = {
        name: round(sum(metric_values) / len(metric_values), 4)
        for name, metric_values in values.items()
        if metric_values
    }
    evaluated_count = sum(1 for item in items if item.get("ragas_scores"))
    status = "success" if evaluated_count and statuses.get("failed", 0) == 0 else "partial" if evaluated_count else "unavailable"
    return {
        "enabled": True,
        "status": status,
        "evaluated_case_count": evaluated_count,
        "scores": scores,
        "case_statuses": statuses,
        "unavailable_reason": unavailable_reason if not evaluated_count else "",
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


def unique_contexts(values) -> list[str]:
    """按正文规范化后去重 Ragas 检索上下文。"""
    seen = set()
    output = []
    for value in values:
        context = str(value or "").strip()
        signature = "".join(context.split()).lower()
        if not context or signature in seen:
            continue
        seen.add(signature)
        output.append(context)
    return output
