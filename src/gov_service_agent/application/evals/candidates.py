"""从生产审计Trace中抽样、脱敏并生成评测候选。"""

from __future__ import annotations

from typing import Any

from sqlalchemy import text

from gov_service_agent.application.evals.pii import sanitize_text, sanitize_value, stable_fingerprint
from gov_service_agent.application.evals.retrieval_ablation import extract_agent_retrieval_queries
from gov_service_agent.application.evals.trajectory import normalize_agent_run


ATTRIBUTION_CATEGORIES = (
    "retrieval_miss",
    "knowledge_missing",
    "intent_mismatch",
    "slot_error",
    "hallucination",
    "tool_error",
    "agent_loop",
    "stale_policy",
    "unknown",
)


def validate_promoted_case(case: dict[str, Any]) -> None:
    """校验由 Badcase 归因页面提升的黄金用例是否具备最小基准。"""
    metadata = dict(case.get("source_metadata") or {})
    category = str(metadata.get("attribution_category") or "").strip()
    if not category or category == "unknown":
        # 兼容历史手工用例：没有归因元数据时沿用原有宽松契约。
        return

    doc_ids = list(case.get("expected_doc_ids") or [])
    chunk_ids = list(case.get("expected_chunk_ids") or [])
    keywords = list(case.get("expected_answer_keywords") or [])
    reference_answer = str(case.get("reference_answer") or "").strip()
    trajectory = dict(case.get("expected_trajectory") or {})
    outcome = dict(case.get("expected_outcome") or {})

    if category in {"retrieval_miss", "knowledge_missing", "stale_policy"}:
        if not doc_ids and not chunk_ids:
            raise ValueError("检索、知识或政策版本类用例必须填写期望文档 ID 或期望片段 ID")
        if not reference_answer and not keywords:
            raise ValueError("检索、知识或政策版本类用例必须填写参考答案或答案关键词")
    elif category == "intent_mismatch" and not str(case.get("expected_intent") or "").strip():
        raise ValueError("意图识别错误用例必须填写期望意图")
    elif category == "hallucination" and not reference_answer:
        raise ValueError("事实幻觉用例必须填写经核实的参考答案")
    elif category in {"tool_error", "slot_error"}:
        required_tools = trajectory.get("required_tools") or []
        if not outcome and not required_tools:
            raise ValueError("工具或槽位类用例必须填写期望业务结果或必需工具")
    elif category == "agent_loop" and not trajectory.get("max_tool_calls"):
        raise ValueError("Agent 循环类用例必须填写最大工具调用次数")


def infer_attribution_category(sample: dict[str, Any]) -> tuple[str, str]:
    """根据线上可观测信号给候选样本一个保守的初始归因。"""
    signals = dict(sample.get("signals") or {})
    trajectory = dict(sample.get("trajectory") or {})
    calls = list(trajectory.get("calls") or [])
    if signals.get("tool_error"):
        return "tool_error", "检测到工具调用失败或参数错误"
    if signals.get("max_iterations"):
        return "agent_loop", "Agent 达到最大迭代次数"
    if not list(sample.get("sources") or []):
        return "retrieval_miss", "线上 Trace 未记录到检索来源"
    if signals.get("downvote"):
        return "unknown", "用户差评，需人工确认具体原因"
    if calls and all(not bool(call.get("ok")) for call in calls):
        return "tool_error", "工具调用轨迹未产生成功结果"
    return "unknown", "自动检测未能确定根因，需人工归因"


def mine_candidates(store, *, salt: str, limit: int = 500) -> list[dict[str, Any]]:
    """兼容原接口：只挖掘明确失败、差评、工具错误或高延迟Trace。"""
    samples = sample_production_traces(
        store,
        salt=salt,
        sample_rate=0.0,
        lookback_hours=24 * 30,
        limit=limit,
        include_failures=True,
    )
    created = []
    for sample in samples:
        payload = candidate_payload(sample)
        signals = sample.get("signals") or {}
        payload["source"] = "feedback" if signals.get("downvote") else "automatic_detector"
        payload["severity"] = "high" if signals.get("run_failed") or signals.get("tool_error") else "medium"
        category, reason = infer_attribution_category(sample)
        payload["attribution_category"] = category
        payload["attribution_reason"] = reason
        created.append(store.create_eval_candidate(payload))
    return created


def sample_production_traces(
    store,
    *,
    salt: str,
    sample_rate: float,
    lookback_hours: int,
    limit: int,
    include_failures: bool = True,
) -> list[dict[str, Any]]:
    """按稳定哈希抽样生产Trace，同时无条件保留失败样本。"""
    engine = getattr(store, "engine", None)
    if engine is None:
        return []
    bounded_limit = max(1, min(int(limit), 2000))
    scan_limit = min(10_000, max(bounded_limit, bounded_limit * 20))
    with engine.begin() as connection:
        rows = connection.execute(
            text(
                """
                SELECT a.audit_uid, a.user_id, a.query, a.answer, a.langfuse_trace_id,
                       a.retrieved_sources, a.retrieval_filter, a.tool_results,
                       a.agent_query_plan, a.region_code, a.biz_type,
                       a.intent, a.created_at,
                       r.run_uid AS agent_run_uid, r.status AS run_status,
                       r.iteration_count, r.error_code, r.error_message,
                       COALESCE(f.rating, '') AS rating,
                       EXISTS (
                           SELECT 1 FROM declaration_qa.tool_calls tc
                           WHERE tc.agent_run_id = r.id
                             AND (tc.status IN ('tool_error', 'invalid_arguments', 'tool_not_found') OR tc.error <> '')
                       ) AS has_tool_error,
                       COALESCE((
                           SELECT MAX(m.latency_ms) FROM declaration_qa.messages m
                           WHERE m.agent_run_id = r.id
                       ), 0) AS latency_ms
                FROM declaration_qa.audit_logs a
                JOIN declaration_qa.agent_runs r ON r.id = a.agent_run_id
                LEFT JOIN LATERAL (
                    SELECT feedback.rating
                    FROM declaration_qa.feedback feedback
                    WHERE feedback.audit_log_id = a.id
                    ORDER BY feedback.created_at DESC, feedback.id DESC
                    LIMIT 1
                ) f ON TRUE
                WHERE a.created_at >= CURRENT_TIMESTAMP - (:lookback_hours * INTERVAL '1 hour')
                  AND a.query <> ''
                  AND NOT EXISTS (
                      SELECT 1 FROM declaration_qa.eval_candidates c WHERE c.audit_uid = a.audit_uid
                  )
                ORDER BY a.created_at DESC
                LIMIT :scan_limit
                """
            ),
            {
                "lookback_hours": max(1, min(int(lookback_hours), 24 * 365)),
                "scan_limit": scan_limit,
            },
        ).mappings().all()
    output = []
    for row in rows:
        fingerprint = stable_fingerprint(str(row["query"] or ""), salt=salt)
        failed = trace_failed(row)
        selected_by_rate = stable_sample_value(fingerprint) < max(0.0, min(1.0, float(sample_rate)))
        if not selected_by_rate and not (include_failures and failed):
            continue
        run = store.get_agent_run(user_id=int(row["user_id"]), run_id=str(row["agent_run_uid"]))
        trajectory = normalize_agent_run(run)
        agent_retrieval_queries = extract_agent_retrieval_queries(trajectory)
        if not agent_retrieval_queries:
            agent_retrieval_queries = extract_queries_from_audit_payload(
                list(row["tool_results"] or []),
                dict(row["agent_query_plan"] or {}),
            )
        signals = {
            "sampled": selected_by_rate,
            "sample_rate": max(0.0, min(1.0, float(sample_rate))),
            "downvote": row["rating"] == "down",
            "run_failed": row["run_status"] == "failed" or bool(row["error_code"]),
            "tool_error": bool(row["has_tool_error"]),
            "max_iterations": int(row["iteration_count"] or 0) >= 4,
            "high_latency": int(row["latency_ms"] or 0) >= 30_000,
            "fingerprint": fingerprint,
        }
        output.append(
            {
                "audit_id": str(row["audit_uid"] or ""),
                "agent_run_uid": str(row["agent_run_uid"] or ""),
                "trace_id": str(row["langfuse_trace_id"] or ""),
                "query": str(row["query"] or ""),
                "answer": str(row["answer"] or ""),
                "sources": list(row["retrieved_sources"] or []),
                "retrieval_filter": dict(row["retrieval_filter"] or {}),
                "region_code": str(row["region_code"] or ""),
                "biz_type": row["biz_type"],
                "intent": str(row["intent"] or ""),
                "created_at": str(row["created_at"] or ""),
                "signals": signals,
                "trajectory": trajectory,
                "agent_retrieval_queries": agent_retrieval_queries,
                "failed": failed,
            }
        )
        if len(output) >= bounded_limit:
            break
    return output


def candidate_payload(
    sample: dict[str, Any],
    *,
    online_scores: dict[str, Any] | None = None,
    faithfulness_threshold: float = 0.9,
) -> dict[str, Any]:
    """把内存中的原始Trace样本转换为只含脱敏文本的持久化候选。"""
    score_payload = dict(online_scores or {})
    scores = dict(score_payload.get("scores") or {})
    faithfulness = scores.get("faithfulness")
    low_faithfulness = isinstance(faithfulness, (int, float)) and float(faithfulness) < faithfulness_threshold
    signals = {
        **dict(sample.get("signals") or {}),
        "low_faithfulness": low_faithfulness,
        "faithfulness_threshold": faithfulness_threshold,
    }
    failed = bool(sample.get("failed")) or low_faithfulness
    category, reason = infer_attribution_category(sample)
    salt = str(signals.get("fingerprint") or "online-evaluation")
    return {
        "audit_uid": str(sample.get("audit_id") or ""),
        "agent_run_uid": str(sample.get("agent_run_uid") or ""),
        "trace_id": str(sample.get("trace_id") or ""),
        "source": "online_failure" if failed else "online_sample",
        "severity": "high" if failed else "low",
        "sanitized_query": sanitize_text(str(sample.get("query") or ""), salt=salt),
        "sanitized_answer": sanitize_text(str(sample.get("answer") or ""), salt=salt),
        "detection_signals": signals,
        "trajectory_snapshot": sanitize_value(dict(sample.get("trajectory") or {}), salt=salt),
        "agent_retrieval_queries": sanitize_value(list(sample.get("agent_retrieval_queries") or []), salt=salt),
        "online_scores": sanitize_value(score_payload, salt=salt),
        "attribution_category": category,
        "attribution_reason": reason,
        "knowledge_draft": {},
    }


def trace_failed(row: Any) -> bool:
    return bool(
        row["rating"] == "down"
        or row["run_status"] == "failed"
        or row["error_code"]
        or int(row["iteration_count"] or 0) >= 4
        or row["has_tool_error"]
        or int(row["latency_ms"] or 0) >= 30_000
    )


def stable_sample_value(fingerprint: str) -> float:
    """把稳定SHA256指纹映射到[0,1)，保证重复周期不会反复抖动。"""
    return int(str(fingerprint or "0")[:16], 16) / float(16**16)


def extract_queries_from_audit_payload(
    tool_results: list[dict[str, Any]],
    agent_query_plan: dict[str, Any],
) -> list[dict[str, Any]]:
    """在工具结果被压缩时从审计表的完整载荷恢复Agent检索Query。"""
    calls = []
    for result in tool_results:
        if not isinstance(result, dict) or str(result.get("tool_name") or "") != "knowledge_search":
            continue
        calls.append({"tool_name": "knowledge_search", "result": result, "arguments": {}})
    recovered = extract_agent_retrieval_queries({"calls": calls})
    if recovered:
        return recovered
    raw_queries = agent_query_plan.get("queries") or agent_query_plan.get("retrieval_queries") or []
    return extract_agent_retrieval_queries(
        {
            "calls": [
                {
                    "tool_name": "knowledge_search",
                    "arguments": {"queries": raw_queries},
                    "result": {},
                }
            ]
        }
    )
