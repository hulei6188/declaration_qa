"""从生产审计中挖掘并脱敏失败候选样本。"""

from __future__ import annotations

from typing import Any

from sqlalchemy import text

from declaration_qa.application.evals.pii import sanitize_text, stable_fingerprint
from declaration_qa.application.evals.trajectory import normalize_agent_run


def mine_candidates(store, *, salt: str, limit: int = 500) -> list[dict[str, Any]]:
    engine = getattr(store, "engine", None)
    if engine is None:
        return []
    with engine.begin() as connection:
        rows = connection.execute(
            text(
                """
                SELECT DISTINCT ON (a.audit_uid)
                       a.audit_uid, a.user_id, a.query, a.answer, a.langfuse_trace_id,
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
                LEFT JOIN declaration_qa.feedback f ON f.audit_log_id = a.id AND f.rating = 'down'
                WHERE (
                    f.rating = 'down' OR r.status = 'failed' OR r.error_code <> ''
                    OR r.iteration_count >= 4
                    OR EXISTS (
                        SELECT 1 FROM declaration_qa.tool_calls tc
                        WHERE tc.agent_run_id = r.id
                          AND (tc.status IN ('tool_error', 'invalid_arguments', 'tool_not_found') OR tc.error <> '')
                    )
                    OR EXISTS (
                        SELECT 1 FROM declaration_qa.messages m
                        WHERE m.agent_run_id = r.id AND m.latency_ms >= 30000
                    )
                )
                AND NOT EXISTS (
                    SELECT 1 FROM declaration_qa.eval_candidates c WHERE c.audit_uid = a.audit_uid
                )
                ORDER BY a.audit_uid, a.created_at DESC
                LIMIT :limit
                """
            ),
            {"limit": max(1, min(int(limit), 2000))},
        ).mappings().all()
    created = []
    for row in rows:
        signals = {
            "downvote": row["rating"] == "down",
            "run_failed": row["run_status"] == "failed" or bool(row["error_code"]),
            "tool_error": bool(row["has_tool_error"]),
            "max_iterations": int(row["iteration_count"] or 0) >= 4,
            "high_latency": int(row["latency_ms"] or 0) >= 30000,
            "fingerprint": stable_fingerprint(str(row["query"] or ""), salt=salt),
        }
        run = store.get_agent_run(user_id=int(row["user_id"]), run_id=str(row["agent_run_uid"]))
        source = "feedback" if signals["downvote"] else "automatic_detector"
        severity = "high" if signals["run_failed"] or signals["tool_error"] else "medium"
        created.append(
            store.create_eval_candidate(
                {
                    "audit_uid": str(row["audit_uid"] or ""),
                    "agent_run_uid": str(row["agent_run_uid"] or ""),
                    "trace_id": str(row["langfuse_trace_id"] or ""),
                    "source": source,
                    "severity": severity,
                    "sanitized_query": sanitize_text(str(row["query"] or ""), salt=salt),
                    "sanitized_answer": sanitize_text(str(row["answer"] or ""), salt=salt),
                    "detection_signals": signals,
                    "trajectory_snapshot": normalize_agent_run(run),
                }
            )
        )
    return created
