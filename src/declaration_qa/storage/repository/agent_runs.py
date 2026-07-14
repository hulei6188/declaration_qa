"""Replayable Agent run, message, tool call, and citation persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from sqlalchemy import JSON, bindparam, text

from declaration_qa.storage.repository.base import EngineBackedRepository
from declaration_qa.storage.session_payloads import query_image_attachments


MAX_INLINE_TOOL_RESULT_CHARS = 100_000


@dataclass(frozen=True)
class PersistedAgentRun:
    """Primary keys created for one replayable Agent run."""

    run_id: int
    final_message_id: int


class AgentRunRepository(EngineBackedRepository):
    """Read complete Agent runs with user ownership enforcement."""

    def get(self, *, user_id: int, run_id: str) -> dict[str, Any] | None:
        """Return an ordered, replayable Agent run."""
        with self.engine.begin() as connection:
            run = connection.execute(
                text(
                    """
                    SELECT r.id, r.run_uid AS run_id, s.session_uid AS session_id,
                           r.user_id, r.status, r.iteration_count, r.trace_id,
                           r.answer_provider, r.idempotency_key, r.error_code,
                           r.error_message, r.prompt_version, r.tool_registry_version,
                           r.model_config AS model_snapshot, r.metadata, r.started_at, r.completed_at,
                           r.created_at, r.updated_at
                    FROM declaration_qa.agent_runs r
                    JOIN declaration_qa.sessions s ON s.id = r.session_id
                    WHERE r.run_uid = :run_uid AND r.user_id = :user_id
                    """
                ),
                {"run_uid": run_id, "user_id": user_id},
            ).mappings().first()
            if not run:
                return None
            run_pk = int(run["id"])
            messages = connection.execute(
                text(
                    """
                    SELECT m.id, m.message_uid AS message_id, m.sequence_no,
                           p.message_uid AS parent_message_id, m.role, m.message_type,
                           m.content, m.content_json, m.status, m.visible, m.model,
                           m.provider, m.input_tokens, m.output_tokens, m.total_tokens,
                           m.latency_ms, m.pii_level, m.retention_until, m.intent,
                           m.refused, m.refuse_reason, m.sources, m.suggested_actions,
                           m.meta, m.created_at
                    FROM declaration_qa.messages m
                    LEFT JOIN declaration_qa.messages p ON p.id = m.parent_message_id
                    WHERE m.agent_run_id = :run_id
                    ORDER BY m.sequence_no, m.id
                    """
                ),
                {"run_id": run_pk},
            ).mappings().all()
            message_pks = [int(item["id"]) for item in messages]
            parts = _load_message_parts(connection, message_pks)
            sources = _load_message_sources(connection, message_pks)
            tool_calls = connection.execute(
                text(
                    """
                    SELECT tc.tool_call_uid AS tool_call_id, tc.external_call_id,
                           tc.sequence_no, tc.tool_name, tc.arguments, tc.status, tc.ok,
                           tc.result, tc.error, tc.latency_ms, tc.started_at,
                           tc.completed_at, tc.metadata,
                           am.message_uid AS assistant_message_id,
                           rm.message_uid AS result_message_id
                    FROM declaration_qa.tool_calls tc
                    JOIN declaration_qa.messages am ON am.id = tc.assistant_message_id
                    LEFT JOIN declaration_qa.messages rm ON rm.id = tc.result_message_id
                    WHERE tc.agent_run_id = :run_id
                    ORDER BY tc.sequence_no, tc.id
                    """
                ),
                {"run_id": run_pk},
            ).mappings().all()

        normalized_messages = []
        for item in messages:
            message = _json_safe_row(item, exclude={"id"})
            message_pk = int(item["id"])
            message["parts"] = parts.get(message_pk, [])
            message["citations"] = sources.get(message_pk, [])
            normalized_messages.append(message)
        payload = _json_safe_row(run, exclude={"id"})
        payload["messages"] = normalized_messages
        payload["tool_calls"] = [_json_safe_row(item) for item in tool_calls]
        return payload

    def recent_messages(self, *, user_id: int, session_id: str, limit_turns: int = 5) -> list[dict[str, Any]]:
        """Return replayable messages from the most recent complete runs in a session."""
        with self.engine.begin() as connection:
            rows = connection.execute(
                text(
                    """
                    WITH recent_runs AS (
                        SELECT r.id, r.created_at
                        FROM declaration_qa.agent_runs r
                        JOIN declaration_qa.sessions s ON s.id = r.session_id
                        WHERE s.session_uid = :session_id
                          AND r.user_id = :user_id
                        ORDER BY r.created_at DESC, r.id DESC
                        LIMIT :limit_turns
                    )
                    SELECT m.role, m.message_type, m.content, m.content_json,
                           tc.external_call_id, rr.created_at AS run_created_at,
                           m.sequence_no, m.id
                    FROM recent_runs rr
                    JOIN declaration_qa.messages m ON m.agent_run_id = rr.id
                    LEFT JOIN declaration_qa.tool_calls tc ON tc.result_message_id = m.id
                    WHERE m.message_type IN ('input', 'tool_call', 'tool_result', 'output')
                    ORDER BY rr.created_at, rr.id, m.sequence_no, m.id
                    """
                ),
                {
                    "session_id": session_id,
                    "user_id": user_id,
                    "limit_turns": max(1, int(limit_turns or 5)),
                },
            ).mappings().all()
        return [message for row in rows if (message := _replay_message(row)) is not None]


def persist_agent_run(
    connection,
    *,
    session_id: int,
    run_uid: str,
    payload: dict[str, Any],
) -> PersistedAgentRun:
    """Persist the observable Agent transcript in deterministic replay order."""
    run_id = _insert_run(connection, session_id=session_id, run_uid=run_uid, payload=payload)
    existing_final = connection.execute(
        text(
            """
            SELECT id
            FROM declaration_qa.messages
            WHERE agent_run_id = :run_id AND message_type = 'output'
            ORDER BY sequence_no DESC
            LIMIT 1
            """
        ),
        {"run_id": run_id},
    ).scalar_one_or_none()
    if existing_final is not None:
        return PersistedAgentRun(run_id=run_id, final_message_id=int(existing_final))

    sequence_no = 0
    parent_message_id: int | None = None
    system_prompt = str(payload.get("agent_system_prompt") or "").strip()
    if system_prompt:
        sequence_no += 1
        parent_message_id = _insert_message(
            connection,
            session_id=session_id,
            run_id=run_id,
            sequence_no=sequence_no,
            parent_message_id=None,
            role="system",
            message_type="system",
            content=system_prompt,
            content_json={"prompt_version": payload.get("agent_prompt_version") or ""},
            visible=False,
            provider="application",
            pii_level="none",
        )
        _insert_part(connection, message_id=parent_message_id, part_index=1, part_type="text", text_content=system_prompt)

    sequence_no += 1
    query_image = payload.get("query_image") if isinstance(payload.get("query_image"), dict) else {}
    user_meta = {
        "query_image": query_image,
        "attachments": query_image_attachments(query_image),
    }
    user_message_id = _insert_message(
        connection,
        session_id=session_id,
        run_id=run_id,
        sequence_no=sequence_no,
        parent_message_id=parent_message_id,
        role="user",
        message_type="input",
        content=str(payload.get("query") or ""),
        content_json={"normalized_query": payload.get("normalized_query") or ""},
        visible=True,
        intent=str(payload.get("intent") or ""),
        meta=user_meta,
        pii_level="personal",
    )
    parent_message_id = user_message_id
    user_part_index = 0
    query = str(payload.get("query") or "")
    if query:
        user_part_index += 1
        _insert_part(connection, message_id=user_message_id, part_index=user_part_index, part_type="text", text_content=query)
    if query_image:
        user_part_index += 1
        _insert_part(
            connection,
            message_id=user_message_id,
            part_index=user_part_index,
            part_type="image",
            asset_uri=str(query_image.get("file_path") or ""),
            mime_type=str(query_image.get("content_type") or ""),
            file_name=str(query_image.get("filename") or ""),
            size_bytes=_safe_int(query_image.get("size_bytes")),
            metadata={"purpose": "user_query"},
        )
    ocr_text = str(payload.get("query_image_ocr_text") or "").strip()
    if ocr_text:
        user_part_index += 1
        _insert_part(
            connection,
            message_id=user_message_id,
            part_index=user_part_index,
            part_type="ocr",
            text_content=ocr_text,
            metadata={"derived_from": "user_query_image"},
        )

    results = [item for item in (payload.get("tool_results") or []) if isinstance(item, dict)]
    used_results: set[int] = set()
    tool_sequence = 0
    for decision in _agent_decisions(payload):
        calls = [item for item in (decision.get("tool_calls") or []) if isinstance(item, dict)]
        if not calls:
            continue
        sequence_no += 1
        assistant_message_id = _insert_message(
            connection,
            session_id=session_id,
            run_id=run_id,
            sequence_no=sequence_no,
            parent_message_id=parent_message_id,
            role="assistant",
            message_type="tool_call",
            content=str(decision.get("assistant_content") or ""),
            content_json={"tool_calls": calls, "iteration": decision.get("iteration")},
            visible=False,
            model=str(decision.get("model") or ""),
            provider=str(decision.get("provider") or ""),
            input_tokens=_token_value(decision, "input_tokens"),
            output_tokens=_token_value(decision, "output_tokens"),
            total_tokens=_token_value(decision, "total_tokens"),
            latency_ms=_nullable_int(decision.get("latency_ms")),
            intent=str(decision.get("intent") or payload.get("intent") or ""),
            pii_level="personal" if any(_is_ticket_call(item) for item in calls) else "none",
        )
        _insert_part(
            connection,
            message_id=assistant_message_id,
            part_index=1,
            part_type="json",
            text_content=_json_text({"tool_calls": calls}),
        )
        parent_message_id = assistant_message_id
        for call in calls:
            tool_sequence += 1
            result_index, result = _match_tool_result(call, results, used_results)
            if result_index is not None:
                used_results.add(result_index)
            tool_call_pk, tool_call_uid = _insert_tool_call(
                connection,
                run_id=run_id,
                assistant_message_id=assistant_message_id,
                sequence_no=tool_sequence,
                call=call,
                result=result,
                audit_uid=run_uid,
            )
            if result is None:
                continue
            sequence_no += 1
            compact_result = compact_tool_result(result, audit_uid=run_uid)
            result_message_id = _insert_message(
                connection,
                session_id=session_id,
                run_id=run_id,
                sequence_no=sequence_no,
                parent_message_id=assistant_message_id,
                role="tool",
                message_type="tool_result",
                content=f"{result.get('tool_name') or call.get('name')}: {result.get('status') or 'completed'}",
                content_json={"tool_call_id": tool_call_uid, "result": compact_result},
                status="completed" if result.get("ok") else "failed",
                visible=False,
                latency_ms=_nullable_int(result.get("latency_ms")),
                pii_level="sensitive" if _is_ticket_call(call) else "none",
            )
            _insert_part(
                connection,
                message_id=result_message_id,
                part_index=1,
                part_type="json",
                text_content=_json_text(compact_result),
            )
            connection.execute(
                text("UPDATE declaration_qa.tool_calls SET result_message_id = :message_id WHERE id = :tool_call_id"),
                {"message_id": result_message_id, "tool_call_id": tool_call_pk},
            )
            parent_message_id = result_message_id

    unmatched = [(index, item) for index, item in enumerate(results) if index not in used_results]
    if unmatched:
        synthetic_calls = [
            {
                "id": str(item.get("call_id") or f"recovered_{index + 1}"),
                "name": str(item.get("tool_name") or "unknown_tool"),
                "arguments": dict(item.get("arguments") or {}),
            }
            for index, item in unmatched
        ]
        sequence_no += 1
        assistant_message_id = _insert_message(
            connection,
            session_id=session_id,
            run_id=run_id,
            sequence_no=sequence_no,
            parent_message_id=parent_message_id,
            role="assistant",
            message_type="tool_call",
            content="",
            content_json={"tool_calls": synthetic_calls, "recovered_from_results": True},
            visible=False,
            provider="recovered_audit",
            intent=str(payload.get("intent") or ""),
            pii_level="personal" if any(_is_ticket_call(item) for item in synthetic_calls) else "none",
        )
        parent_message_id = assistant_message_id
        for (result_index, result), call in zip(unmatched, synthetic_calls):
            used_results.add(result_index)
            tool_sequence += 1
            tool_call_pk, tool_call_uid = _insert_tool_call(
                connection,
                run_id=run_id,
                assistant_message_id=assistant_message_id,
                sequence_no=tool_sequence,
                call=call,
                result=result,
                audit_uid=run_uid,
            )
            sequence_no += 1
            compact_result = compact_tool_result(result, audit_uid=run_uid)
            result_message_id = _insert_message(
                connection,
                session_id=session_id,
                run_id=run_id,
                sequence_no=sequence_no,
                parent_message_id=assistant_message_id,
                role="tool",
                message_type="tool_result",
                content=f"{result.get('tool_name') or call.get('name')}: {result.get('status') or 'completed'}",
                content_json={"tool_call_id": tool_call_uid, "result": compact_result},
                status="completed" if result.get("ok") else "failed",
                visible=False,
                latency_ms=_nullable_int(result.get("latency_ms")),
                pii_level="sensitive" if _is_ticket_call(call) else "none",
            )
            _insert_part(
                connection,
                message_id=result_message_id,
                part_index=1,
                part_type="json",
                text_content=_json_text(compact_result),
            )
            connection.execute(
                text("UPDATE declaration_qa.tool_calls SET result_message_id = :message_id WHERE id = :tool_call_id"),
                {"message_id": result_message_id, "tool_call_id": tool_call_pk},
            )
            parent_message_id = result_message_id

    final_provider = str(payload.get("agent_answer_provider") or "")
    if final_provider in {"agent_multimodal_final", "agent_text_final"}:
        sequence_no += 1
        final_prompt = str(payload.get("final_prompt") or "")
        final_input_message_id = _insert_message(
            connection,
            session_id=session_id,
            run_id=run_id,
            sequence_no=sequence_no,
            parent_message_id=parent_message_id,
            role="user",
            message_type="model_input",
            content=final_prompt,
            content_json={"stage": "final_answer"},
            visible=False,
            provider="application",
            pii_level="personal",
        )
        final_part_index = 1
        _insert_part(
            connection,
            message_id=final_input_message_id,
            part_index=final_part_index,
            part_type="text",
            text_content=final_prompt,
        )
        if query_image:
            final_part_index += 1
            _insert_part(
                connection,
                message_id=final_input_message_id,
                part_index=final_part_index,
                part_type="image",
                asset_uri=str(query_image.get("file_path") or ""),
                mime_type=str(query_image.get("content_type") or ""),
                file_name=str(query_image.get("filename") or ""),
                size_bytes=_safe_int(query_image.get("size_bytes")),
                metadata={"purpose": "final_user_query_image"},
            )
        generation_plan = payload.get("answer_generation_plan") if isinstance(payload.get("answer_generation_plan"), dict) else {}
        selected_chunk_ids = {str(item) for item in (generation_plan.get("evidence_chunk_ids") or []) if str(item)}
        for source in payload.get("retrieved_sources") or []:
            if not isinstance(source, dict) or str(source.get("chunk_id") or "") not in selected_chunk_ids:
                continue
            metadata = source.get("metadata") if isinstance(source.get("metadata"), dict) else {}
            asset_path = str(metadata.get("asset_path") or "")
            if not asset_path:
                continue
            final_part_index += 1
            _insert_part(
                connection,
                message_id=final_input_message_id,
                part_index=final_part_index,
                part_type="image",
                asset_uri=asset_path,
                mime_type=str(metadata.get("asset_content_type") or ""),
                file_name=str(metadata.get("asset_filename") or metadata.get("filename") or ""),
                size_bytes=_safe_int(metadata.get("asset_size_bytes")),
                metadata={
                    "purpose": "final_knowledge_evidence",
                    "chunk_id": str(source.get("chunk_id") or ""),
                    "source_asset_id": str(metadata.get("source_asset_id") or metadata.get("asset_id") or ""),
                },
            )
        parent_message_id = final_input_message_id

    sequence_no += 1
    final_usage = payload.get("agent_final_token_usage") if isinstance(payload.get("agent_final_token_usage"), dict) else {}
    final_message_id = _insert_message(
        connection,
        session_id=session_id,
        run_id=run_id,
        sequence_no=sequence_no,
        parent_message_id=parent_message_id,
        role="assistant",
        message_type="output",
        content=str(payload.get("answer") or ""),
        content_json={"evidence_decision": payload.get("evidence_decision") or {}},
        visible=True,
        model=str(payload.get("agent_final_model") or ""),
        provider=str(payload.get("agent_answer_provider") or ""),
        input_tokens=_nullable_int(final_usage.get("input_tokens")),
        output_tokens=_nullable_int(final_usage.get("output_tokens")),
        total_tokens=_nullable_int(final_usage.get("total_tokens")),
        latency_ms=_nullable_int(payload.get("agent_final_latency_ms")),
        intent=str(payload.get("intent") or ""),
        refused=bool(payload.get("refused")),
        refuse_reason=str(payload.get("refuse_reason") or ""),
        sources=list(payload.get("retrieved_sources") or []),
        suggested_actions=list(payload.get("suggested_actions") or []),
        meta={
            "retrieval_chain": payload.get("retrieval_chain") or "",
            "agent_iteration": _safe_int(payload.get("agent_iteration")),
            "evidence_ok": bool(payload.get("evidence_ok")),
            "langfuse_trace_id": payload.get("langfuse_trace_id") or "",
        },
        pii_level="personal" if any(_is_ticket_result(item) for item in results) else "none",
    )
    answer = str(payload.get("answer") or "")
    if answer:
        _insert_part(connection, message_id=final_message_id, part_index=1, part_type="text", text_content=answer)
    _insert_sources(connection, message_id=final_message_id, sources=list(payload.get("retrieved_sources") or []))
    return PersistedAgentRun(run_id=run_id, final_message_id=final_message_id)


def compact_tool_result(result: dict[str, Any], *, audit_uid: str) -> dict[str, Any]:
    """Keep normal results inline and replace oversized payloads with an audit reference."""
    normalized = dict(result)
    if len(_json_text(normalized)) <= MAX_INLINE_TOOL_RESULT_CHARS:
        return normalized
    return {
        "tool_name": normalized.get("tool_name") or "",
        "call_id": normalized.get("call_id") or "",
        "ok": bool(normalized.get("ok")),
        "status": normalized.get("status") or "",
        "error": normalized.get("error") or "",
        "truncated": True,
        "result_ref": {"table": "audit_logs", "audit_uid": audit_uid, "field": "tool_results"},
    }


def _insert_run(connection, *, session_id: int, run_uid: str, payload: dict[str, Any]) -> int:
    stmt = text(
        """
        INSERT INTO declaration_qa.agent_runs (
            run_uid, session_id, user_id, status, iteration_count, trace_id,
            answer_provider, idempotency_key, error_code, error_message,
            prompt_version, tool_registry_version, model_config, metadata,
            completed_at, updated_at
        )
        VALUES (
            :run_uid, :session_id, :user_id, :status, :iteration_count, :trace_id,
            :answer_provider, :idempotency_key, :error_code, :error_message,
            :prompt_version, :tool_registry_version, :model_config, :metadata,
            CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
        )
        ON CONFLICT (run_uid) DO UPDATE SET
            status = EXCLUDED.status,
            iteration_count = EXCLUDED.iteration_count,
            trace_id = EXCLUDED.trace_id,
            answer_provider = EXCLUDED.answer_provider,
            error_code = EXCLUDED.error_code,
            error_message = EXCLUDED.error_message,
            model_config = EXCLUDED.model_config,
            metadata = EXCLUDED.metadata,
            completed_at = EXCLUDED.completed_at,
            updated_at = CURRENT_TIMESTAMP
        RETURNING id
        """
    ).bindparams(bindparam("model_config", type_=JSON), bindparam("metadata", type_=JSON))
    row = connection.execute(
        stmt,
        {
            "run_uid": run_uid,
            "session_id": session_id,
            "user_id": _safe_int(payload.get("user_id")),
            "status": "succeeded" if not payload.get("run_error") else "failed",
            "iteration_count": _safe_int(payload.get("agent_iteration")),
            "trace_id": str(payload.get("langfuse_trace_id") or ""),
            "answer_provider": str(payload.get("agent_answer_provider") or ""),
            "idempotency_key": str(payload.get("idempotency_key") or ""),
            "error_code": str(payload.get("run_error_code") or ""),
            "error_message": str(payload.get("run_error") or ""),
            "prompt_version": str(payload.get("agent_prompt_version") or ""),
            "tool_registry_version": str(payload.get("tool_registry_version") or ""),
            "model_config": payload.get("agent_model_config") or {},
            "metadata": {
                "route": payload.get("route") or "",
                "intent": payload.get("intent") or "",
                "required_evidence": (payload.get("evidence_decision") or {}).get("required_evidence") or [],
            },
        },
    ).mappings().one()
    return int(row["id"])


def _insert_message(
    connection,
    *,
    session_id: int,
    run_id: int,
    sequence_no: int,
    parent_message_id: int | None,
    role: str,
    message_type: str,
    content: str,
    content_json: dict[str, Any] | None = None,
    status: str = "completed",
    visible: bool,
    model: str = "",
    provider: str = "",
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    total_tokens: int | None = None,
    latency_ms: int | None = None,
    pii_level: str = "none",
    intent: str = "",
    refused: bool = False,
    refuse_reason: str = "",
    sources: list[dict[str, Any]] | None = None,
    suggested_actions: list[dict[str, Any]] | None = None,
    meta: dict[str, Any] | None = None,
) -> int:
    stmt = text(
        """
        INSERT INTO declaration_qa.messages (
            message_uid, session_id, agent_run_id, sequence_no, parent_message_id,
            role, message_type, content, content_json, status, visible, model,
            provider, input_tokens, output_tokens, total_tokens, latency_ms,
            pii_level, intent, refused, refuse_reason,
            sources, suggested_actions, meta
        )
        VALUES (
            :message_uid, :session_id, :agent_run_id, :sequence_no, :parent_message_id,
            :role, :message_type, :content, :content_json, :status, :visible, :model,
            :provider, :input_tokens, :output_tokens, :total_tokens, :latency_ms,
            :pii_level, :intent, :refused, :refuse_reason,
            :sources, :suggested_actions, :meta
        )
        RETURNING id
        """
    ).bindparams(
        bindparam("content_json", type_=JSON),
        bindparam("sources", type_=JSON),
        bindparam("suggested_actions", type_=JSON),
        bindparam("meta", type_=JSON),
    )
    row = connection.execute(
        stmt,
        {
            "message_uid": uuid4().hex,
            "session_id": session_id,
            "agent_run_id": run_id,
            "sequence_no": sequence_no,
            "parent_message_id": parent_message_id,
            "role": role,
            "message_type": message_type,
            "content": content,
            "content_json": content_json or {},
            "status": status,
            "visible": visible,
            "model": model,
            "provider": provider,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "latency_ms": latency_ms,
            "pii_level": pii_level,
            "intent": intent,
            "refused": refused,
            "refuse_reason": refuse_reason,
            "sources": sources or [],
            "suggested_actions": suggested_actions or [],
            "meta": meta or {},
        },
    ).mappings().one()
    return int(row["id"])


def _insert_tool_call(
    connection,
    *,
    run_id: int,
    assistant_message_id: int,
    sequence_no: int,
    call: dict[str, Any],
    result: dict[str, Any] | None,
    audit_uid: str,
) -> tuple[int, str]:
    tool_call_uid = uuid4().hex
    compact_result = compact_tool_result(result or {}, audit_uid=audit_uid) if result else {}
    stmt = text(
        """
        INSERT INTO declaration_qa.tool_calls (
            tool_call_uid, agent_run_id, assistant_message_id, external_call_id,
            sequence_no, tool_name, arguments, status, ok, result, error,
            latency_ms, started_at, completed_at, metadata
        )
        VALUES (
            :tool_call_uid, :agent_run_id, :assistant_message_id, :external_call_id,
            :sequence_no, :tool_name, :arguments, :status, :ok, :result, :error,
            :latency_ms, CAST(:started_at AS timestamptz), CAST(:completed_at AS timestamptz), :metadata
        )
        RETURNING id
        """
    ).bindparams(
        bindparam("arguments", type_=JSON),
        bindparam("result", type_=JSON),
        bindparam("metadata", type_=JSON),
    )
    row = connection.execute(
        stmt,
        {
            "tool_call_uid": tool_call_uid,
            "agent_run_id": run_id,
            "assistant_message_id": assistant_message_id,
            "external_call_id": str(call.get("id") or (result or {}).get("call_id") or ""),
            "sequence_no": sequence_no,
            "tool_name": str(call.get("name") or (result or {}).get("tool_name") or "unknown_tool"),
            "arguments": dict(call.get("arguments") or {}),
            "status": str((result or {}).get("status") or "pending"),
            "ok": bool((result or {}).get("ok")),
            "result": compact_result,
            "error": str((result or {}).get("error") or ""),
            "latency_ms": _nullable_int((result or {}).get("latency_ms")),
            "started_at": (result or {}).get("started_at"),
            "completed_at": (result or {}).get("completed_at"),
            "metadata": {"audit_uid": audit_uid, "result_truncated": bool(compact_result.get("truncated"))},
        },
    ).mappings().one()
    return int(row["id"]), tool_call_uid


def _insert_part(
    connection,
    *,
    message_id: int,
    part_index: int,
    part_type: str,
    text_content: str = "",
    asset_uri: str = "",
    mime_type: str = "",
    file_name: str = "",
    size_bytes: int = 0,
    metadata: dict[str, Any] | None = None,
) -> None:
    stmt = text(
        """
        INSERT INTO declaration_qa.message_parts (
            message_id, part_index, part_type, text_content, asset_uri,
            mime_type, file_name, size_bytes, metadata
        )
        VALUES (
            :message_id, :part_index, :part_type, :text_content, :asset_uri,
            :mime_type, :file_name, :size_bytes, :metadata
        )
        """
    ).bindparams(bindparam("metadata", type_=JSON))
    connection.execute(
        stmt,
        {
            "message_id": message_id,
            "part_index": part_index,
            "part_type": part_type,
            "text_content": text_content,
            "asset_uri": asset_uri,
            "mime_type": mime_type,
            "file_name": file_name,
            "size_bytes": max(0, size_bytes),
            "metadata": metadata or {},
        },
    )


def _insert_sources(connection, *, message_id: int, sources: list[dict[str, Any]]) -> None:
    stmt = text(
        """
        INSERT INTO declaration_qa.message_sources (
            message_id, citation_no, chunk_uid, document_uid, source_asset_id,
            title, source_url, page_no, relevance_score, metadata
        )
        VALUES (
            :message_id, :citation_no, :chunk_uid, :document_uid, :source_asset_id,
            :title, :source_url, :page_no, :relevance_score, :metadata
        )
        """
    ).bindparams(bindparam("metadata", type_=JSON))
    for citation_no, source in enumerate((item for item in sources if isinstance(item, dict)), start=1):
        metadata = source.get("metadata") if isinstance(source.get("metadata"), dict) else {}
        page_no = metadata.get("attachment_page_no") or metadata.get("page_no")
        connection.execute(
            stmt,
            {
                "message_id": message_id,
                "citation_no": citation_no,
                "chunk_uid": str(source.get("chunk_id") or metadata.get("chunk_id") or ""),
                "document_uid": str(source.get("doc_id") or metadata.get("document_id") or ""),
                "source_asset_id": str(metadata.get("source_asset_id") or metadata.get("asset_id") or ""),
                "title": str(source.get("title") or ""),
                "source_url": str(source.get("source_url") or metadata.get("source_url") or ""),
                "page_no": _nullable_int(page_no),
                "relevance_score": _safe_float(source.get("rerank_score") or source.get("score")),
                "metadata": {
                    "chunk_type": metadata.get("chunk_type") or metadata.get("semantic_type") or "",
                    "modality": metadata.get("modality") or "",
                    "region_code": source.get("region_code") or metadata.get("region_code") or "",
                    "policy_version": source.get("policy_version") or metadata.get("policy_version") or "",
                },
            },
        )


def _agent_decisions(payload: dict[str, Any]) -> list[dict[str, Any]]:
    decisions = []
    for event in payload.get("events") or []:
        if not isinstance(event, dict) or event.get("event") != "agent_decision":
            continue
        data = event.get("data") if isinstance(event.get("data"), dict) else {}
        if data.get("tool_calls"):
            decisions.append(dict(data))
    if decisions:
        return decisions
    plan = payload.get("agent_plan") if isinstance(payload.get("agent_plan"), dict) else {}
    return [dict(plan)] if plan.get("tool_calls") else []


def _match_tool_result(
    call: dict[str, Any],
    results: list[dict[str, Any]],
    used: set[int],
) -> tuple[int | None, dict[str, Any] | None]:
    call_id = str(call.get("id") or "")
    tool_name = str(call.get("name") or "")
    for index, result in enumerate(results):
        if index in used:
            continue
        if call_id and str(result.get("call_id") or "") == call_id and str(result.get("tool_name") or "") == tool_name:
            return index, result
    for index, result in enumerate(results):
        if index not in used and str(result.get("tool_name") or "") == tool_name:
            return index, result
    return None, None


def _load_message_parts(connection, message_ids: list[int]) -> dict[int, list[dict[str, Any]]]:
    if not message_ids:
        return {}
    rows = connection.execute(
        text(
            """
            SELECT message_id, part_index, part_type, text_content, asset_uri,
                   mime_type, file_name, size_bytes, metadata, created_at
            FROM declaration_qa.message_parts
            WHERE message_id = ANY(:message_ids)
            ORDER BY message_id, part_index
            """
        ),
        {"message_ids": message_ids},
    ).mappings().all()
    output: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        output.setdefault(int(row["message_id"]), []).append(_json_safe_row(row, exclude={"message_id"}))
    return output


def _load_message_sources(connection, message_ids: list[int]) -> dict[int, list[dict[str, Any]]]:
    if not message_ids:
        return {}
    rows = connection.execute(
        text(
            """
            SELECT message_id, citation_no, chunk_uid AS chunk_id,
                   document_uid AS document_id, source_asset_id, title,
                   source_url, page_no, relevance_score, metadata, created_at
            FROM declaration_qa.message_sources
            WHERE message_id = ANY(:message_ids)
            ORDER BY message_id, citation_no
            """
        ),
        {"message_ids": message_ids},
    ).mappings().all()
    output: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        output.setdefault(int(row["message_id"]), []).append(_json_safe_row(row, exclude={"message_id"}))
    return output


def _replay_message(row: Any) -> dict[str, Any] | None:
    """Convert one persisted message row into the model-facing message shape."""
    payload = dict(row)
    message_type = str(payload.get("message_type") or "")
    content_json = payload.get("content_json") if isinstance(payload.get("content_json"), dict) else {}
    if message_type == "input":
        return {"role": "user", "content": str(payload.get("content") or "")}
    if message_type == "output":
        return {"role": "assistant", "content": str(payload.get("content") or "")}
    if message_type == "tool_call":
        calls = []
        for call in content_json.get("tool_calls") or []:
            if not isinstance(call, dict) or not str(call.get("name") or ""):
                continue
            calls.append(
                {
                    "id": str(call.get("id") or call.get("call_id") or call.get("name") or ""),
                    "name": str(call.get("name") or ""),
                    "args": dict(call.get("args") or call.get("arguments") or {}),
                    "type": "tool_call",
                }
            )
        return {
            "role": "assistant",
            "content": str(payload.get("content") or ""),
            "tool_calls": calls,
        }
    if message_type == "tool_result":
        result = content_json.get("result") if isinstance(content_json.get("result"), dict) else {}
        return {
            "role": "tool",
            "tool_call_id": str(payload.get("external_call_id") or content_json.get("tool_call_id") or ""),
            "name": str(result.get("tool_name") or ""),
            "content": result,
        }
    return None


def _json_safe_row(row: Any, *, exclude: set[str] | None = None) -> dict[str, Any]:
    excluded = exclude or set()
    output = {}
    for key, value in dict(row).items():
        if key in excluded:
            continue
        if hasattr(value, "isoformat"):
            value = value.isoformat()
        output[str(key)] = value
    return output


def _is_ticket_call(call: dict[str, Any]) -> bool:
    return str(call.get("name") or "") in {"ticket_list_query", "ticket_detail_query"}


def _is_ticket_result(result: dict[str, Any]) -> bool:
    return str(result.get("tool_name") or "") in {"ticket_list_query", "ticket_detail_query"}


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _nullable_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _token_value(decision: dict[str, Any], key: str) -> int | None:
    usage = decision.get("token_usage") if isinstance(decision.get("token_usage"), dict) else {}
    return _nullable_int(usage.get(key))


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0
