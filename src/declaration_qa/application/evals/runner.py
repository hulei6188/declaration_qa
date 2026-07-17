"""生产评测数据集回放、评分和结果持久化。"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from time import perf_counter
from typing import Any

from declaration_qa.api.schemas import ChatRequest, HumanConfirmationRequest
from declaration_qa.application.evaluation import evaluate_with_ragas, ratio, unique_contexts, unique_doc_ids
from declaration_qa.application.evals.evaluators import (
    evaluate_answer,
    evaluate_outcome,
    evaluate_retrieval,
    performance_scores,
)
from declaration_qa.application.evals.fixtures import EvaluationFixtureManager
from declaration_qa.application.evals.gate import evaluate_gate
from declaration_qa.application.evals.comparison import compare_runs
from declaration_qa.application.evals.trajectory import (
    evaluate_trajectory,
    merge_turn_trajectories,
    normalize_agent_run,
)
from declaration_qa.observability.langfuse import score_trace


@dataclass
class CaseExecution:
    query: str
    answer: str
    refused: bool
    intent: str
    audit_id: str
    sources: list[dict[str, Any]]
    trajectory: dict[str, Any]
    human_confirmation: dict[str, Any]
    status: str
    elapsed_ms: int


class EvaluationRunner:
    """在独立工作流中回放用例并把评测事实写入主评测仓储。"""

    def __init__(
        self,
        *,
        workflow,
        store,
        ragas_evaluator=None,
        ragas_requested: bool = False,
        ragas_unavailable_reason: str = "",
        judge=None,
        langfuse_settings=None,
    ) -> None:
        self.workflow = workflow
        self.store = store
        self.ragas_evaluator = ragas_evaluator
        self.ragas_requested = ragas_requested
        self.ragas_unavailable_reason = ragas_unavailable_reason
        self.judge = judge
        self.langfuse_settings = langfuse_settings
        self.fixtures = EvaluationFixtureManager(workflow.audit_store)

    def run(
        self,
        *,
        dataset_id: str,
        run_id: str | None = None,
        note: str = "",
        top_k: int = 4,
        runtime_snapshot: dict[str, Any] | None = None,
        baseline_run_id: str | None = None,
        trigger_source: str = "manual",
        git_commit: str = "",
    ) -> dict[str, Any]:
        if run_id is None:
            run = self.store.create_eval_run(
                dataset_id=dataset_id,
                note=note,
                top_k=top_k,
                runtime_snapshot=runtime_snapshot or {},
                baseline_run_id=baseline_run_id,
                trigger_source=trigger_source,
                git_commit=git_commit,
            )
            run_id = run["run_id"]
        else:
            marker = getattr(self.store, "mark_eval_run_running", None)
            if callable(marker):
                marker(run_id)
            existing_run = self.store.get_eval_run(run_id) or {}
            baseline_run_id = baseline_run_id or existing_run.get("baseline_run_id")
        items = []
        try:
            for case in self.store.list_eval_cases(dataset_id):
                if not case.get("enabled", True):
                    continue
                self.fixtures.prepare(case)
                try:
                    execution = self.execute_case(case)
                    item, score_records = self.score_case(case, execution, top_k=top_k)
                    persisted = self.store.append_eval_run_item(run_id=run_id, item=item)
                    append_scores = getattr(self.store, "append_eval_scores", None)
                    if callable(append_scores):
                        append_scores(item_id=persisted["item_id"], scores=score_records)
                    self._score_langfuse_traces(case, item, score_records)
                    items.append(item)
                finally:
                    self.fixtures.cleanup(case)
            metrics = summarize_items(items, ragas_requested=self.ragas_requested)
            reasons = [reason for item in items for reason in item.get("failure_reasons") or []]
            baseline = self.store.get_eval_run(baseline_run_id) if baseline_run_id else None
            comparison = compare_runs(baseline, {"run_id": run_id, "metrics": metrics}) if baseline else None
            gate = evaluate_gate(metrics, failure_reasons=reasons, comparison=comparison)
            if comparison:
                gate["comparison"] = comparison
            return self.store.finish_eval_run(
                run_id=run_id,
                status="succeeded",
                metrics=metrics,
                gate_result=gate,
            )
        except Exception as exc:
            return self.store.finish_eval_run(
                run_id=run_id,
                status="failed",
                metrics={},
                error_message=f"{type(exc).__name__}: {str(exc)[:1000]}",
                gate_result={"passed": False, "hard_failures": ["evaluation_runtime_failure"]},
            )

    def execute_case(self, case: dict[str, Any]) -> CaseExecution:
        started = perf_counter()
        turns = list(case.get("turns") or []) or [{"type": "user", "content": case.get("query") or ""}]
        session_id = None
        trajectories = []
        latest: dict[str, Any] = {
            "query": str(case.get("query") or ""),
            "answer": "",
            "refused": False,
            "intent": "unknown",
            "audit_id": "",
            "sources": [],
            "human_confirmation": {},
            "status": "completed",
        }
        pending_confirmation: dict[str, Any] = {}
        turn_index = 0
        for turn in turns:
            turn_type = str(turn.get("type") or "user")
            if turn_type == "user":
                turn_index += 1
                request = ChatRequest(
                    query=str(turn.get("content") or case.get("query") or ""),
                    user_id=int(case.get("user_id") or 1),
                    session_id=session_id,
                    region_code=str(case.get("region_code") or "340000"),
                    policy_version=str(case.get("policy_version") or "2025"),
                    biz_type=case.get("biz_type"),
                )
                response = self.workflow.invoke(request)
                session_id = response.session_id
                latest = response.model_dump(mode="json")
                latest["query"] = request.query
                pending_confirmation = dict(response.human_confirmation or {})
                if response.audit_id:
                    run = self.workflow.audit_store.get_agent_run(
                        user_id=int(case.get("user_id") or 1),
                        run_id=response.audit_id,
                    )
                    trajectories.append(normalize_agent_run(run, turn_index=turn_index))
                elif pending_confirmation:
                    trajectories.append(trajectory_from_pending_response(response, turn_index=turn_index))
                    latest["status"] = "pending_confirmation"
            elif turn_type == "confirmation":
                if not pending_confirmation:
                    raise ValueError("confirmation turn has no pending human confirmation")
                result = self._resume_confirmation(
                    pending_confirmation,
                    user_id=int(case.get("user_id") or 1),
                    approved=str(turn.get("action") or "approve") == "approve",
                    selected_slot_ids=dict(turn.get("selected_slot_ids") or {}),
                )
                latest.update(result)
                pending_confirmation = dict(result.get("human_confirmation") or {})
                if result.get("audit_id"):
                    run = self.workflow.audit_store.get_agent_run(
                        user_id=int(case.get("user_id") or 1),
                        run_id=str(result["audit_id"]),
                    )
                    if trajectories and trajectories[-1].get("run_id") == "":
                        trajectories.pop()
                    trajectories.append(normalize_agent_run(run, turn_index=turn_index))
        trajectory = merge_turn_trajectories(trajectories)
        return CaseExecution(
            query=str(latest.get("query") or case.get("query") or ""),
            answer=str(latest.get("answer") or ""),
            refused=bool(latest.get("refused")),
            intent=str(latest.get("intent") or "unknown"),
            audit_id=str(latest.get("audit_id") or ""),
            sources=list(latest.get("sources") or []),
            trajectory=trajectory,
            human_confirmation=dict(latest.get("human_confirmation") or pending_confirmation or {}),
            status=str(latest.get("status") or "completed"),
            elapsed_ms=int((perf_counter() - started) * 1000),
        )

    def _resume_confirmation(
        self,
        confirmation: dict[str, Any],
        *,
        user_id: int,
        approved: bool,
        selected_slot_ids: dict[str, int],
    ) -> dict[str, Any]:
        request = HumanConfirmationRequest(
            session_id=str(confirmation.get("session_id") or ""),
            interrupt_id=str(confirmation.get("interrupt_id") or ""),
            user_id=user_id,
            approved=approved,
            selected_slot_ids=selected_slot_ids,
        )
        answer_parts = []
        sources = []
        done = {}
        pending = {}
        for event in self.workflow.resume_stream(request):
            name = str(event.get("event") or "")
            data = dict(event.get("data") or {})
            if name == "token":
                answer_parts.append(str(data.get("content") or ""))
            elif name == "sources":
                sources = list(data.get("items") or [])
            elif name == "done":
                done = data
            elif name == "human_confirmation_required":
                pending = data
        return {
            **done,
            "answer": "".join(answer_parts),
            "sources": sources,
            "human_confirmation": pending,
            "status": "pending_confirmation" if pending else "completed",
        }

    def score_case(
        self, case: dict[str, Any], execution: CaseExecution, *, top_k: int
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        retrieval, retrieval_failures, retrieval_records = evaluate_retrieval(
            execution.sources,
            expected_doc_ids=list(case.get("expected_doc_ids") or []),
            expected_chunk_ids=list(case.get("expected_chunk_ids") or []),
            top_k=top_k,
        )
        answer, answer_failures, answer_records = evaluate_answer(
            answer=execution.answer,
            refused=execution.refused,
            expected_refused=case.get("expected_refused"),
            expected_keywords=list(case.get("expected_answer_keywords") or []),
            sources=execution.sources,
        )
        trajectory_scores, trajectory_failures, trajectory_records = evaluate_trajectory(
            execution.trajectory,
            dict(case.get("expected_trajectory") or {}),
            user_id=int(case.get("user_id") or 1),
        )
        outcome, outcome_failures, outcome_records = evaluate_outcome(
            {
                "status": execution.status,
                "human_confirmation": execution.human_confirmation,
                "trajectory": execution.trajectory,
            },
            dict(case.get("expected_outcome") or {}),
        )
        contexts = unique_contexts(
            str(source.get("evidence_text") or source.get("snippet") or "")
            for source in execution.sources[:top_k]
        )
        ragas = evaluate_with_ragas(
            self.ragas_evaluator,
            query=execution.query,
            answer=execution.answer,
            retrieved_contexts=contexts,
            reference_answer=str(case.get("reference_answer") or ""),
            reference_contexts=list(case.get("reference_contexts") or []),
            refused=execution.refused,
            requested=self.ragas_requested,
            unavailable_reason=self.ragas_unavailable_reason,
        )
        judge = (
            self.judge.evaluate(
                query=execution.query,
                answer=execution.answer,
                contexts=contexts,
                reference_answer=str(case.get("reference_answer") or ""),
            )
            if self.judge is not None
            else {"scores": {}, "status": {"status": "disabled"}, "records": []}
        )
        failures = list(dict.fromkeys(retrieval_failures + answer_failures + trajectory_failures + outcome_failures))
        if case.get("expected_intent") and str(case.get("expected_intent")) != execution.intent:
            failures.append("intent_mismatch")
        applicable = [value for value in {**retrieval, **answer, **trajectory_scores, **outcome}.values() if isinstance(value, (int, float))]
        passed = not failures and all(value >= 1.0 for value in applicable if value in {0.0, 1.0})
        performance = performance_scores(execution.trajectory)
        performance["latency_ms"] = max(performance["latency_ms"], execution.elapsed_ms)
        item = {
            "case_id": str(case.get("case_id") or ""),
            "query": execution.query,
            "expected_intent": str(case.get("expected_intent") or ""),
            "actual_intent": execution.intent,
            "expected_refused": case.get("expected_refused"),
            "actual_refused": execution.refused,
            "expected_doc_ids": list(case.get("expected_doc_ids") or []),
            "retrieved_doc_ids": unique_doc_ids(source.get("doc_id") for source in execution.sources)[:top_k],
            "expected_answer_keywords": list(case.get("expected_answer_keywords") or []),
            "reference_answer": str(case.get("reference_answer") or ""),
            "reference_contexts": list(case.get("reference_contexts") or []),
            "retrieved_contexts": contexts,
            "answer": execution.answer,
            "audit_id": execution.audit_id,
            "recall_hit": retrieval.get("retrieval_hit_at_k") == 1.0 if retrieval.get("retrieval_hit_at_k") is not None else None,
            "refusal_match": answer.get("refusal_accuracy") == 1.0 if answer.get("refusal_accuracy") is not None else None,
            "intent_match": (str(case.get("expected_intent")) == execution.intent) if case.get("expected_intent") else None,
            "keyword_hit": answer.get("answer_keyword_coverage") == 1.0 if answer.get("answer_keyword_coverage") is not None else None,
            "ragas_scores": dict(ragas.get("scores") or {}),
            "ragas_status": dict(ragas.get("status") or {}),
            "agent_run_uid": execution.audit_id,
            "actual_trajectory": execution.trajectory,
            "rule_scores": {**retrieval, **answer, **trajectory_scores, **outcome},
            "judge_scores": {"scores": judge.get("scores") or {}, "status": judge.get("status") or {}},
            "failure_reasons": failures,
            "passed": passed,
            **performance,
            "estimated_cost": None,
        }
        ragas_records = [
            {
                "metric_name": name,
                "evaluator_type": "ragas",
                "evaluator_version": "0.3.9",
                "score": value,
                "label": "pass" if float(value) >= 0.95 else "fail",
                "reason": "",
                "raw_result": ragas.get("status") or {},
            }
            for name, value in (ragas.get("scores") or {}).items()
        ]
        records = retrieval_records + answer_records + trajectory_records + outcome_records + ragas_records + list(judge.get("records") or [])
        return item, records

    def _score_langfuse_traces(
        self,
        case: dict[str, Any],
        item: dict[str, Any],
        records: list[dict[str, Any]],
    ) -> None:
        trace_ids = list((item.get("actual_trajectory") or {}).get("trace_ids") or [])
        for trace_id in trace_ids:
            for record in records:
                value = record.get("score")
                if not isinstance(value, (int, float)):
                    continue
                score_trace(
                    str(trace_id),
                    name=f"eval.{record.get('metric_name')}",
                    value=float(value),
                    comment=str(record.get("reason") or "")[:500],
                    settings=self.langfuse_settings,
                )
            score_trace(
                str(trace_id),
                name="eval.case_passed",
                value=1 if item.get("passed") else 0,
                comment=f"case_id={case.get('case_id')}; severity={case.get('severity')}",
                settings=self.langfuse_settings,
            )


def trajectory_from_pending_response(response, *, turn_index: int) -> dict[str, Any]:
    calls = []
    for index, result in enumerate(response.tool_results or [], start=1):
        calls.append(
            {
                "turn_index": turn_index,
                "sequence_no": index,
                "tool_name": str(result.get("tool_name") or ""),
                "arguments": {},
                "status": str(result.get("status") or ""),
                "ok": bool(result.get("ok")),
                "result": dict(result),
                "error": str(result.get("error") or ""),
                "latency_ms": 0,
            }
        )
    for action in (response.human_confirmation or {}).get("actions") or []:
        calls.append(
            {
                "turn_index": turn_index,
                "sequence_no": len(calls) + 1,
                "tool_name": str(action.get("tool_name") or ""),
                "arguments": dict(action.get("arguments") or {}),
                "status": "pending_confirmation",
                "ok": False,
                "result": {},
                "error": "",
                "latency_ms": 0,
            }
        )
    return {"run_id": "", "status": "paused", "iteration_count": 0, "calls": calls, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "latency_ms": 0}


def summarize_items(items: list[dict[str, Any]], *, ragas_requested: bool) -> dict[str, Any]:
    rules: dict[str, list[float]] = {}
    ragas: dict[str, list[float]] = {}
    judge: dict[str, list[float]] = {}
    for item in items:
        for name, value in (item.get("rule_scores") or {}).items():
            if isinstance(value, (int, float)):
                rules.setdefault(name, []).append(float(value))
        for name, value in (item.get("ragas_scores") or {}).items():
            if isinstance(value, (int, float)):
                ragas.setdefault(name, []).append(float(value))
        for name, value in ((item.get("judge_scores") or {}).get("scores") or {}).items():
            if isinstance(value, (int, float)):
                judge.setdefault(name, []).append(float(value))
    passed = sum(1 for item in items if item.get("passed") is True)
    latencies = sorted(int(item.get("latency_ms") or 0) for item in items)
    metrics = {
        "case_count": len(items),
        "recall_at_k": ratio(item.get("recall_hit") for item in items),
        "refusal_accuracy": ratio(item.get("refusal_match") for item in items),
        "intent_accuracy": ratio(item.get("intent_match") for item in items),
        "answer_keyword_hit_rate": ratio(item.get("keyword_hit") for item in items),
        "task_success_rate": round(passed / len(items), 6) if items else None,
        "rules": {name: round(mean(values), 6) for name, values in rules.items()},
        "ragas": {
            "enabled": ragas_requested,
            "evaluated_case_count": max((len(values) for values in ragas.values()), default=0),
            "scores": {name: round(mean(values), 6) for name, values in ragas.items()},
        },
        "judge": {name: round(mean(values), 6) for name, values in judge.items()},
        "performance": {
            "total_tokens": sum(int(item.get("total_tokens") or 0) for item in items),
            "average_tool_calls": round(mean([len((item.get("actual_trajectory") or {}).get("calls") or []) for item in items]), 6) if items else 0,
            "p50_latency_ms": percentile(latencies, 0.50),
            "p95_latency_ms": percentile(latencies, 0.95),
        },
    }
    for name, value in metrics["rules"].items():
        metrics.setdefault(name, value)
    for name, value in metrics["ragas"]["scores"].items():
        metrics.setdefault(name, value)
    for name, value in metrics["judge"].items():
        metrics.setdefault(name, value)
    return metrics


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    return values[min(len(values) - 1, max(0, round((len(values) - 1) * fraction)))]
