"""评测数据集、运行、评分和线上候选样本存储。"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from sqlalchemy import JSON, bindparam, text

from gov_service_agent.storage.repository.base import EngineBackedRepository
from gov_service_agent.storage.repository.normalizers import (
    normalize_eval_candidate,
    normalize_eval_case,
    normalize_eval_dataset,
    normalize_eval_run,
    normalize_eval_run_item,
)


CASE_JSON_FIELDS = (
    "expected_doc_ids",
    "expected_answer_keywords",
    "reference_contexts",
    "tags",
    "turns",
    "expected_chunk_ids",
    "expected_trajectory",
    "expected_outcome",
)

RUN_ITEM_JSON_FIELDS = (
    "expected_doc_ids",
    "retrieved_doc_ids",
    "expected_answer_keywords",
    "reference_contexts",
    "retrieved_contexts",
    "ragas_scores",
    "ragas_status",
    "actual_trajectory",
    "rule_scores",
    "judge_scores",
    "failure_reasons",
)


class EvaluationRepository(EngineBackedRepository):
    """生产级评测数据集、运行、评分和标注仓储。"""

    def create_eval_dataset(
        self,
        *,
        name: str,
        description: str,
        cases: list[dict[str, Any]],
        version: int = 1,
        status: str = "draft",
        source: str = "manual",
        parent_dataset_id: str | None = None,
        knowledge_revision: str = "",
    ) -> dict[str, Any]:
        dataset_uid = f"eval-ds-{uuid4().hex}"
        with self.engine.begin() as connection:
            row = connection.execute(
                text(
                    """
                    INSERT INTO declaration_qa.eval_datasets (
                        dataset_uid, name, description, version, status, source,
                        parent_dataset_id, knowledge_revision
                    )
                    SELECT :dataset_uid, :name, :description, :version, :status, :source,
                           parent.id, :knowledge_revision
                    FROM (SELECT 1) seed
                    LEFT JOIN declaration_qa.eval_datasets parent
                      ON parent.dataset_uid = :parent_dataset_uid
                    RETURNING dataset_uid AS dataset_id, name, description, version, status,
                              source, :parent_dataset_uid AS parent_dataset_id,
                              knowledge_revision, created_at
                    """
                ),
                {
                    "dataset_uid": dataset_uid,
                    "name": name,
                    "description": description,
                    "version": int(version),
                    "status": "draft" if cases and status == "published" else status,
                    "source": source,
                    "parent_dataset_uid": parent_dataset_id or "",
                    "knowledge_revision": knowledge_revision,
                },
            ).mappings().first()
        for case in cases:
            self.add_eval_case(dataset_id=dataset_uid, case=case)
        if cases and status == "published":
            self.publish_eval_dataset(dataset_uid)
        result = normalize_eval_dataset(row)
        result["status"] = status
        result["case_count"] = len(cases)
        return result

    def publish_eval_dataset(self, dataset_id: str) -> dict[str, Any]:
        """发布数据集；发布后用例不可变。"""
        with self.engine.begin() as connection:
            row = connection.execute(
                text(
                    """
                    UPDATE declaration_qa.eval_datasets
                    SET status = 'published', updated_at = CURRENT_TIMESTAMP
                    WHERE dataset_uid = :dataset_uid AND status = 'draft'
                    RETURNING dataset_uid AS dataset_id, name, description, version, status,
                              source, knowledge_revision, created_at,
                              (SELECT dataset_uid FROM declaration_qa.eval_datasets p
                               WHERE p.id = parent_dataset_id) AS parent_dataset_id
                    """
                ),
                {"dataset_uid": dataset_id},
            ).mappings().first()
        if not row:
            raise ValueError("Evaluation dataset not found or is not draft")
        return {**normalize_eval_dataset(row), "case_count": len(self.list_eval_cases(dataset_id))}

    def list_eval_datasets(self) -> list[dict[str, Any]]:
        with self.engine.begin() as connection:
            rows = connection.execute(
                text(
                    """
                    SELECT d.dataset_uid AS dataset_id, d.name, d.description, d.version,
                           d.status, d.source, p.dataset_uid AS parent_dataset_id,
                           d.knowledge_revision, d.created_at, COUNT(c.id)::int AS case_count
                    FROM declaration_qa.eval_datasets d
                    LEFT JOIN declaration_qa.eval_datasets p ON p.id = d.parent_dataset_id
                    LEFT JOIN declaration_qa.eval_cases c ON c.dataset_id = d.id
                    GROUP BY d.id, p.dataset_uid
                    ORDER BY d.created_at DESC
                    """
                )
            ).mappings().all()
        return [normalize_eval_dataset(row) for row in rows]

    def add_eval_case(self, *, dataset_id: str, case: dict[str, Any]) -> dict[str, Any]:
        case_uid = f"eval-case-{uuid4().hex}"
        stmt = text(
            """
            INSERT INTO declaration_qa.eval_cases (
                case_uid, dataset_id, query, user_id, region_code, policy_version,
                biz_type, apply_id, apply_context, expected_intent, expected_refused, expected_doc_ids,
                expected_answer_keywords, reference_answer, reference_contexts, tags,
                case_type, turns, fixture_key, expected_chunk_ids, expected_trajectory,
                expected_outcome, severity, enabled
            )
            SELECT :case_uid, d.id, :query, :user_id, :region_code, :policy_version,
                   :biz_type, :apply_id, :apply_context, :expected_intent, :expected_refused, :expected_doc_ids,
                   :expected_answer_keywords, :reference_answer, :reference_contexts, :tags,
                   :case_type, :turns, :fixture_key, :expected_chunk_ids,
                   :expected_trajectory, :expected_outcome, :severity, :enabled
            FROM declaration_qa.eval_datasets d
            WHERE d.dataset_uid = :dataset_uid AND d.status = 'draft'
            RETURNING id AS db_id, case_uid AS case_id, :dataset_uid AS dataset_id,
                      query, user_id, region_code, policy_version, biz_type, apply_id, apply_context, expected_intent,
                      expected_refused, expected_doc_ids, expected_answer_keywords,
                      reference_answer, reference_contexts, tags, case_type, turns,
                      fixture_key, expected_chunk_ids, expected_trajectory,
                      expected_outcome, severity, enabled, created_at
            """
        )
        for field in CASE_JSON_FIELDS:
            stmt = stmt.bindparams(bindparam(field, type_=JSON))
        stmt = stmt.bindparams(bindparam("apply_context", type_=JSON))
        params = self._case_params(case_uid, dataset_id, case)
        with self.engine.begin() as connection:
            row = connection.execute(stmt, params).mappings().first()
            if not row:
                raise ValueError("Evaluation dataset not found or is immutable")
            connection.execute(
                text(
                    """
                    INSERT INTO declaration_qa.eval_case_revisions (
                        revision_uid, case_id, revision_no, snapshot, change_note
                    ) VALUES (:revision_uid, :case_id, 1, :snapshot, 'initial')
                    """
                ).bindparams(bindparam("snapshot", type_=JSON)),
                {
                    "revision_uid": f"eval-rev-{uuid4().hex}",
                    "case_id": row["db_id"],
                    "snapshot": json_safe({key: value for key, value in dict(row).items() if key != "db_id"}),
                },
            )
        return normalize_eval_case(row)

    def list_eval_cases(self, dataset_id: str) -> list[dict[str, Any]]:
        with self.engine.begin() as connection:
            rows = connection.execute(
                text(
                    """
                    SELECT c.case_uid AS case_id, d.dataset_uid AS dataset_id, c.query,
                           c.user_id, c.region_code, c.policy_version, c.biz_type,
                           c.apply_id, c.apply_context,
                           c.expected_intent, c.expected_refused, c.expected_doc_ids,
                           c.expected_answer_keywords, c.reference_answer,
                           c.reference_contexts, c.tags, c.case_type, c.turns,
                           c.fixture_key, c.expected_chunk_ids, c.expected_trajectory,
                           c.expected_outcome, c.severity, c.enabled, c.created_at
                    FROM declaration_qa.eval_cases c
                    JOIN declaration_qa.eval_datasets d ON d.id = c.dataset_id
                    WHERE d.dataset_uid = :dataset_uid
                    ORDER BY c.created_at, c.id
                    """
                ),
                {"dataset_uid": dataset_id},
            ).mappings().all()
        return [normalize_eval_case(row) for row in rows]

    def create_eval_run(
        self,
        *,
        dataset_id: str,
        note: str,
        top_k: int,
        baseline_run_id: str | None = None,
        trigger_source: str = "manual",
        runtime_snapshot: dict[str, Any] | None = None,
        git_commit: str = "",
        status: str = "running",
    ) -> dict[str, Any]:
        run_uid = f"eval-run-{uuid4().hex}"
        stmt = text(
            """
            INSERT INTO declaration_qa.eval_runs (
                run_uid, dataset_id, status, note, top_k, baseline_run_id,
                trigger_source, runtime_snapshot, git_commit
            )
            SELECT :run_uid, d.id, :status, :note, :top_k, baseline.id,
                   :trigger_source, :runtime_snapshot, :git_commit
            FROM declaration_qa.eval_datasets d
            LEFT JOIN declaration_qa.eval_runs baseline
              ON baseline.run_uid = :baseline_run_uid
            WHERE d.dataset_uid = :dataset_uid
            RETURNING run_uid AS run_id, :dataset_uid AS dataset_id, status, metrics,
                      :baseline_run_uid AS baseline_run_id, trigger_source,
                      runtime_snapshot, gate_result, git_commit, error_message,
                      created_at, completed_at
            """
        ).bindparams(bindparam("runtime_snapshot", type_=JSON))
        with self.engine.begin() as connection:
            row = connection.execute(
                stmt,
                {
                    "run_uid": run_uid,
                    "dataset_uid": dataset_id,
                    "status": status,
                    "note": note,
                    "top_k": int(top_k),
                    "baseline_run_uid": baseline_run_id or "",
                    "trigger_source": trigger_source,
                    "runtime_snapshot": runtime_snapshot or {},
                    "git_commit": git_commit,
                },
            ).mappings().first()
        if not row:
            raise ValueError("Evaluation dataset not found")
        return normalize_eval_run(row, items=[])

    def mark_eval_run_running(self, run_id: str) -> None:
        with self.engine.begin() as connection:
            connection.execute(
                text("UPDATE declaration_qa.eval_runs SET status = 'running' WHERE run_uid = :run_uid AND status = 'queued'"),
                {"run_uid": run_id},
            )

    def append_eval_run_item(self, *, run_id: str, item: dict[str, Any]) -> dict[str, Any]:
        item_uid = f"eval-item-{uuid4().hex}"
        stmt = text(
            """
            INSERT INTO declaration_qa.eval_run_items (
                item_uid, run_id, case_id, query, expected_intent, actual_intent,
                expected_refused, actual_refused, expected_doc_ids, retrieved_doc_ids,
                expected_answer_keywords, reference_answer, reference_contexts,
                retrieved_contexts, answer, audit_uid, recall_hit, refusal_match,
                intent_match, keyword_hit, ragas_scores, ragas_status, agent_run_uid,
                actual_trajectory, rule_scores, judge_scores, failure_reasons, passed,
                input_tokens, output_tokens, total_tokens, latency_ms, estimated_cost
            )
            SELECT :item_uid, r.id, c.id, :query, :expected_intent, :actual_intent,
                   :expected_refused, :actual_refused, :expected_doc_ids, :retrieved_doc_ids,
                   :expected_answer_keywords, :reference_answer, :reference_contexts,
                   :retrieved_contexts, :answer, :audit_uid, :recall_hit, :refusal_match,
                   :intent_match, :keyword_hit, :ragas_scores, :ragas_status, :agent_run_uid,
                   :actual_trajectory, :rule_scores, :judge_scores, :failure_reasons, :passed,
                   :input_tokens, :output_tokens, :total_tokens, :latency_ms, :estimated_cost
            FROM declaration_qa.eval_runs r
            LEFT JOIN declaration_qa.eval_cases c ON c.case_uid = :case_uid
            WHERE r.run_uid = :run_uid
            RETURNING item_uid AS item_id, :case_uid AS case_id, query, expected_intent,
                      actual_intent, expected_refused, actual_refused, expected_doc_ids,
                      retrieved_doc_ids, expected_answer_keywords, reference_answer,
                      reference_contexts, retrieved_contexts, answer, audit_uid,
                      recall_hit, refusal_match, intent_match, keyword_hit, ragas_scores,
                      ragas_status, agent_run_uid, actual_trajectory, rule_scores,
                      judge_scores, failure_reasons, passed, input_tokens, output_tokens,
                      total_tokens, latency_ms, estimated_cost, annotation_label,
                      annotation_comment
            """
        )
        for field in RUN_ITEM_JSON_FIELDS:
            stmt = stmt.bindparams(bindparam(field, type_=JSON))
        params = {field: item.get(field) for field in item}
        params.update(
            {
                "item_uid": item_uid,
                "run_uid": run_id,
                "case_uid": str(item.get("case_id") or ""),
                "query": str(item.get("query") or ""),
                "expected_intent": str(item.get("expected_intent") or ""),
                "actual_intent": str(item.get("actual_intent") or ""),
                "expected_refused": item.get("expected_refused"),
                "actual_refused": bool(item.get("actual_refused")),
                "expected_doc_ids": list(item.get("expected_doc_ids") or []),
                "retrieved_doc_ids": list(item.get("retrieved_doc_ids") or []),
                "expected_answer_keywords": list(item.get("expected_answer_keywords") or []),
                "reference_answer": str(item.get("reference_answer") or ""),
                "reference_contexts": list(item.get("reference_contexts") or []),
                "retrieved_contexts": list(item.get("retrieved_contexts") or []),
                "answer": str(item.get("answer") or ""),
                "audit_uid": str(item.get("audit_id") or ""),
                "ragas_scores": dict(item.get("ragas_scores") or {}),
                "ragas_status": dict(item.get("ragas_status") or {}),
                "agent_run_uid": str(item.get("agent_run_uid") or ""),
                "actual_trajectory": dict(item.get("actual_trajectory") or {}),
                "rule_scores": dict(item.get("rule_scores") or {}),
                "judge_scores": dict(item.get("judge_scores") or {}),
                "failure_reasons": list(item.get("failure_reasons") or []),
                "recall_hit": item.get("recall_hit"),
                "refusal_match": item.get("refusal_match"),
                "intent_match": item.get("intent_match"),
                "keyword_hit": item.get("keyword_hit"),
                "passed": item.get("passed"),
                "input_tokens": item.get("input_tokens"),
                "output_tokens": item.get("output_tokens"),
                "total_tokens": item.get("total_tokens"),
                "latency_ms": item.get("latency_ms"),
                "estimated_cost": item.get("estimated_cost"),
            }
        )
        with self.engine.begin() as connection:
            row = connection.execute(stmt, params).mappings().first()
        if not row:
            raise ValueError("Evaluation run not found")
        return normalize_eval_run_item(row)

    def append_eval_scores(self, *, item_id: str, scores: list[dict[str, Any]]) -> None:
        stmt = text(
            """
            INSERT INTO declaration_qa.eval_scores (
                score_uid, run_item_id, metric_name, evaluator_type, evaluator_version,
                score, label, reason, raw_result
            )
            SELECT :score_uid, i.id, :metric_name, :evaluator_type, :evaluator_version,
                   :score, :label, :reason, :raw_result
            FROM declaration_qa.eval_run_items i WHERE i.item_uid = :item_uid
            """
        ).bindparams(bindparam("raw_result", type_=JSON))
        with self.engine.begin() as connection:
            for score in scores:
                connection.execute(
                    stmt,
                    {
                        "score_uid": f"eval-score-{uuid4().hex}",
                        "item_uid": item_id,
                        "metric_name": str(score.get("metric_name") or ""),
                        "evaluator_type": str(score.get("evaluator_type") or "rule"),
                        "evaluator_version": str(score.get("evaluator_version") or "v1"),
                        "score": score.get("score"),
                        "label": str(score.get("label") or ""),
                        "reason": str(score.get("reason") or ""),
                        "raw_result": dict(score.get("raw_result") or {}),
                    },
                )

    def finish_eval_run(
        self,
        *,
        run_id: str,
        status: str,
        metrics: dict[str, Any],
        error_message: str = "",
        gate_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        stmt = text(
            """
            UPDATE declaration_qa.eval_runs SET status = :status, metrics = :metrics,
                error_message = :error_message, gate_result = :gate_result,
                completed_at = CURRENT_TIMESTAMP
            WHERE run_uid = :run_uid
            """
        ).bindparams(bindparam("metrics", type_=JSON), bindparam("gate_result", type_=JSON))
        with self.engine.begin() as connection:
            result = connection.execute(
                stmt,
                {
                    "run_uid": run_id,
                    "status": status,
                    "metrics": metrics,
                    "error_message": error_message,
                    "gate_result": gate_result or {},
                },
            )
        if result.rowcount != 1:
            raise ValueError("Evaluation run not found")
        return self.get_eval_run(run_id) or {}

    def get_eval_run(self, run_id: str) -> dict[str, Any] | None:
        with self.engine.begin() as connection:
            run = connection.execute(
                text(
                    """
                    SELECT r.run_uid AS run_id, d.dataset_uid AS dataset_id, r.status,
                           r.metrics, r.error_message, baseline.run_uid AS baseline_run_id,
                           r.trigger_source, r.runtime_snapshot, r.gate_result, r.git_commit,
                           r.created_at, r.completed_at
                    FROM declaration_qa.eval_runs r
                    JOIN declaration_qa.eval_datasets d ON d.id = r.dataset_id
                    LEFT JOIN declaration_qa.eval_runs baseline ON baseline.id = r.baseline_run_id
                    WHERE r.run_uid = :run_uid
                    """
                ),
                {"run_uid": run_id},
            ).mappings().first()
            if not run:
                return None
            items = connection.execute(
                text(
                    """
                    SELECT i.item_uid AS item_id, COALESCE(c.case_uid, '') AS case_id,
                           i.query, i.expected_intent, i.actual_intent,
                           i.expected_refused, i.actual_refused, i.expected_doc_ids,
                           i.retrieved_doc_ids, i.expected_answer_keywords,
                           i.reference_answer, i.reference_contexts, i.retrieved_contexts,
                           i.answer, i.audit_uid, i.recall_hit, i.refusal_match,
                           i.intent_match, i.keyword_hit, i.ragas_scores, i.ragas_status,
                           i.agent_run_uid, i.actual_trajectory, i.rule_scores,
                           i.judge_scores, i.failure_reasons, i.passed, i.input_tokens,
                           i.output_tokens, i.total_tokens, i.latency_ms, i.estimated_cost,
                           i.annotation_label, i.annotation_comment
                    FROM declaration_qa.eval_run_items i
                    LEFT JOIN declaration_qa.eval_cases c ON c.id = i.case_id
                    WHERE i.run_id = (SELECT id FROM declaration_qa.eval_runs WHERE run_uid = :run_uid)
                    ORDER BY i.id
                    """
                ),
                {"run_uid": run_id},
            ).mappings().all()
        return normalize_eval_run(run, items=[normalize_eval_run_item(item) for item in items])

    def list_eval_runs(self, *, dataset_id: str, limit: int = 50) -> list[dict[str, Any]]:
        with self.engine.begin() as connection:
            rows = connection.execute(
                text(
                    """
                    SELECT r.run_uid AS run_id, d.dataset_uid AS dataset_id, r.status,
                           r.metrics, r.error_message, baseline.run_uid AS baseline_run_id,
                           r.trigger_source, r.runtime_snapshot, r.gate_result, r.git_commit,
                           r.created_at, r.completed_at
                    FROM declaration_qa.eval_runs r
                    JOIN declaration_qa.eval_datasets d ON d.id = r.dataset_id
                    LEFT JOIN declaration_qa.eval_runs baseline ON baseline.id = r.baseline_run_id
                    WHERE d.dataset_uid = :dataset_uid
                    ORDER BY r.created_at DESC LIMIT :limit
                    """
                ),
                {"dataset_uid": dataset_id, "limit": max(1, min(int(limit), 200))},
            ).mappings().all()
        return [normalize_eval_run(row, items=[]) for row in rows]

    def annotate_eval_item(self, *, item_id: str, label: str, comment: str) -> dict[str, Any]:
        with self.engine.begin() as connection:
            result = connection.execute(
                text(
                    """
                    UPDATE declaration_qa.eval_run_items SET annotation_label = :label,
                        annotation_comment = :comment, annotated_at = CURRENT_TIMESTAMP
                    WHERE item_uid = :item_uid
                    """
                ),
                {"item_uid": item_id, "label": label, "comment": comment},
            )
        if result.rowcount != 1:
            raise ValueError("Evaluation run item not found")
        return self._get_eval_item(item_id)

    def create_eval_candidate(self, candidate: dict[str, Any]) -> dict[str, Any]:
        candidate_uid = f"eval-candidate-{uuid4().hex}"
        stmt = text(
            """
            INSERT INTO declaration_qa.eval_candidates (
                candidate_uid, audit_uid, agent_run_uid, trace_id, source, severity,
                sanitized_query, sanitized_answer, detection_signals, trajectory_snapshot
            ) VALUES (
                :candidate_uid, :audit_uid, :agent_run_uid, :trace_id, :source, :severity,
                :sanitized_query, :sanitized_answer, :detection_signals, :trajectory_snapshot
            )
            RETURNING candidate_uid AS candidate_id, *
            """
        ).bindparams(
            bindparam("detection_signals", type_=JSON),
            bindparam("trajectory_snapshot", type_=JSON),
        )
        with self.engine.begin() as connection:
            row = connection.execute(stmt, {"candidate_uid": candidate_uid, **candidate}).mappings().first()
        return normalize_eval_candidate(row)

    def list_eval_candidates(self, *, status: str = "", limit: int = 100) -> list[dict[str, Any]]:
        with self.engine.begin() as connection:
            rows = connection.execute(
                text(
                    """
                    SELECT candidate_uid AS candidate_id, *
                    FROM declaration_qa.eval_candidates
                    WHERE (:status = '' OR status = :status)
                    ORDER BY created_at DESC LIMIT :limit
                    """
                ),
                {"status": status, "limit": max(1, min(int(limit), 500))},
            ).mappings().all()
        return [normalize_eval_candidate(row) for row in rows]

    def review_eval_candidate(
        self,
        *,
        candidate_id: str,
        status: str,
        severity: str,
        labels: list[str],
        comment: str,
        reviewer_id: int,
    ) -> dict[str, Any]:
        stmt = text(
            """
            UPDATE declaration_qa.eval_candidates SET status = :status, severity = :severity,
                labels = :labels, reviewer_comment = :comment, reviewed_by = :reviewer_id,
                reviewed_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
            WHERE candidate_uid = :candidate_uid AND status <> 'promoted'
            RETURNING candidate_uid AS candidate_id, *
            """
        ).bindparams(bindparam("labels", type_=JSON))
        with self.engine.begin() as connection:
            row = connection.execute(
                stmt,
                {
                    "candidate_uid": candidate_id,
                    "status": status,
                    "severity": severity,
                    "labels": labels,
                    "comment": comment,
                    "reviewer_id": reviewer_id,
                },
            ).mappings().first()
        if not row:
            raise ValueError("Evaluation candidate not found or already promoted")
        return normalize_eval_candidate(row)

    def promote_eval_candidate(self, *, candidate_id: str, dataset_id: str, case: dict[str, Any]) -> dict[str, Any]:
        with self.engine.begin() as connection:
            promotable = connection.execute(
                text("SELECT 1 FROM declaration_qa.eval_candidates WHERE candidate_uid = :uid AND status IN ('triaged', 'labeled')"),
                {"uid": candidate_id},
            ).first()
        if not promotable:
            raise ValueError("Evaluation candidate must be reviewed before promotion")
        created = self.add_eval_case(dataset_id=dataset_id, case=case)
        with self.engine.begin() as connection:
            row = connection.execute(
                text(
                    """
                    UPDATE declaration_qa.eval_candidates SET status = 'promoted',
                        promoted_case_id = (SELECT id FROM declaration_qa.eval_cases WHERE case_uid = :case_uid),
                        updated_at = CURRENT_TIMESTAMP, reviewed_at = CURRENT_TIMESTAMP
                    WHERE candidate_uid = :candidate_uid AND status IN ('triaged', 'labeled')
                    RETURNING candidate_uid
                    """
                ),
                {"candidate_uid": candidate_id, "case_uid": created["case_id"]},
            ).first()
        if not row:
            raise ValueError("Evaluation candidate must be reviewed before promotion")
        return created

    def _get_eval_item(self, item_id: str) -> dict[str, Any]:
        with self.engine.begin() as connection:
            row = connection.execute(
                text(
                    """
                    SELECT i.item_uid AS item_id, COALESCE(c.case_uid, '') AS case_id,
                           i.query, i.expected_intent, i.actual_intent,
                           i.expected_refused, i.actual_refused, i.expected_doc_ids,
                           i.retrieved_doc_ids, i.expected_answer_keywords,
                           i.reference_answer, i.reference_contexts, i.retrieved_contexts,
                           i.answer, i.audit_uid, i.recall_hit, i.refusal_match,
                           i.intent_match, i.keyword_hit, i.ragas_scores, i.ragas_status,
                           i.agent_run_uid, i.actual_trajectory, i.rule_scores,
                           i.judge_scores, i.failure_reasons, i.passed, i.input_tokens,
                           i.output_tokens, i.total_tokens, i.latency_ms, i.estimated_cost,
                           i.annotation_label, i.annotation_comment
                    FROM declaration_qa.eval_run_items i
                    LEFT JOIN declaration_qa.eval_cases c ON c.id = i.case_id
                    WHERE i.item_uid = :item_uid
                    """
                ),
                {"item_uid": item_id},
            ).mappings().first()
        if not row:
            raise ValueError("Evaluation run item not found")
        return normalize_eval_run_item(row)

    @staticmethod
    def _case_params(case_uid: str, dataset_id: str, case: dict[str, Any]) -> dict[str, Any]:
        return {
            "case_uid": case_uid,
            "dataset_uid": dataset_id,
            "query": str(case.get("query") or ""),
            "user_id": int(case.get("user_id") or 1),
            "region_code": str(case.get("region_code") or "340000"),
            "policy_version": str(case.get("policy_version") or "2025"),
            "biz_type": case.get("biz_type"),
            "apply_id": case.get("apply_id"),
            "apply_context": case.get("apply_context"),
            "expected_intent": str(case.get("expected_intent") or ""),
            "expected_refused": case.get("expected_refused"),
            "expected_doc_ids": list(case.get("expected_doc_ids") or []),
            "expected_answer_keywords": list(case.get("expected_answer_keywords") or []),
            "reference_answer": str(case.get("reference_answer") or ""),
            "reference_contexts": list(case.get("reference_contexts") or []),
            "tags": list(case.get("tags") or []),
            "case_type": str(case.get("case_type") or "rag"),
            "turns": list(case.get("turns") or []),
            "fixture_key": str(case.get("fixture_key") or ""),
            "expected_chunk_ids": list(case.get("expected_chunk_ids") or []),
            "expected_trajectory": dict(case.get("expected_trajectory") or {}),
            "expected_outcome": dict(case.get("expected_outcome") or {}),
            "severity": str(case.get("severity") or "medium"),
            "enabled": bool(case.get("enabled", True)),
        }


def json_safe(value: Any) -> Any:
    """将数据库返回值转换为可持久化到JSONB的结构。"""
    if isinstance(value, dict):
        return {str(key): json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(child) for child in value]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
