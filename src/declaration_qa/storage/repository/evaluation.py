"""评测数据集和运行记录存储。"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from sqlalchemy import JSON, bindparam, text

from declaration_qa.storage.repository.base import EngineBackedRepository
from declaration_qa.storage.repository.normalizers import (
    normalize_eval_case,
    normalize_eval_dataset,
    normalize_eval_run,
    normalize_eval_run_item,
)


class EvaluationRepository(EngineBackedRepository):
    """评测数据集、用例、运行和标注仓储。"""
    def create_eval_dataset(self, *, name: str, description: str, cases: list[dict[str, Any]]) -> dict[str, Any]:
        """创建评测数据集并写入初始用例。"""
        dataset_uid = f"eval-ds-{uuid4().hex}"
        with self.engine.begin() as connection:
            row = connection.execute(
                text(
                    """
                    INSERT INTO declaration_qa.eval_datasets (dataset_uid, name, description)
                    VALUES (:dataset_uid, :name, :description)
                    RETURNING dataset_uid AS dataset_id, name, description, created_at
                    """
                ),
                {"dataset_uid": dataset_uid, "name": name, "description": description},
            ).mappings().first()
        for case in cases:
            self.add_eval_case(dataset_id=dataset_uid, case=case)
        return {**normalize_eval_dataset(row), "case_count": len(self.list_eval_cases(dataset_uid))}

    def list_eval_datasets(self) -> list[dict[str, Any]]:
        """返回评测数据集列表及用例数量。"""
        with self.engine.begin() as connection:
            rows = connection.execute(
                text(
                    """
                    SELECT d.dataset_uid AS dataset_id, d.name, d.description, d.created_at,
                           COUNT(c.id)::int AS case_count
                    FROM declaration_qa.eval_datasets d
                    LEFT JOIN declaration_qa.eval_cases c ON c.dataset_id = d.id
                    GROUP BY d.id
                    ORDER BY d.created_at DESC
                    """
                )
            ).mappings().all()
        return [normalize_eval_dataset(row) for row in rows]

    def add_eval_case(self, *, dataset_id: str, case: dict[str, Any]) -> dict[str, Any]:
        """向指定评测数据集添加用例。"""
        case_uid = f"eval-case-{uuid4().hex}"
        stmt = text(
            """
            INSERT INTO declaration_qa.eval_cases (
                case_uid, dataset_id, query, user_id, region_code, policy_version,
                biz_type, expected_intent, expected_refused,
                expected_doc_ids, expected_answer_keywords, tags
            )
            SELECT :case_uid, id, :query, :user_id, :region_code, :policy_version,
                   :biz_type, :expected_intent, :expected_refused,
                   :expected_doc_ids, :expected_answer_keywords, :tags
            FROM declaration_qa.eval_datasets
            WHERE dataset_uid = :dataset_uid
            RETURNING case_uid AS case_id, :dataset_uid AS dataset_id, query, user_id,
                      region_code, policy_version, biz_type,
                      expected_intent, expected_refused, expected_doc_ids,
                      expected_answer_keywords, tags, created_at
            """
        ).bindparams(
            bindparam("expected_doc_ids", type_=JSON),
            bindparam("expected_answer_keywords", type_=JSON),
            bindparam("tags", type_=JSON),
        )
        with self.engine.begin() as connection:
            row = connection.execute(
                stmt,
                {
                    "case_uid": case_uid,
                    "dataset_uid": dataset_id,
                    "query": str(case.get("query") or ""),
                    "user_id": int(case.get("user_id") or 1),
                    "region_code": str(case.get("region_code") or "340000"),
                    "policy_version": str(case.get("policy_version") or "2025"),
                    "biz_type": case.get("biz_type"),
                    "expected_intent": str(case.get("expected_intent") or ""),
                    "expected_refused": case.get("expected_refused"),
                    "expected_doc_ids": list(case.get("expected_doc_ids") or []),
                    "expected_answer_keywords": list(case.get("expected_answer_keywords") or []),
                    "tags": list(case.get("tags") or []),
                },
            ).mappings().first()
        if not row:
            raise ValueError("Evaluation dataset not found")
        return normalize_eval_case(row)

    def list_eval_cases(self, dataset_id: str) -> list[dict[str, Any]]:
        """查询指定评测数据集下的用例列表。"""
        with self.engine.begin() as connection:
            rows = connection.execute(
                text(
                    """
                    SELECT c.case_uid AS case_id, d.dataset_uid AS dataset_id, c.query,
                           c.user_id, c.region_code, c.policy_version, c.biz_type,
                           c.expected_intent, c.expected_refused, c.expected_doc_ids,
                           c.expected_answer_keywords, c.tags, c.created_at
                    FROM declaration_qa.eval_cases c
                    JOIN declaration_qa.eval_datasets d ON d.id = c.dataset_id
                    WHERE d.dataset_uid = :dataset_uid
                    ORDER BY c.created_at ASC, c.id ASC
                    """
                ),
                {"dataset_uid": dataset_id},
            ).mappings().all()
        return [normalize_eval_case(row) for row in rows]

    def create_eval_run(self, *, dataset_id: str, note: str, top_k: int) -> dict[str, Any]:
        """创建一次评测运行记录。"""
        run_uid = f"eval-run-{uuid4().hex}"
        with self.engine.begin() as connection:
            row = connection.execute(
                text(
                    """
                    INSERT INTO declaration_qa.eval_runs (run_uid, dataset_id, status, note, top_k)
                    SELECT :run_uid, id, 'running', :note, :top_k
                    FROM declaration_qa.eval_datasets
                    WHERE dataset_uid = :dataset_uid
                    RETURNING run_uid AS run_id, :dataset_uid AS dataset_id, status, note,
                              top_k, metrics, error_message, created_at, completed_at
                    """
                ),
                {"run_uid": run_uid, "dataset_uid": dataset_id, "note": note, "top_k": int(top_k)},
            ).mappings().first()
        if not row:
            raise ValueError("Evaluation dataset not found")
        return normalize_eval_run(row, items=[])

    def append_eval_run_item(self, *, run_id: str, item: dict[str, Any]) -> dict[str, Any]:
        """追加单条评测运行明细。"""
        item_uid = f"eval-item-{uuid4().hex}"
        stmt = text(
            """
            INSERT INTO declaration_qa.eval_run_items (
                item_uid, run_id, case_id, query, expected_intent, actual_intent,
                expected_refused, actual_refused, expected_doc_ids, retrieved_doc_ids,
                expected_answer_keywords, answer, audit_uid, recall_hit, refusal_match,
                intent_match, keyword_hit
            )
            SELECT :item_uid, r.id, c.id, :query, :expected_intent, :actual_intent,
                   :expected_refused, :actual_refused, :expected_doc_ids, :retrieved_doc_ids,
                   :expected_answer_keywords, :answer, :audit_uid, :recall_hit,
                   :refusal_match, :intent_match, :keyword_hit
            FROM declaration_qa.eval_runs r
            LEFT JOIN declaration_qa.eval_cases c ON c.case_uid = :case_uid
            WHERE r.run_uid = :run_uid
            RETURNING item_uid AS item_id, :case_uid AS case_id, query, expected_intent,
                      actual_intent, expected_refused, actual_refused, expected_doc_ids,
                      retrieved_doc_ids, expected_answer_keywords, answer, audit_uid,
                      recall_hit, refusal_match, intent_match, keyword_hit,
                      annotation_label, annotation_comment
            """
        ).bindparams(
            bindparam("expected_doc_ids", type_=JSON),
            bindparam("retrieved_doc_ids", type_=JSON),
            bindparam("expected_answer_keywords", type_=JSON),
        )
        with self.engine.begin() as connection:
            row = connection.execute(
                stmt,
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
                    "answer": str(item.get("answer") or ""),
                    "audit_uid": str(item.get("audit_id") or ""),
                    "recall_hit": item.get("recall_hit"),
                    "refusal_match": item.get("refusal_match"),
                    "intent_match": item.get("intent_match"),
                    "keyword_hit": item.get("keyword_hit"),
                },
            ).mappings().first()
        if not row:
            raise ValueError("Evaluation run not found")
        return normalize_eval_run_item(row)

    def finish_eval_run(self, *, run_id: str, status: str, metrics: dict[str, Any], error_message: str = "") -> dict[str, Any]:
        """结束评测运行并写入指标或错误信息。"""
        stmt = text(
            """
            UPDATE declaration_qa.eval_runs SET
                status = :status,
                metrics = :metrics,
                error_message = :error_message,
                completed_at = CURRENT_TIMESTAMP
            WHERE run_uid = :run_uid
            RETURNING run_uid AS run_id, status, metrics, error_message, completed_at
            """
        ).bindparams(bindparam("metrics", type_=JSON))
        with self.engine.begin() as connection:
            row = connection.execute(
                stmt,
                {"run_uid": run_id, "status": status, "metrics": metrics, "error_message": error_message},
            ).mappings().first()
        if not row:
            raise ValueError("Evaluation run not found")
        return self.get_eval_run(run_id) or normalize_eval_run(row, items=[])

    def get_eval_run(self, run_id: str) -> dict[str, Any] | None:
        """查询评测运行及其明细。"""
        with self.engine.begin() as connection:
            run = connection.execute(
                text(
                    """
                    SELECT r.run_uid AS run_id, d.dataset_uid AS dataset_id, r.status,
                           r.note, r.top_k, r.metrics, r.error_message,
                           r.created_at, r.completed_at
                    FROM declaration_qa.eval_runs r
                    JOIN declaration_qa.eval_datasets d ON d.id = r.dataset_id
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
                    SELECT i.item_uid AS item_id, c.case_uid AS case_id, i.query,
                           i.expected_intent, i.actual_intent, i.expected_refused,
                           i.actual_refused, i.expected_doc_ids, i.retrieved_doc_ids,
                           i.expected_answer_keywords, i.answer, i.audit_uid,
                           i.recall_hit, i.refusal_match, i.intent_match, i.keyword_hit,
                           i.annotation_label, i.annotation_comment
                    FROM declaration_qa.eval_run_items i
                    LEFT JOIN declaration_qa.eval_cases c ON c.id = i.case_id
                    WHERE i.run_id = (
                        SELECT id FROM declaration_qa.eval_runs WHERE run_uid = :run_uid
                    )
                    ORDER BY i.id ASC
                    """
                ),
                {"run_uid": run_id},
            ).mappings().all()
        return normalize_eval_run(run, items=[normalize_eval_run_item(item) for item in items])

    def annotate_eval_item(self, *, item_id: str, label: str, comment: str) -> dict[str, Any]:
        """保存人工标注的评测明细结果。"""
        with self.engine.begin() as connection:
            row = connection.execute(
                text(
                    """
                    UPDATE declaration_qa.eval_run_items SET
                        annotation_label = :label,
                        annotation_comment = :comment,
                        annotated_at = CURRENT_TIMESTAMP
                    WHERE item_uid = :item_uid
                    RETURNING item_uid AS item_id, query, expected_intent, actual_intent,
                              expected_refused, actual_refused, expected_doc_ids,
                              retrieved_doc_ids, expected_answer_keywords, answer, audit_uid,
                              recall_hit, refusal_match, intent_match, keyword_hit,
                              annotation_label, annotation_comment,
                              '' AS case_id
                    """
                ),
                {"item_uid": item_id, "label": label, "comment": comment},
            ).mappings().first()
        if not row:
            raise ValueError("Evaluation run item not found")
        return normalize_eval_run_item(row)

