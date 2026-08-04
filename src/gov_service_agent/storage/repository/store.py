"""PostgreSQL 政务服务智能体存储门面。"""

from __future__ import annotations

from datetime import date
from typing import Any

from gov_service_agent.core.domain import KnowledgeChunk, KnowledgeDocument
from gov_service_agent.storage.repository.agent_runs import AgentRunRepository
from gov_service_agent.storage.repository.audit import AuditRepository
from gov_service_agent.storage.repository.base import PostgresStoreBase
from gov_service_agent.storage.repository.business import BusinessRepository
from gov_service_agent.storage.repository.evaluation import EvaluationRepository
from gov_service_agent.storage.repository.jobs import IndexJobRepository
from gov_service_agent.storage.repository.knowledge import KnowledgeRepository
from gov_service_agent.storage.repository.sessions import SessionRepository


class PostgresGovServiceAgentStore(PostgresStoreBase):
    """PostgreSQL 持久化存储和业务数据适配器门面。"""

    def __init__(self, database_url: str) -> None:
        """初始化PostgreSQL 存储。"""
        super().__init__(database_url)
        self.audit = AuditRepository(self.engine)
        self.agent_runs = AgentRunRepository(self.engine)
        self.business = BusinessRepository(self.engine)
        self.evaluation = EvaluationRepository(self.engine)
        self.knowledge = KnowledgeRepository(self.engine)
        self.index_jobs = IndexJobRepository(self.engine)
        self.sessions = SessionRepository(self.engine)

    def append(self, payload: dict[str, Any]) -> str:
        """追加一条审计记录并返回记录 ID。"""
        return self.audit.append(payload)

    def get(self, audit_id: str) -> dict[str, Any] | None:
        """按 ID 查询审计记录。"""
        return self.audit.get(audit_id)

    def get_agent_run(self, *, user_id: int, run_id: str) -> dict[str, Any] | None:
        """按用户读取可重放的完整 Agent 执行记录。"""
        return self.agent_runs.get(user_id=user_id, run_id=run_id)

    def add_feedback(self, *, audit_id: str, rating: str, comment: str) -> dict[str, Any]:
        """保存用户对问答结果的反馈。"""
        return self.audit.add_feedback(audit_id=audit_id, rating=rating, comment=comment)

    def list_ticket_items(self, *, user_id: int, region_code: str | None = None) -> list[dict[str, Any]]:
        """按事项汇总当前用户的申请。"""
        return self.business.list_ticket_items(user_id=user_id, region_code=region_code)

    def get_applications_by_items(
        self,
        *,
        user_id: int,
        item_codes: list[str],
    ) -> list[dict[str, Any]]:
        """批量返回多个事项下的申请和完整状态历史。"""
        return self.business.get_applications_by_items(
            user_id=user_id,
            item_codes=item_codes,
        )

    def list_appointments(
        self,
        *,
        user_id: int,
        region_code: str | None = None,
        status: str = "active",
    ) -> list[dict[str, Any]]:
        return self.business.list_appointments(user_id=user_id, region_code=region_code, status=status)

    def query_appointment_slots(
        self,
        *,
        item_code: str,
        region_code: str | None = None,
        service_date: str | None = None,
        hall_code: str | None = None,
    ) -> list[dict[str, Any]]:
        return self.business.query_appointment_slots(
            item_code=item_code,
            region_code=region_code,
            service_date=service_date,
            hall_code=hall_code,
        )

    def book_appointment(
        self,
        *,
        user_id: int,
        region_code: str | None,
        slot_id: int,
        idempotency_key: str,
    ) -> dict[str, Any]:
        return self.business.book_appointment(
            user_id=user_id,
            region_code=region_code,
            slot_id=slot_id,
            idempotency_key=idempotency_key,
        )

    def cancel_appointment(
        self,
        *,
        user_id: int,
        region_code: str | None,
        appointment_id: int,
    ) -> dict[str, Any]:
        return self.business.cancel_appointment(
            user_id=user_id,
            region_code=region_code,
            appointment_id=appointment_id,
        )

    def get_queue_status(self, *, user_id: int, appointment_id: int) -> dict[str, Any] | None:
        return self.business.get_queue_status(user_id=user_id, appointment_id=appointment_id)

    def get_payment_orders_by_applications(
        self,
        *,
        user_id: int,
        application_ids: list[int],
    ) -> list[dict[str, Any]]:
        return self.business.get_payment_orders_by_applications(
            user_id=user_id,
            application_ids=application_ids,
        )

    def get_payment_receipts_by_targets(
        self,
        *,
        user_id: int,
        targets: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return self.business.get_payment_receipts_by_targets(
            user_id=user_id,
            targets=targets,
        )

    def get_submission_receipts_by_applications(
        self, *, user_id: int, application_ids: list[int]
    ) -> list[dict[str, Any]]:
        return self.business.get_submission_receipts_by_applications(
            user_id=user_id, application_ids=application_ids
        )

    def get_result_certificates_by_applications(
        self, *, user_id: int, application_ids: list[int]
    ) -> list[dict[str, Any]]:
        return self.business.get_result_certificates_by_applications(
            user_id=user_id, application_ids=application_ids
        )

    def get_delivery_statuses_by_applications(
        self, *, user_id: int, application_ids: list[int]
    ) -> list[dict[str, Any]]:
        return self.business.get_delivery_statuses_by_applications(
            user_id=user_id, application_ids=application_ids
        )

    def list_consultation_records(
        self,
        *,
        user_id: int,
        status: str = "all",
        consultation_id: int | None = None,
    ) -> list[dict[str, Any]]:
        return self.business.list_consultation_records(
            user_id=user_id,
            status=status,
            consultation_id=consultation_id,
        )

    def get_application_form_schema(
        self,
        *,
        item_code: str,
        region_code: str | None = None,
    ) -> dict[str, Any] | None:
        return self.business.get_application_form_schema(item_code=item_code, region_code=region_code)

    def submit_application(
        self,
        *,
        user_id: int,
        region_code: str | None,
        item_code: str,
        item_name: str,
        biz_type: str,
        schema_version: str,
        form_data: dict[str, Any],
        idempotency_key: str,
    ) -> dict[str, Any]:
        return self.business.submit_application(
            user_id=user_id,
            region_code=region_code,
            item_code=item_code,
            item_name=item_name,
            biz_type=biz_type,
            schema_version=schema_version,
            form_data=form_data,
            idempotency_key=idempotency_key,
        )

    def create_complaint(
        self,
        *,
        user_id: int,
        region_code: str | None,
        values: dict[str, Any],
        schema_version: str,
        idempotency_key: str,
    ) -> dict[str, Any]:
        return self.business.create_complaint(
            user_id=user_id,
            region_code=region_code,
            values=values,
            schema_version=schema_version,
            idempotency_key=idempotency_key,
        )

    def create_eval_dataset(self, *, name: str, description: str, cases: list[dict[str, Any]], **metadata) -> dict[str, Any]:
        """创建评测数据集并写入初始用例。"""
        return self.evaluation.create_eval_dataset(name=name, description=description, cases=cases, **metadata)

    def publish_eval_dataset(self, dataset_id: str) -> dict[str, Any]:
        return self.evaluation.publish_eval_dataset(dataset_id)

    def list_eval_datasets(self) -> list[dict[str, Any]]:
        """返回评测数据集列表及用例数量。"""
        return self.evaluation.list_eval_datasets()

    def add_eval_case(self, *, dataset_id: str, case: dict[str, Any]) -> dict[str, Any]:
        """向指定评测数据集添加用例。"""
        return self.evaluation.add_eval_case(dataset_id=dataset_id, case=case)

    def list_eval_cases(self, dataset_id: str) -> list[dict[str, Any]]:
        """返回指定评测数据集的用例。"""
        return self.evaluation.list_eval_cases(dataset_id)

    def create_eval_run(self, *, dataset_id: str, note: str, top_k: int, **metadata) -> dict[str, Any]:
        """创建评测运行记录。"""
        return self.evaluation.create_eval_run(dataset_id=dataset_id, note=note, top_k=top_k, **metadata)

    def mark_eval_run_running(self, run_id: str) -> None:
        self.evaluation.mark_eval_run_running(run_id)

    def append_eval_run_item(self, *, run_id: str, item: dict[str, Any]) -> dict[str, Any]:
        """追加评测运行明细。"""
        return self.evaluation.append_eval_run_item(run_id=run_id, item=item)

    def append_eval_scores(self, *, item_id: str, scores: list[dict[str, Any]]) -> None:
        self.evaluation.append_eval_scores(item_id=item_id, scores=scores)

    def finish_eval_run(
        self,
        *,
        run_id: str,
        status: str,
        metrics: dict[str, Any],
        error_message: str = "",
        gate_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """完成评测运行。"""
        return self.evaluation.finish_eval_run(
            run_id=run_id,
            status=status,
            metrics=metrics,
            error_message=error_message,
            gate_result=gate_result,
        )

    def get_eval_run(self, run_id: str) -> dict[str, Any] | None:
        """查询评测运行结果。"""
        return self.evaluation.get_eval_run(run_id)

    def list_eval_runs(self, *, dataset_id: str, limit: int = 50) -> list[dict[str, Any]]:
        return self.evaluation.list_eval_runs(dataset_id=dataset_id, limit=limit)

    def annotate_eval_item(self, *, item_id: str, label: str, comment: str) -> dict[str, Any]:
        """标注评测运行明细。"""
        return self.evaluation.annotate_eval_item(item_id=item_id, label=label, comment=comment)

    def create_eval_candidate(self, candidate: dict[str, Any]) -> dict[str, Any]:
        return self.evaluation.create_eval_candidate(candidate)

    def list_eval_candidates(self, *, status: str = "", limit: int = 100) -> list[dict[str, Any]]:
        return self.evaluation.list_eval_candidates(status=status, limit=limit)

    def review_eval_candidate(self, **payload) -> dict[str, Any]:
        return self.evaluation.review_eval_candidate(**payload)

    def promote_eval_candidate(self, **payload) -> dict[str, Any]:
        return self.evaluation.promote_eval_candidate(**payload)

    def persist_document(self, document: KnowledgeDocument, chunks: list[KnowledgeChunk]) -> None:
        """持久化知识文档及其片段。"""
        self.knowledge.persist_document(document, chunks)

    def list_documents(self) -> list[KnowledgeDocument]:
        """返回全部已发布知识文档。"""
        return self.knowledge.list_documents()

    def list_indexed_knowledge(self) -> tuple[list[KnowledgeDocument], list[KnowledgeChunk]]:
        """返回已持久化的知识文档和片段。"""
        return self.knowledge.list_indexed_knowledge()

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[KnowledgeChunk]:
        """按 chunk id 读取知识片段。"""
        return self.knowledge.get_chunks_by_ids(chunk_ids)

    def get_parent_chunks(self, parent_ids: list[str]) -> list[KnowledgeChunk]:
        """读取父级知识片段。"""
        return self.knowledge.get_parent_chunks(parent_ids)

    def knowledge_base_revision(self) -> str:
        """计算知识库修订标识。"""
        return self.knowledge.knowledge_base_revision()

    def delete_document(self, document_uid: str) -> bool:
        """删除知识文档。"""
        return self.knowledge.delete_document(document_uid)

    def list_document_records(self, *, status: str | None = None) -> list[dict[str, Any]]:
        """查询后台管理用知识文档记录。"""
        return self.knowledge.list_document_records(status=status)

    def get_document_record(self, document_uid: str) -> dict[str, Any] | None:
        """查询单条后台管理用知识文档记录。"""
        return self.knowledge.get_document_record(document_uid)

    def set_document_status(self, document_uid: str, status: str) -> dict[str, Any]:
        """更新知识文档发布状态。"""
        return self.knowledge.set_document_status(document_uid, status)

    def update_document_rollout(
        self,
        *,
        document_uid: str,
        rollout_percent: int,
        effective_start_time: date | None,
        effective_end_time: date | None,
    ) -> dict[str, Any]:
        """更新知识文档灰度比例和生效时间。"""
        return self.knowledge.update_document_rollout(
            document_uid=document_uid,
            rollout_percent=rollout_percent,
            effective_start_time=effective_start_time,
            effective_end_time=effective_end_time,
        )

    def create_document_version(
        self,
        *,
        source_document_uid: str,
        document_uid: str,
        content: str | None,
        title: str | None,
        change_note: str,
        policy_version: str | None = None,
    ) -> dict[str, Any]:
        """创建知识文档新版本。"""
        return self.knowledge.create_document_version(
            source_document_uid=source_document_uid,
            document_uid=document_uid,
            content=content,
            title=title,
            change_note=change_note,
            policy_version=policy_version,
        )

    def rollback_document(self, *, target_document_uid: str, change_note: str) -> dict[str, Any]:
        """回滚到指定知识文档版本。"""
        return self.knowledge.rollback_document(target_document_uid=target_document_uid, change_note=change_note)

    def create_index_job(self, *, job_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        """创建索引任务。"""
        return self.index_jobs.create_index_job(job_type=job_type, payload=payload)

    def update_index_job(
        self,
        *,
        job_uid: str,
        status: str,
        progress: int | None = None,
        message: str = "",
        error_message: str = "",
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """更新索引任务状态。"""
        return self.index_jobs.update_index_job(
            job_uid=job_uid,
            status=status,
            progress=progress,
            message=message,
            error_message=error_message,
            payload=payload,
        )

    def get_index_job(self, job_uid: str) -> dict[str, Any] | None:
        """查询索引任务状态。"""
        return self.index_jobs.get_index_job(job_uid)

    def list_sessions(self, *, user_id: int) -> list[dict[str, Any]]:
        """按用户查询历史会话摘要列表。"""
        return self.sessions.list_sessions(user_id=user_id)

    def get_session(self, *, user_id: int, session_id: str) -> dict[str, Any] | None:
        """按用户和会话 ID 查询完整会话详情。"""
        return self.sessions.get_session(user_id=user_id, session_id=session_id)

    def recent_messages(self, *, user_id: int, session_id: str, limit: int = 10) -> list[dict[str, Any]]:
        """按用户和会话 ID 查询最近消息。"""
        return self.sessions.recent_messages(user_id=user_id, session_id=session_id, limit=limit)

    def delete_session(self, *, user_id: int, session_id: str) -> bool:
        """按用户删除指定历史会话。"""
        return self.sessions.delete_session(user_id=user_id, session_id=session_id)
