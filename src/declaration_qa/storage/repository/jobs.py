"""索引任务状态存储。"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from sqlalchemy import JSON, bindparam, text

from declaration_qa.storage.repository.base import EngineBackedRepository


class IndexJobRepository(EngineBackedRepository):
    """索引任务创建、更新和查询仓储。"""
    def create_index_job(self, *, job_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        """创建索引任务记录。"""
        from uuid import uuid4

        job_uid = uuid4().hex
        stmt = text(
            """
            INSERT INTO declaration_qa.index_jobs (job_uid, job_type, status, progress, payload)
            VALUES (:job_uid, :job_type, 'pending', 0, :payload)
            RETURNING *
            """
        ).bindparams(bindparam("payload", type_=JSON))
        with self.engine.begin() as connection:
            row = connection.execute(stmt, {"job_uid": job_uid, "job_type": job_type, "payload": payload}).mappings().first()
        return dict(row)

    def update_index_job(
        self,
        *,
        job_uid: str,
        status: str,
        progress: int,
        message: str = "",
        error_message: str = "",
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """更新索引任务状态和进度。"""
        payload_assignment = "payload = :payload," if payload is not None else ""
        stmt = text(
            f"""
            UPDATE declaration_qa.index_jobs SET
                status = CAST(:status AS VARCHAR),
                progress = :progress,
                message = :message,
                error_message = :error_message,
                {payload_assignment}
                started_at = CASE WHEN CAST(:status AS VARCHAR) = 'running' AND started_at IS NULL THEN CURRENT_TIMESTAMP ELSE started_at END,
                completed_at = CASE WHEN CAST(:status AS VARCHAR) IN ('succeeded', 'failed', 'cancelled') THEN CURRENT_TIMESTAMP ELSE completed_at END,
                updated_at = CURRENT_TIMESTAMP
            WHERE job_uid = :job_uid
            RETURNING *
            """
        )
        if payload is not None:
            stmt = stmt.bindparams(bindparam("payload", type_=JSON))
        params = {
            "job_uid": job_uid,
            "status": status,
            "progress": max(0, min(100, int(progress))),
            "message": message,
            "error_message": error_message,
        }
        if payload is not None:
            params["payload"] = payload
        with self.engine.begin() as connection:
            row = connection.execute(stmt, params).mappings().first()
        if not row:
            raise ValueError("Index job not found")
        return dict(row)

    def get_index_job(self, job_uid: str) -> dict[str, Any] | None:
        """查询索引任务状态。"""
        with self.engine.begin() as connection:
            row = connection.execute(
                text("SELECT * FROM declaration_qa.index_jobs WHERE job_uid = :job_uid"),
                {"job_uid": job_uid},
            ).mappings().first()
        return dict(row) if row else None

