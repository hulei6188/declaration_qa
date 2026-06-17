"""知识库索引任务存储协议和内存实现。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol
from uuid import uuid4


class IndexJobStore(Protocol):
    """索引任务状态存储协议。"""
    def create_index_job(self, *, job_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        """创建索引任务记录。"""
        ...

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
        ...

    def get_index_job(self, job_uid: str) -> dict[str, Any] | None:
        """查询索引任务状态。"""
        ...


@dataclass
class InMemoryIndexJobStore:
    """进程内索引任务状态存储。"""
    _jobs: dict[str, dict[str, Any]] = field(default_factory=dict)

    def create_index_job(self, *, job_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        """创建索引任务记录。"""
        job_uid = uuid4().hex
        now = datetime.now(timezone.utc).isoformat()
        job = {
            "job_uid": job_uid,
            "job_type": job_type,
            "status": "pending",
            "progress": 0,
            "message": "",
            "error_message": "",
            "payload": payload,
            "started_at": None,
            "completed_at": None,
            "created_at": now,
            "updated_at": now,
        }
        self._jobs[job_uid] = job
        return dict(job)

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
        job = self._jobs[job_uid]
        now = datetime.now(timezone.utc).isoformat()
        job.update(
            {
                "status": status,
                "progress": max(0, min(100, int(progress))),
                "message": message,
                "error_message": error_message,
                "payload": payload if payload is not None else job.get("payload") or {},
                "updated_at": now,
            }
        )
        if status == "running" and not job.get("started_at"):
            job["started_at"] = now
        if status in {"succeeded", "failed", "cancelled"}:
            job["completed_at"] = now
        return dict(job)

    def get_index_job(self, job_uid: str) -> dict[str, Any] | None:
        """查询索引任务状态。"""
        job = self._jobs.get(job_uid)
        return dict(job) if job else None
