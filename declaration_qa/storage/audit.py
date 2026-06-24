"""审计日志、用户反馈与评测记录的内存存储实现。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


class InMemoryAuditStore:
    """用于本地开发和测试的内存审计与评测存储。"""
    def __init__(self) -> None:
        """初始化内存审计存储。"""
        self._items: dict[str, dict[str, Any]] = {}
        self._feedback: list[dict[str, Any]] = []
        self._eval_datasets: dict[str, dict[str, Any]] = {}
        self._eval_cases: dict[str, dict[str, Any]] = {}
        self._eval_runs: dict[str, dict[str, Any]] = {}
        self._eval_run_items: dict[str, dict[str, Any]] = {}

    def append(self, payload: dict[str, Any]) -> str:
        """追加一条审计记录并返回记录 ID。"""
        audit_id = uuid4().hex
        self._items[audit_id] = {
            "id": audit_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            **payload,
        }
        return audit_id

    def get(self, audit_id: str) -> dict[str, Any] | None:
        """按 ID 查询记录并返回副本。"""
        item = self._items.get(audit_id)
        return dict(item) if item else None

    def add_feedback(self, *, audit_id: str, rating: str, comment: str) -> dict[str, Any]:
        """保存用户对问答结果的反馈。"""
        payload = {
            "audit_id": audit_id,
            "rating": rating,
            "comment": comment,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._feedback.append(payload)
        return payload

    def feedback_items(self) -> list[dict[str, Any]]:
        """返回已收集的反馈列表。"""
        return list(self._feedback)

    def list_sessions(self, *, user_id: int) -> list[dict[str, Any]]:
        """按用户查询历史会话摘要列表。"""
        sessions: dict[str, dict[str, Any]] = {}
        for item in self._items.values():
            if int(item.get("user_id") or 0) != int(user_id):
                continue
            session_id = str(item.get("session_id") or "")
            if not session_id:
                continue
            session = sessions.setdefault(
                session_id,
                {
                    "session_id": session_id,
                    "user_id": int(user_id),
                    "region_code": str(item.get("region_code") or ""),
                    "biz_type": item.get("biz_type"),
                    "apply_id": str(item.get("apply_id") or ""),
                    "title": str(item.get("query") or "申报智能问答")[:80],
                    "message_count": 0,
                    "updated_at": str(item.get("created_at") or ""),
                },
            )
            session["message_count"] += 2
            session["updated_at"] = max(str(session.get("updated_at") or ""), str(item.get("created_at") or ""))
        return sorted(sessions.values(), key=lambda item: item.get("updated_at") or "", reverse=True)

    def get_session(self, *, user_id: int, session_id: str) -> dict[str, Any] | None:
        """按用户和会话 ID 查询完整会话详情。"""
        items = [
            item
            for item in self._items.values()
            if int(item.get("user_id") or 0) == int(user_id) and str(item.get("session_id") or "") == session_id
        ]
        if not items:
            return None
        items = sorted(items, key=lambda item: item.get("created_at") or "")
        first = items[0]
        messages = []
        for item in items:
            messages.append({"role": "user", "content": item.get("query") or "", "created_at": item.get("created_at") or ""})
            messages.append(
                {
                    "role": "assistant",
                    "content": item.get("answer") or "",
                    "intent": item.get("intent") or "",
                    "refused": bool(item.get("refused")),
                    "sources": item.get("retrieved_sources") or [],
                    "suggested_actions": item.get("suggested_actions") or [],
                    "created_at": item.get("created_at") or "",
                }
            )
        return {
            "session_id": session_id,
            "user_id": int(user_id),
            "region_code": str(first.get("region_code") or ""),
            "biz_type": first.get("biz_type"),
            "apply_id": str(first.get("apply_id") or ""),
            "title": str(first.get("query") or "申报智能问答")[:80],
            "message_count": len(messages),
            "updated_at": str(items[-1].get("created_at") or ""),
            "messages": messages,
        }

    def delete_session(self, *, user_id: int, session_id: str) -> bool:
        """按用户删除指定历史会话。"""
        keys = [
            audit_id
            for audit_id, item in self._items.items()
            if int(item.get("user_id") or 0) == int(user_id) and str(item.get("session_id") or "") == session_id
        ]
        for audit_id in keys:
            del self._items[audit_id]
        return bool(keys)

    def create_eval_dataset(self, *, name: str, description: str, cases: list[dict[str, Any]]) -> dict[str, Any]:
        """创建评测数据集并写入初始用例。"""
        dataset_id = f"eval-ds-{uuid4().hex}"
        now = datetime.now(timezone.utc).isoformat()
        self._eval_datasets[dataset_id] = {
            "dataset_id": dataset_id,
            "name": name,
            "description": description,
            "created_at": now,
        }
        for case in cases:
            self.add_eval_case(dataset_id=dataset_id, case=case)
        return {**self._eval_datasets[dataset_id], "case_count": len(self.list_eval_cases(dataset_id))}

    def list_eval_datasets(self) -> list[dict[str, Any]]:
        """返回评测数据集列表及用例数量。"""
        output = []
        for dataset in self._eval_datasets.values():
            output.append({**dataset, "case_count": len(self.list_eval_cases(str(dataset["dataset_id"])))})
        return sorted(output, key=lambda item: item.get("created_at") or "", reverse=True)

    def add_eval_case(self, *, dataset_id: str, case: dict[str, Any]) -> dict[str, Any]:
        """向指定评测数据集添加用例。"""
        if dataset_id not in self._eval_datasets:
            raise ValueError("Evaluation dataset not found")
        case_id = f"eval-case-{uuid4().hex}"
        payload = {
            "case_id": case_id,
            "dataset_id": dataset_id,
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
            "tags": list(case.get("tags") or []),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._eval_cases[case_id] = payload
        return dict(payload)

    def list_eval_cases(self, dataset_id: str) -> list[dict[str, Any]]:
        """查询指定评测数据集下的用例列表。"""
        return [
            dict(item)
            for item in sorted(self._eval_cases.values(), key=lambda case: case.get("created_at") or "")
            if str(item.get("dataset_id") or "") == dataset_id
        ]

    def create_eval_run(self, *, dataset_id: str, note: str, top_k: int) -> dict[str, Any]:
        """创建一次评测运行记录。"""
        if dataset_id not in self._eval_datasets:
            raise ValueError("Evaluation dataset not found")
        run_id = f"eval-run-{uuid4().hex}"
        payload = {
            "run_id": run_id,
            "dataset_id": dataset_id,
            "status": "running",
            "note": note,
            "top_k": int(top_k),
            "metrics": {},
            "error_message": "",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": "",
        }
        self._eval_runs[run_id] = payload
        return dict(payload)

    def append_eval_run_item(self, *, run_id: str, item: dict[str, Any]) -> dict[str, Any]:
        """追加单条评测运行明细。"""
        if run_id not in self._eval_runs:
            raise ValueError("Evaluation run not found")
        item_id = f"eval-item-{uuid4().hex}"
        payload = {
            "item_id": item_id,
            "run_id": run_id,
            "annotation_label": "",
            "annotation_comment": "",
            **item,
        }
        self._eval_run_items[item_id] = payload
        return dict(payload)

    def finish_eval_run(self, *, run_id: str, status: str, metrics: dict[str, Any], error_message: str = "") -> dict[str, Any]:
        """结束评测运行并写入指标或错误信息。"""
        if run_id not in self._eval_runs:
            raise ValueError("Evaluation run not found")
        self._eval_runs[run_id].update(
            {
                "status": status,
                "metrics": metrics,
                "error_message": error_message,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        return self.get_eval_run(run_id) or {}

    def get_eval_run(self, run_id: str) -> dict[str, Any] | None:
        """查询评测运行及其明细。"""
        run = self._eval_runs.get(run_id)
        if not run:
            return None
        items = [
            dict(item)
            for item in self._eval_run_items.values()
            if str(item.get("run_id") or "") == run_id
        ]
        return {**run, "items": items}

    def annotate_eval_item(self, *, item_id: str, label: str, comment: str) -> dict[str, Any]:
        """保存人工标注的评测明细结果。"""
        item = self._eval_run_items.get(item_id)
        if not item:
            raise ValueError("Evaluation run item not found")
        item["annotation_label"] = label
        item["annotation_comment"] = comment
        item["annotated_at"] = datetime.now(timezone.utc).isoformat()
        return dict(item)
