"""PostgreSQL 申报问答存储实现。"""

from __future__ import annotations

from declaration_qa.storage.repository.audit import AuditStoreMixin
from declaration_qa.storage.repository.base import PostgresStoreBase
from declaration_qa.storage.repository.business import BusinessStoreMixin
from declaration_qa.storage.repository.evaluation import EvaluationStoreMixin
from declaration_qa.storage.repository.jobs import IndexJobStoreMixin
from declaration_qa.storage.repository.knowledge import KnowledgeStoreMixin
from declaration_qa.storage.repository.sessions import SessionStoreMixin


class PostgresDeclarationQAStore(
    AuditStoreMixin,
    BusinessStoreMixin,
    EvaluationStoreMixin,
    KnowledgeStoreMixin,
    IndexJobStoreMixin,
    SessionStoreMixin,
    PostgresStoreBase,
):
    """PostgreSQL 持久化存储和业务数据适配器。"""
