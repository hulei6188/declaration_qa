"""PostgreSQL 存储基础设施。"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


class PostgresStoreBase:
    """PostgreSQL 存储基础连接和 schema 初始化。"""

    def __init__(self, database_url: str) -> None:
        """初始化PostgreSQL 存储。"""
        self.engine: Engine = create_engine(database_url, future=True, pool_pre_ping=True)

    def init_schema(self) -> None:
        """初始化数据库表结构。"""
        schema_path = Path(__file__).resolve().parents[3] / "sql" / "postgresql" / "003_declaration_qa_schema.sql"
        with self.engine.begin() as connection:
            connection.exec_driver_sql(schema_path.read_text(encoding="utf-8"))
