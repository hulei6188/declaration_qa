from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_COLUMNS = {
    "reference_answer",
    "reference_contexts",
    "retrieved_contexts",
    "ragas_scores",
    "ragas_status",
}


def test_ragas_evaluation_columns_exist_in_fresh_schema_and_upgrade_migration():
    fresh_schema = (ROOT / "sql/postgresql/declaration_qa_postgresql_all.sql").read_text(encoding="utf-8")
    migration = (ROOT / "sql/postgresql/004_ragas_evaluation.sql").read_text(encoding="utf-8")

    for column in REQUIRED_COLUMNS:
        assert column in fresh_schema
        assert column in migration
    assert "IF NOT EXISTS" in migration
    assert "DELETE FROM" not in migration.upper()
    assert "DROP " not in migration.upper()
