"""将版本化JSON评测集导入主评测事实库。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from declaration_qa.api.schemas import EvaluationCaseRequest
from declaration_qa.core.config import get_settings
from declaration_qa.storage.repository.store import PostgresDeclarationQAStore


def import_dataset(path: Path) -> dict:
    settings = get_settings()
    if not settings.effective_database_url:
        raise ValueError("DECLARATION_QA_DATABASE_URL is required")
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = [EvaluationCaseRequest.model_validate(case).model_dump(mode="json") for case in payload.get("cases") or []]
    store = PostgresDeclarationQAStore(settings.effective_database_url)
    try:
        return store.create_eval_dataset(
            name=str(payload.get("name") or path.stem),
            description=str(payload.get("description") or ""),
            cases=cases,
            version=int(payload.get("version") or 1),
            status=str(payload.get("status") or "draft"),
            source=str(payload.get("source") or "import"),
            parent_dataset_id=payload.get("parent_dataset_id"),
            knowledge_revision=str(payload.get("knowledge_revision") or ""),
        )
    finally:
        store.engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description="Import a Gov Service Agent evaluation dataset")
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    result = import_dataset(args.path.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
