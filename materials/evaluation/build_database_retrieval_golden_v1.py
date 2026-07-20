"""用人工策划清单绑定数据库中的真实正文与附件证据。"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

from sqlalchemy import bindparam, create_engine, text

from gov_service_agent.api.schemas import EvaluationCaseRequest
from gov_service_agent.application.evals.pii import sanitize_text
from gov_service_agent.core.config import get_settings


OUTPUT = Path(__file__).with_name("database-retrieval-golden-v1.json")
CURATION = Path(__file__).with_name("database-retrieval-golden-v1-curation.json")
DATASET_NAME = "database-retrieval-golden-v1"

CHUNKS_SQL = text(
    """
    SELECT d.document_uid,
           d.title AS source_file,
           COALESCE(NULLIF(d.metadata->>'item_name', ''), d.title) AS item_name,
           COALESCE(d.metadata->>'item_code', '') AS item_code,
           COALESCE(NULLIF(d.metadata->>'region_code', ''), '340000') AS region_code,
           COALESCE(NULLIF(d.metadata->>'region_name', ''), '安徽省') AS region_name,
           COALESCE(NULLIF(d.metadata->>'policy_version', ''), '2026') AS policy_version,
           COALESCE(d.metadata->>'knowledge_revision', '') AS knowledge_revision,
           c.chunk_uid,
           c.chunk_index,
           c.chunk_type,
           c.content,
           COALESCE(c.metadata->>'heading_path', '') AS heading_path,
           COALESCE(c.metadata->>'attachment_name', '') AS attachment_name,
           COALESCE(c.metadata->>'attachment_type', '') AS attachment_type,
           COALESCE(c.metadata->>'attachment_role', '') AS attachment_role,
           COALESCE(c.metadata->>'section_path', '') AS section_path
    FROM declaration_qa.documents d
    JOIN declaration_qa.chunks c ON c.document_id = d.id
    WHERE d.status = 'published'
      AND c.status = 'published'
      AND d.title IN :source_files
    ORDER BY d.title, c.chunk_index, c.id
    """
).bindparams(bindparam("source_files", expanding=True))


def main() -> None:
    curation = json.loads(CURATION.read_text(encoding="utf-8"))
    specs = list(curation.get("cases") or [])
    validate_curation(specs)
    rows = load_rows({str(spec["source_file"]) for spec in specs})
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[str(row["source_file"])].append(row)

    cases = [build_case(spec, by_source.get(str(spec["source_file"]), [])) for spec in specs]
    validate_cases(cases)
    scope_counts = Counter(case["apply_context"]["evidence_scope"] for case in cases)
    attachment_cases = [case for case in cases if case["apply_context"]["evidence_scope"] == "attachment"]
    attachment_type_counts = Counter(case["apply_context"]["attachment_type"] for case in attachment_cases)
    intent_counts = Counter(case["apply_context"]["intent"] for case in cases)
    gold_counts = Counter(len(case["expected_chunk_ids"]) for case in cases)
    payload = {
        "name": DATASET_NAME,
        "description": (
            "基于主知识库真实办事指南 Markdown 正文及其 DOC/DOCX/PDF/XLS/XLSX/图片附件人工策划的"
            "150 条检索黄金集。每条 Query 只包含一个用户问题；Query 文本不由脚本生成，脚本只负责"
            "将人工清单绑定到已发布 doc_id/chunk_id 证据。"
        ),
        "version": 1,
        "status": "draft",
        "source": "manual-curated-docs-attachments",
        "knowledge_revision": dataset_revision(cases),
        "generation": {
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "document_count": len({case["apply_context"]["document_uid"] for case in cases}),
            "case_count": len(cases),
            "evidence_scope_counts": clean_counter(scope_counts),
            "attachment_type_counts": clean_counter(attachment_type_counts),
            "intent_counts": clean_counter(intent_counts),
            "gold_evidence_count_distribution": clean_counter(gold_counts),
            "max_gold_evidence_per_case": max((len(case["expected_chunk_ids"]) for case in cases), default=0),
            "multi_gold_case_count": sum(len(case["expected_chunk_ids"]) > 1 for case in cases),
            "gold_level": "doc_id+chunk_id",
            "query_style": "manually-authored single-intent user query",
            "query_source": "human reading of real markdown body and parsed attachments",
            "target_metrics": ["Recall@5", "Recall@10", "MRR@10", "P95"],
            "pii_policy": "phone/id-card/email masked in reference contexts",
            "review_status": "content_curated_evidence_bound",
        },
        "cases": cases,
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["generation"], ensure_ascii=False, indent=2))
    print(f"wrote {len(cases)} cases to {OUTPUT}")


def load_rows(source_files: set[str]) -> list[dict[str, Any]]:
    settings = get_settings()
    if not settings.effective_database_url:
        raise ValueError("GOV_SERVICE_AGENT_DATABASE_URL is required")
    engine = create_engine(settings.effective_database_url, future=True)
    try:
        with engine.connect() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    CHUNKS_SQL,
                    {"source_files": sorted(source_files)},
                ).mappings()
            ]
    finally:
        engine.dispose()


def build_case(spec: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    source_file = str(spec["source_file"])
    if not rows:
        raise ValueError(f"published database document not found: {source_file}")
    evidence_spec = dict(spec["evidence"])
    evidence_rows = resolve_evidence(source_file, rows, evidence_spec)
    first = evidence_rows[0]
    scope = str(evidence_spec["scope"])
    query = str(spec["query"]).strip()
    contexts = [sanitize_text(clean_text(row["content"]), salt=DATASET_NAME) for row in evidence_rows]
    contexts = unique_strings(context for context in contexts if context)
    expected_chunk_ids = unique_strings(str(row["chunk_uid"]) for row in evidence_rows)
    attachment_name = str(first.get("attachment_name") or "")
    attachment_type = str(first.get("attachment_type") or "").lower()
    apply_context = {
        "document_uid": str(first["document_uid"]),
        "source_file": source_file,
        "item_code": str(first.get("item_code") or ""),
        "item_name": str(first.get("item_name") or ""),
        "region_code": str(first.get("region_code") or ""),
        "region_name": str(first.get("region_name") or ""),
        "knowledge_revision": str(first.get("knowledge_revision") or ""),
        "evidence_scope": scope,
        "intent": str(spec["intent"]),
        "gold_chunk_uids": expected_chunk_ids,
        "gold_evidence_count": len(expected_chunk_ids),
        "query_provenance": "manual_from_real_markdown_or_attachment",
        "gold_provenance": "published_database_chunk_binding",
        "review_status": "content_curated_evidence_bound",
    }
    if scope == "body":
        apply_context["heading_paths"] = unique_strings(str(row.get("heading_path") or "") for row in evidence_rows)
    else:
        apply_context.update(
            {
                "attachment_name": attachment_name,
                "attachment_type": attachment_type,
                "attachment_role": str(first.get("attachment_role") or ""),
                "section_path": str(first.get("section_path") or ""),
            }
        )
    return {
        "query": query,
        "user_id": stable_user_id(query),
        "region_code": str(first.get("region_code") or "340000"),
        "policy_version": str(first.get("policy_version") or "2026"),
        "biz_type": None,
        "apply_id": None,
        "apply_context": apply_context,
        "case_type": "rag",
        "turns": [],
        "fixture_key": "",
        "expected_intent": str(spec["intent"]),
        "expected_refused": False,
        "expected_doc_ids": [str(first["document_uid"])],
        "expected_chunk_ids": expected_chunk_ids,
        "expected_answer_keywords": list(spec.get("answer_keywords") or []),
        "reference_answer": "\n\n".join(contexts),
        "reference_contexts": contexts,
        "expected_trajectory": {"required_tools": ["knowledge_search"], "forbidden_tools": []},
        "expected_outcome": {},
        "severity": "medium",
        "enabled": True,
        "tags": [
            DATASET_NAME,
            "manual-content-curation",
            "single-intent-query",
            f"evidence_scope:{scope}",
            f"intent:{spec['intent']}",
            f"source_file:{source_file}",
            *( [f"attachment_type:{attachment_type}"] if attachment_type else [] ),
        ],
    }


def resolve_evidence(
    source_file: str,
    rows: list[dict[str, Any]],
    evidence_spec: dict[str, Any],
) -> list[dict[str, Any]]:
    scope = str(evidence_spec.get("scope") or "")
    if scope == "body":
        candidates = [row for row in rows if not str(row.get("attachment_name") or "")]
        terms = [str(term) for term in evidence_spec.get("heading_contains") or []]
        mode = str(evidence_spec.get("mode") or "first")
        if mode == "first_per_term":
            selected = []
            for term in terms:
                matches = [row for row in candidates if term in str(row.get("heading_path") or "")]
                if not matches:
                    raise ValueError(f"body heading not found: {source_file} / {term}")
                selected.append(best_body_row(matches))
            return unique_rows(selected)
        matches = [row for row in candidates if all(term in str(row.get("heading_path") or "") for term in terms)]
        if not matches:
            raise ValueError(f"body evidence not found: {source_file} / {terms}")
        return [best_body_row(matches)]
    if scope == "attachment":
        attachment_name = str(evidence_spec.get("attachment_name") or "")
        matches = [row for row in rows if str(row.get("attachment_name") or "") == attachment_name]
        content_contains = str(evidence_spec.get("content_contains") or "").strip()
        if content_contains:
            narrowed = [row for row in matches if normalized_contains(str(row.get("content") or ""), content_contains)]
            if narrowed:
                matches = narrowed
            else:
                raise ValueError(
                    f"attachment content not found: {source_file} / {attachment_name} / {content_contains}"
                )
        if not matches:
            raise ValueError(f"attachment evidence not found: {source_file} / {attachment_name}")
        return unique_content_rows(matches)[:5]
    raise ValueError(f"unsupported evidence scope: {scope}")


def best_body_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def rank(row: dict[str, Any]) -> tuple[int, int, int]:
        content = str(row.get("content") or "")
        chunk_type = str(row.get("chunk_type") or "")
        type_rank = 0 if "table" in chunk_type else 1 if "text" in chunk_type else 2 if "list" in chunk_type else 3
        return (type_rank, -len(content), int(row.get("chunk_index") or 0))
    return min(rows, key=rank)


def unique_content_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    seen = set()
    for row in rows:
        fingerprint = re.sub(r"\s+", "", str(row.get("content") or ""))
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        selected.append(row)
    return selected


def unique_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    seen = set()
    for row in rows:
        uid = str(row["chunk_uid"])
        if uid in seen:
            continue
        seen.add(uid)
        selected.append(row)
    return selected


def validate_curation(specs: list[dict[str, Any]]) -> None:
    if len(specs) != 150:
        raise ValueError(f"curation must contain 150 cases, got {len(specs)}")
    queries = [str(spec.get("query") or "").strip() for spec in specs]
    if len(set(queries)) != len(queries):
        raise ValueError("curation contains duplicate queries")
    for query in queries:
        if query.count("？") + query.count("?") != 1:
            raise ValueError(f"query must contain exactly one question mark: {query}")
        if not query.endswith(("？", "?")):
            raise ValueError(f"query must end with a question mark: {query}")
    scope_counts = Counter(str(spec.get("evidence", {}).get("scope") or "") for spec in specs)
    if scope_counts != Counter({"body": 75, "attachment": 75}):
        raise ValueError(f"expected 75 body and 75 attachment cases, got {dict(scope_counts)}")


def validate_cases(cases: list[dict[str, Any]]) -> None:
    for case in cases:
        if not case["expected_chunk_ids"]:
            raise ValueError(f"case has no evidence: {case['query']}")
        if len(case["expected_chunk_ids"]) > 5:
            raise ValueError(f"case has more than 5 gold chunks: {case['query']}")
        EvaluationCaseRequest.model_validate(case)


def dataset_revision(cases: list[dict[str, Any]]) -> str:
    payload = "\n".join(
        f"{case['query']}|{'|'.join(case['expected_chunk_ids'])}|{case['apply_context']['knowledge_revision']}"
        for case in cases
    )
    return "golden-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def stable_user_id(query: str) -> int:
    return 920000 + int(hashlib.sha256(query.encode("utf-8")).hexdigest()[:6], 16) % 80000


def normalized_contains(content: str, term: str) -> bool:
    return re.sub(r"\s+", "", term).lower() in re.sub(r"\s+", "", content).lower()


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def unique_strings(values: Iterable[str]) -> list[str]:
    result = []
    seen = set()
    for value in values:
        value = str(value or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def clean_counter(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda item: str(item[0]))}


if __name__ == "__main__":
    main()
