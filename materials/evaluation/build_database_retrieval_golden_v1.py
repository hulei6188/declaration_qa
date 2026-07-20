"""从主知识库生成同时覆盖正文与附件证据的检索评测集。"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

from sqlalchemy import create_engine, text

from gov_service_agent.api.schemas import EvaluationCaseRequest
from gov_service_agent.application.evals.pii import sanitize_text
from gov_service_agent.core.config import get_settings


OUTPUT = Path(__file__).with_name("database-retrieval-golden-v1.json")
DATASET_NAME = "database-retrieval-golden-v1"
TARGET_DOCUMENT_CASES = 60
TARGET_ATTACHMENT_CASES = 60
PII_SALT = DATASET_NAME

BODY_SQL = text(
    """
    SELECT d.document_uid,
           COALESCE(NULLIF(d.metadata->>'item_name', ''), d.title) AS item_name,
           d.metadata->>'item_code' AS item_code,
           COALESCE(NULLIF(d.metadata->>'region_code', ''), '340000') AS region_code,
           COALESCE(NULLIF(d.metadata->>'region_name', ''), '安徽省') AS region_name,
           COALESCE(NULLIF(d.metadata->>'policy_version', ''), '2026') AS policy_version,
           COALESCE(NULLIF(d.metadata->>'knowledge_revision', ''), '') AS knowledge_revision,
           c.chunk_uid,
           c.content,
           COALESCE(c.metadata->>'heading_path', '') AS heading_path,
           COALESCE(c.metadata->>'node_title', '') AS node_title,
           COALESCE(NULLIF(c.metadata->>'chunk_type', ''), c.chunk_type) AS semantic_type
    FROM declaration_qa.documents d
    JOIN declaration_qa.chunks c ON c.document_id = d.id
    WHERE d.status = 'published'
      AND c.status = 'published'
      AND COALESCE(d.metadata->>'item_code', '') <> ''
      AND COALESCE(c.metadata->>'attachment_name', '') = ''
      AND COALESCE(NULLIF(c.metadata->>'chunk_type', ''), c.chunk_type)
          IN ('chonkie_text', 'chonkie_list', 'chonkie_table', 'child')
      AND length(btrim(c.content)) >= 80
    ORDER BY item_name, d.document_uid, c.chunk_index
    """
)

ASSET_SQL = text(
    """
    SELECT d.document_uid,
           COALESCE(NULLIF(d.metadata->>'item_name', ''), d.title) AS item_name,
           d.metadata->>'item_code' AS item_code,
           COALESCE(NULLIF(d.metadata->>'region_code', ''), '340000') AS region_code,
           COALESCE(NULLIF(d.metadata->>'region_name', ''), '安徽省') AS region_name,
           COALESCE(NULLIF(d.metadata->>'policy_version', ''), '2026') AS policy_version,
           COALESCE(NULLIF(d.metadata->>'knowledge_revision', ''), '') AS knowledge_revision,
           a.asset_uid,
           a.attachment_name,
           a.attachment_role,
           a.attachment_type,
           a.asset_kind,
           a.page_no,
           c.chunk_uid,
           c.content,
           COALESCE(c.metadata->>'material_name', '') AS material_name,
           COALESCE(c.metadata->>'section_path', '') AS section_path,
           COALESCE(c.metadata->>'visual_summary_status', '') AS visual_summary_status
    FROM declaration_qa.documents d
    JOIN declaration_qa.assets a
      ON a.document_id = d.id AND a.extraction_status = 'success'
    JOIN declaration_qa.chunks c
      ON c.id = a.chunk_id AND c.status = 'published'
    WHERE d.status = 'published'
      AND COALESCE(d.metadata->>'item_code', '') <> ''
      AND COALESCE(a.attachment_name, '') <> ''
      AND length(btrim(c.content)) >= 40
    ORDER BY item_name, d.document_uid, a.attachment_index, a.id
    """
)

EXCLUDED_BODY_PATHS = ("办事评价", "附件清单", "流程图")
BODY_CATEGORY_ROTATION = ("material", "eligibility", "process", "faq", "basic")
BODY_CATEGORY_PRIORITY = {
    "material": 0,
    "eligibility": 1,
    "process": 2,
    "faq": 3,
    "fee_time": 4,
    "basic": 5,
    "other": 6,
}
SAMPLE_TYPE_ROTATION = ("png", "jpg", "docx", "pdf", "png", "jpg", "docx", "pdf")


def main() -> None:
    settings = get_settings()
    if not settings.effective_database_url:
        raise ValueError("GOV_SERVICE_AGENT_DATABASE_URL is required")
    engine = create_engine(settings.effective_database_url, future=True)
    try:
        with engine.connect() as connection:
            body_rows = [dict(row) for row in connection.execute(BODY_SQL).mappings()]
            asset_rows = [dict(row) for row in connection.execute(ASSET_SQL).mappings()]
    finally:
        engine.dispose()

    document_cases = build_document_cases(body_rows, target=TARGET_DOCUMENT_CASES)
    attachment_cases = build_attachment_cases(asset_rows, target=TARGET_ATTACHMENT_CASES)
    cases = interleave(document_cases, attachment_cases)
    validate_cases(cases)
    document_uids = sorted({case["apply_context"]["document_uid"] for case in cases})
    knowledge_revision = database_revision(cases)
    attachment_role_counts = Counter(case["apply_context"].get("attachment_role") for case in attachment_cases)
    attachment_type_counts = Counter(case["apply_context"].get("attachment_type") for case in attachment_cases)
    payload = {
        "name": DATASET_NAME,
        "description": (
            "从主知识库已发布事项生成的120条检索草稿集：60条正文证据、60条附件证据；"
            "覆盖流程图、样表及PNG/JPG/DOCX/PDF附件。发布为黄金集前需人工复核Query和证据金标。"
        ),
        "version": 1,
        "status": "draft",
        "source": "database-derived",
        "knowledge_revision": knowledge_revision,
        "generation": {
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "document_count": len(document_uids),
            "case_count": len(cases),
            "document_case_count": len(document_cases),
            "attachment_case_count": len(attachment_cases),
            "attachment_role_counts": clean_counter(attachment_role_counts),
            "attachment_type_counts": clean_counter(attachment_type_counts),
            "gold_level": "doc_id+chunk_id",
            "pii_policy": "phone/id-card/email masked; raw OCR excluded",
            "review_status": "manual_review_required",
        },
        "cases": cases,
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["generation"], ensure_ascii=False, indent=2))
    print(f"wrote {len(cases)} cases to {OUTPUT}")


def build_document_cases(rows: list[dict[str, Any]], *, target: int) -> list[dict[str, Any]]:
    by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        path = normalized_path(row)
        if not path or any(excluded in path for excluded in EXCLUDED_BODY_PATHS):
            continue
        row = {**row, "evidence_category": body_category(path)}
        by_document[str(row["document_uid"])].append(row)
    if target < len(by_document):
        raise ValueError(f"document target {target} is smaller than eligible document count {len(by_document)}")

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    selected_queries: set[str] = set()
    ordered_documents = sorted(by_document, key=lambda uid: (str(by_document[uid][0]["item_name"]), uid))
    for index, document_uid in enumerate(ordered_documents):
        candidates = sorted_body_candidates(by_document[document_uid])
        desired = BODY_CATEGORY_ROTATION[index % len(BODY_CATEGORY_ROTATION)]
        desired_candidates = [item for item in candidates if item["evidence_category"] == desired]
        candidate = next(
            (
                item
                for item in desired_candidates + candidates
                if document_query(
                    clean_text(item["item_name"]),
                    normalized_path(item),
                    str(item["evidence_category"]),
                )
                not in selected_queries
            ),
            None,
        )
        if candidate is None:
            raise ValueError(f"document {document_uid} has no candidate with a unique query")
        selected.append(candidate)
        selected_ids.add(str(candidate["chunk_uid"]))
        selected_queries.add(
            document_query(
                clean_text(candidate["item_name"]),
                normalized_path(candidate),
                str(candidate["evidence_category"]),
            )
        )

    while len(selected) < target:
        progressed = False
        for document_uid in ordered_documents:
            for candidate in sorted_body_candidates(by_document[document_uid]):
                chunk_uid = str(candidate["chunk_uid"])
                if chunk_uid in selected_ids:
                    continue
                query = document_query(
                    clean_text(candidate["item_name"]),
                    normalized_path(candidate),
                    str(candidate["evidence_category"]),
                )
                if query in selected_queries:
                    continue
                selected.append(candidate)
                selected_ids.add(chunk_uid)
                selected_queries.add(query)
                progressed = True
                break
            if len(selected) >= target:
                break
        if not progressed:
            raise ValueError(f"only {len(selected)} unique document cases are available, target={target}")
    return [make_document_case(row, ordinal=index) for index, row in enumerate(selected, start=1)]


def build_attachment_cases(rows: list[dict[str, Any]], *, target: int) -> list[dict[str, Any]]:
    by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_document[str(row["document_uid"])].append(row)
    if target < len(by_document):
        raise ValueError(f"attachment target {target} is smaller than eligible document count {len(by_document)}")

    selected: list[dict[str, Any]] = []
    selected_assets: set[str] = set()
    ordered_documents = sorted(by_document, key=lambda uid: (str(by_document[uid][0]["item_name"]), uid))
    for document_uid in ordered_documents:
        candidates = sorted(by_document[document_uid], key=attachment_sort_key)
        flowchart = next((item for item in candidates if item["attachment_role"] == "flowchart"), None)
        candidate = flowchart or candidates[0]
        selected.append(candidate)
        selected_assets.add(str(candidate["asset_uid"]))

    sample_rows = [row for row in rows if row["attachment_role"] == "sample_file"]
    used_extra_documents: set[str] = set()
    for attachment_type in SAMPLE_TYPE_ROTATION:
        if len(selected) >= target:
            break
        candidate = next(
            (
                row
                for row in sorted(sample_rows, key=attachment_sort_key)
                if str(row["attachment_type"]).lower() == attachment_type
                and str(row["asset_uid"]) not in selected_assets
                and str(row["document_uid"]) not in used_extra_documents
            ),
            None,
        )
        if candidate is None:
            candidate = next(
                (
                    row
                    for row in sorted(sample_rows, key=attachment_sort_key)
                    if str(row["attachment_type"]).lower() == attachment_type
                    and str(row["asset_uid"]) not in selected_assets
                ),
                None,
            )
        if candidate is None:
            continue
        selected.append(candidate)
        selected_assets.add(str(candidate["asset_uid"]))
        used_extra_documents.add(str(candidate["document_uid"]))

    for candidate in sorted(sample_rows, key=attachment_sort_key):
        if len(selected) >= target:
            break
        asset_uid = str(candidate["asset_uid"])
        if asset_uid in selected_assets:
            continue
        selected.append(candidate)
        selected_assets.add(asset_uid)
    if len(selected) < target:
        raise ValueError(f"only {len(selected)} unique attachment cases are available, target={target}")
    return [make_attachment_case(row, ordinal=index) for index, row in enumerate(selected[:target], start=1)]


def sorted_body_candidates(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            BODY_CATEGORY_PRIORITY.get(str(row["evidence_category"]), 99),
            len(str(row["content"])),
            str(row["chunk_uid"]),
        ),
    )


def attachment_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    role_order = {"flowchart": 0, "sample_file": 1, "other_attachment": 2}
    summary_order = 0 if row.get("visual_summary_status") == "success" else 1
    return (
        role_order.get(str(row.get("attachment_role") or ""), 9),
        summary_order,
        -len(str(row.get("content") or "")),
        str(row.get("attachment_name") or ""),
        str(row.get("asset_uid") or ""),
    )


def make_document_case(row: dict[str, Any], *, ordinal: int) -> dict[str, Any]:
    category = str(row["evidence_category"])
    item_name = clean_text(row["item_name"])
    path = normalized_path(row)
    evidence = sanitized_evidence(row["content"])
    labels = base_labels(row, evidence_scope="document_body")
    labels.update(
        {
            "body_category": category,
            "heading_path": path,
            "query_provenance": "document_derived_candidate",
            "gold_provenance": "database_relation_candidate",
            "review_status": "manual_review_required",
        }
    )
    return {
        "query": document_query(item_name, path, category),
        "user_id": 920000 + ordinal,
        "region_code": normalized_region_code(row["region_code"]),
        "policy_version": clean_text(row["policy_version"]) or "2026",
        "biz_type": None,
        "apply_id": None,
        "apply_context": labels,
        "case_type": "rag",
        "turns": [],
        "fixture_key": "",
        "expected_intent": "",
        "expected_refused": False,
        "expected_doc_ids": [str(row["document_uid"])],
        "expected_chunk_ids": [str(row["chunk_uid"])],
        "expected_answer_keywords": [],
        "reference_answer": evidence_excerpt(evidence),
        "reference_contexts": [evidence],
        "expected_trajectory": {"required_tools": ["knowledge_search"], "forbidden_tools": []},
        "expected_outcome": {},
        "severity": "medium",
        "enabled": True,
        "tags": tag_values(labels, dataset=DATASET_NAME),
    }


def make_attachment_case(row: dict[str, Any], *, ordinal: int) -> dict[str, Any]:
    item_name = clean_text(row["item_name"])
    evidence = sanitized_evidence(row["content"])
    labels = base_labels(row, evidence_scope="attachment")
    labels.update(
        {
            "asset_uid": str(row["asset_uid"]),
            "attachment_name": clean_text(row["attachment_name"]),
            "attachment_role": clean_text(row["attachment_role"]),
            "attachment_type": clean_text(row["attachment_type"]).lower(),
            "asset_kind": clean_text(row["asset_kind"]),
            "page_no": row.get("page_no"),
            "material_name": clean_text(row.get("material_name")),
            "query_provenance": "document_derived_candidate",
            "gold_provenance": "database_relation_candidate",
            "review_status": "manual_review_required",
        }
    )
    return {
        "query": attachment_query(item_name, row),
        "user_id": 930000 + ordinal,
        "region_code": normalized_region_code(row["region_code"]),
        "policy_version": clean_text(row["policy_version"]) or "2026",
        "biz_type": None,
        "apply_id": None,
        "apply_context": labels,
        "case_type": "rag",
        "turns": [],
        "fixture_key": "",
        "expected_intent": "",
        "expected_refused": False,
        "expected_doc_ids": [str(row["document_uid"])],
        "expected_chunk_ids": [str(row["chunk_uid"])],
        "expected_answer_keywords": [],
        "reference_answer": evidence_excerpt(evidence),
        "reference_contexts": [evidence],
        "expected_trajectory": {"required_tools": ["knowledge_search"], "forbidden_tools": []},
        "expected_outcome": {},
        "severity": "high",
        "enabled": True,
        "tags": tag_values(labels, dataset=DATASET_NAME),
    }


def document_query(item_name: str, path: str, category: str) -> str:
    leaf = clean_heading(path.rsplit(">", 1)[-1])
    if category == "material":
        return f"办理{item_name}需要提交{leaf}吗，相关材料在哪里查看或下载？"
    if category == "eligibility":
        return f"申请{item_name}需要满足哪些受理条件？"
    if category == "process":
        if leaf not in {"办理流程", "办理流程说明", "办理环节"}:
            return f"{item_name}在“{leaf}”环节主要办理什么，时限和职责是什么？"
        return f"{item_name}从申请到办结要经过哪些步骤？"
    if category == "faq":
        question = re.sub(r"^Q\s*\d+\s*[:：]?\s*", "", leaf, flags=re.I).strip()
        return question if question.endswith(("？", "?")) else f"{question}？"
    if category == "fee_time":
        if "收费" in leaf:
            return f"办理{item_name}是否收费，收费依据和标准是什么？"
        return f"{item_name}承诺多长时间办结，法定办结时限是多少？"
    if category == "basic":
        return f"{item_name}属于什么事项类型，面向哪些服务对象？"
    return f"办理{item_name}时，关于“{leaf}”有哪些具体规定？"


def attachment_query(item_name: str, row: dict[str, Any]) -> str:
    name = clean_text(row["attachment_name"])
    role = clean_text(row["attachment_role"])
    material_name = clean_text(row.get("material_name"))
    page_no = row.get("page_no")
    page_hint = f"第{page_no}页" if page_no else ""
    if role == "flowchart":
        return f"根据{item_name}的流程图附件《{name}》，办理过程有哪些主要步骤？"
    if role == "sample_file":
        return f"{item_name}的材料示例附件《{name}》{page_hint}展示了哪些主要内容，办理时应关注哪些信息？"
    subject = material_name or name
    return f"办理{item_name}时，附件《{name}》中的{subject}主要用于证明什么、包含哪些关键信息？"


def body_category(path: str) -> str:
    if "申请材料" in path:
        return "material"
    if "受理条件" in path:
        return "eligibility"
    if "办理流程" in path or "办理环节" in path:
        return "process"
    if "常见问题" in path or re.search(r"(?:^|>)\s*Q\s*\d+", path, flags=re.I):
        return "faq"
    if "收费" in path or "办理时限" in path:
        return "fee_time"
    if "基本信息" in path:
        return "basic"
    return "other"


def normalized_path(row: dict[str, Any]) -> str:
    return clean_text(row.get("heading_path") or row.get("node_title"))


def base_labels(row: dict[str, Any], *, evidence_scope: str) -> dict[str, Any]:
    return {
        "document_uid": str(row["document_uid"]),
        "chunk_uid": str(row["chunk_uid"]),
        "item_code": clean_text(row["item_code"]),
        "item_name": clean_text(row["item_name"]),
        "region_code": normalized_region_code(row["region_code"]),
        "region_name": clean_text(row["region_name"]),
        "knowledge_revision": clean_text(row["knowledge_revision"]),
        "evidence_scope": evidence_scope,
    }


def tag_values(labels: dict[str, Any], *, dataset: str) -> list[str]:
    keys = (
        "evidence_scope",
        "item_code",
        "region_code",
        "body_category",
        "attachment_role",
        "attachment_type",
        "asset_kind",
    )
    values = [dataset, "database-derived", "manual-review-required"]
    values.extend(f"{key}:{labels[key]}" for key in keys if labels.get(key) not in (None, ""))
    return values


def sanitized_evidence(value: Any) -> str:
    text_value = sanitize_text(clean_text(value), salt=PII_SALT)
    return text_value[:8_000]


def evidence_excerpt(value: str) -> str:
    return value[:1_500]


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def clean_heading(value: str) -> str:
    return re.sub(r"^\s*(?:\d+[\.、．\)]\s*|Q\s*\d+\s*[:：]\s*)", "", clean_text(value), flags=re.I)


def normalized_region_code(value: Any) -> str:
    region_code = clean_text(value)
    return region_code[:12] or "340000"


def interleave(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index in range(max(len(left), len(right))):
        if index < len(left):
            output.append(left[index])
        if index < len(right):
            output.append(right[index])
    return output


def validate_cases(cases: list[dict[str, Any]]) -> None:
    if len(cases) != TARGET_DOCUMENT_CASES + TARGET_ATTACHMENT_CASES:
        raise ValueError(f"unexpected case count: {len(cases)}")
    queries: set[str] = set()
    for index, case in enumerate(cases, start=1):
        EvaluationCaseRequest.model_validate(case)
        query = str(case["query"])
        if query in queries:
            raise ValueError(f"duplicate query at case {index}: {query}")
        queries.add(query)
        if len(case["expected_doc_ids"]) != 1 or len(case["expected_chunk_ids"]) != 1:
            raise ValueError(f"case {index} must contain one doc and one chunk gold label")
        if not case["reference_contexts"] or not case["reference_contexts"][0].strip():
            raise ValueError(f"case {index} has no reference evidence")


def database_revision(cases: list[dict[str, Any]]) -> str:
    values = sorted(
        {
            f"{case['apply_context']['document_uid']}:{case['apply_context'].get('knowledge_revision') or ''}"
            for case in cases
        }
    )
    digest = hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()[:16]
    return f"database-{digest}"


def clean_counter(counter: Counter) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items()) if key not in (None, "")}


if __name__ == "__main__":
    main()
