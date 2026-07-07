"""Parent and structured-section context expansion."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Any

from declaration_qa.core.domain import KnowledgeChunk, RetrievedSource
from declaration_qa.rag.engine.text import safe_int
from declaration_qa.rag.retrieval_plan import backfill_chunk_types_from_plan


def expand_parent_context(sources: list[RetrievedSource], chunks: dict[str, KnowledgeChunk]) -> list[RetrievedSource]:
    """Expand child hits with parent chunk content."""
    expanded = []
    for source in sources:
        chunk = source.chunk
        parent = chunks.get(chunk.parent_id) if chunk.chunk_type == "child" else None
        if parent and parent.content and parent.content != chunk.content:
            expanded_chunk = replace(
                chunk,
                content=parent.content,
                metadata={
                    **chunk.metadata,
                    "retrieved_child_chunk_id": chunk.id,
                    "expanded_parent_chunk_id": parent.id,
                },
            )
            expanded.append(replace(source, chunk=expanded_chunk))
        else:
            expanded.append(source)
    return expanded


def expand_structured_section_context(
    query: str,
    sources: list[RetrievedSource],
    chunks: dict[str, KnowledgeChunk],
    *,
    retrieval_plan: dict[str, Any] | None = None,
) -> tuple[list[RetrievedSource], dict[str, Any]]:
    """Backfill structured section siblings."""
    target_chunk_types = structured_backfill_chunk_types(query, retrieval_plan)
    if not target_chunk_types:
        return sources, {"applied": False}
    expanded: list[RetrievedSource] = []
    seen_chunk_ids: set[str] = set()
    seen_node_keys: set[tuple[str, str, str]] = set()
    backfilled_sections: set[tuple[str, str]] = set()
    added_count = 0
    for source in sources:
        parent_section_id = str(source.chunk.metadata.get("parent_section_id") or "")
        for target_chunk_type in target_chunk_types:
            section_key = (parent_section_id, target_chunk_type)
            if parent_section_id and section_key not in backfilled_sections and structured_source_matches(source, target_chunk_type):
                siblings = structured_section_sibling_sources(source, chunks, parent_section_id, target_chunk_type)
                for sibling in siblings:
                    if append_unique_source(expanded, sibling, seen_chunk_ids, seen_node_keys):
                        added_count += 1
                backfilled_sections.add(section_key)
        append_unique_source(expanded, source, seen_chunk_ids, seen_node_keys)
    return expanded, {
        "applied": bool(backfilled_sections),
        "target_chunk_type": target_chunk_types[0],
        "target_chunk_types": target_chunk_types,
        "section_count": len(backfilled_sections),
        "added_source_count": added_count,
    }


def structured_backfill_chunk_types(query: str, retrieval_plan: dict[str, Any] | None = None) -> list[str]:
    """Infer structured chunk types to backfill."""
    plan_types = backfill_chunk_types_from_plan(retrieval_plan)
    if plan_types:
        return plan_types
    text = str(query or "")
    if any(keyword in text for keyword in ["材料", "申请材料", "要交啥", "要准备什么", "提交什么", "上传什么", "需要哪些材料", "要带", "需要什么"]):
        return ["material_item"]
    if any(keyword in text for keyword in ["流程", "怎么办", "怎么申请", "办理步骤", "提交后", "审核", "如何办理"]):
        return ["process_step"]
    return []


def structured_backfill_chunk_type(query: str, retrieval_plan: dict[str, Any] | None = None) -> str:
    """Infer the primary structured chunk type to backfill."""
    target_chunk_types = structured_backfill_chunk_types(query, retrieval_plan)
    return target_chunk_types[0] if target_chunk_types else ""


def structured_source_matches(source: RetrievedSource, target_chunk_type: str) -> bool:
    """Return whether a source can trigger structured backfill."""
    metadata = source.chunk.metadata or {}
    semantic_chunk_type = str(metadata.get("chunk_type") or "")
    section_title = str(metadata.get("section_title") or "")
    if semantic_chunk_type == target_chunk_type:
        return True
    if target_chunk_type == "material_item":
        return section_title == "申请材料"
    if target_chunk_type == "process_step":
        return section_title == "办理流程"
    return False


def structured_section_sibling_sources(
    trigger: RetrievedSource,
    chunks: dict[str, KnowledgeChunk],
    parent_section_id: str,
    target_chunk_type: str,
) -> list[RetrievedSource]:
    """Return sibling sources in a structured section."""
    grouped: dict[tuple[int, str], list[KnowledgeChunk]] = defaultdict(list)
    for chunk in chunks.values():
        metadata = chunk.metadata or {}
        if chunk.chunk_type != "child":
            continue
        if metadata.get("indexable") is False:
            continue
        if str(metadata.get("parent_section_id") or "") != parent_section_id:
            continue
        if str(metadata.get("chunk_type") or "") != target_chunk_type:
            continue
        node_title = str(metadata.get("node_title") or "")
        grouped[(safe_int(metadata.get("node_order")), node_title)].append(chunk)
    return [
        merged_structured_node_source(trigger, node_chunks)
        for _, node_chunks in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1]))
    ]


def merged_structured_node_source(trigger: RetrievedSource, node_chunks: list[KnowledgeChunk]) -> RetrievedSource:
    """Merge one structured node into a single source."""
    ordered_chunks = sorted(node_chunks, key=structured_chunk_sort_key)
    first = ordered_chunks[0]
    merged_content = merge_structured_node_content(ordered_chunks)
    merged_chunk = replace(
        first,
        parent_id="",
        content=merged_content,
        metadata={
            **first.metadata,
            "merged_chunk_ids": [chunk.id for chunk in ordered_chunks],
            "merged_chunk_count": len(ordered_chunks),
            "structured_backfill_merged": True,
        },
    )
    return RetrievedSource(
        chunk=merged_chunk,
        score=trigger.score,
        dense_score=0.0,
        bm25_score=0.0,
        rerank_score=0.0,
        retrieval_channel=f"{trigger.retrieval_channel}+structured_backfill",
    )


def structured_chunk_sort_key(chunk: KnowledgeChunk) -> tuple[int, int, int, str]:
    """Sort key for structured chunks."""
    metadata = chunk.metadata or {}
    return (
        safe_int(metadata.get("node_order")),
        safe_int(metadata.get("parent_window_index")),
        safe_int(metadata.get("child_window_index")),
        chunk.id,
    )


def merge_structured_node_content(chunks: list[KnowledgeChunk]) -> str:
    """Merge chunk contents without duplicates."""
    parts: list[str] = []
    seen_content: set[str] = set()
    for chunk in chunks:
        content = chunk.content.strip()
        if not content or content in seen_content:
            continue
        parts.append(content)
        seen_content.add(content)
    return "\n\n".join(parts)


def append_unique_source(
    output: list[RetrievedSource],
    source: RetrievedSource,
    seen_chunk_ids: set[str],
    seen_node_keys: set[tuple[str, str, str]],
) -> bool:
    """Append a source once."""
    chunk = source.chunk
    if chunk.id in seen_chunk_ids:
        return False
    metadata = chunk.metadata or {}
    parent_section_id = str(metadata.get("parent_section_id") or "")
    semantic_chunk_type = str(metadata.get("chunk_type") or "")
    node_title = str(metadata.get("node_title") or "")
    if parent_section_id and semantic_chunk_type and node_title:
        node_key = (parent_section_id, semantic_chunk_type, node_title)
        if node_key in seen_node_keys:
            return False
        seen_node_keys.add(node_key)
    seen_chunk_ids.add(chunk.id)
    output.append(source)
    return True
