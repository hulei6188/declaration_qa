"""知识文档切片策略。"""

from __future__ import annotations

from dataclasses import replace
import re
from typing import Any

from declaration_qa.core.domain import KnowledgeChunk, KnowledgeDocument
from declaration_qa.rag.engine.constants import (
    CHUNKING_DOCLING,
    CHUNKING_HIERARCHICAL,
    CHUNKING_PARENT_CHILD,
    SUPPORTED_CHUNKING_STRATEGIES,
)
from declaration_qa.rag.retrieval import sanitize_metadata


def normalize_chunking_strategy(value: Any) -> str:
    """归一化文档切片策略。"""
    strategy = str(value or CHUNKING_PARENT_CHILD).strip().lower().replace("-", "_")
    return strategy if strategy in SUPPORTED_CHUNKING_STRATEGIES else CHUNKING_PARENT_CHILD


def split_document(document: KnowledgeDocument) -> list[KnowledgeChunk]:
    """按配置的策略切分文档。"""
    strategy = normalize_chunking_strategy(document.metadata.get("chunking_strategy"))
    if strategy == CHUNKING_DOCLING:
        chunks = []
    elif strategy == CHUNKING_HIERARCHICAL:
        chunks = split_hierarchical(document)
    else:
        chunks = split_parent_child(document)
    chunks.extend(derived_chunks_from_document(document))
    return attach_sibling_ids(chunks)


def split_parent_child(
    document: KnowledgeDocument,
    *,
    parent_size: int = 1200,
    child_size: int = 420,
    overlap: int = 80,
) -> list[KnowledgeChunk]:
    """将文档切分为父窗口和子窗口。"""
    content = document.content.strip()
    if not content:
        return []
    chunks: list[KnowledgeChunk] = []
    parent_texts = split_windows(content, parent_size, max(0, min(overlap, parent_size // 2)))
    for parent_index, parent_text in enumerate(parent_texts):
        parent_id = f"{document.id}:p:{parent_index}"
        chunks.append(
            knowledge_chunk_from_document(
                document,
                chunk_id=parent_id,
                parent_id=parent_id,
                chunk_type="parent",
                content=parent_text,
            )
        )
        child_texts = split_windows(parent_text, child_size, max(0, min(overlap, child_size // 2)))
        for child_index, child_text in enumerate(child_texts):
            child_id = f"{parent_id}:c:{child_index}"
            chunks.append(
                knowledge_chunk_from_document(
                    document,
                    chunk_id=child_id,
                    parent_id=parent_id,
                    chunk_type="child",
                    content=child_text,
                )
            )
    return chunks


def split_hierarchical(
    document: KnowledgeDocument,
    *,
    parent_size: int = 1200,
    child_size: int = 420,
    overlap: int = 80,
) -> list[KnowledgeChunk]:
    """按标题层级切分结构化 Markdown 文档。"""
    content = document.content.strip()
    if not content:
        return []
    chunks: list[KnowledgeChunk] = []
    for section_index, section in enumerate(markdown_sections(content)):
        base_metadata = hierarchical_section_metadata(document, section, section_index)
        section_content = section["content"].strip()
        if len(section_content) <= child_size:
            chunks.append(
                knowledge_chunk_from_document(
                    document,
                    chunk_id=f"{document.id}:h:{section_index}:c:0",
                    parent_id="",
                    chunk_type="child",
                    content=section_content,
                    extra_metadata={**base_metadata, "child_window_index": 0, "indexable": True},
                )
            )
            continue
        parent_texts = split_windows(section_content, parent_size, max(0, min(overlap, parent_size // 2)))
        for parent_index, parent_text in enumerate(parent_texts):
            child_texts = split_windows(parent_text, child_size, max(0, min(overlap, child_size // 2)))
            if len(child_texts) == 1 and child_texts[0].strip() == parent_text.strip():
                chunks.append(
                    knowledge_chunk_from_document(
                        document,
                        chunk_id=f"{document.id}:h:{section_index}:p:{parent_index}:c:0",
                        parent_id="",
                        chunk_type="child",
                        content=parent_text,
                        extra_metadata={
                            **base_metadata,
                            "parent_window_index": parent_index,
                            "child_window_index": 0,
                            "indexable": True,
                        },
                    )
                )
                continue
            parent_id = f"{document.id}:h:{section_index}:p:{parent_index}"
            chunks.append(
                knowledge_chunk_from_document(
                    document,
                    chunk_id=parent_id,
                    parent_id=parent_id,
                    chunk_type="parent",
                    content=parent_text,
                    extra_metadata={
                        **base_metadata,
                        "parent_window_index": parent_index,
                        "context_role": "section_context",
                        "indexable": False,
                        "context_only": True,
                    },
                )
            )
            for child_index, child_text in enumerate(child_texts):
                chunks.append(
                    knowledge_chunk_from_document(
                        document,
                        chunk_id=f"{parent_id}:c:{child_index}",
                        parent_id=parent_id,
                        chunk_type="child",
                        content=child_text,
                        extra_metadata={
                            **base_metadata,
                            "parent_window_index": parent_index,
                            "child_window_index": child_index,
                            "indexable": True,
                        },
                    )
                )
    return chunks


def hierarchical_section_metadata(document: KnowledgeDocument, section: dict[str, Any], section_index: int) -> dict[str, Any]:
    """构建层级片段元数据。"""
    section_title = str(section.get("section_title") or section.get("title") or "Document")
    node_title = str(section.get("node_title") or section.get("title") or section_title)
    node_order = parse_heading_order(node_title)
    if node_order is None:
        node_order = section_index
    return {
        "chunking_strategy": CHUNKING_HIERARCHICAL,
        "heading_path": section["path"],
        "heading_level": section["level"],
        "section_index": section_index,
        "section_title": section_title,
        "node_title": node_title,
        "parent_section_id": f"{document.id}#{section_title}",
        "node_order": node_order,
        "chunk_type": classify_hierarchical_chunk_type(section_title, node_title),
    }


def classify_hierarchical_chunk_type(section_title: str, node_title: str) -> str:
    """分类结构化片段的语义类型。"""
    if section_title == "申请材料" and parse_heading_order(node_title) is not None:
        return "material_item"
    normalized_node_title = strip_heading_order(node_title)
    if section_title == "办理流程" and normalized_node_title in {"受理", "审查", "决定", "办结", "送达"}:
        return "process_step"
    return "section_leaf"


def parse_heading_order(title: str) -> int | None:
    """解析数字标题序号。"""
    match = re.match(r"^\s*(\d+)(?:[\.、．\)]\s*|\s+)", str(title or ""))
    return int(match.group(1)) if match else None


def strip_heading_order(title: str) -> str:
    """移除开头的数字标题序号。"""
    return re.sub(r"^\s*\d+(?:[\.、．\)]\s*|\s+)", "", str(title or "")).strip()


def markdown_sections(text: str) -> list[dict[str, Any]]:
    """将 Markdown 标题解析为章节。"""
    sections: list[dict[str, Any]] = []
    stack: list[tuple[int, str]] = []
    current_lines: list[str] = []
    current_path: list[str] = []
    current_level = 0
    current_title = ""
    heading_pattern = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")

    def flush_current() -> None:
        body = "\n".join(current_lines).strip()
        if not body:
            return
        path = " > ".join(current_path) if current_path else "Document"
        section_text = f"{path}\n\n{body}".strip() if current_path else body
        section_title = markdown_section_title(stack, current_title)
        node_title = current_title or section_title
        sections.append(
            {
                "path": path,
                "level": current_level,
                "title": current_title,
                "section_title": section_title,
                "node_title": node_title,
                "body": body,
                "content": section_text,
            }
        )

    for line in text.splitlines():
        heading = heading_pattern.match(line.strip())
        if heading:
            flush_current()
            level = len(heading.group(1))
            title = heading.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            current_path = [item[1] for item in stack]
            current_level = level
            current_title = title
            current_lines = []
            continue
        current_lines.append(line)
    flush_current()
    if sections:
        return [section for section in sections if section["content"].strip()]
    return [{"path": "Document", "level": 0, "content": text.strip()}]


def markdown_section_title(stack: list[tuple[int, str]], current_title: str) -> str:
    """返回当前 Markdown 栈对应的语义章节标题。"""
    for level, title in stack:
        if level == 2:
            return title
    if current_title:
        return current_title
    if stack:
        return stack[-1][1]
    return "Document"


def split_windows(text: str, size: int, overlap: int) -> list[str]:
    """将文本切分为重叠窗口。"""
    content = str(text or "").strip()
    if len(content) <= size:
        return [content]
    units = split_window_units(content, max(1, size))
    output = []
    current: list[str] = []
    for unit in units:
        candidate = join_window_units([*current, unit])
        if current and len(candidate) > size:
            output.append(join_window_units(current))
            current = trailing_overlap_units(current, overlap)
        current.append(unit)
    if current:
        output.append(join_window_units(current))
    return [item for item in output if item]


def split_window_units(text: str, size: int) -> list[str]:
    """将文本切分为按行组织的单元。"""
    units: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        units.extend(split_long_line(line.rstrip(), size))
    return units


def split_long_line(line: str, size: int) -> list[str]:
    """在软边界处切分过长行。"""
    if len(line) <= size:
        return [line]
    if "http://" in line or "https://" in line:
        return [line]
    parts: list[str] = []
    remaining = line.strip()
    while len(remaining) > size:
        split_at = best_soft_split(remaining, size)
        parts.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()
    if remaining:
        parts.append(remaining)
    return parts


def best_soft_split(text: str, size: int) -> int:
    """在目标大小附近寻找软切分点。"""
    search_start = max(1, size // 2)
    candidates = [text.rfind(separator, search_start, size) for separator in ["。", "；", "，", "、", " ", "\t"]]
    split_at = max(candidates)
    return size if split_at <= 0 else split_at + 1


def trailing_overlap_units(units: list[str], overlap: int) -> list[str]:
    """返回保留尾部重叠所需的单元。"""
    if overlap <= 0:
        return []
    retained: list[str] = []
    total = 0
    for unit in reversed(units):
        total += len(unit)
        if retained and total > overlap:
            break
        retained.insert(0, unit)
    return retained


def join_window_units(units: list[str]) -> str:
    """拼接窗口单元。"""
    return "\n".join(unit for unit in units if unit.strip()).strip()


def knowledge_chunk_from_document(
    document: KnowledgeDocument,
    *,
    chunk_id: str,
    parent_id: str,
    chunk_type: str,
    content: str,
    extra_metadata: dict[str, Any] | None = None,
) -> KnowledgeChunk:
    """根据文档创建知识片段。"""
    document_metadata = sanitize_metadata(document.metadata)
    metadata = {
        **document_metadata,
        "region_code": document.region_code,
        "region_key": document_metadata.get("region_key") or document.region_code,
        "policy_version": document.policy_version,
        "source_version": document_metadata.get("source_version") or document.policy_version,
        "audit_node": document.audit_node,
        "version_group_uid": document.version_group_uid or document.id,
        "version_no": document.version_no,
        "rollout_percent": document.rollout_percent,
        "doc_status": document_metadata.get("doc_status") or document.status,
    }
    metadata.update(sanitize_metadata(extra_metadata or {}))
    return KnowledgeChunk(
        id=chunk_id,
        doc_id=document.id,
        parent_id=parent_id,
        chunk_type=chunk_type,
        title=document.title,
        content=content,
        region_code=document.region_code,
        policy_version=document.policy_version,
        source_url=document.source_url,
        audit_node=document.audit_node,
        status=document.status,
        effective_start_time=document.effective_start_time,
        effective_end_time=document.effective_end_time,
        embedding_id=f"vec:{chunk_id}",
        metadata=metadata,
    )


def derived_chunks_from_document(document: KnowledgeDocument) -> list[KnowledgeChunk]:
    """将索引流水线生成的派生片段转换为知识片段。"""
    chunks: list[KnowledgeChunk] = []
    for index, spec in enumerate(document.derived_chunks or []):
        content = str(spec.get("content") or "").strip()
        if not content:
            continue
        id_suffix = str(spec.get("id_suffix") or f"derived:{index}")
        chunk_type = normalize_derived_chunk_type(spec.get("chunk_type"))
        chunk_id = f"{document.id}:{id_suffix}"
        parent_id = str(spec.get("parent_id") or "")
        parent_id_suffix = str(spec.get("parent_id_suffix") or "")
        if not parent_id and parent_id_suffix:
            parent_id = f"{document.id}:{parent_id_suffix}"
        if chunk_type == "parent" and not parent_id:
            parent_id = chunk_id
        metadata = dict(spec.get("metadata") or {})
        chunk = knowledge_chunk_from_document(
            document,
            chunk_id=chunk_id,
            parent_id=parent_id,
            chunk_type=chunk_type,
            content=content,
            extra_metadata={
                **metadata,
                "derived_chunk": True,
                "derived_chunk_index": index,
            },
        )
        source_url = str(spec.get("source_url") or chunk.source_url)
        if source_url != chunk.source_url:
            chunk = replace(chunk, source_url=source_url)
        chunks.append(chunk)
    return chunks


def normalize_derived_chunk_type(value: Any) -> str:
    """归一化派生片段的物理 chunk 类型。"""
    return "parent" if str(value or "").strip().lower() == "parent" else "child"


def attach_sibling_ids(chunks: list[KnowledgeChunk]) -> list[KnowledgeChunk]:
    """附加用于章节级材料回填的同级片段 ID。"""
    groups: dict[str, list[str]] = {}
    for chunk in chunks:
        if chunk.chunk_type != "child":
            continue
        group_key = str(chunk.metadata.get("parent_section_id") or chunk.parent_id or chunk.doc_id)
        groups.setdefault(group_key, []).append(chunk.id)
    enriched: list[KnowledgeChunk] = []
    for chunk in chunks:
        if chunk.chunk_type != "child":
            enriched.append(chunk)
            continue
        group_key = str(chunk.metadata.get("parent_section_id") or chunk.parent_id or chunk.doc_id)
        sibling_ids = [item for item in groups.get(group_key, []) if item != chunk.id]
        enriched.append(replace(chunk, metadata={**chunk.metadata, "sibling_ids": sibling_ids}))
    return enriched
