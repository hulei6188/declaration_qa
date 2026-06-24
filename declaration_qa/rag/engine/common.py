"""Shared helpers for RAG engines."""

from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from dataclasses import replace
from typing import Any, Iterable

from declaration_qa.core.domain import KnowledgeChunk, KnowledgeDocument, RetrievedSource
from declaration_qa.rag.retrieval import sanitize_metadata


EMBEDDING_DIMS = 96
CHUNKING_PARENT_CHILD = "parent_child"
CHUNKING_HIERARCHICAL = "hierarchical"
SUPPORTED_CHUNKING_STRATEGIES = {CHUNKING_PARENT_CHILD, CHUNKING_HIERARCHICAL}


def normalize_text(value: Any) -> str:
    """Normalize text to compact whitespace."""
    return re.sub(r"\s+", " ", str(value or "").strip())


def tokenize(text: str) -> list[str]:
    """Tokenize text for local retrieval scoring."""
    normalized = normalize_text(text).lower()
    try:
        import jieba

        tokens = [token.strip() for token in jieba.lcut(normalized) if token.strip()]
    except Exception:
        tokens = re.findall(r"[\w\u4e00-\u9fff]+", normalized)
    expanded = []
    for token in tokens:
        expanded.append(token)
        if len(token) >= 4 and re.search(r"[\u4e00-\u9fff]", token):
            expanded.extend(token[index : index + 2] for index in range(len(token) - 1))
    return [token for token in expanded if token]


def unique_ordered_strings(values: list[str]) -> list[str]:
    """Return non-empty strings in first-seen order."""
    seen = set()
    output = []
    for value in values:
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def safe_int(value: Any) -> int:
    """Convert a value to int, defaulting to 0."""
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def cosine(left: Iterable[float], right: Iterable[float]) -> float:
    """Return cosine dot product for normalized vectors."""
    left_values = list(left)
    right_values = list(right)
    return sum(a * b for a, b in zip(left_values, right_values))


def embed_text(text: str) -> list[float]:
    """Generate a local hashing embedding."""
    values = [0.0 for _ in range(EMBEDDING_DIMS)]
    for token in tokenize(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:2], "big") % EMBEDDING_DIMS
        sign = 1.0 if digest[2] % 2 == 0 else -1.0
        values[index] += sign
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]


def normalize_chunking_strategy(value: Any) -> str:
    """Normalize document chunking strategy."""
    strategy = str(value or CHUNKING_PARENT_CHILD).strip().lower().replace("-", "_")
    return strategy if strategy in SUPPORTED_CHUNKING_STRATEGIES else CHUNKING_PARENT_CHILD


def split_document(document: KnowledgeDocument) -> list[KnowledgeChunk]:
    """Split a document with the configured strategy."""
    strategy = normalize_chunking_strategy(document.metadata.get("chunking_strategy"))
    if strategy == CHUNKING_HIERARCHICAL:
        return split_hierarchical(document)
    return split_parent_child(document)


def split_parent_child(
    document: KnowledgeDocument,
    *,
    parent_size: int = 1200,
    child_size: int = 420,
    overlap: int = 80,
) -> list[KnowledgeChunk]:
    """Split a document into parent and child windows."""
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
    """Split a structured markdown document by heading hierarchy."""
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
    """Build metadata for hierarchical chunks."""
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
    """Classify structured chunk semantic type."""
    if section_title == "申请材料" and parse_heading_order(node_title) is not None:
        return "material_item"
    normalized_node_title = strip_heading_order(node_title)
    if section_title == "办理流程" and normalized_node_title in {"受理", "审查", "决定", "办结", "送达"}:
        return "process_step"
    return "section_leaf"


def parse_heading_order(title: str) -> int | None:
    """Parse numeric heading order."""
    match = re.match(r"^\s*(\d+)(?:[\.、．\)]\s*|\s+)", str(title or ""))
    return int(match.group(1)) if match else None


def strip_heading_order(title: str) -> str:
    """Remove leading numeric heading order."""
    return re.sub(r"^\s*\d+(?:[\.、．\)]\s*|\s+)", "", str(title or "")).strip()


def markdown_sections(text: str) -> list[dict[str, Any]]:
    """Parse markdown headings into sections."""
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
    """Return the semantic section title for current markdown stack."""
    for level, title in stack:
        if level == 2:
            return title
    if current_title:
        return current_title
    if stack:
        return stack[-1][1]
    return "Document"


def split_windows(text: str, size: int, overlap: int) -> list[str]:
    """Split text into overlapping windows."""
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
    """Split text into line-sized units."""
    units: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        units.extend(split_long_line(line.rstrip(), size))
    return units


def split_long_line(line: str, size: int) -> list[str]:
    """Split an overlong line at a soft boundary."""
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
    """Find a soft split near the desired size."""
    search_start = max(1, size // 2)
    candidates = [text.rfind(separator, search_start, size) for separator in ["。", "；", "，", "、", " ", "\t"]]
    split_at = max(candidates)
    return size if split_at <= 0 else split_at + 1


def trailing_overlap_units(units: list[str], overlap: int) -> list[str]:
    """Return units needed to preserve trailing overlap."""
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
    """Join window units."""
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
    """Create a knowledge chunk from a document."""
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


def embed_chunk_batch(embedding_provider: Any, texts: list[str], *, batch_size: int = 10) -> list[list[float]]:
    """Create embeddings in batches."""
    vectors: list[list[float]] = []
    for index in range(0, len(texts), batch_size):
        vectors.extend(embedding_provider.embed_documents(texts[index : index + batch_size]))
    return vectors


def dense_retrieve(
    query: str,
    candidates: list[KnowledgeChunk],
    vectors: dict[str, list[float]],
    *,
    limit: int,
    embedding_provider: Any,
) -> list[RetrievedSource]:
    """Run local dense retrieval over in-memory vectors."""
    query_vector = embedding_provider.embed_query(query)
    hits = []
    for chunk in candidates:
        vector = vectors.get(chunk.embedding_id)
        if not vector:
            continue
        score = cosine(query_vector, vector)
        if score > 0:
            hits.append(RetrievedSource(chunk=chunk, score=score, dense_score=score, retrieval_channel="dense"))
    return sorted(hits, key=lambda item: item.score, reverse=True)[:limit]


def bm25_retrieve(query: str, candidates: list[KnowledgeChunk], *, limit: int) -> list[RetrievedSource]:
    """Run local BM25 keyword retrieval."""
    query_tokens = tokenize(query)
    corpus = [tokenize(chunk.content) for chunk in candidates]
    if not query_tokens:
        return []
    scores = bm25_scores(query_tokens, corpus)
    max_score = max(scores or [0.0]) or 1.0
    hits = []
    for chunk, score in zip(candidates, scores):
        normalized_score = float(score) / max_score if score > 0 else 0.0
        if normalized_score <= 0:
            continue
        hits.append(RetrievedSource(chunk=chunk, score=normalized_score, bm25_score=normalized_score, retrieval_channel="bm25"))
    return sorted(hits, key=lambda item: item.score, reverse=True)[:limit]


def bm25_scores(query_tokens: list[str], corpus: list[list[str]]) -> list[float]:
    """Calculate BM25 scores with lexical fallback."""
    try:
        from rank_bm25 import BM25Okapi

        scores = [float(score) for score in BM25Okapi(corpus).get_scores(query_tokens)]
        if any(score > 0 for score in scores):
            return scores
    except Exception:
        pass
    query_set = set(query_tokens)
    return [float(len(query_set.intersection(tokens))) for tokens in corpus]


def rrf_fuse(hit_lists: list[list[RetrievedSource]], *, rrf_k: int) -> list[RetrievedSource]:
    """Fuse retrieval channels with reciprocal rank fusion."""
    chunk_scores: dict[str, float] = defaultdict(float)
    best_source: dict[str, RetrievedSource] = {}
    dense_scores: dict[str, float] = defaultdict(float)
    bm25_scores_by_chunk: dict[str, float] = defaultdict(float)
    channels: dict[str, set[str]] = defaultdict(set)
    for hit_list in hit_lists:
        for rank, hit in enumerate(hit_list, start=1):
            chunk_id = hit.chunk.id
            chunk_scores[chunk_id] += 1.0 / (rrf_k + rank)
            best_source.setdefault(chunk_id, hit)
            dense_scores[chunk_id] = max(dense_scores[chunk_id], hit.dense_score)
            bm25_scores_by_chunk[chunk_id] = max(bm25_scores_by_chunk[chunk_id], hit.bm25_score)
            channels[chunk_id].add(hit.retrieval_channel)
    fused = []
    for chunk_id, score in chunk_scores.items():
        source = best_source[chunk_id]
        fused.append(
            RetrievedSource(
                chunk=source.chunk,
                score=score,
                dense_score=dense_scores[chunk_id],
                bm25_score=bm25_scores_by_chunk[chunk_id],
                retrieval_channel="+".join(sorted(channels[chunk_id])),
            )
        )
    return sorted(fused, key=lambda item: item.score, reverse=True)


def metadata_match_boost(chunk: KnowledgeChunk, filters: dict[str, Any]) -> float:
    """Boost rerank score when metadata matches filters."""
    if not filters:
        return 0.0
    metadata = chunk.metadata or {}
    matched = sum(
        1
        for key, value in filters.items()
        if key not in {"as_of", "rollout_key"} and metadata_value_matches(chunk_filter_value(chunk, metadata, key), value)
    )
    return min(0.2, matched * 0.05)


def matches_filter(chunk: KnowledgeChunk, filters: dict[str, Any]) -> bool:
    """Return whether a chunk satisfies metadata filters."""
    if not filters:
        return True
    metadata = chunk.metadata or {}
    for key, value in filters.items():
        if key in {"as_of", "rollout_key"}:
            continue
        if not metadata_value_matches(chunk_filter_value(chunk, metadata, key), value):
            return False
    return True


def chunk_filter_value(chunk: KnowledgeChunk, metadata: dict[str, Any], key: str) -> Any:
    """Return the value used for metadata filter matching."""
    if key == "status":
        return chunk.status
    if key == "doc_status":
        return metadata.get("doc_status") or chunk.status
    if key == "audit_node":
        return chunk.audit_node or metadata.get("audit_node")
    return metadata.get(key)


def metadata_value_matches(candidate: Any, expected: Any) -> bool:
    """Match metadata values with list support."""
    if isinstance(expected, list):
        return any(metadata_value_matches(candidate, item) for item in expected)
    if isinstance(candidate, list):
        return any(metadata_value_matches(item, expected) for item in candidate)
    return normalize_text(candidate) == normalize_text(expected)


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
) -> tuple[list[RetrievedSource], dict[str, Any]]:
    """Backfill structured section siblings."""
    target_chunk_type = structured_backfill_chunk_type(query)
    if not target_chunk_type:
        return sources, {"applied": False}
    expanded: list[RetrievedSource] = []
    seen_chunk_ids: set[str] = set()
    seen_node_keys: set[tuple[str, str, str]] = set()
    backfilled_sections: set[str] = set()
    added_count = 0
    for source in sources:
        parent_section_id = str(source.chunk.metadata.get("parent_section_id") or "")
        if parent_section_id and parent_section_id not in backfilled_sections and structured_source_matches(source, target_chunk_type):
            siblings = structured_section_sibling_sources(source, chunks, parent_section_id, target_chunk_type)
            for sibling in siblings:
                if append_unique_source(expanded, sibling, seen_chunk_ids, seen_node_keys):
                    added_count += 1
            backfilled_sections.add(parent_section_id)
        append_unique_source(expanded, source, seen_chunk_ids, seen_node_keys)
    return expanded, {
        "applied": bool(backfilled_sections),
        "target_chunk_type": target_chunk_type,
        "section_count": len(backfilled_sections),
        "added_source_count": added_count,
    }


def structured_backfill_chunk_type(query: str) -> str:
    """Infer structured chunk type to backfill."""
    text = str(query or "")
    if any(keyword in text for keyword in ["材料", "申请材料", "要交啥", "要准备什么", "提交什么", "上传什么", "需要哪些材料", "要带", "需要什么"]):
        return "material_item"
    if any(keyword in text for keyword in ["流程", "怎么办", "怎么申请", "办理步骤", "提交后", "审核", "如何办理"]):
        return "process_step"
    return ""


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
