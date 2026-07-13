"""Structure-aware Chonkie adapter for parsed knowledge documents."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from html import unescape
from typing import Any

class ChonkieChunkingError(RuntimeError):
    """Raised when Chonkie cannot produce retrieval chunks."""


@dataclass(frozen=True)
class StructuredElement:
    """One normalized parser element with source provenance."""

    source_index: int
    element_type: str
    text: str
    heading_level: int = 0
    page_idx: int | None = None
    bbox: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _ElementSpan:
    start: int
    end: int
    element: StructuredElement


@dataclass(frozen=True)
class _TextSlice:
    text: str
    start: int
    end: int


def normalize_extracted_text(text: str) -> str:
    """Normalize parser text without importing the parser package initializer."""
    normalized = unescape(str(text or ""))
    normalized = re.sub(r"[ \t\r\f\v]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def build_chonkie_chunk_specs(
    elements: list[StructuredElement],
    *,
    text_max_chars: int = 900,
    text_min_chars: int = 120,
    table_max_chars: int = 3500,
) -> list[dict[str, Any]]:
    """Build final retrieval chunks without synthetic parent, summary, or KV chunks."""
    text_max_chars = max(100, int(text_max_chars))
    text_min_chars = max(1, min(int(text_min_chars), text_max_chars))
    table_max_chars = max(200, int(table_max_chars))
    normalized_elements = _dedupe_elements(elements)
    specs: list[dict[str, Any]] = []
    seen_chunks: set[tuple[str, str]] = set()
    heading_stack: list[tuple[int, str]] = []
    text_buffer: list[StructuredElement] = []
    heading_elements: list[StructuredElement] = []

    def heading_path() -> str:
        return " > ".join(title for _, title in heading_stack)

    def append_spec(
        *,
        kind: str,
        body: str,
        source_elements: list[StructuredElement],
        sub_index: int,
        extra_metadata: dict[str, Any] | None = None,
    ) -> None:
        normalized_body = normalize_extracted_text(body)
        if not normalized_body:
            return
        path = heading_path()
        content = _contextualize_content(normalized_body, heading_path=path)
        dedupe_key = (kind, " ".join(content.split()))
        if dedupe_key in seen_chunks:
            return
        seen_chunks.add(dedupe_key)
        metadata = _chunk_metadata(
            kind=kind,
            heading_path=path,
            source_elements=source_elements,
            extra_metadata=extra_metadata,
        )
        source_start = min((item.source_index for item in source_elements), default=0)
        source_end = max((item.source_index for item in source_elements), default=source_start)
        digest = hashlib.sha1(f"{kind}\n{path}\n{normalized_body}".encode("utf-8")).hexdigest()[:12]
        specs.append(
            {
                "id_suffix": f"chonkie:{kind}:{source_start}-{source_end}:{sub_index}:{digest}",
                "chunk_type": "child",
                "content": content,
                "metadata": metadata,
            }
        )

    def flush_text_buffer() -> None:
        nonlocal text_buffer
        if not text_buffer:
            return
        combined, spans = _join_elements(text_buffer)
        slices = _recursive_slices(
            combined,
            max_chars=text_max_chars,
            min_chars=text_min_chars,
        )
        for slice_index, text_slice in enumerate(slices):
            source_elements = _elements_for_slice(spans, text_slice.start, text_slice.end) or list(text_buffer)
            append_spec(
                kind=_text_chunk_kind(source_elements),
                body=text_slice.text,
                source_elements=source_elements,
                sub_index=slice_index,
            )
        text_buffer = []

    for element in normalized_elements:
        element_type = str(element.element_type or "text").lower()
        if element.heading_level > 0 or element_type == "title":
            flush_text_buffer()
            level = max(1, int(element.heading_level or 1))
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, element.text.strip()))
            heading_elements.append(element)
            continue
        if element_type == "table":
            flush_text_buffer()
            table_source = str(element.metadata.get("table_body") or element.text).strip()
            caption = str(element.metadata.get("table_caption") or "").strip()
            table_slices = _table_slices(table_source, max_chars=table_max_chars)
            for table_index, table_slice in enumerate(table_slices):
                body = "\n\n".join(part for part in [caption, table_slice] if part)
                append_spec(
                    kind="table",
                    body=body,
                    source_elements=[element],
                    sub_index=table_index,
                    extra_metadata={
                        "table_source": str(element.metadata.get("table_source") or "structured"),
                        "table_caption": caption,
                        "table_html": table_slice if table_slice.lstrip().lower().startswith("<table") else "",
                        "table_split_index": table_index,
                        "table_split_count": len(table_slices),
                    },
                )
            continue
        if element_type in {"image", "chart"}:
            flush_text_buffer()
            append_spec(
                kind=element_type,
                body=element.text,
                source_elements=[element],
                sub_index=0,
            )
            continue
        text_buffer.append(element)

    flush_text_buffer()
    if not specs and heading_elements:
        last_heading = heading_elements[-1]
        append_spec(
            kind="text",
            body=last_heading.text,
            source_elements=[last_heading],
            sub_index=0,
        )
    return specs


def build_markdown_chunk_specs(
    markdown: str,
    *,
    text_max_chars: int = 900,
    text_min_chars: int = 120,
    table_max_chars: int = 3500,
) -> list[dict[str, Any]]:
    """Parse Markdown structure and build Chonkie retrieval chunks."""
    return build_chonkie_chunk_specs(
        markdown_elements(markdown),
        text_max_chars=text_max_chars,
        text_min_chars=text_min_chars,
        table_max_chars=table_max_chars,
    )


def build_fallback_chunk_specs(text: str) -> list[dict[str, Any]]:
    """Build one final child chunk when Chonkie is unexpectedly unavailable."""
    content = normalize_extracted_text(text)
    if not content:
        return []
    digest = hashlib.sha1(content.encode("utf-8")).hexdigest()[:12]
    return [
        {
            "id_suffix": f"chonkie:fallback:text:{digest}",
            "chunk_type": "child",
            "content": content,
            "metadata": {
                "chunk_type": "chonkie_fallback_text",
                "chunking_strategy": "chonkie",
                "chonkie_chunk_kind": "text",
                "indexable": True,
            },
        }
    ]


def markdown_elements(markdown: str) -> list[StructuredElement]:
    """Convert Markdown headings, paragraphs, and pipe tables into elements."""
    lines = str(markdown or "").splitlines()
    elements: list[StructuredElement] = []
    paragraph_lines: list[str] = []

    def append_element(element_type: str, text: str, *, heading_level: int = 0, metadata: dict[str, Any] | None = None) -> None:
        normalized = normalize_extracted_text(text)
        if not normalized:
            return
        elements.append(
            StructuredElement(
                source_index=len(elements),
                element_type=element_type,
                text=normalized,
                heading_level=heading_level,
                metadata=dict(metadata or {}),
            )
        )

    def flush_paragraph() -> None:
        nonlocal paragraph_lines
        append_element("text", "\n".join(paragraph_lines))
        paragraph_lines = []

    index = 0
    heading_pattern = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
    while index < len(lines):
        line = lines[index]
        heading = heading_pattern.match(line.strip())
        if heading:
            flush_paragraph()
            append_element("title", heading.group(2), heading_level=len(heading.group(1)))
            index += 1
            continue
        if _looks_like_markdown_table(lines, index):
            flush_paragraph()
            table_lines = [line]
            index += 1
            while index < len(lines) and "|" in lines[index] and lines[index].strip():
                table_lines.append(lines[index])
                index += 1
            table_text = "\n".join(table_lines)
            append_element(
                "table",
                table_text,
                metadata={"table_body": table_text, "table_source": "markdown"},
            )
            continue
        if not line.strip():
            flush_paragraph()
        else:
            paragraph_lines.append(line)
        index += 1
    flush_paragraph()
    return elements


def _recursive_slices(text: str, *, max_chars: int, min_chars: int) -> list[_TextSlice]:
    if len(text) <= max_chars:
        return [_TextSlice(text=text, start=0, end=len(text))]
    try:
        from chonkie import RecursiveChunker, RecursiveLevel, RecursiveRules

        rules = RecursiveRules(
            levels=[
                RecursiveLevel(delimiters=["\n\n", "\r\n\r\n"], include_delim="prev"),
                RecursiveLevel(delimiters=["\n", "\r\n"], include_delim="prev"),
                RecursiveLevel(delimiters=["。", "！", "？", "；", ". ", "! ", "? ", "; "], include_delim="prev"),
                RecursiveLevel(delimiters=["，", "、", "：", ", ", ": "], include_delim="prev"),
                RecursiveLevel(whitespace=True),
                RecursiveLevel(),
            ]
        )
        chunks = RecursiveChunker(
            tokenizer="character",
            chunk_size=max_chars,
            rules=rules,
            min_characters_per_chunk=min_chars,
        ).chunk(text)
    except Exception as exc:
        raise ChonkieChunkingError(f"Chonkie recursive chunking failed: {type(exc).__name__}: {str(exc)[:240]}") from exc
    slices = [
        _TextSlice(
            text=normalize_extracted_text(getattr(chunk, "text", "")),
            start=max(0, int(getattr(chunk, "start_index", 0) or 0)),
            end=max(0, int(getattr(chunk, "end_index", 0) or 0)),
        )
        for chunk in chunks
        if str(getattr(chunk, "text", "") or "").strip()
    ]
    if not slices:
        raise ChonkieChunkingError("Chonkie recursive chunking returned no content")
    return _merge_short_slices(slices, max_chars=max_chars, min_chars=min_chars)


def _table_slices(table: str, *, max_chars: int) -> list[str]:
    normalized = str(table or "").strip()
    if not normalized or len(normalized) <= max_chars:
        return [normalized] if normalized else []
    try:
        from chonkie import TableChunker

        chunks = TableChunker(tokenizer="character", chunk_size=max_chars).chunk(normalized)
        output = [normalize_extracted_text(getattr(chunk, "text", "")) for chunk in chunks]
        output = [item for item in output if item]
        return output or [normalized]
    except Exception:
        return [normalized]


def _merge_short_slices(slices: list[_TextSlice], *, max_chars: int, min_chars: int) -> list[_TextSlice]:
    output: list[_TextSlice] = []
    for item in slices:
        if output and (len(item.text) < min_chars or len(output[-1].text) < min_chars):
            combined = normalize_extracted_text(f"{output[-1].text}\n{item.text}")
            if len(combined) <= max_chars:
                previous = output.pop()
                output.append(_TextSlice(text=combined, start=previous.start, end=max(previous.end, item.end)))
                continue
        output.append(item)
    return output


def _join_elements(elements: list[StructuredElement]) -> tuple[str, list[_ElementSpan]]:
    parts: list[str] = []
    spans: list[_ElementSpan] = []
    cursor = 0
    for element in elements:
        text = normalize_extracted_text(element.text)
        if not text:
            continue
        if parts:
            parts.append("\n\n")
            cursor += 2
        start = cursor
        parts.append(text)
        cursor += len(text)
        spans.append(_ElementSpan(start=start, end=cursor, element=element))
    return "".join(parts), spans


def _elements_for_slice(spans: list[_ElementSpan], start: int, end: int) -> list[StructuredElement]:
    return [span.element for span in spans if span.end > start and span.start < end]


def _chunk_metadata(
    *,
    kind: str,
    heading_path: str,
    source_elements: list[StructuredElement],
    extra_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    source_indexes = sorted({item.source_index for item in source_elements})
    pages = sorted({item.page_idx for item in source_elements if item.page_idx is not None})
    section_name = heading_path or "Document"
    section_digest = hashlib.sha1(section_name.encode("utf-8")).hexdigest()[:12]
    metadata: dict[str, Any] = {
        "chunk_type": f"chonkie_{kind}",
        "chunking_strategy": "chonkie",
        "chonkie_chunk_kind": kind,
        "heading_path": heading_path,
        "parent_section_id": f"section:{section_digest}",
        "source_element_indexes": source_indexes,
        "source_element_start": source_indexes[0] if source_indexes else 0,
        "source_element_end": source_indexes[-1] if source_indexes else 0,
        "indexable": True,
    }
    source_types = sorted({str(item.element_type or "text") for item in source_elements})
    metadata["source_element_types"] = source_types
    metadata["mineru_item_type"] = "table" if kind == "table" else (source_types[0] if len(source_types) == 1 else "text")
    if pages:
        metadata.update(
            {
                "page_idx": pages[0],
                "page_no": pages[0] + 1,
                "page_start": pages[0] + 1,
                "page_end": pages[-1] + 1,
                "page_numbers": [page + 1 for page in pages],
            }
        )
    if len(source_elements) == 1 and source_elements[0].bbox is not None:
        metadata["bbox"] = list(source_elements[0].bbox or [])
    metadata.update(dict(extra_metadata or {}))
    return metadata


def _text_chunk_kind(elements: list[StructuredElement]) -> str:
    types = {str(item.element_type or "text").lower() for item in elements}
    if types == {"list"}:
        return "list"
    if types == {"code"}:
        return "code"
    if types == {"equation"}:
        return "equation"
    return "text"


def _contextualize_content(body: str, *, heading_path: str) -> str:
    if not heading_path or body.startswith(heading_path):
        return body
    return f"{heading_path}\n\n{body}".strip()


def _dedupe_elements(elements: list[StructuredElement]) -> list[StructuredElement]:
    output: list[StructuredElement] = []
    seen: set[tuple[str, int | None, tuple[float, ...], str]] = set()
    for element in elements:
        text = normalize_extracted_text(element.text)
        if not text:
            continue
        key = (
            str(element.element_type or "text").lower(),
            element.page_idx,
            tuple(float(value) for value in (element.bbox or [])),
            " ".join(text.split()),
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(
            StructuredElement(
                source_index=element.source_index,
                element_type=element.element_type,
                text=text,
                heading_level=element.heading_level,
                page_idx=element.page_idx,
                bbox=element.bbox,
                metadata=dict(element.metadata or {}),
            )
        )
    return output


def _looks_like_markdown_table(lines: list[str], index: int) -> bool:
    if index + 1 >= len(lines) or "|" not in lines[index]:
        return False
    separator = lines[index + 1].strip().strip("|")
    cells = [cell.strip() for cell in separator.split("|")]
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell or "") for cell in cells)
