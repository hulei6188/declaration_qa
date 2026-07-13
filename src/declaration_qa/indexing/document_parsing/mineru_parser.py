"""MinerU document parser adapter."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from declaration_qa.indexing.document_parsing.models import (
    DocumentParseError,
    ParsedBlock,
    ParsedDocument,
)
from declaration_qa.indexing.document_parsing.native import parse_metadata
from declaration_qa.indexing.document_parsing.structured import temp_document_file
from declaration_qa.indexing.document_parsing.text import normalize_extracted_text


SUPPORTED_MINERU_SUFFIXES = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".docx",
    ".pptx",
    ".xlsx",
}


class MinerUDocumentParser:
    """Run MinerU CLI and adapt its native content list to the project model."""

    name = "mineru"

    def __init__(
        self,
        *,
        cli_path: str = "mineru",
        backend: str = "pipeline",
        method: str = "auto",
        language: str = "ch",
        model_source: str = "modelscope",
        timeout_seconds: int = 3600,
    ) -> None:
        self.cli_path = str(cli_path or "mineru").strip() or "mineru"
        self.backend = str(backend or "pipeline").strip() or "pipeline"
        self.method = str(method or "auto").strip() or "auto"
        self.language = str(language or "ch").strip()
        self.model_source = str(model_source or "modelscope").strip()
        self.timeout_seconds = max(1, int(timeout_seconds))

    def parse(self, *, filename: str, content_type: str, data: bytes) -> ParsedDocument:
        """Parse a supported document and preserve MinerU content blocks one-to-one."""
        suffix = Path(filename or "").suffix.lower()
        if suffix not in SUPPORTED_MINERU_SUFFIXES:
            raise DocumentParseError(f"MinerU does not support this input type: {suffix or content_type or 'unknown'}")
        executable = shutil.which(self.cli_path)
        if executable is None and not Path(self.cli_path).is_file():
            raise DocumentParseError(f"MinerU CLI is not installed or not found: {self.cli_path}")

        with temp_document_file(filename=filename, data=data) as input_path:
            with tempfile.TemporaryDirectory(prefix="declaration-qa-mineru-") as output_value:
                output_path = Path(output_value)
                self._run_cli(executable or self.cli_path, input_path=input_path, output_path=output_path)
                markdown_path = find_mineru_markdown(output_path, input_stem=input_path.stem)
                content_list_path = find_mineru_content_list(output_path, input_stem=input_path.stem)
                content_items = load_mineru_content_items(content_list_path) if content_list_path else []
                chunk_specs = mineru_chunk_specs(content_items)
                markdown = read_text_file(markdown_path) if markdown_path else ""
                if not markdown.strip():
                    markdown = "\n\n".join(str(spec.get("content") or "") for spec in chunk_specs).strip()
                if not markdown.strip():
                    raise DocumentParseError("MinerU returned no readable Markdown or content blocks")
                blocks = mineru_parsed_blocks(
                    content_items,
                    content_list_dir=content_list_path.parent if content_list_path else output_path,
                    output_dir=output_path,
                )

        normalized_markdown = normalize_extracted_text(markdown)
        metadata = {
            **parse_metadata(self.name, blocks),
            "parser_output_format": "markdown+content_list",
            "mineru_backend": self.backend,
            "mineru_method": self.method,
            "mineru_language": self.language,
            "mineru_model_source": self.model_source,
            "mineru_chunk_count": len(chunk_specs),
            "mineru_chunk_specs": chunk_specs,
            "mineru_content_item_count": len(content_items),
        }
        return ParsedDocument(
            filename=filename,
            content_type=(content_type or "").lower(),
            text=normalized_markdown,
            parser_name=self.name,
            blocks=blocks,
            metadata=metadata,
        )

    def _run_cli(self, executable: str, *, input_path: Path, output_path: Path) -> None:
        command = [
            executable,
            "-p",
            str(input_path),
            "-o",
            str(output_path),
            "-b",
            self.backend,
            "-m",
            self.method,
        ]
        if self.language and self.backend == "pipeline":
            command.extend(["-l", self.language])
        environment = os.environ.copy()
        if self.model_source:
            environment["MINERU_MODEL_SOURCE"] = self.model_source
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_seconds,
                env=environment,
            )
        except subprocess.TimeoutExpired as exc:
            raise DocumentParseError(f"MinerU parse timed out after {self.timeout_seconds} seconds") from exc
        except OSError as exc:
            raise DocumentParseError(f"MinerU CLI could not start: {type(exc).__name__}: {str(exc)[:240]}") from exc
        if completed.returncode != 0:
            details = (completed.stderr or completed.stdout or "unknown error").strip()
            raise DocumentParseError(f"MinerU parse failed with exit code {completed.returncode}: {details[-500:]}")


def find_mineru_markdown(output_dir: Path, *, input_stem: str) -> Path | None:
    """Find the primary Markdown generated by MinerU."""
    paths = [path for path in output_dir.rglob("*.md") if path.is_file()]
    return preferred_output_path(paths, input_stem=input_stem)


def find_mineru_content_list(output_dir: Path, *, input_stem: str) -> Path | None:
    """Find the stable v1 content list, with v2 as a compatibility fallback."""
    v1_paths = [path for path in output_dir.rglob("*_content_list.json") if path.is_file()]
    if v1_paths:
        return preferred_output_path(v1_paths, input_stem=input_stem)
    v2_paths = [path for path in output_dir.rglob("*_content_list_v2.json") if path.is_file()]
    return preferred_output_path(v2_paths, input_stem=input_stem)


def preferred_output_path(paths: list[Path], *, input_stem: str) -> Path | None:
    """Prefer an output whose filename starts with the temporary input stem."""
    if not paths:
        return None
    normalized_stem = input_stem.lower()
    return sorted(
        paths,
        key=lambda path: (not path.stem.lower().startswith(normalized_stem), len(path.parts), str(path).lower()),
    )[0]


def read_text_file(path: Path) -> str:
    """Read MinerU text output with a tolerant UTF-8 decoder."""
    return path.read_text(encoding="utf-8", errors="replace")


def load_mineru_content_items(path: Path) -> list[dict[str, Any]]:
    """Load either MinerU content list format into a flat sequence."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DocumentParseError(f"MinerU content list is invalid: {type(exc).__name__}: {str(exc)[:240]}") from exc
    if isinstance(payload, dict):
        payload = payload.get("content_list") or payload.get("data") or []
    if not isinstance(payload, list):
        return []
    if payload and all(isinstance(page, list) for page in payload):
        output: list[dict[str, Any]] = []
        for page_index, page in enumerate(payload):
            for item in page:
                if isinstance(item, dict):
                    output.append({"page_idx": page_index, **item})
        return output
    return [item for item in payload if isinstance(item, dict)]


def mineru_chunk_specs(content_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert each unique readable MinerU content item into exactly one chunk spec."""
    specs: list[dict[str, Any]] = []
    heading_stack: list[tuple[int, str]] = []
    seen: set[tuple[str, int | None, tuple[float, ...], str]] = set()
    for item_index, item in enumerate(content_items):
        item_type = mineru_item_type(item)
        text = mineru_item_text(item)
        if not text:
            continue
        text_level = mineru_text_level(item)
        if item_type in {"text", "title"} and text_level > 0:
            while heading_stack and heading_stack[-1][0] >= text_level:
                heading_stack.pop()
            heading_stack.append((text_level, text))
        page_idx = integer_or_none(item.get("page_idx"))
        bbox = mineru_bbox(item.get("bbox"))
        dedupe_key = (item_type, page_idx, tuple(bbox or []), " ".join(text.split()))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        metadata: dict[str, Any] = {
            "chunk_type": f"mineru_{item_type}",
            "chunking_strategy": "mineru",
            "mineru_chunk_index": item_index,
            "mineru_item_type": item_type,
            "mineru_sub_type": str(item.get("sub_type") or ""),
            "mineru_text_level": text_level,
            "heading_path": " > ".join(title for _, title in heading_stack),
            "indexable": True,
        }
        if page_idx is not None:
            metadata.update({"page_idx": page_idx, "page_no": page_idx + 1})
        if bbox is not None:
            metadata["bbox"] = bbox
        if item_type == "table":
            metadata.update(
                {
                    "table_source": "mineru_native",
                    "table_html": str(item.get("table_body") or ""),
                    "table_caption": join_text_values(item.get("table_caption")),
                }
            )
        specs.append(
            {
                "id_suffix": mineru_chunk_suffix(item_index, item_type=item_type, text=text),
                "chunk_type": "child",
                "content": text,
                "metadata": metadata,
            }
        )
    return specs


def mineru_parsed_blocks(
    content_items: list[dict[str, Any]],
    *,
    content_list_dir: Path,
    output_dir: Path,
) -> list[ParsedBlock]:
    """Preserve MinerU text, table, equation and image blocks for downstream assets."""
    blocks: list[ParsedBlock] = []
    for item_index, item in enumerate(content_items):
        item_type = mineru_item_type(item)
        text = mineru_item_text(item)
        page_idx = integer_or_none(item.get("page_idx"))
        image_path = resolve_mineru_image_path(
            str(item.get("img_path") or item.get("image_path") or ""),
            content_list_dir=content_list_dir,
            output_dir=output_dir,
        )
        binary_data = b""
        if image_path is not None:
            try:
                binary_data = image_path.read_bytes()
            except OSError:
                binary_data = b""
        if not text and not binary_data:
            continue
        block_type = {
            "table": "table",
            "image": "picture",
            "chart": "picture",
            "equation": "equation",
        }.get(item_type, "text")
        image_content_type = mimetypes.guess_type(str(image_path or ""))[0] or ""
        blocks.append(
            ParsedBlock(
                block_type=block_type,
                text=text,
                html=str(item.get("table_body") or "") if item_type == "table" else "",
                json_value=item,
                page_no=(page_idx + 1) if page_idx is not None else None,
                bbox=mineru_bbox(item.get("bbox")),
                binary_data=binary_data,
                content_type=image_content_type,
                metadata={
                    "mineru_item_index": item_index,
                    "mineru_item_type": item_type,
                    "mineru_sub_type": str(item.get("sub_type") or ""),
                    "mineru_image_path": str(item.get("img_path") or item.get("image_path") or ""),
                    "picture_caption": mineru_picture_caption(item),
                    "table_source": "mineru_native" if item_type == "table" else "",
                    "table_index": item_index if item_type == "table" else None,
                },
            )
        )
    return blocks


def mineru_item_type(item: dict[str, Any]) -> str:
    """Normalize v1 and v2 MinerU item types."""
    value = str(item.get("type") or "text").strip().lower()
    aliases = {
        "paragraph": "text",
        "equation_interline": "equation",
        "algorithm": "code",
        "index": "list",
    }
    return aliases.get(value, value or "text")


def mineru_item_text(item: dict[str, Any]) -> str:
    """Serialize one MinerU item without creating synthetic summary or KV chunks."""
    item_type = mineru_item_type(item)
    content = item.get("content")
    if isinstance(content, dict):
        content = mineru_v2_content_text(item_type, content)
    candidates: list[Any]
    if item_type == "table":
        candidates = [item.get("table_caption"), item.get("table_body"), item.get("table_footnote")]
    elif item_type in {"image", "chart"}:
        candidates = [
            item.get("image_caption") or item.get("chart_caption"),
            content,
            item.get("image_footnote") or item.get("chart_footnote"),
        ]
    elif item_type == "code":
        candidates = [item.get("code_caption"), item.get("code_body"), item.get("code_footnote")]
    elif item_type == "list":
        candidates = [item.get("list_items"), item.get("text"), content]
    else:
        candidates = [item.get("text"), content, item.get("math_content")]
    return normalize_extracted_text("\n".join(value for value in (join_text_values(item) for item in candidates) if value))


def mineru_v2_content_text(item_type: str, content: dict[str, Any]) -> str:
    """Extract readable text from MinerU content_list_v2 entries."""
    keys_by_type = {
        "title": ["title_content"],
        "text": ["paragraph_content", "text"],
        "equation": ["math_content"],
        "table": ["table_caption", "table_body", "table_footnote"],
        "image": ["image_caption", "image_content", "image_footnote"],
        "chart": ["chart_caption", "chart_content", "chart_footnote"],
        "code": ["code_caption", "code_content", "algorithm_content", "code_footnote"],
        "list": ["list_items"],
    }
    values = [content.get(key) for key in keys_by_type.get(item_type, [])]
    return "\n".join(value for value in (join_text_values(item) for item in values) if value)


def join_text_values(value: Any) -> str:
    """Flatten MinerU strings, caption arrays and span dictionaries."""
    if value in (None, ""):
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return "\n".join(part for part in (join_text_values(item) for item in value) if part)
    if isinstance(value, dict):
        direct = value.get("content") or value.get("text")
        if direct not in (None, ""):
            return join_text_values(direct)
        return join_text_values(value.get("children"))
    return str(value).strip()


def mineru_text_level(item: dict[str, Any]) -> int:
    """Return heading level from either content list format."""
    value = item.get("text_level")
    if value is None and isinstance(item.get("content"), dict):
        value = item["content"].get("level")
    parsed = integer_or_none(value)
    return max(0, parsed or 0)


def mineru_picture_caption(item: dict[str, Any]) -> str:
    """Return image or chart caption text."""
    return join_text_values(item.get("image_caption") or item.get("chart_caption"))


def mineru_bbox(value: Any) -> list[float] | None:
    """Normalize MinerU's 0-1000 bounding box."""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def integer_or_none(value: Any) -> int | None:
    """Parse an optional integer value."""
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def resolve_mineru_image_path(value: str, *, content_list_dir: Path, output_dir: Path) -> Path | None:
    """Resolve an image path referenced by MinerU output."""
    normalized = str(value or "").strip().replace("/", os.sep)
    if not normalized:
        return None
    candidate = Path(normalized)
    candidates = [candidate] if candidate.is_absolute() else [content_list_dir / candidate, output_dir / candidate]
    for path in candidates:
        if path.is_file():
            return path
    filename = candidate.name
    return next((path for path in output_dir.rglob(filename) if path.is_file()), None)


def mineru_chunk_suffix(index: int, *, item_type: str, text: str) -> str:
    """Build a stable native chunk suffix."""
    digest = hashlib.sha1(f"{item_type}\n{text}".encode("utf-8")).hexdigest()[:12]
    return f"mineru:{index}:{digest}"
