"""MinerU 文档解析适配器。"""

from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
from pathlib import Path
from typing import Any

import requests

from declaration_qa.indexing.chunking import (
    ChonkieChunkingError,
    StructuredElement,
    build_chonkie_chunk_specs,
)
from declaration_qa.indexing.document_parsing.models import (
    DocumentParseError,
    ParsedBlock,
    ParsedDocument,
)
from declaration_qa.indexing.document_parsing.native import parse_metadata
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
SUPPORTED_MINERU_CONTENT_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "image/bmp",
    "image/jpeg",
    "image/png",
    "image/tiff",
    "image/webp",
}


class MinerUDocumentParser:
    """调用 MinerU API，并将其原生内容列表适配为项目模型。"""

    name = "mineru"

    def __init__(
        self,
        *,
        api_url: str = "http://127.0.0.1:8011",
        backend: str = "pipeline",
        method: str = "auto",
        language: str = "ch",
        model_source: str = "modelscope",
        formula_enabled: bool = False,
        timeout_seconds: int = 3600,
        chonkie_text_max_chars: int = 900,
        chonkie_text_min_chars: int = 120,
        chonkie_table_max_chars: int = 3500,
    ) -> None:
        self.api_url = str(api_url or "").strip().rstrip("/")
        self.backend = str(backend or "pipeline").strip() or "pipeline"
        self.method = str(method or "auto").strip() or "auto"
        self.language = str(language or "ch").strip()
        self.model_source = str(model_source or "modelscope").strip()
        self.formula_enabled = bool(formula_enabled)
        self.timeout_seconds = max(1, int(timeout_seconds))
        self.chonkie_text_max_chars = max(100, int(chonkie_text_max_chars))
        self.chonkie_text_min_chars = max(1, int(chonkie_text_min_chars))
        self.chonkie_table_max_chars = max(200, int(chonkie_table_max_chars))

    def supports(self, *, filename: str, content_type: str) -> bool:
        """判断 MinerU 是否接受该文件类型。"""
        suffix = Path(filename or "").suffix.lower()
        normalized_content_type = str(content_type or "").partition(";")[0].strip().lower()
        return suffix in SUPPORTED_MINERU_SUFFIXES or normalized_content_type in SUPPORTED_MINERU_CONTENT_TYPES

    def parse(self, *, filename: str, content_type: str, data: bytes) -> ParsedDocument:
        """解析受支持的文档，并将 MinerU 元素转换为最终 Chonkie 分块。"""
        suffix = Path(filename or "").suffix.lower()
        if not self.supports(filename=filename, content_type=content_type):
            raise DocumentParseError(f"MinerU does not support this input type: {suffix or content_type or 'unknown'}")
        if not self.api_url:
            raise DocumentParseError("MinerU API URL is not configured")
        result = self._request_parse(filename=filename, content_type=content_type, data=data)
        content_items = parse_mineru_content_items(result.get("content_list"))
        chunking_fallback_reason = ""
        try:
            chunk_specs = mineru_chunk_specs(
                content_items,
                text_max_chars=self.chonkie_text_max_chars,
                text_min_chars=self.chonkie_text_min_chars,
                table_max_chars=self.chonkie_table_max_chars,
            )
        except ChonkieChunkingError as exc:
            chunk_specs = mineru_native_chunk_specs(content_items)
            chunking_fallback_reason = str(exc)
        markdown = str(result.get("md_content") or "")
        if not markdown.strip():
            markdown = "\n\n".join(str(spec.get("content") or "") for spec in chunk_specs).strip()
        if not markdown.strip():
            raise DocumentParseError("MinerU returned no readable Markdown or content blocks")
        images = result.get("images") if isinstance(result.get("images"), dict) else {}
        blocks = mineru_parsed_blocks(content_items, images=images)

        normalized_markdown = normalize_extracted_text(markdown)
        metadata = {
            **parse_metadata(self.name, blocks),
            "parser_output_format": "markdown+content_list",
            "mineru_backend": self.backend,
            "mineru_method": self.method,
            "mineru_language": self.language,
            "mineru_model_source": self.model_source,
            "mineru_api_url": self.api_url,
            "chonkie_chunk_count": len(chunk_specs),
            "chonkie_chunk_specs": chunk_specs,
            "chonkie_chunking_fallback_reason": chunking_fallback_reason,
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

    def _request_parse(self, *, filename: str, content_type: str, data: bytes) -> dict[str, Any]:
        form_data: list[tuple[str, str]] = [
            ("backend", self.backend),
            ("parse_method", self.method),
            ("formula_enable", str(self.formula_enabled).lower()),
            ("table_enable", "true"),
            ("return_md", "true"),
            ("return_content_list", "true"),
            ("return_images", "true"),
        ]
        if self.language and self.backend == "pipeline":
            form_data.append(("lang_list", self.language))
        try:
            response = requests.post(
                f"{self.api_url}/file_parse",
                files=[("files", (filename, data, content_type or "application/octet-stream"))],
                data=form_data,
                timeout=(10, self.timeout_seconds),
            )
            response.raise_for_status()
        except requests.Timeout as exc:
            raise DocumentParseError(f"MinerU parse timed out after {self.timeout_seconds} seconds") from exc
        except requests.RequestException as exc:
            raise DocumentParseError(
                f"MinerU API request failed at {self.api_url}: {type(exc).__name__}: {str(exc)[:300]}"
            ) from exc
        try:
            payload = response.json()
        except ValueError as exc:
            raise DocumentParseError("MinerU API returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise DocumentParseError("MinerU API returned an unexpected response")
        if payload.get("error"):
            raise DocumentParseError(f"MinerU parse failed: {str(payload['error'])[:500]}")
        results = payload.get("results")
        if not isinstance(results, dict) or not results:
            raise DocumentParseError("MinerU API response contains no document result")
        result = next(iter(results.values()))
        if not isinstance(result, dict):
            raise DocumentParseError("MinerU API document result is invalid")
        return result


def parse_mineru_content_items(value: Any) -> list[dict[str, Any]]:
    """将任一 MinerU 内容列表格式解析为扁平序列。"""
    payload = value
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise DocumentParseError(f"MinerU content list is invalid: {str(exc)[:240]}") from exc
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


def mineru_chunk_specs(
    content_items: list[dict[str, Any]],
    *,
    text_max_chars: int = 900,
    text_min_chars: int = 120,
    table_max_chars: int = 3500,
) -> list[dict[str, Any]]:
    """将 MinerU 内容列表元素转换为最终 Chonkie 分块。"""
    elements: list[StructuredElement] = []
    for item_index, item in enumerate(content_items):
        item_type = mineru_item_type(item)
        text = mineru_item_text(item)
        if not text:
            continue
        text_level = mineru_text_level(item)
        is_heading = item_type == "title" or text_level > 0
        metadata: dict[str, Any] = {
            "mineru_item_index": item_index,
            "mineru_item_type": item_type,
            "mineru_sub_type": str(item.get("sub_type") or ""),
            "mineru_text_level": text_level,
        }
        if item_type == "table":
            metadata.update(
                {
                    "table_source": "mineru_native",
                    "table_body": str(item.get("table_body") or text),
                    "table_caption": join_text_values(item.get("table_caption")),
                }
            )
        elements.append(
            StructuredElement(
                source_index=item_index,
                element_type="title" if is_heading else item_type,
                text=text,
                heading_level=max(1, text_level) if is_heading else 0,
                page_idx=integer_or_none(item.get("page_idx")),
                bbox=mineru_bbox(item.get("bbox")),
                metadata=metadata,
            )
        )
    return build_chonkie_chunk_specs(
        elements,
        text_max_chars=text_max_chars,
        text_min_chars=text_min_chars,
        table_max_chars=table_max_chars,
    )


def mineru_native_chunk_specs(content_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Chonkie 无法切分文档时仍保持入库能力可用。"""
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
            "chunk_type": f"chonkie_fallback_{item_type}",
            "chunking_strategy": "chonkie",
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
    images: dict[str, Any],
) -> list[ParsedBlock]:
    """保留 MinerU 文本、表格、公式和图片块，供下游资产处理。"""
    blocks: list[ParsedBlock] = []
    for item_index, item in enumerate(content_items):
        item_type = mineru_item_type(item)
        text = mineru_item_text(item)
        page_idx = integer_or_none(item.get("page_idx"))
        image_name = str(item.get("img_path") or item.get("image_path") or "")
        binary_data = mineru_image_bytes(images, image_name)
        if not text and not binary_data:
            continue
        block_type = {
            "table": "table",
            "image": "picture",
            "chart": "picture",
            "equation": "equation",
        }.get(item_type, "text")
        image_content_type = mimetypes.guess_type(image_name)[0] or ""
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
    """统一 MinerU 第一版和第二版的元素类型。"""
    value = str(item.get("type") or "text").strip().lower()
    aliases = {
        "paragraph": "text",
        "equation_interline": "equation",
        "algorithm": "code",
        "index": "list",
    }
    return aliases.get(value, value or "text")


def mineru_item_text(item: dict[str, Any]) -> str:
    """序列化单个 MinerU 元素，不创建合成摘要块或键值块。"""
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
    """从 MinerU 第二版内容列表元素中提取可读文本。"""
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
    """扁平化 MinerU 字符串、标题数组和文本片段字典。"""
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
    """从任一内容列表格式中读取标题层级。"""
    value = item.get("text_level")
    if value is None and isinstance(item.get("content"), dict):
        value = item["content"].get("level")
    parsed = integer_or_none(value)
    return max(0, parsed or 0)


def mineru_picture_caption(item: dict[str, Any]) -> str:
    """返回图片或图表的标题文本。"""
    return join_text_values(item.get("image_caption") or item.get("chart_caption"))


def mineru_bbox(value: Any) -> list[float] | None:
    """规范化 MinerU 的 0 到 1000 坐标边界框。"""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def integer_or_none(value: Any) -> int | None:
    """解析可选整数值。"""
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def mineru_image_bytes(images: dict[str, Any], image_name: str) -> bytes:
    """解码 MinerU API 内联返回的图片。"""
    normalized_name = str(image_name or "").strip().replace("\\", "/")
    if not normalized_name or not images:
        return b""
    normalized_images = {str(key).replace("\\", "/"): value for key, value in images.items()}
    value = normalized_images.get(normalized_name)
    if value is None:
        basename = Path(normalized_name).name
        value = next((item for key, item in normalized_images.items() if Path(key).name == basename), None)
    if isinstance(value, bytes):
        return value
    if isinstance(value, dict):
        value = value.get("data") or value.get("base64") or value.get("content")
    if not isinstance(value, str) or not value.strip():
        return b""
    encoded = value.split(",", 1)[1] if value.lstrip().startswith("data:") and "," in value else value
    try:
        return base64.b64decode(encoded)
    except (ValueError, TypeError):
        return b""


def mineru_chunk_suffix(index: int, *, item_type: str, text: str) -> str:
    """构建稳定的原生分块后缀。"""
    digest = hashlib.sha1(f"{item_type}\n{text}".encode("utf-8")).hexdigest()[:12]
    return f"mineru:{index}:{digest}"
