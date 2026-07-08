"""将抽取后的附件转换为派生 chunk 描述。"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from declaration_qa.indexing.attachments.form_fields import FormField, extract_form_fields
from declaration_qa.indexing.attachments.models import ExtractedAttachment
from declaration_qa.rag.engine.chunking import markdown_sections, normalize_chunking_strategy, split_windows
from declaration_qa.rag.engine.constants import CHUNKING_HIERARCHICAL, CHUNKING_PARENT_CHILD


def build_attachment_chunk_specs(
    extracted_items: list[ExtractedAttachment],
    *,
    document_id: str,
    document_metadata: dict[str, Any],
    max_chunk_size: int = 900,
    overlap: int = 120,
) -> list[dict[str, Any]]:
    """构建可交给 RAG 切片层合并的附件 chunk 描述。"""
    specs: list[dict[str, Any]] = []
    chunking_strategy = normalize_chunking_strategy(document_metadata.get("chunking_strategy"))
    for extracted in extracted_items:
        if extracted.status != "success" or not extracted.text.strip():
            continue
        semantic_type = str(extracted.metadata.get("semantic_chunk_type") or "attachment_text")
        if semantic_type == "attachment_form":
            specs.extend(
                build_attachment_form_chunk_specs(
                    extracted,
                    document_metadata=document_metadata,
                    max_chunk_size=max_chunk_size,
                    overlap=overlap,
                )
            )
            specs.extend(build_form_field_chunk_specs(extracted, document_metadata=document_metadata))
            non_table_text = extract_non_table_attachment_text(extracted.text)
            if should_index_attachment_text(non_table_text):
                specs.extend(
                    build_attachment_text_chunk_specs(
                        extracted,
                        document_id=document_id,
                        document_metadata=document_metadata,
                        text=non_table_text,
                        semantic_type="attachment_text",
                        source_attachment_chunk_type=semantic_type,
                        chunking_strategy=chunking_strategy,
                        parent_size=max(900, max_chunk_size),
                        child_size=min(520, max_chunk_size),
                        overlap=overlap,
                    )
                )
            continue
        specs.extend(
            build_attachment_text_chunk_specs(
                extracted,
                document_id=document_id,
                document_metadata=document_metadata,
                text=extracted.text,
                semantic_type=semantic_type,
                source_attachment_chunk_type=semantic_type,
                chunking_strategy=chunking_strategy,
                parent_size=max(900, max_chunk_size),
                child_size=min(520, max_chunk_size),
                overlap=overlap,
            )
        )
    return specs


def build_attachment_form_chunk_specs(
    extracted: ExtractedAttachment,
    *,
    document_metadata: dict[str, Any],
    max_chunk_size: int,
    overlap: int,
) -> list[dict[str, Any]]:
    """构建表单附件整体 chunk 描述。"""
    specs: list[dict[str, Any]] = []
    parts = split_attachment_text(
        extracted.text,
        semantic_type="attachment_form",
        max_chunk_size=max_chunk_size,
        overlap=overlap,
    )
    for part_index, part in enumerate(parts):
        specs.append(
            {
                "id_suffix": attachment_chunk_suffix(extracted, part_index),
                "content": attachment_chunk_content(extracted, part, part_index=part_index, part_count=len(parts)),
                "source_url": extracted.ref.url,
                "metadata": {
                    **attachment_base_metadata(extracted, document_metadata, semantic_type="attachment_form"),
                    "attachment_part_index": part_index,
                    "attachment_part_count": len(parts),
                    "indexable": True,
                    **extracted.metadata,
                },
            }
        )
    return specs


def build_attachment_text_chunk_specs(
    extracted: ExtractedAttachment,
    *,
    document_id: str,
    document_metadata: dict[str, Any],
    text: str,
    semantic_type: str,
    source_attachment_chunk_type: str,
    chunking_strategy: str,
    parent_size: int,
    child_size: int,
    overlap: int,
) -> list[dict[str, Any]]:
    """按主文档切分策略构建附件正文父子/层级 chunk。"""
    normalized = str(text or "").strip()
    if not normalized:
        return []
    if chunking_strategy == CHUNKING_HIERARCHICAL:
        return build_hierarchical_attachment_text_specs(
            extracted,
            document_id=document_id,
            document_metadata=document_metadata,
            text=normalized,
            semantic_type=semantic_type,
            source_attachment_chunk_type=source_attachment_chunk_type,
            parent_size=parent_size,
            child_size=child_size,
            overlap=overlap,
        )
    return build_parent_child_attachment_text_specs(
        extracted,
        document_id=document_id,
        document_metadata=document_metadata,
        text=normalized,
        semantic_type=semantic_type,
        source_attachment_chunk_type=source_attachment_chunk_type,
        parent_size=parent_size,
        child_size=child_size,
        overlap=overlap,
    )


def build_parent_child_attachment_text_specs(
    extracted: ExtractedAttachment,
    *,
    document_id: str,
    document_metadata: dict[str, Any],
    text: str,
    semantic_type: str,
    source_attachment_chunk_type: str,
    parent_size: int,
    child_size: int,
    overlap: int,
) -> list[dict[str, Any]]:
    """将附件正文按父子窗口切分。"""
    specs: list[dict[str, Any]] = []
    parent_texts = split_windows(text, parent_size, max(0, min(overlap, parent_size // 2)))
    for parent_index, parent_text in enumerate(parent_texts):
        parent_suffix = attachment_text_parent_suffix(extracted, section_index=0, parent_index=parent_index)
        parent_chunk_id = f"{document_id}:{parent_suffix}"
        parent_metadata = attachment_text_metadata(
            extracted,
            document_metadata,
            text=parent_text,
            semantic_type=semantic_type,
            source_attachment_chunk_type=source_attachment_chunk_type,
            chunking_strategy=CHUNKING_PARENT_CHILD,
            section_index=0,
            section_title=extracted.ref.node_title or extracted.ref.name,
            section_path=extracted.ref.section_path,
            parent_index=parent_index,
            child_index=None,
            indexable=False,
        )
        specs.append(
            {
                "id_suffix": parent_suffix,
                "parent_id": parent_chunk_id,
                "chunk_type": "parent",
                "content": attachment_chunk_content(extracted, parent_text, part_index=parent_index, part_count=len(parent_texts)),
                "source_url": extracted.ref.url,
                "metadata": parent_metadata,
            }
        )
        child_texts = split_windows(parent_text, child_size, max(0, min(overlap, child_size // 2)))
        for child_index, child_text in enumerate(child_texts):
            specs.append(
                {
                    "id_suffix": attachment_text_child_suffix(
                        extracted,
                        section_index=0,
                        parent_index=parent_index,
                        child_index=child_index,
                    ),
                    "parent_id": parent_chunk_id,
                    "chunk_type": "child",
                    "content": attachment_chunk_content(
                        extracted,
                        child_text,
                        part_index=child_index,
                        part_count=len(child_texts),
                    ),
                    "source_url": extracted.ref.url,
                    "metadata": attachment_text_metadata(
                        extracted,
                        document_metadata,
                        text=child_text,
                        semantic_type=semantic_type,
                        source_attachment_chunk_type=source_attachment_chunk_type,
                        chunking_strategy=CHUNKING_PARENT_CHILD,
                        section_index=0,
                        section_title=extracted.ref.node_title or extracted.ref.name,
                        section_path=extracted.ref.section_path,
                        parent_index=parent_index,
                        child_index=child_index,
                        indexable=True,
                    ),
                }
            )
    return specs


def build_hierarchical_attachment_text_specs(
    extracted: ExtractedAttachment,
    *,
    document_id: str,
    document_metadata: dict[str, Any],
    text: str,
    semantic_type: str,
    source_attachment_chunk_type: str,
    parent_size: int,
    child_size: int,
    overlap: int,
) -> list[dict[str, Any]]:
    """将附件正文按标题层级切分，长章节继续生成父子窗口。"""
    specs: list[dict[str, Any]] = []
    sections = markdown_sections(normalize_attachment_text_headings(text))
    for section_index, section in enumerate(sections):
        section_text = str(section.get("content") or "").strip()
        if not section_text:
            continue
        section_title = attachment_section_title(section, extracted)
        section_path = str(section.get("path") or extracted.ref.section_path or section_title)
        if len(section_text) <= child_size:
            specs.append(
                {
                    "id_suffix": attachment_text_child_suffix(
                        extracted,
                        section_index=section_index,
                        parent_index=0,
                        child_index=0,
                    ),
                    "chunk_type": "child",
                    "content": attachment_chunk_content(extracted, section_text, part_index=0, part_count=1),
                    "source_url": extracted.ref.url,
                    "metadata": attachment_text_metadata(
                        extracted,
                        document_metadata,
                        text=section_text,
                        semantic_type=semantic_type,
                        source_attachment_chunk_type=source_attachment_chunk_type,
                        chunking_strategy=CHUNKING_HIERARCHICAL,
                        section_index=section_index,
                        section_title=section_title,
                        section_path=section_path,
                        parent_index=None,
                        child_index=0,
                        indexable=True,
                    ),
                }
            )
            continue
        parent_texts = split_windows(section_text, parent_size, max(0, min(overlap, parent_size // 2)))
        for parent_index, parent_text in enumerate(parent_texts):
            child_texts = split_windows(parent_text, child_size, max(0, min(overlap, child_size // 2)))
            if len(child_texts) == 1 and child_texts[0].strip() == parent_text.strip():
                specs.append(
                    {
                        "id_suffix": attachment_text_child_suffix(
                            extracted,
                            section_index=section_index,
                            parent_index=parent_index,
                            child_index=0,
                        ),
                        "chunk_type": "child",
                        "content": attachment_chunk_content(extracted, parent_text, part_index=0, part_count=1),
                        "source_url": extracted.ref.url,
                        "metadata": attachment_text_metadata(
                            extracted,
                            document_metadata,
                            text=parent_text,
                            semantic_type=semantic_type,
                            source_attachment_chunk_type=source_attachment_chunk_type,
                            chunking_strategy=CHUNKING_HIERARCHICAL,
                            section_index=section_index,
                            section_title=section_title,
                            section_path=section_path,
                            parent_index=parent_index,
                            child_index=0,
                            indexable=True,
                        ),
                    }
                )
                continue
            parent_suffix = attachment_text_parent_suffix(
                extracted,
                section_index=section_index,
                parent_index=parent_index,
            )
            parent_chunk_id = f"{document_id}:{parent_suffix}"
            specs.append(
                {
                    "id_suffix": parent_suffix,
                    "parent_id": parent_chunk_id,
                    "chunk_type": "parent",
                    "content": attachment_chunk_content(
                        extracted,
                        parent_text,
                        part_index=parent_index,
                        part_count=len(parent_texts),
                    ),
                    "source_url": extracted.ref.url,
                    "metadata": attachment_text_metadata(
                        extracted,
                        document_metadata,
                        text=parent_text,
                        semantic_type=semantic_type,
                        source_attachment_chunk_type=source_attachment_chunk_type,
                        chunking_strategy=CHUNKING_HIERARCHICAL,
                        section_index=section_index,
                        section_title=section_title,
                        section_path=section_path,
                        parent_index=parent_index,
                        child_index=None,
                        indexable=False,
                    ),
                }
            )
            for child_index, child_text in enumerate(child_texts):
                specs.append(
                    {
                        "id_suffix": attachment_text_child_suffix(
                            extracted,
                            section_index=section_index,
                            parent_index=parent_index,
                            child_index=child_index,
                        ),
                        "parent_id": parent_chunk_id,
                        "chunk_type": "child",
                        "content": attachment_chunk_content(
                            extracted,
                            child_text,
                            part_index=child_index,
                            part_count=len(child_texts),
                        ),
                        "source_url": extracted.ref.url,
                        "metadata": attachment_text_metadata(
                            extracted,
                            document_metadata,
                            text=child_text,
                            semantic_type=semantic_type,
                            source_attachment_chunk_type=source_attachment_chunk_type,
                            chunking_strategy=CHUNKING_HIERARCHICAL,
                            section_index=section_index,
                            section_title=section_title,
                            section_path=section_path,
                            parent_index=parent_index,
                            child_index=child_index,
                            indexable=True,
                        ),
                    }
                )
    return specs


def build_form_field_chunk_specs(
    extracted: ExtractedAttachment,
    *,
    document_metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    """从表单附件中构建字段级 chunk 描述。"""
    specs: list[dict[str, Any]] = []
    fields = extract_form_fields(extracted.text)
    for field in fields:
        specs.append(
            {
                "id_suffix": form_field_chunk_suffix(extracted, field),
                "content": form_field_chunk_content(extracted, field),
                "source_url": extracted.ref.url,
                "metadata": {
                    **attachment_base_metadata(extracted, document_metadata, semantic_type="form_field"),
                    "chunk_type": "form_field",
                    "node_title": f"表单字段：{field.field_name}",
                    "node_order": field.index,
                    "form_name": extracted.ref.material_name or extracted.ref.node_title or extracted.ref.name,
                    "field_name": field.field_name,
                    "field_type": field.field_type,
                    "field_options": field.options,
                    "field_multiple": field.multiple,
                    "field_required": field.required,
                    "field_raw_label": field.raw_label,
                    "field_raw_value": field.raw_value,
                    "field_table_index": field.table_index,
                    "field_row_index": field.row_index,
                    "field_column_index": field.column_index,
                    "field_confidence": round(field.confidence, 4),
                    "source_attachment_chunk_type": "attachment_form",
                    "extraction_method": extracted.metadata.get("extraction_method") or "native",
                    "indexable": True,
                },
            }
        )
    return specs


def attachment_base_metadata(
    extracted: ExtractedAttachment,
    document_metadata: dict[str, Any],
    *,
    semantic_type: str,
) -> dict[str, Any]:
    """构建附件 chunk 共享元数据。"""
    return {
        **document_metadata,
        "chunk_type": semantic_type,
        "attachment_index": extracted.ref.index,
        "attachment_name": extracted.ref.name,
        "attachment_type": extracted.ref.extension,
        "attachment_role": extracted.ref.role,
        "attachment_content_type": extracted.content_type,
        "attachment_sha256": extracted.sha256,
        "attachment_cache_path": extracted.cache_path,
        "attachment_status": extracted.status,
        "section_path": extracted.ref.section_path,
        "section_title": extracted.ref.section_title,
        "node_title": extracted.ref.node_title,
        "parent_section_id": extracted.ref.parent_section_id,
        "material_name": extracted.ref.material_name,
    }


def attachment_text_metadata(
    extracted: ExtractedAttachment,
    document_metadata: dict[str, Any],
    *,
    text: str,
    semantic_type: str,
    source_attachment_chunk_type: str,
    chunking_strategy: str,
    section_index: int,
    section_title: str,
    section_path: str,
    parent_index: int | None,
    child_index: int | None,
    indexable: bool,
) -> dict[str, Any]:
    """构建附件正文层级/父子 chunk 元数据。"""
    metadata = {
        **attachment_base_metadata(extracted, document_metadata, semantic_type=semantic_type),
        "source_attachment_chunk_type": source_attachment_chunk_type,
        "attachment_text_chunking_strategy": chunking_strategy,
        "attachment_text_section_index": section_index,
        "attachment_text_section_title": section_title,
        "attachment_text_section_path": section_path,
        "node_title": section_title,
        "node_order": section_index,
        "indexable": indexable,
        **extracted.metadata,
    }
    if parent_index is not None:
        metadata["parent_window_index"] = parent_index
    if child_index is not None:
        metadata["child_window_index"] = child_index
    page_metadata = attachment_page_metadata("\n".join([section_title, section_path, text]))
    if page_metadata:
        metadata.update(page_metadata)
    if not indexable:
        metadata["context_only"] = True
        metadata["context_role"] = "attachment_text_context"
    return metadata


def attachment_page_metadata(text: str) -> dict[str, Any]:
    """从附件正文标题中提取 PDF 页码元数据。"""
    page_numbers = attachment_page_numbers(text)
    if not page_numbers:
        return {}
    metadata: dict[str, Any] = {
        "attachment_page_numbers": page_numbers,
        "attachment_page_start": page_numbers[0],
        "attachment_page_end": page_numbers[-1],
        "attachment_page_range": compact_page_range(page_numbers),
    }
    if len(page_numbers) == 1:
        metadata["attachment_page_no"] = page_numbers[0]
    return metadata


def attachment_page_numbers(text: str) -> list[int]:
    """识别 OCR/PDF 页标题中的页码。"""
    numbers: list[int] = []
    patterns = [
        r"(?:OCR\s*)?PDF\s*第\s*([0-9]+)\s*页",
        r"第\s*([0-9]+)\s*页",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, str(text or ""), flags=re.IGNORECASE):
            page_no = int(match.group(1))
            if page_no > 0 and page_no not in numbers:
                numbers.append(page_no)
    return sorted(numbers)


def compact_page_range(page_numbers: list[int]) -> str:
    """将页码列表压缩成人类可读范围。"""
    numbers = sorted({int(number) for number in page_numbers if int(number) > 0})
    if not numbers:
        return ""
    ranges: list[str] = []
    start = numbers[0]
    previous = numbers[0]
    for number in numbers[1:]:
        if number == previous + 1:
            previous = number
            continue
        ranges.append(f"{start}-{previous}" if start != previous else str(start))
        start = previous = number
    ranges.append(f"{start}-{previous}" if start != previous else str(start))
    return ",".join(ranges)


def split_attachment_text(text: str, *, semantic_type: str, max_chunk_size: int, overlap: int) -> list[str]:
    """按附件语义切分文本。"""
    normalized = str(text or "").strip()
    if not normalized:
        return []
    if semantic_type == "attachment_form":
        sections = split_form_sections(normalized)
        if sections:
            return [item for section in sections for item in split_windows(section, max_chunk_size, overlap)]
    return split_windows(normalized, max_chunk_size, overlap)


def split_form_sections(text: str) -> list[str]:
    """将表格式申请书按标题和表格块拆分。"""
    blocks = [block.strip() for block in re.split(r"\n{2,}", text) if block.strip()]
    sections: list[str] = []
    current: list[str] = []
    for block in blocks:
        if is_form_section_boundary(block) and current:
            sections.append("\n\n".join(current).strip())
            current = [block]
        else:
            current.append(block)
    if current:
        sections.append("\n\n".join(current).strip())
    return sections


def is_form_section_boundary(block: str) -> bool:
    """判断文本块是否像表单区块标题。"""
    head = block.strip().splitlines()[0]
    if head.startswith("|"):
        return False
    if head.startswith("#"):
        return True
    if len(head) <= 40 and any(token in head for token in ["基本情况", "申请", "项目", "取水", "水源", "用途", "计量", "签章"]):
        return True
    return False


def extract_non_table_attachment_text(text: str) -> str:
    """从附件文本中剥离 Markdown 表格，保留说明性正文。"""
    lines: list[str] = []
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if line.startswith("|") and line.endswith("|") and line.count("|") >= 2:
            continue
        lines.append(raw_line.rstrip())
    normalized = "\n".join(lines)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def should_index_attachment_text(text: str) -> bool:
    """判断非表格正文是否值得单独入库。"""
    compact = re.sub(r"\s+", "", str(text or ""))
    if len(compact) < 80:
        return False
    return bool(re.search(r"[\u4e00-\u9fffA-Za-z0-9]", compact))


def normalize_attachment_text_headings(text: str) -> str:
    """为缺少 Word 标题样式的附件正文补充轻量级标题边界。"""
    lines = []
    previous_blank = True
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            lines.append(raw_line)
            previous_blank = True
            continue
        if line.startswith(("#", "-", "•", "|")):
            lines.append(raw_line)
        elif previous_blank and is_plain_attachment_heading(line):
            lines.append(f"## {line}")
        else:
            lines.append(raw_line)
        previous_blank = False
    return "\n".join(lines).strip()


def is_plain_attachment_heading(line: str) -> bool:
    """识别 DOCX/PDF 抽取后常见的裸标题行。"""
    if len(line) > 40:
        return False
    if re.search(r"[。；;，,：:]", line):
        return False
    if re.match(r"^第[一二三四五六七八九十百千万0-9]+[章节部分]", line):
        return True
    if re.match(r"^[一二三四五六七八九十百千万0-9]+[、.．]\s*.+", line):
        return True
    return any(token in line for token in ["说明", "须知", "承诺", "目录", "概况", "总论"])


def attachment_section_title(section: dict[str, Any], extracted: ExtractedAttachment) -> str:
    """返回附件正文小节标题。"""
    title = str(section.get("node_title") or section.get("title") or "").strip()
    if title and title != "Document":
        return title
    return extracted.ref.node_title or extracted.ref.material_name or extracted.ref.name


def attachment_chunk_content(extracted: ExtractedAttachment, text: str, *, part_index: int, part_count: int) -> str:
    """补充附件上下文，形成可直接检索和引用的 chunk 文本。"""
    ref = extracted.ref
    header = [
        f"附件：{ref.name}",
        f"附件角色：{role_label(ref.role)}",
        f"所属章节：{ref.section_path}",
    ]
    if ref.material_name:
        header.append(f"关联材料：{ref.material_name}")
    if part_count > 1:
        header.append(f"附件分段：{part_index + 1}/{part_count}")
    return "\n".join(header) + "\n\n" + text.strip()


def form_field_chunk_content(extracted: ExtractedAttachment, field: FormField) -> str:
    """生成字段级证据文本。"""
    ref = extracted.ref
    lines = [
        f"表单字段：{field.field_name}",
        f"所属表单：{ref.material_name or ref.node_title or ref.name}",
        f"来源附件：{ref.name}",
        f"所属章节：{ref.section_path}",
        f"字段类型：{field_type_label(field.field_type)}",
    ]
    if field.multiple is not None:
        lines.append(f"是否多选：{'是' if field.multiple else '否'}")
    if field.required is not None:
        lines.append(f"是否必填：{'是' if field.required else '否'}")
    if field.options:
        lines.append(f"可选项：{'、'.join(field.options)}")
    if field.raw_label and field.raw_label != field.field_name:
        lines.append(f"原始标签：{field.raw_label}")
    if field.raw_value:
        lines.append(f"原始内容：{field.raw_value}")
    if field.raw_cells:
        lines.append(f"原始表格行：{' | '.join(field.raw_cells)}")
    return "\n".join(lines)


def field_type_label(field_type: str) -> str:
    """返回字段类型中文标签。"""
    return {
        "checkbox": "勾选项",
        "signature": "签字盖章",
        "date": "日期时间",
        "text": "文本填写",
    }.get(field_type, field_type or "文本填写")


def role_label(role: str) -> str:
    """返回附件角色中文标签。"""
    return {
        "blank_form": "空表文件",
        "sample_file": "样本文件",
        "flowchart": "流程图",
        "other_attachment": "其他附件",
    }.get(role, role or "其他附件")


def attachment_chunk_suffix(extracted: ExtractedAttachment, part_index: int) -> str:
    """构建稳定的附件 chunk 后缀。"""
    key = f"{extracted.ref.url}|{extracted.ref.name}|{part_index}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    return f"att:{extracted.ref.index}:{digest}:{part_index}"


def attachment_text_parent_suffix(extracted: ExtractedAttachment, *, section_index: int, parent_index: int) -> str:
    """构建稳定的附件正文父 chunk 后缀。"""
    key = f"{extracted.ref.url}|{extracted.ref.name}|text-parent|{section_index}|{parent_index}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    return f"att_text_parent:{extracted.ref.index}:{digest}:{section_index}:{parent_index}"


def attachment_text_child_suffix(
    extracted: ExtractedAttachment,
    *,
    section_index: int,
    parent_index: int,
    child_index: int,
) -> str:
    """构建稳定的附件正文子 chunk 后缀。"""
    key = f"{extracted.ref.url}|{extracted.ref.name}|text-child|{section_index}|{parent_index}|{child_index}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    return f"att_text_child:{extracted.ref.index}:{digest}:{section_index}:{parent_index}:{child_index}"


def form_field_chunk_suffix(extracted: ExtractedAttachment, field: FormField) -> str:
    """构建稳定的字段级 chunk 后缀。"""
    key = f"{extracted.ref.url}|{field.table_index}|{field.row_index}|{field.column_index}|{field.field_name}|{field.raw_value}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    return f"field:{extracted.ref.index}:{digest}:{field.index}"
