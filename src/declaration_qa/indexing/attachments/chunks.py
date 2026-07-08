"""将抽取后的附件转换为派生 chunk 描述。"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from declaration_qa.indexing.attachments.form_fields import FormField, extract_form_fields
from declaration_qa.indexing.attachments.models import ExtractedAttachment
from declaration_qa.rag.engine.chunking import split_windows


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
    for extracted in extracted_items:
        if extracted.status != "success" or not extracted.text.strip():
            continue
        semantic_type = str(extracted.metadata.get("semantic_chunk_type") or "attachment_text")
        parts = split_attachment_text(extracted.text, semantic_type=semantic_type, max_chunk_size=max_chunk_size, overlap=overlap)
        for part_index, part in enumerate(parts):
            content = attachment_chunk_content(extracted, part, part_index=part_index, part_count=len(parts))
            specs.append(
                {
                    "id_suffix": attachment_chunk_suffix(extracted, part_index),
                    "content": content,
                    "source_url": extracted.ref.url,
                    "metadata": {
                        **document_metadata,
                        "chunk_type": semantic_type,
                        "attachment_index": extracted.ref.index,
                        "attachment_name": extracted.ref.name,
                        "attachment_type": extracted.ref.extension,
                        "attachment_role": extracted.ref.role,
                        "attachment_content_type": extracted.content_type,
                        "attachment_sha256": extracted.sha256,
                        "attachment_cache_path": extracted.cache_path,
                        "attachment_part_index": part_index,
                        "attachment_part_count": len(parts),
                        "attachment_status": extracted.status,
                        "section_path": extracted.ref.section_path,
                        "section_title": extracted.ref.section_title,
                        "node_title": extracted.ref.node_title,
                        "parent_section_id": extracted.ref.parent_section_id,
                        "material_name": extracted.ref.material_name,
                        "indexable": True,
                        **extracted.metadata,
                    },
                }
            )
        if semantic_type == "attachment_form":
            specs.extend(build_form_field_chunk_specs(extracted, document_metadata=document_metadata))
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
                    **document_metadata,
                    "chunk_type": "form_field",
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
                    "node_title": f"表单字段：{field.field_name}",
                    "node_order": field.index,
                    "parent_section_id": extracted.ref.parent_section_id,
                    "material_name": extracted.ref.material_name,
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


def form_field_chunk_suffix(extracted: ExtractedAttachment, field: FormField) -> str:
    """构建稳定的字段级 chunk 后缀。"""
    key = f"{extracted.ref.url}|{field.table_index}|{field.row_index}|{field.column_index}|{field.field_name}|{field.raw_value}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    return f"field:{extracted.ref.index}:{digest}:{field.index}"
