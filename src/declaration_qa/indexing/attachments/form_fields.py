"""表单附件字段抽取。"""

from __future__ import annotations

from dataclasses import dataclass, field
import re


CHECKBOX_MARKERS = "□☑☒☐√○〇"


@dataclass(frozen=True)
class FormField:
    """从申请表表格中抽取出的字段。"""

    index: int
    table_index: int
    row_index: int
    column_index: int
    field_name: str
    raw_label: str
    raw_value: str
    raw_cells: list[str]
    field_type: str
    options: list[str] = field(default_factory=list)
    multiple: bool | None = None
    required: bool | None = None
    confidence: float = 0.7


def extract_form_fields(text: str) -> list[FormField]:
    """从 Markdown 表格文本中抽取表单字段。"""
    fields: list[FormField] = []
    seen: set[tuple[str, str, int, int]] = set()
    for table_index, table_rows in enumerate(markdown_tables(text)):
        data_rows = [row for row in table_rows if not is_separator_row(row)]
        for row_index, row in enumerate(data_rows):
            for column_index, label, value in field_pairs_from_row(row):
                field = build_form_field(
                    label,
                    value,
                    raw_cells=row,
                    index=len(fields),
                    table_index=table_index,
                    row_index=row_index,
                    column_index=column_index,
                )
                if not field:
                    continue
                key = (field.field_name, field.raw_value, table_index, row_index)
                if key in seen:
                    continue
                seen.add(key)
                fields.append(field)
    return fields


def markdown_tables(text: str) -> list[list[list[str]]]:
    """解析 Markdown 表格块为行列数组。"""
    tables: list[list[list[str]]] = []
    current: list[list[str]] = []
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if is_markdown_table_line(line):
            current.append(parse_markdown_row(line))
            continue
        if current:
            if len(current) >= 2:
                tables.append(current)
            current = []
    if current and len(current) >= 2:
        tables.append(current)
    return tables


def field_pairs_from_row(row: list[str]) -> list[tuple[int, str, str]]:
    """从一行表格中生成候选字段和值。"""
    pairs: list[tuple[int, str, str]] = []
    for column_index, cell in enumerate(row):
        inline_pair = split_inline_checkbox_field(cell)
        if inline_pair:
            pairs.append((column_index, inline_pair[0], inline_pair[1]))
    if len(row) == 2:
        pairs.append((0, row[0], row[1]))
        return pairs
    column_index = 0
    while column_index < len(row):
        label = row[column_index]
        value = row[column_index + 1] if column_index + 1 < len(row) else ""
        if value_is_likely_value(value) or is_candidate_label(label, value):
            pairs.append((column_index, label, value))
            column_index += 2
        else:
            column_index += 1
    return pairs


def build_form_field(
    label: str,
    value: str,
    *,
    raw_cells: list[str],
    index: int,
    table_index: int,
    row_index: int,
    column_index: int,
) -> FormField | None:
    """构建表单字段对象。"""
    raw_label = clean_cell_text(label)
    raw_value = clean_cell_text(value)
    field_name = normalize_field_name(raw_label)
    if not is_candidate_label(raw_label, raw_value):
        return None
    options = checkbox_options(f"{raw_label} {raw_value}")
    combined = f"{raw_label} {raw_value}"
    field_type = infer_field_type(combined, options)
    required = infer_required(combined)
    multiple = infer_multiple(combined, field_type)
    confidence = field_confidence(raw_label, raw_value, field_type, options)
    return FormField(
        index=index,
        table_index=table_index,
        row_index=row_index,
        column_index=column_index,
        field_name=field_name,
        raw_label=raw_label,
        raw_value=raw_value,
        raw_cells=[clean_cell_text(cell) for cell in raw_cells],
        field_type=field_type,
        options=options,
        multiple=multiple,
        required=required,
        confidence=confidence,
    )


def is_markdown_table_line(line: str) -> bool:
    """判断一行是否是 Markdown 表格行。"""
    return line.startswith("|") and line.endswith("|") and line.count("|") >= 2


def parse_markdown_row(line: str) -> list[str]:
    """解析 Markdown 表格行，兼容转义竖线。"""
    placeholder = "\uE000"
    normalized = line.strip().strip("|").replace("\\|", placeholder)
    return [clean_cell_text(cell.replace(placeholder, "|")) for cell in normalized.split("|")]


def is_separator_row(row: list[str]) -> bool:
    """判断是否是 Markdown 表格分隔行。"""
    return bool(row) and all(re.fullmatch(r":?-{2,}:?", cell.strip()) for cell in row if cell.strip())


def clean_cell_text(text: str) -> str:
    """清理表格单元格文本。"""
    normalized = str(text or "").replace("\\|", "|")
    normalized = re.sub(r"<[^>]+>", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def normalize_field_name(label: str) -> str:
    """将字段标签归一化为字段名。"""
    normalized = clean_cell_text(label)
    normalized = re.split(f"[{CHECKBOX_MARKERS}]", normalized, maxsplit=1)[0]
    normalized = re.sub(r"[（(]\s*(?:可)?多选\s*[）)]", "", normalized)
    normalized = re.sub(r"[（(]\s*(?:单选|必填|选填|必选)\s*[）)]", "", normalized)
    normalized = normalized.strip(" ：:，,；;、")
    normalized = re.sub(r"\s+", "", normalized)
    return normalized


def split_inline_checkbox_field(cell: str) -> tuple[str, str] | None:
    """处理“字段名 □选项1 □选项2”在同一格的情况。"""
    cleaned = clean_cell_text(cell)
    match = re.search(f"[{CHECKBOX_MARKERS}]", cleaned)
    if not match:
        return None
    label = cleaned[: match.start()].strip(" ：:，,；;、")
    value = cleaned[match.start() :].strip()
    if len(normalize_field_name(label)) < 2:
        return None
    return label, value


def is_candidate_label(label: str, value: str) -> bool:
    """判断单元格是否像表单字段标签。"""
    field_name = normalize_field_name(label)
    if len(field_name) < 2 or len(field_name) > 40:
        return False
    if not re.search(r"[\u4e00-\u9fffA-Za-z0-9]", field_name):
        return False
    if any(marker in field_name for marker in CHECKBOX_MARKERS):
        return False
    if is_section_only_label(field_name, value):
        return False
    return True


def is_section_only_label(field_name: str, value: str) -> bool:
    """过滤表单分组标题。"""
    if field_name in {"申请人基本情况", "项目基本情况", "取水基本情况", "其他事项", "备注"}:
        return True
    if field_name.startswith("省（自治区") or field_name.startswith("省(自治区"):
        return True
    if "街道" in field_name and "村" in field_name and "号" in field_name:
        return True
    if clean_cell_text(value):
        return False
    return field_name.endswith("情况") and len(field_name) <= 10


def value_is_likely_value(value: str) -> bool:
    """判断右侧单元格是否像字段值。"""
    cleaned = clean_cell_text(value)
    if not cleaned:
        return True
    if any(marker in cleaned for marker in CHECKBOX_MARKERS):
        return True
    if any(token in cleaned for token in ["____", "年 月 日", "年  月  日", "%"]):
        return True
    return len(cleaned) <= 80


def checkbox_options(text: str) -> list[str]:
    """从文本中提取勾选项。"""
    output: list[str] = []
    pattern = re.compile(f"[{CHECKBOX_MARKERS}]\\s*([^{CHECKBOX_MARKERS}]+)")
    for match in pattern.finditer(str(text or "")):
        option = clean_option_text(match.group(1))
        if option and option not in output:
            output.append(option)
    return output


def clean_option_text(text: str) -> str:
    """清理勾选项文本。"""
    option = re.split(r"[；;，,]", clean_cell_text(text), maxsplit=1)[0]
    option = option.strip(" ：:，,；;、()（）[]【】")
    if len(option) > 40:
        option = option[:40].strip(" ：:，,；;、()（）[]【】")
    return option


def infer_field_type(text: str, options: list[str]) -> str:
    """推断字段类型。"""
    if options:
        return "checkbox"
    if any(token in text for token in ["签字", "签名", "签章", "盖章", "印章"]):
        return "signature"
    if re.search(r"年\s*月\s*日", text) or any(token in text for token in ["日期", "时间", "期限"]):
        return "date"
    return "text"


def infer_multiple(text: str, field_type: str) -> bool | None:
    """推断字段是否多选。"""
    if field_type != "checkbox":
        return None
    if "单选" in text:
        return False
    if "多选" in text or "可多选" in text:
        return True
    return None


def infer_required(text: str) -> bool | None:
    """推断字段是否必填。"""
    if any(token in text for token in ["非必填", "选填", "非必选"]):
        return False
    if any(token in text for token in ["必填", "必选", "必须"]):
        return True
    return None


def field_confidence(label: str, value: str, field_type: str, options: list[str]) -> float:
    """根据规则命中特征给字段置信度。"""
    score = 0.68
    if value:
        score += 0.08
    if options:
        score += 0.12
    if field_type in {"signature", "date"}:
        score += 0.08
    if len(normalize_field_name(label)) <= 20:
        score += 0.04
    return min(score, 0.95)
