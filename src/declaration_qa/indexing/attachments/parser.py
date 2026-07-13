"""从 Markdown 办事指南中解析附件链接。"""

from __future__ import annotations

import re
from urllib.parse import unquote, urlparse

from declaration_qa.indexing.attachments.models import AttachmentRef
from declaration_qa.rag.engine.chunking import parse_heading_order, strip_heading_order


SUPPORTED_ATTACHMENT_EXTENSIONS = {
    "pdf",
    "docx",
    "doc",
    "png",
    "jpg",
    "jpeg",
    "webp",
    "bmp",
    "tif",
    "tiff",
}


def parse_markdown_attachments(content: str, *, document_id: str) -> list[AttachmentRef]:
    """使用 Marko GFM AST 解析官方附件并保留所在章节上下文。"""
    try:
        return parse_marko_attachments(content, document_id=document_id)
    except Exception:
        return parse_legacy_markdown_attachments(content, document_id=document_id)


def parse_marko_attachments(content: str, *, document_id: str) -> list[AttachmentRef]:
    """从 Marko AST 中读取标题、列表和链接节点。"""
    import marko
    from marko import block, inline
    from marko.ext.gfm import GFM
    from marko.md_renderer import MarkdownRenderer

    parser = marko.Markdown(renderer=MarkdownRenderer, extensions=[GFM])
    document = parser.parse(str(content or ""))
    attachments: list[AttachmentRef] = []
    heading_stack: list[tuple[int, str]] = []
    current_label = ""
    seen: set[tuple[str, str]] = set()

    with parser.renderer:
        parser.renderer.root_node = document
        for node in document.children:
            if isinstance(node, (block.Heading, block.SetextHeading)):
                level = max(1, int(node.level))
                title = marko_plain_text(node).strip()
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, title))
                current_label = ""
                continue
            if isinstance(node, block.BlankLine):
                continue

            rendered = parser.renderer.render(node).strip()
            if isinstance(node, block.Paragraph) and not marko_link_nodes(node, inline=inline):
                plain_text = marko_plain_text(node).strip()
                if plain_text.endswith(("：", ":")):
                    current_label = plain_text.rstrip("：:").strip()

            candidates = marko_attachment_candidates(
                node,
                rendered=rendered,
                inline=inline,
                renderer=parser.renderer,
            )
            for name, url, extension in candidates:
                signature = (name, url)
                if extension not in SUPPORTED_ATTACHMENT_EXTENSIONS or signature in seen:
                    continue
                seen.add(signature)
                attachments.append(
                    attachment_ref(
                        index=len(attachments),
                        name=name,
                        url=url,
                        extension=extension,
                        document_id=document_id,
                        heading_stack=heading_stack,
                        current_label=current_label,
                    )
                )
    return attachments


def marko_attachment_candidates(node, *, rendered: str, inline, renderer) -> list[tuple[str, str, str]]:
    """读取标准 Markdown Link，并兼容历史 ``文件名: URL`` 行。"""
    output: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for link in marko_link_nodes(node, inline=inline):
        name = marko_plain_text(link).strip()
        url = normalize_attachment_url(str(link.dest or ""))
        extension = attachment_extension(name, url)
        signature = (name, url)
        if name and url.startswith(("http://", "https://")) and extension and signature not in seen:
            seen.add(signature)
            output.append((name, url, extension))
    for line in str(rendered or "").splitlines():
        candidates = [line.strip()]
        if line.strip().startswith("|"):
            candidates.extend(cell.strip() for cell in line.strip().strip("|").split("|"))
        for candidate in candidates:
            parsed = parse_attachment_bullet(candidate)
            if not parsed or (parsed[0], parsed[1]) in seen:
                continue
            seen.add((parsed[0], parsed[1]))
            output.append(parsed)
    return output


def marko_link_nodes(node, *, inline) -> list:
    """按文档顺序返回 AST 中的普通链接节点。"""
    output = []

    def visit(item) -> None:
        if isinstance(item, inline.Link):
            output.append(item)
            return
        children = getattr(item, "children", None)
        if isinstance(children, list):
            for child in children:
                visit(child)

    visit(node)
    return output


def marko_plain_text(node) -> str:
    """提取 Marko 节点的可读文本，不带 Markdown 标记。"""
    children = getattr(node, "children", None)
    if isinstance(children, str):
        return children
    if isinstance(children, list):
        return "".join(marko_plain_text(child) for child in children)
    return str(children or "")


def attachment_ref(
    *,
    index: int,
    name: str,
    url: str,
    extension: str,
    document_id: str,
    heading_stack: list[tuple[int, str]],
    current_label: str,
) -> AttachmentRef:
    """根据统一候选构造带章节语义的附件引用。"""
    section_path = " > ".join(title for _, title in heading_stack)
    section_title = first_level_title(heading_stack, 2)
    node_title = heading_stack[-1][1] if heading_stack else section_title
    material_name = infer_material_name(section_title, node_title, name)
    role = infer_attachment_role(name, current_label, section_title, node_title, extension)
    return AttachmentRef(
        index=index,
        name=name,
        url=url,
        extension=extension,
        section_path=section_path,
        section_title=section_title,
        node_title=node_title,
        parent_section_id=f"{document_id}#{section_title or node_title or 'Document'}",
        material_name=material_name,
        role=role,
    )


def parse_legacy_markdown_attachments(content: str, *, document_id: str) -> list[AttachmentRef]:
    """Marko 不可用或解析异常时使用原行级解析器。"""
    attachments: list[AttachmentRef] = []
    heading_stack: list[tuple[int, str]] = []
    current_label = ""
    for raw_line in str(content or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        heading = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if heading:
            level = len(heading.group(1))
            title = heading.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            current_label = ""
            continue
        if line.endswith(("：", ":")) and not line.startswith(("|", "-", "*")):
            current_label = line.rstrip("：:").strip()
            continue
        parsed = parse_attachment_bullet(line)
        if not parsed:
            continue
        name, url, extension = parsed
        if extension not in SUPPORTED_ATTACHMENT_EXTENSIONS:
            continue
        attachments.append(
            attachment_ref(
                index=len(attachments),
                name=name,
                url=url,
                extension=extension,
                document_id=document_id,
                heading_stack=heading_stack,
                current_label=current_label,
            )
        )
    return attachments


def parse_attachment_bullet(line: str) -> tuple[str, str, str] | None:
    """解析带或不带列表符号的 ``文件名.ext: URL`` 附件行。"""
    match = re.match(
        r"^(?:[-*]\s*)?(.+?)\s*[:：]\s*(https?://\S+?)(?:\s+(?:样本文件|空表文件)\s*[:：])?\s*$",
        line,
    )
    if not match:
        return None
    name = match.group(1).strip()
    url = normalize_attachment_url(match.group(2))
    extension = attachment_extension(name, url)
    if not extension:
        return None
    return name, url, extension


def normalize_attachment_url(value: str) -> str:
    """还原 Markdown 为 URL 中标点加入的反斜杠转义。"""
    return re.sub(r"\\([_()\[\]])", r"\1", str(value or "").strip())


def attachment_extension(name: str, url: str) -> str:
    """从文件名、attrname 或 URL 路径中提取附件扩展名。"""
    candidates = [name]
    parsed = urlparse(url)
    query = parsed.query
    attr_match = re.search(r"(?:^|&)attrname=([^&]+)", query)
    if attr_match:
        candidates.append(unquote(attr_match.group(1)))
    candidates.append(unquote(parsed.path.rsplit("/", 1)[-1]))
    for candidate in candidates:
        suffix_match = re.search(r"\.([A-Za-z0-9]{2,8})(?:$|\?)", candidate)
        if suffix_match:
            return suffix_match.group(1).lower()
    return ""


def first_level_title(stack: list[tuple[int, str]], level: int) -> str:
    """返回指定级别的当前标题。"""
    for item_level, title in stack:
        if item_level == level:
            return title
    return stack[-1][1] if stack else ""


def infer_material_name(section_title: str, node_title: str, attachment_name: str = "") -> str:
    """从申请材料节点标题和附件文件名中推断材料名称。"""
    if section_title != "申请材料":
        return ""
    attachment_material = normalize_attachment_material_name(attachment_name)
    if parse_heading_order(node_title) is None:
        return attachment_material
    node_material = strip_heading_order(node_title)
    if not attachment_material:
        return node_material
    if attachment_material in node_material or node_material in attachment_material:
        return node_material
    return attachment_material


def normalize_attachment_material_name(name: str) -> str:
    """从附件文件名中提取更接近材料名称的文本。"""
    stem = re.sub(r"\.[A-Za-z0-9]{2,8}$", "", str(name or "").strip())
    stem = re.sub(r"^(?:示例样表|空白样表|空表文件|样本文件)[-—_：:]*", "", stem)
    stem = re.sub(r"(?:样本|空表|模板)$", "", stem).strip("-—_：: ")
    return stem


def infer_attachment_role(name: str, label: str, section_title: str, node_title: str, extension: str) -> str:
    """根据附件名、上下文标签和扩展名推断附件角色。"""
    text = f"{name} {label} {section_title} {node_title}".lower()
    if "空表" in text or "空白" in text:
        return "blank_form"
    if "样本" in text or "示例" in text or "样表" in text:
        return "sample_file"
    if "流程图" in text or (section_title == "办理流程" and extension in {"png", "jpg", "jpeg", "webp"}):
        return "flowchart"
    return "other_attachment"
