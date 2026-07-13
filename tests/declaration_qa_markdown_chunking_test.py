from declaration_qa.indexing.chunking import chonkie_chunker
from declaration_qa.indexing.chunking.chonkie_chunker import build_markdown_chunk_specs, markdown_elements


COMPLEX_MARKDOWN = """# 办事指南

## 申请材料

- 身份证明
  - 护照原件
  - 有效签证
- 住所证明

> 所有外文材料应附中文译本。

| 材料 | 要求 |
| --- | --- |
| 收养人照片<br>房产证&lt;br&gt;出入境记录 | 原件核验 |

```text
材料编号：A-01
```
"""


def test_marko_gfm_parser_preserves_structured_markdown_blocks():
    elements = markdown_elements(COMPLEX_MARKDOWN)

    assert [element.element_type for element in elements] == ["title", "title", "list", "quote", "table", "code"]
    assert [element.heading_level for element in elements[:2]] == [1, 2]
    assert "- 身份证明" in elements[2].text
    assert "- 护照原件" in elements[2].text
    assert elements[3].text == "> 所有外文材料应附中文译本。"
    assert "收养人照片；房产证；出入境记录" in elements[4].text
    assert "<br>" not in elements[4].text
    assert "材料编号：A-01" in elements[5].text
    assert all(element.metadata["markdown_parser"] == "marko_gfm" for element in elements)


def test_marko_elements_keep_heading_context_and_table_boundary_for_chonkie():
    specs = build_markdown_chunk_specs(COMPLEX_MARKDOWN, text_max_chars=900, text_min_chars=1)

    table = next(spec for spec in specs if spec["metadata"]["chonkie_chunk_kind"] == "table")
    assert table["metadata"]["heading_path"] == "办事指南 > 申请材料"
    assert table["metadata"]["table_source"] == "markdown_marko_gfm"
    assert "收养人照片；房产证；出入境记录" in table["content"]
    assert "护照原件" not in table["content"]

    contextual_chunk = next(spec for spec in specs if "护照原件" in spec["content"])
    assert contextual_chunk["metadata"]["heading_path"] == "办事指南 > 申请材料"
    assert contextual_chunk["metadata"]["source_element_types"] == ["list", "quote"]
    assert contextual_chunk["metadata"]["markdown_parser"] == "marko_gfm"


def test_marko_parser_preserves_unknown_html_blocks_as_retrievable_text():
    elements = markdown_elements("# 标题\n\n<section data-kind=\"notice\">保留这段内容</section>\n")

    html = next(element for element in elements if element.element_type == "html")
    assert "<section" in html.text
    assert "保留这段内容" in html.text


def test_markdown_parser_falls_back_to_line_parser_when_marko_fails(monkeypatch):
    monkeypatch.setattr(
        chonkie_chunker,
        "_marko_markdown_elements",
        lambda markdown: (_ for _ in ()).throw(RuntimeError("invalid markdown ast")),
    )

    elements = markdown_elements("# 标题\n\n正文\n")

    assert [element.element_type for element in elements] == ["title", "text"]
    assert all(element.metadata["markdown_parser"] == "legacy_line_parser" for element in elements)
