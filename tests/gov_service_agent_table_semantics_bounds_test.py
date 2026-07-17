from gov_service_agent.indexing.document_parsing.table_semantics import (
    MAX_INDEXABLE_TABLE_CHUNK_CHARS,
    build_table_semantic_chunk_specs,
)


def test_indexable_table_chunks_are_bounded_for_pathological_table_values():
    huge = "超长内容" * 5000
    specs = build_table_semantic_chunk_specs(
        id_prefix="table:test",
        table_index=0,
        table_json_value={
            "title": huge,
            "headers": ["字段", "内容"],
            "rows": [[huge, huge]],
        },
        table_text_value=huge,
        table_html_value="",
        base_metadata={},
        source_lines=[huge],
    )

    indexable = [spec for spec in specs if spec["metadata"].get("indexable")]
    assert indexable
    assert all(len(spec["content"]) <= MAX_INDEXABLE_TABLE_CHUNK_CHARS for spec in indexable)
