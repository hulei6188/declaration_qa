from declaration_qa.source_types import (
    MATERIAL_RULE_SOURCE_TYPE,
    canonicalize_source_type,
    expand_source_type_filter,
)
from declaration_qa.vector_index import milvus_filter_expr


def test_reject_rule_is_canonicalized_to_material_rule():
    assert canonicalize_source_type("reject_rule") == MATERIAL_RULE_SOURCE_TYPE
    assert expand_source_type_filter("reject_rule") == [MATERIAL_RULE_SOURCE_TYPE, "reject_rule"]


def test_milvus_material_rule_filter_includes_legacy_reject_rule_alias():
    expr = milvus_filter_expr({"source_type": MATERIAL_RULE_SOURCE_TYPE, "status": "published"})

    assert 'source_type in ["material_rule", "reject_rule"]' in expr
    assert 'status == "published"' in expr
