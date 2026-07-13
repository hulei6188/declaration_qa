"""本地内存 RAG 使用的内置种子知识。"""

from __future__ import annotations

from declaration_qa.core.domain import KnowledgeDocument
from declaration_qa.rag.engine.in_memory import InMemoryRAGEngine
from declaration_qa.rag.model_providers.embeddings.embedding_provider import EmbeddingProvider
from declaration_qa.rag.model_providers.rerankers.reranker import Reranker
from declaration_qa.search.keyword import KeywordIndex
from declaration_qa.search.vector.base import VectorIndex
from declaration_qa.storage.cache import CacheStore


def build_seed_rag_engine(
    *,
    cache_store: CacheStore | None = None,
    vector_index: VectorIndex | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    reranker: Reranker | None = None,
    keyword_index: KeywordIndex | None = None,
    cache_ttl_seconds: int = 600,
    rerank_score_threshold: float = 0.0,
    rerank_max_sources: int = 20,
    retrieval_candidates_per_query: int = 30,
    rerank_top_n_per_query: int = 6,
    include_seed: bool = True,
) -> InMemoryRAGEngine:
    """创建内存 RAG 引擎，并可选择加载种子知识。"""
    engine = InMemoryRAGEngine(
        cache_store=cache_store,
        vector_index=vector_index,
        embedding_provider=embedding_provider,
        reranker=reranker,
        keyword_index=keyword_index,
        cache_ttl_seconds=cache_ttl_seconds,
        rerank_score_threshold=rerank_score_threshold,
        rerank_max_sources=rerank_max_sources,
        retrieval_candidates_per_query=retrieval_candidates_per_query,
        rerank_top_n_per_query=rerank_top_n_per_query,
    )
    if include_seed:
        for document in seed_documents():
            engine.index_document(document)
    return engine


def seed_documents() -> list[KnowledgeDocument]:
    """返回内置种子知识文档。"""
    return [
        KnowledgeDocument(
            id="doc-residence-guide-2025",
            title="安徽省居住证办理材料指南",
            content=(
                "办理居住证通常需要本人有效身份证件、居住地址证明、近期照片等材料。"
                "居住地址证明可以包括房屋租赁合同、用人单位住宿证明、学校住宿证明或社区出具的居住证明。"
                "申请人应确保材料真实、完整，材料不齐全时需要补正后重新提交。"
            ),
            region_code="340000",
            policy_version="2025",
            source_url="internal://guides/residence-permit",
        ),
        KnowledgeDocument(
            id="doc-health-process-2025",
            title="健康证办理流程节点说明",
            content=(
                "健康证申请提交后会进入材料受理、部门审核、制证和办结等节点。"
                "部门审核中表示申请材料已进入业务部门审核环节，用户通常无需重复提交。"
                "如审核发现材料问题，系统会退回并提示补正原因。"
            ),
            region_code="340000",
            policy_version="2025",
            source_url="internal://process/health-certificate",
        ),
        KnowledgeDocument(
            id="doc-id-card-photo-rule-2025",
            title="身份证照片上传规范",
            content=(
                "身份证正面照片应保持边框完整、文字清晰、无遮挡、无反光。"
                "如果照片模糊、证件号码无法识别、边缘缺失或存在明显反光，审核人员可退回申请。"
                "用户需要重新上传清晰的身份证正面照片后再次提交。"
            ),
            region_code="340000",
            policy_version="2025",
            source_url="internal://rules/id-card-photo",
            audit_node="人工初审",
        ),
        KnowledgeDocument(
            id="doc-residence-proof-rule-2025",
            title="居住证明补正规则",
            content=(
                "居住证明材料缺失时，申请人应补充租赁合同、单位住宿证明、学校住宿证明或社区居住证明。"
                "补交材料需要与当前申请地区一致，并能证明申请人在当地实际居住。"
            ),
            region_code="340000",
            policy_version="2025",
            source_url="internal://rules/residence-proof",
            audit_node="材料初审",
        ),
    ]
