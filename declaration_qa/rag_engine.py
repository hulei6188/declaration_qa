from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import replace
from datetime import date
from typing import Any, Iterable

from declaration_qa.cache import CacheStore, InMemoryCacheStore
from declaration_qa.domain import KnowledgeChunk, KnowledgeDocument, RetrievalResult, RetrievedSource
from declaration_qa.model_providers import EmbeddingProvider, HashEmbeddingProvider, HeuristicReranker, Reranker
from declaration_qa.retrieval import KNOWLEDGE_RETRIEVAL_CHAIN, metadata_coverage, sanitize_metadata
from declaration_qa.vector_index import MemoryVectorIndex, VectorIndex


EMBEDDING_DIMS = 96
CHUNKING_PARENT_CHILD = "parent_child"
CHUNKING_HIERARCHICAL = "hierarchical"
SUPPORTED_CHUNKING_STRATEGIES = {CHUNKING_PARENT_CHILD, CHUNKING_HIERARCHICAL}


class InMemoryRAGEngine:
    """Isolated RAG engine for the declaration QA workflow.

    The class keeps the same retrieval shape expected in production
    (dense retrieval -> BM25 -> RRF -> rerank), while retaining metadata only
    for source display and auditing. It uses an in-memory vector index so it can
    run without Milvus/Redis locally.
    """

    def __init__(
        self,
        *,
        cache_store: CacheStore | None = None,
        vector_index: VectorIndex | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        reranker: Reranker | None = None,
        cache_ttl_seconds: int = 600,
    ) -> None:
        self._documents: dict[str, KnowledgeDocument] = {}
        self._chunks: dict[str, KnowledgeChunk] = {}
        self._vectors: dict[str, list[float]] = {}
        self.cache_store = cache_store or InMemoryCacheStore()
        self.vector_index = vector_index or MemoryVectorIndex()
        self.embedding_provider = embedding_provider or HashEmbeddingProvider(dimension=EMBEDDING_DIMS)
        self.reranker = reranker or HeuristicReranker()
        self.cache_ttl_seconds = max(1, int(cache_ttl_seconds))

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def documents(self) -> list[KnowledgeDocument]:
        return list(self._documents.values())

    def index_document(self, document: KnowledgeDocument) -> int:
        self.delete_document(document.id)
        self._documents[document.id] = document
        chunks = split_document(document)
        for chunk in chunks:
            self._chunks[chunk.id] = chunk
            vector = self.embedding_provider.embed_query(chunk.content)
            self._vectors[chunk.embedding_id] = vector
            if chunk.chunk_type == "child":
                self.vector_index.upsert(chunk, vector)
        self.cache_store.clear()
        return len(chunks)

    def delete_document(self, doc_id: str) -> None:
        self._documents.pop(doc_id, None)
        for chunk_id, chunk in list(self._chunks.items()):
            if chunk.doc_id == doc_id:
                self._chunks.pop(chunk_id, None)
                self._vectors.pop(chunk.embedding_id, None)
        self.vector_index.delete_by_doc(doc_id)
        self.cache_store.clear()

    def rebuild(self, documents: list[KnowledgeDocument]) -> int:
        self._documents.clear()
        self._chunks.clear()
        self._vectors.clear()
        self.vector_index.clear()
        count = 0
        for document in documents:
            count += self.index_document(document)
        self.cache_store.clear()
        return count

    def document_chunks(self, document: KnowledgeDocument) -> list[KnowledgeChunk]:
        return split_document(document)

    def retrieve(self, query: str, filters: dict[str, Any], *, top_k: int = 4) -> RetrievalResult:
        normalized_query = normalize_text(query)
        effective_filter: dict[str, Any] = {}
        cache_key = json.dumps(
            {"query": normalized_query, "filters": effective_filter, "top_k": top_k},
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        cache_key = f"declaration_qa:rag:{hashlib.sha256(cache_key.encode('utf-8')).hexdigest()}"
        cached_payload = self.cache_store.get_json(cache_key)
        if cached_payload:
            cached_sources = [
                RetrievedSource(
                    chunk=self._chunks[item["chunk_id"]],
                    score=float(item.get("score") or 0.0),
                    dense_score=float(item.get("dense_score") or 0.0),
                    bm25_score=float(item.get("bm25_score") or 0.0),
                    rerank_score=float(item.get("rerank_score") or 0.0),
                    retrieval_channel=str(item.get("retrieval_channel") or "cache"),
                )
                for item in cached_payload.get("sources", [])
                if item.get("chunk_id") in self._chunks
            ]
            return RetrievalResult(
                sources=cached_sources,
                filter=cached_payload.get("filter") or effective_filter,
                status={**(cached_payload.get("status") or {}), "cache": {"hit": True, "backend": self.cache_store.backend}},
            )

        candidates = [chunk for chunk in self._chunks.values() if chunk.chunk_type == "child"]
        prefilter_child_count = sum(1 for chunk in self._chunks.values() if chunk.chunk_type == "child")
        coverage = metadata_coverage(effective_filter, chain=str(effective_filter.get("chain") or KNOWLEDGE_RETRIEVAL_CHAIN))
        if not candidates:
            result = RetrievalResult(
                sources=[],
                filter=effective_filter,
                status={
                    "query": normalized_query,
                    "prefilter_child_count": prefilter_child_count,
                    "candidate_count": 0,
                    "metadata_coverage": coverage,
                    "cache": {"hit": False, "backend": self.cache_store.backend},
                    "embedding_provider": self.embedding_provider.name,
                    "rerank_provider": self.reranker.name,
                    "metadata_filtering": False,
                    "channels": {"dense": 0, "bm25": 0, "rrf": 0, "rerank": 0},
                    "filter": effective_filter,
                },
            )
            self._cache_result(cache_key, result)
            return result

        dense_hits = self._dense_retrieve(normalized_query, candidates, effective_filter, limit=max(top_k * 3, 8))
        bm25_hits = bm25_retrieve(normalized_query, candidates, limit=max(top_k * 3, 8))
        fused_hits = rrf_fuse([dense_hits, bm25_hits], rrf_k=60)
        reranked = self.reranker.rerank(normalized_query, fused_hits, effective_filter, top_n=max(top_k, 1))
        expanded_sources = expand_parent_context(reranked, self._chunks)
        result = RetrievalResult(
            sources=expanded_sources,
            filter=effective_filter,
            status={
                "query": normalized_query,
                "prefilter_child_count": prefilter_child_count,
                "candidate_count": len(candidates),
                "metadata_coverage": coverage,
                "cache": {"hit": False, "backend": self.cache_store.backend},
                "vector_backend": self.vector_index.backend,
                "embedding_provider": self.embedding_provider.name,
                "rerank_provider": self.reranker.name,
                "metadata_filtering": False,
                "channels": {
                    "dense": len(dense_hits),
                    "bm25": len(bm25_hits),
                    "rrf": len(fused_hits),
                    "rerank": len(reranked),
                },
                "filter": effective_filter,
            },
        )
        self._cache_result(cache_key, result)
        return result

    def _dense_retrieve(
        self,
        query: str,
        candidates: list[KnowledgeChunk],
        filters: dict[str, Any],
        *,
        limit: int,
    ) -> list[RetrievedSource]:
        query_vector = self.embedding_provider.embed_query(query)
        by_id = {chunk.id: chunk for chunk in candidates}
        vector_hits = self.vector_index.search(query_vector, filters, limit=limit)
        hits = [
            RetrievedSource(
                chunk=by_id[chunk_id],
                score=score,
                dense_score=score,
                retrieval_channel="dense",
            )
            for chunk_id, score in vector_hits
            if chunk_id in by_id and score > 0
        ]
        if hits:
            return hits[:limit]
        return dense_retrieve(query, candidates, self._vectors, limit=limit, embedding_provider=self.embedding_provider)

    def _cache_result(self, cache_key: str, result: RetrievalResult) -> None:
        self.cache_store.set_json(
            cache_key,
            {
                "sources": [source.payload() for source in result.sources],
                "filter": result.filter,
                "status": result.status,
            },
            ttl_seconds=self.cache_ttl_seconds,
        )


def split_document(document: KnowledgeDocument) -> list[KnowledgeChunk]:
    strategy = normalize_chunking_strategy(document.metadata.get("chunking_strategy"))
    if strategy == CHUNKING_HIERARCHICAL:
        return split_hierarchical(document)
    return split_parent_child(document)


def normalize_chunking_strategy(value: Any) -> str:
    strategy = str(value or CHUNKING_PARENT_CHILD).strip().lower().replace("-", "_")
    return strategy if strategy in SUPPORTED_CHUNKING_STRATEGIES else CHUNKING_PARENT_CHILD


def split_parent_child(document: KnowledgeDocument, *, parent_size: int = 1200, child_size: int = 420, overlap: int = 80) -> list[KnowledgeChunk]:
    content = document.content.strip()
    if not content:
        return []
    parent_texts = split_windows(content, parent_size, max(0, min(overlap, parent_size // 2)))
    chunks: list[KnowledgeChunk] = []
    for parent_index, parent_text in enumerate(parent_texts):
        parent_id = f"{document.id}:p:{parent_index}"
        parent_chunk = knowledge_chunk_from_document(
            document,
            chunk_id=parent_id,
            parent_id=parent_id,
            chunk_type="parent",
            content=parent_text,
        )
        chunks.append(parent_chunk)
        child_texts = split_windows(parent_text, child_size, max(0, min(overlap, child_size // 2)))
        for child_index, child_text in enumerate(child_texts):
            child_id = f"{parent_id}:c:{child_index}"
            chunks.append(
                knowledge_chunk_from_document(
                    document,
                    chunk_id=child_id,
                    parent_id=parent_id,
                    chunk_type="child",
                    content=child_text,
                )
            )
    return chunks


def split_hierarchical(document: KnowledgeDocument, *, parent_size: int = 1200, child_size: int = 420, overlap: int = 80) -> list[KnowledgeChunk]:
    content = document.content.strip()
    if not content:
        return []
    chunks: list[KnowledgeChunk] = []
    for section_index, section in enumerate(markdown_sections(content)):
        parent_texts = split_windows(section["content"], parent_size, max(0, min(overlap, parent_size // 2)))
        for parent_index, parent_text in enumerate(parent_texts):
            parent_id = f"{document.id}:h:{section_index}:p:{parent_index}"
            extra_metadata = {
                "chunking_strategy": CHUNKING_HIERARCHICAL,
                "heading_path": section["path"],
                "heading_level": section["level"],
                "section_index": section_index,
                "parent_window_index": parent_index,
            }
            parent_chunk = knowledge_chunk_from_document(
                document,
                chunk_id=parent_id,
                parent_id=parent_id,
                chunk_type="parent",
                content=parent_text,
                extra_metadata=extra_metadata,
            )
            chunks.append(parent_chunk)
            child_texts = split_windows(parent_text, child_size, max(0, min(overlap, child_size // 2)))
            for child_index, child_text in enumerate(child_texts):
                child_id = f"{parent_id}:c:{child_index}"
                chunks.append(
                    knowledge_chunk_from_document(
                        document,
                        chunk_id=child_id,
                        parent_id=parent_id,
                        chunk_type="child",
                        content=child_text,
                        extra_metadata={**extra_metadata, "child_window_index": child_index},
                    )
                )
    return chunks


def markdown_sections(text: str) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    stack: list[tuple[int, str]] = []
    current_lines: list[str] = []
    current_path: list[str] = []
    current_level = 0
    heading_pattern = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")

    def flush_current() -> None:
        body = "\n".join(current_lines).strip()
        if not body and not current_path:
            return
        path = " > ".join(current_path) if current_path else "Document"
        section_text = f"{path}\n{body}".strip() if body else path
        sections.append({"path": path, "level": current_level, "content": section_text})

    for line in text.splitlines():
        heading = heading_pattern.match(line.strip())
        if heading:
            flush_current()
            level = len(heading.group(1))
            title = heading.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            current_path = [item[1] for item in stack]
            current_level = level
            current_lines = []
            continue
        current_lines.append(line)
    flush_current()

    if sections:
        return [section for section in sections if section["content"].strip()]
    return [{"path": "Document", "level": 0, "content": text.strip()}]


def split_windows(text: str, size: int, overlap: int) -> list[str]:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= size:
        return [compact]
    output = []
    start = 0
    step = max(1, size - overlap)
    while start < len(compact):
        output.append(compact[start : start + size].strip())
        start += step
    return [item for item in output if item]


def knowledge_chunk_from_document(
    document: KnowledgeDocument,
    *,
    chunk_id: str,
    parent_id: str,
    chunk_type: str,
    content: str,
    extra_metadata: dict[str, Any] | None = None,
) -> KnowledgeChunk:
    document_metadata = sanitize_metadata(document.metadata)
    metadata = {
        **document_metadata,
        "region_code": document.region_code,
        "region_key": document_metadata.get("region_key") or document.region_code,
        "policy_version": document.policy_version,
        "source_version": document_metadata.get("source_version") or document.policy_version,
        "audit_node": document.audit_node,
        "version_group_uid": document.version_group_uid or document.id,
        "version_no": document.version_no,
        "rollout_percent": document.rollout_percent,
        "doc_status": document_metadata.get("doc_status") or document.status,
    }
    metadata.update(sanitize_metadata(extra_metadata or {}))
    return KnowledgeChunk(
        id=chunk_id,
        doc_id=document.id,
        parent_id=parent_id,
        chunk_type=chunk_type,
        title=document.title,
        content=content,
        region_code=document.region_code,
        policy_version=document.policy_version,
        source_url=document.source_url,
        audit_node=document.audit_node,
        status=document.status,
        effective_start_time=document.effective_start_time,
        effective_end_time=document.effective_end_time,
        embedding_id=f"vec:{chunk_id}",
        metadata=metadata,
    )


def dense_retrieve(
    query: str,
    candidates: list[KnowledgeChunk],
    vectors: dict[str, list[float]],
    *,
    limit: int,
    embedding_provider: EmbeddingProvider | None = None,
) -> list[RetrievedSource]:
    embedding_provider = embedding_provider or HashEmbeddingProvider(dimension=EMBEDDING_DIMS)
    query_vector = embedding_provider.embed_query(query)
    hits = []
    for chunk in candidates:
        score = cosine(query_vector, vectors.get(chunk.embedding_id) or embedding_provider.embed_query(chunk.content))
        if score > 0:
            hits.append(
                RetrievedSource(
                    chunk=chunk,
                    score=score,
                    dense_score=score,
                    retrieval_channel="dense",
                )
            )
    return sorted(hits, key=lambda item: item.score, reverse=True)[:limit]


def expand_parent_context(sources: list[RetrievedSource], chunks: dict[str, KnowledgeChunk]) -> list[RetrievedSource]:
    expanded = []
    for source in sources:
        chunk = source.chunk
        parent = chunks.get(chunk.parent_id) if chunk.chunk_type == "child" else None
        if parent and parent.content and parent.content != chunk.content:
            expanded_chunk = replace(
                chunk,
                content=parent.content,
                metadata={
                    **chunk.metadata,
                    "retrieved_child_chunk_id": chunk.id,
                    "expanded_parent_chunk_id": parent.id,
                },
            )
            expanded.append(replace(source, chunk=expanded_chunk))
        else:
            expanded.append(source)
    return expanded


def bm25_retrieve(query: str, candidates: list[KnowledgeChunk], *, limit: int) -> list[RetrievedSource]:
    query_tokens = tokenize(query)
    corpus = [tokenize(chunk.content) for chunk in candidates]
    if not query_tokens:
        return []
    scores = bm25_scores(query_tokens, corpus)
    max_score = max(scores or [0.0]) or 1.0
    hits = []
    for chunk, score in zip(candidates, scores, strict=True):
        normalized_score = float(score) / max_score if score > 0 else 0.0
        if normalized_score <= 0:
            continue
        hits.append(
            RetrievedSource(
                chunk=chunk,
                score=normalized_score,
                bm25_score=normalized_score,
                retrieval_channel="bm25",
            )
        )
    return sorted(hits, key=lambda item: item.score, reverse=True)[:limit]


def bm25_scores(query_tokens: list[str], corpus: list[list[str]]) -> list[float]:
    try:
        from rank_bm25 import BM25Okapi

        return [float(score) for score in BM25Okapi(corpus).get_scores(query_tokens)]
    except Exception:
        query_set = set(query_tokens)
        return [float(len(query_set.intersection(tokens))) for tokens in corpus]


def rrf_fuse(hit_lists: list[list[RetrievedSource]], *, rrf_k: int) -> list[RetrievedSource]:
    chunk_scores: dict[str, float] = defaultdict(float)
    best_source: dict[str, RetrievedSource] = {}
    dense_scores: dict[str, float] = defaultdict(float)
    bm25_scores_by_chunk: dict[str, float] = defaultdict(float)
    channels: dict[str, set[str]] = defaultdict(set)

    for hit_list in hit_lists:
        for rank, hit in enumerate(hit_list, start=1):
            chunk_id = hit.chunk.id
            chunk_scores[chunk_id] += 1.0 / (rrf_k + rank)
            best_source.setdefault(chunk_id, hit)
            dense_scores[chunk_id] = max(dense_scores[chunk_id], hit.dense_score)
            bm25_scores_by_chunk[chunk_id] = max(bm25_scores_by_chunk[chunk_id], hit.bm25_score)
            channels[chunk_id].add(hit.retrieval_channel)

    fused = []
    for chunk_id, score in chunk_scores.items():
        source = best_source[chunk_id]
        fused.append(
            RetrievedSource(
                chunk=source.chunk,
                score=score,
                dense_score=dense_scores[chunk_id],
                bm25_score=bm25_scores_by_chunk[chunk_id],
                retrieval_channel="+".join(sorted(channels[chunk_id])),
            )
        )
    return sorted(fused, key=lambda item: item.score, reverse=True)


def rerank(query: str, hits: list[RetrievedSource], filters: dict[str, Any], *, top_n: int) -> list[RetrievedSource]:
    query_tokens = set(tokenize(query))
    reranked = []
    for hit in hits:
        content_tokens = set(tokenize(hit.chunk.content))
        overlap = len(query_tokens.intersection(content_tokens)) / max(1, len(query_tokens))
        metadata_boost = metadata_match_boost(hit.chunk, filters)
        rerank_score = min(1.0, 0.55 * overlap + 0.35 * hit.dense_score + 0.25 * hit.bm25_score + metadata_boost)
        reranked.append(
            RetrievedSource(
                chunk=hit.chunk,
                score=max(hit.score, rerank_score),
                dense_score=hit.dense_score,
                bm25_score=hit.bm25_score,
                rerank_score=rerank_score,
                retrieval_channel=f"{hit.retrieval_channel}+rerank",
            )
        )
    return sorted(reranked, key=lambda item: (item.rerank_score, item.score), reverse=True)[:top_n]


def metadata_match_boost(chunk: KnowledgeChunk, filters: dict[str, Any]) -> float:
    return 0.0


def matches_filter(chunk: KnowledgeChunk, filters: dict[str, Any]) -> bool:
    return True


def parse_filter_date(value: Any) -> date | None:
    if isinstance(value, date):
        return value
    if isinstance(value, str) and value:
        try:
            return date.fromisoformat(value)
        except ValueError:
            return None
    return None


def embed_text(text: str) -> list[float]:
    values = [0.0 for _ in range(EMBEDDING_DIMS)]
    for token in tokenize(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:2], "big") % EMBEDDING_DIMS
        sign = 1.0 if digest[2] % 2 == 0 else -1.0
        values[index] += sign
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]


def cosine(left: Iterable[float], right: Iterable[float]) -> float:
    left_values = list(left)
    right_values = list(right)
    return sum(a * b for a, b in zip(left_values, right_values))


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def tokenize(text: str) -> list[str]:
    normalized = normalize_text(text).lower()
    try:
        import jieba

        tokens = [token.strip() for token in jieba.lcut(normalized) if token.strip()]
    except Exception:
        tokens = re.findall(r"[\w\u4e00-\u9fff]+", normalized)
    expanded = []
    for token in tokens:
        expanded.append(token)
        if len(token) >= 4 and re.search(r"[\u4e00-\u9fff]", token):
            expanded.extend(token[index : index + 2] for index in range(len(token) - 1))
    return [token for token in expanded if token]


def build_seed_rag_engine(
    *,
    cache_store: CacheStore | None = None,
    vector_index: VectorIndex | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    reranker: Reranker | None = None,
    cache_ttl_seconds: int = 600,
) -> InMemoryRAGEngine:
    engine = InMemoryRAGEngine(
        cache_store=cache_store,
        vector_index=vector_index,
        embedding_provider=embedding_provider,
        reranker=reranker,
        cache_ttl_seconds=cache_ttl_seconds,
    )
    for document in seed_documents():
        engine.index_document(document)
    return engine


def seed_documents() -> list[KnowledgeDocument]:
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
