"""FastAPI 请求和响应模型定义。"""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, Field


class ApplyContext(BaseModel):
    """办件上下文请求模型。"""
    apply_id: str = ""
    biz_type: str | None = None
    region_code: str | None = None
    status: str | None = None
    current_node: str | None = None
    node_name: str | None = None
    reject_code: str | None = None
    reject_reason: str | None = None
    material_code: str | None = None
    material_name: str | None = None
    audit_node: str | None = None


class TicketSummary(BaseModel):
    """用户办件列表中的摘要模型。"""
    ticket_id: str
    biz_type: str = ""
    region_code: str = ""
    status: str = ""
    current_node: str = ""
    node_name: str = ""
    title: str = ""
    updated_at: str = ""


class ChatRequest(BaseModel):
    """申报问答请求模型。"""
    query: str = Field(min_length=1, max_length=1000)
    user_id: int = Field(default=1, ge=1)
    session_id: str | None = None
    region_code: str = Field(default="340000", min_length=2, max_length=12)
    app_version: str = Field(default="", max_length=40)
    policy_version: str = Field(default="2025", max_length=80)
    biz_type: str | None = Field(default=None, max_length=80)
    apply_id: str | None = Field(default=None, max_length=80)
    apply_context: ApplyContext | None = None
    metadata_filter: dict[str, Any] = Field(default_factory=dict)


class SourcePayload(BaseModel):
    """回答引用来源的响应模型。"""
    chunk_id: str
    doc_id: str
    title: str
    snippet: str
    score: float
    dense_score: float = 0.0
    bm25_score: float = 0.0
    rerank_score: float = 0.0
    retrieval_channel: str
    region_code: str
    policy_version: str
    region_key: str = ""
    region_name: str = ""
    catalog_name: str = ""
    catalog_subitem_name: str = ""
    item_code: str = ""
    item_name: str = ""
    item_type: str = ""
    theme_code: str = ""
    service_object: str = ""
    source_version: str = ""
    source_url: str = ""
    audit_node: str | None = None
    effective_start_time: str | None = None
    effective_end_time: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SuggestedAction(BaseModel):
    """回答后续建议动作模型。"""
    type: str
    label: str
    payload: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    """申报问答响应模型。"""
    session_id: str
    audit_id: str
    intent: str
    route: str = ""
    sub_intent: str = ""
    route_decision: dict[str, Any] = Field(default_factory=dict)
    need_ticket_selection: bool = False
    ticket_selection_reason: str = ""
    tickets: list[TicketSummary] = Field(default_factory=list)
    ticket_context: dict[str, Any] = Field(default_factory=dict)
    retrieval_chain: str = ""
    answer: str
    refused: bool = False
    refuse_reason: str | None = None
    sources: list[SourcePayload] = Field(default_factory=list)
    business_snapshot: dict[str, Any] = Field(default_factory=dict)
    rag_status: dict[str, Any] = Field(default_factory=dict)
    evidence_decision: dict[str, Any] = Field(default_factory=dict)
    suggested_actions: list[SuggestedAction] = Field(default_factory=list)
    events: list[dict[str, Any]] = Field(default_factory=list)


class KnowledgeDocumentRequest(BaseModel):
    """文本知识文档入库请求模型。"""
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1)
    region_code: str = Field(default="340000", min_length=2, max_length=12)
    policy_version: str = Field(default="2025", max_length=80)
    chunking_strategy: str = Field(default="parent_child", max_length=40)
    source_url: str = Field(default="", max_length=500)
    audit_node: str = Field(default="", max_length=80)
    status: str = Field(default="published", max_length=40)
    effective_start_time: date | None = None
    effective_end_time: date | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class KnowledgeDocumentResponse(BaseModel):
    """知识文档入库响应模型。"""
    document_id: str
    chunks_indexed: int
    job_id: str | None = None


class KnowledgeUploadResponse(KnowledgeDocumentResponse):
    """单文件知识上传响应模型。"""
    filename: str
    content_type: str
    chunking_strategy: str = "parent_child"


class KnowledgeBatchUploadResponse(BaseModel):
    """批量知识上传响应模型。"""
    total_files: int
    documents: list[KnowledgeUploadResponse]


class ReindexResponse(BaseModel):
    """知识库重建索引响应模型。"""
    job_id: str
    documents_indexed: int
    chunks_indexed: int


class IndexJobResponse(BaseModel):
    """索引任务状态响应模型。"""
    job_uid: str
    job_type: str
    status: str
    progress: int
    message: str = ""
    error_message: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)


class RetryIndexJobResponse(IndexJobResponse):
    """索引任务重试响应模型。"""
    retried: bool = True


class DeleteDocumentResponse(BaseModel):
    """知识文档删除响应模型。"""
    document_id: str
    deleted: bool
    job_id: str | None = None


class KnowledgeDocumentAdminRecord(BaseModel):
    """知识文档后台管理记录模型。"""
    document_id: str
    title: str
    filename: str = ""
    region_code: str
    policy_version: str
    audit_node: str = ""
    status: str
    version_group_uid: str = ""
    version_no: int = 1
    rollout_percent: int = 100
    effective_start_time: str | None = None
    effective_end_time: str | None = None
    chunk_count: int = 0
    change_note: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""


class DocumentStatusResponse(BaseModel):
    """知识文档状态变更响应模型。"""
    document_id: str
    status: str
    job_id: str | None = None


class DocumentRolloutRequest(BaseModel):
    """知识文档灰度比例调整请求模型。"""
    rollout_percent: int = Field(ge=0, le=100)
    effective_start_time: date | None = None
    effective_end_time: date | None = None


class DocumentRolloutResponse(BaseModel):
    """知识文档灰度比例调整响应模型。"""
    document_id: str
    rollout_percent: int
    effective_start_time: str | None = None
    effective_end_time: str | None = None


class DocumentVersionRequest(BaseModel):
    """知识文档新版本创建请求模型。"""
    title: str | None = Field(default=None, max_length=200)
    content: str | None = None
    policy_version: str | None = Field(default=None, max_length=80)
    change_note: str = Field(default="", max_length=1000)


class DocumentVersionResponse(BaseModel):
    """知识文档版本创建响应模型。"""
    document_id: str
    version_group_uid: str
    version_no: int
    status: str
    job_id: str | None = None


class DocumentCompareResponse(BaseModel):
    """知识文档版本差异响应模型。"""
    left_document_id: str
    right_document_id: str
    added: list[str] = Field(default_factory=list)
    removed: list[str] = Field(default_factory=list)
    changed: bool = False


class DocumentRollbackRequest(BaseModel):
    """知识文档回滚请求模型。"""
    target_document_id: str
    change_note: str = Field(default="rollback", max_length=1000)


class SessionSummary(BaseModel):
    """历史会话列表摘要模型。"""
    session_id: str
    user_id: int
    region_code: str = ""
    biz_type: str | None = None
    apply_id: str = ""
    title: str = ""
    message_count: int = 0
    updated_at: str = ""


class SessionDetail(SessionSummary):
    """历史会话详情模型。"""
    messages: list[dict[str, Any]] = Field(default_factory=list)


class FeedbackRequest(BaseModel):
    """回答反馈提交请求模型。"""
    audit_id: str
    rating: str = Field(pattern="^(up|down)$")
    comment: str = Field(default="", max_length=1000)


class EvaluationCaseRequest(BaseModel):
    """评测用例创建请求模型。"""
    query: str = Field(min_length=1, max_length=1000)
    user_id: int = Field(default=1, ge=1)
    region_code: str = Field(default="340000", min_length=2, max_length=12)
    policy_version: str = Field(default="2025", max_length=80)
    biz_type: str | None = Field(default=None, max_length=80)
    apply_id: str | None = Field(default=None, max_length=80)
    apply_context: ApplyContext | None = None
    expected_intent: str = Field(default="", max_length=40)
    expected_refused: bool | None = None
    expected_doc_ids: list[str] = Field(default_factory=list)
    expected_answer_keywords: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class EvaluationDatasetRequest(BaseModel):
    """评测数据集创建请求模型。"""
    name: str = Field(min_length=1, max_length=160)
    description: str = Field(default="", max_length=1000)
    cases: list[EvaluationCaseRequest] = Field(default_factory=list)


class EvaluationCaseResponse(BaseModel):
    """评测用例响应模型。"""
    case_id: str
    dataset_id: str
    query: str
    expected_intent: str = ""
    expected_refused: bool | None = None
    expected_doc_ids: list[str] = Field(default_factory=list)
    expected_answer_keywords: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    created_at: str = ""


class EvaluationDatasetResponse(BaseModel):
    """评测数据集响应模型。"""
    dataset_id: str
    name: str
    description: str = ""
    case_count: int = 0
    created_at: str = ""


class EvaluationRunRequest(BaseModel):
    """评测运行请求模型。"""
    note: str = Field(default="", max_length=1000)
    top_k: int = Field(default=4, ge=1, le=20)


class EvaluationAnnotationRequest(BaseModel):
    """评测结果标注请求模型。"""
    label: str = Field(pattern="^(pass|fail|needs_review)$")
    comment: str = Field(default="", max_length=1000)


class EvaluationRunItemResponse(BaseModel):
    """评测运行明细响应模型。"""
    item_id: str
    case_id: str
    query: str
    expected_intent: str = ""
    actual_intent: str = ""
    expected_refused: bool | None = None
    actual_refused: bool = False
    expected_doc_ids: list[str] = Field(default_factory=list)
    retrieved_doc_ids: list[str] = Field(default_factory=list)
    expected_answer_keywords: list[str] = Field(default_factory=list)
    answer: str = ""
    audit_id: str = ""
    recall_hit: bool | None = None
    refusal_match: bool | None = None
    intent_match: bool | None = None
    keyword_hit: bool | None = None
    annotation_label: str = ""
    annotation_comment: str = ""


class EvaluationRunResponse(BaseModel):
    """评测运行汇总响应模型。"""
    run_id: str
    dataset_id: str
    status: str
    metrics: dict[str, Any] = Field(default_factory=dict)
    items: list[EvaluationRunItemResponse] = Field(default_factory=list)
    created_at: str = ""
    completed_at: str = ""
