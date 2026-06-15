# Declaration QA Workflow

This is an isolated FastAPI + LangGraph implementation of the declaration
question-answering workflow. It does not import or mount the existing
AgentBase application.

Start it with:

```bash
python -m uvicorn declaration_qa.server:app --host 127.0.0.1 --port 8010 --reload
```

Start the asynchronous index worker when `DECLARATION_QA_INDEX_QUEUE_BACKEND=rq`:

```bash
python -m declaration_qa.worker
```

Initialize the isolated PostgreSQL tables with:

```bash
psql "$DECLARATION_QA_DATABASE_URL" -f sql/postgresql/003_declaration_qa_schema.sql
```

Or initialize through the service when `DECLARATION_QA_DATABASE_URL` is set:

```bash
curl -X POST http://127.0.0.1:8010/api/declaration-qa/admin/init-schema
```

Development mode can use in-memory business data when no database is configured.
When `DECLARATION_QA_DATABASE_URL` is set, application, status, and rejection
data are read from `declaration_qa.business_applications`,
`declaration_qa.business_application_statuses`, and
`declaration_qa.business_reject_infos`.

Production mode:

```bash
set DECLARATION_QA_ENV=production
```

In production the service fails fast if real dependencies are missing. It will
not silently use in-memory audit storage, in-memory cache, local hash
embeddings, deterministic answers, or heuristic reranking.

Required production environment variables:

- `DECLARATION_QA_DATABASE_URL`: isolated PostgreSQL URL for sessions, messages, audit logs, feedback, documents, chunks, and local business application/status/reject tables.
- `DECLARATION_QA_REDIS_URL`: Redis cache for retrieval/answer cache.
- `DECLARATION_QA_VECTOR_BACKEND=milvus`: enable Milvus dense vector search.
- `DECLARATION_QA_MILVUS_URI`, `DECLARATION_QA_MILVUS_TOKEN`, `DECLARATION_QA_MILVUS_COLLECTION`: Milvus connection.
- `DECLARATION_QA_AUTH_REQUIRED=true` plus `DECLARATION_QA_JWT_SECRET` or `DECLARATION_QA_INTERNAL_API_KEY`.
- `DECLARATION_QA_LLM_PROVIDER=openai`, `DECLARATION_QA_OPENAI_API_KEY`, `DECLARATION_QA_OPENAI_API_BASE`, `DECLARATION_QA_LLM_MODEL`.
- `DECLARATION_QA_EMBEDDING_PROVIDER=openai`, `DECLARATION_QA_EMBEDDING_MODEL`, `DECLARATION_QA_EMBEDDING_DIMENSIONS`.
- `DECLARATION_QA_RERANK_PROVIDER=http`, `DECLARATION_QA_RERANK_API_URL`, `DECLARATION_QA_RERANK_API_KEY`.
- `DECLARATION_QA_OCR_API_URL`: required for image uploads that need OCR.
- `DECLARATION_QA_OCR_MODEL`: optional model name sent to the OCR HTTP service.
- `DECLARATION_QA_INDEX_QUEUE_BACKEND=rq`, `DECLARATION_QA_INDEX_QUEUE_NAME`: Redis RQ queue for document indexing jobs.
- `DECLARATION_QA_UPLOAD_DIR`: shared upload staging directory. In container deployments the API and worker must mount the same directory.
- `DECLARATION_QA_SEED_KNOWLEDGE=false`: production must not answer from sample seed documents.

Core endpoints:

- `GET /api/declaration-qa/health`
- `POST /api/declaration-qa/chat`
- `POST /api/declaration-qa/chat/stream`
- `POST /api/declaration-qa/knowledge/documents`
- `POST /api/declaration-qa/knowledge/upload`
- `POST /api/declaration-qa/knowledge/reindex`
- `GET /api/declaration-qa/knowledge/documents`
- `POST /api/declaration-qa/knowledge/documents/{document_id}/publish`
- `POST /api/declaration-qa/knowledge/documents/{document_id}/unpublish`
- `POST /api/declaration-qa/knowledge/documents/{document_id}/rollout`
- `POST /api/declaration-qa/knowledge/documents/{document_id}/versions`
- `GET /api/declaration-qa/knowledge/documents/{left_document_id}/compare/{right_document_id}`
- `POST /api/declaration-qa/knowledge/documents/rollback`
- `GET /api/declaration-qa/index-jobs/{job_id}`
- `POST /api/declaration-qa/index-jobs/{job_id}/retry`
- `DELETE /api/declaration-qa/knowledge/documents/{document_id}`
- `GET /api/declaration-qa/audit/{audit_id}`
- `POST /api/declaration-qa/feedback`
- `POST /api/declaration-qa/evaluation/datasets`
- `GET /api/declaration-qa/evaluation/datasets`
- `POST /api/declaration-qa/evaluation/datasets/{dataset_id}/cases`
- `POST /api/declaration-qa/evaluation/datasets/{dataset_id}/runs`
- `GET /api/declaration-qa/evaluation/runs/{run_id}`
- `POST /api/declaration-qa/evaluation/run-items/{item_id}/annotation`

The workflow keeps declaration business state and RAG knowledge separate:

- status and reject questions run on the `ticket_augmented` chain: an application context is mandatory, real business status/reject records are queried first, and RAG only explains the verified business snapshot;
- policy/material/process questions run on the `knowledge_retrieval` chain and use RAG filters based on region, policy version, effective time, audit node, rollout, and status;
- `biz_type`, `source_type`, `material_code`, and `reject_code` are removed from knowledge/RAG metadata, filters, source payloads, and vector-index fields; similarly named business fields remain only in the real ticket snapshot;
- request `metadata_filter` accepts camelCase aliases such as `regionCode`, `policyVersion`, `auditNode`, `asOf`, and `rolloutKey`; removed metadata fields are ignored;
- BM25 and Milvus/vector hits are fused with RRF, reranked, and passed through evidence gates before an answer is generated;
- evidence decisions and retrieval diagnostics are written to responses and audit logs for traceability;
- `appVersion` is only used in answer post-processing for entry/action compatibility.
- asynchronous indexing moves upload parsing, embedding, Milvus upsert, PostgreSQL persistence, reindex, retry, and delete work out of the request path;
- knowledge management supports draft versions, publish/unpublish, rollout percentage, effective dates, version comparison, and rollback;
- evaluation datasets run through the real workflow and record recall@k, refusal accuracy, intent accuracy, answer keyword hit rate, run items, audit IDs, and human annotations.

Supported upload formats:

- text/Markdown/JSON/XML/CSV
- PDF through `pypdf`
- DOCX and XLSX through package XML parsing
- images through a configured OCR HTTP service
