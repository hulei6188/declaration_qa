# Declaration QA

Independent FastAPI + LangGraph declaration question-answering service with an
isolated static frontend, PostgreSQL schema, RQ worker, and Docker deployment
assets.

## Project Layout

- `declaration_qa/`: backend service and workflow code.
- `frontend/`: static admin/chat frontend for `/api/declaration-qa/*`.
- `sql/postgresql/003_declaration_qa_schema.sql`: isolated PostgreSQL schema.
- `tests/`: service and frontend regression tests.

## Local Development

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
```

Copy the example environment and adjust any production integration values:

```bash
copy .env.example .env
```

Start the API:

```bash
python -m uvicorn declaration_qa.server:app --host 127.0.0.1 --port 8010 --reload
```

Start the static frontend in another terminal:

```bash
python -m http.server 8020 -d frontend
```

When `DECLARATION_QA_INDEX_QUEUE_BACKEND=rq`, start the worker separately:

```bash
python -m declaration_qa.worker
```

Initialize PostgreSQL manually:

```bash
psql "$DECLARATION_QA_DATABASE_URL" -f sql/postgresql/003_declaration_qa_schema.sql
```

Or initialize through the service:

```bash
curl -X POST http://127.0.0.1:8010/api/declaration-qa/admin/init-schema
```

The schema also includes local business application, status, and rejection
tables used by status/reject intents; the QA workflow reads those tables
directly instead of calling declaration/application HTTP APIs.

## Production QA Design

The service is split into two explicit answer chains:

- `knowledge_retrieval`: policy, material, process, and FAQ questions go
  directly through RAG over the full knowledge corpus. Knowledge metadata is
  retained for display, auditing, and source traceability, but is not used as a
  retrieval filter; BM25 and Milvus/vector retrieval are fused with RRF and
  reranked.
- `ticket_augmented`: application progress, rejection, and correction
  questions must first resolve the user's real application record through the
  business adapter. The model receives a business snapshot from the application
  status/reject tables, then RAG retrieves process or material-rule evidence to
  explain the status. If the business snapshot is missing, the workflow refuses
  instead of inventing progress or rejection reasons.

Evidence checks require traceable sources for knowledge answers and a verified
business snapshot for ticket-augmented answers. Knowledge metadata is retained
for display, auditing, and source traceability, but it is not used as a RAG
retrieval filter. The response and audit log include `retrieval_chain`,
`business_snapshot`, `rag_status`, and `evidence_decision` so production issues
can be traced to business data, retrieval, rerank, or evidence gates.

## Docker

Compose only starts local infrastructure. Start the API, worker, and frontend
manually from your Python environment.

```bash
docker compose up -d
```

The compose stack exposes:

- PostgreSQL: `localhost:5433`
- Redis: `localhost:6380`
- Milvus: `localhost:19530`

## Core Endpoints

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

## Tests

Project policy in the source repository may restrict automatic test execution.
When you do want to run them manually:

```bash
python -m pytest
```
