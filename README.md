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
