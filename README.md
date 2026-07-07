# Declaration QA

政务申报辅助问答系统，用来回答办事指南、申请材料、办理流程、办理条件、办理时限、咨询电话，以及工单进度、驳回原因、补正建议等问题。

核心设计：

- `QuestionResolver`：拆问题，判断是否需要工单上下文，输出 `retrieval_plan`。
- `QueryRewriter`：只根据 `retrieval_plan` 生成 1 到 3 条检索 query。
- `RAGEngine`：按 metadata filter 和 `retrieval_plan` 检索，并做 section-aware backfill。
- `AnswerGenerator`：只基于证据和业务事实回答。

## 快速启动

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements-dev.txt
pip install -e .
copy .env.example .env
```

启动基础设施：

```powershell
docker compose up -d
```

启动 API：

```powershell
python -m uvicorn declaration_qa.api.server.app:app --host 127.0.0.1 --port 8000 --reload
```

启动前端：

```powershell
cd frontend
npm install
npm run dev
```

异步入库 worker：

```powershell
python -m declaration_qa.indexing.worker
```

## 运行模式

本地轻量模式：

- `DECLARATION_QA_VECTOR_BACKEND=memory`
- `DECLARATION_QA_KEYWORD_BACKEND=memory`
- `DECLARATION_QA_LLM_PROVIDER=deterministic`
- `DECLARATION_QA_EMBEDDING_PROVIDER=hashing`
- `DECLARATION_QA_RERANK_PROVIDER=heuristic`

外部索引模式：

- `DECLARATION_QA_VECTOR_BACKEND=milvus`
- `DECLARATION_QA_KEYWORD_BACKEND=elasticsearch`
- `DECLARATION_QA_LLM_PROVIDER=openai` 或 `dashscope`
- `DECLARATION_QA_EMBEDDING_PROVIDER=openai` 或 `dashscope`
- `DECLARATION_QA_RERANK_PROVIDER=dashscope`

## 关键配置

配置参考 `.env.example`。

常用环境变量：

- `DECLARATION_QA_DATABASE_URL`: PostgreSQL 连接地址。
- `DECLARATION_QA_REDIS_URL`: Redis 连接地址。
- `DECLARATION_QA_VECTOR_BACKEND`: `memory` 或 `milvus`。
- `DECLARATION_QA_MILVUS_URI` / `DECLARATION_QA_MILVUS_COLLECTION`
- `DECLARATION_QA_KEYWORD_BACKEND`: `memory` 或 `elasticsearch`。
- `DECLARATION_QA_ELASTICSEARCH_URL` / `DECLARATION_QA_ELASTICSEARCH_INDEX`
- `DECLARATION_QA_LLM_PROVIDER` / `DECLARATION_QA_LLM_MODEL`
- `DECLARATION_QA_QUESTION_RESOLVER_MODEL`
- `DECLARATION_QA_QUERY_REWRITER_MODEL`
- `DECLARATION_QA_EMBEDDING_PROVIDER` / `DECLARATION_QA_EMBEDDING_MODEL`
- `DECLARATION_QA_RERANK_PROVIDER` / `DECLARATION_QA_RERANK_MODEL`
- `DECLARATION_QA_RERANK_SCORE_THRESHOLD`
- `DECLARATION_QA_INDEX_QUEUE_BACKEND`: `inline` 或 `rq`。

## Langfuse

打开 Langfuse 可观测：

```env
DECLARATION_QA_LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=http://localhost:3000
```

本地 Langfuse 可用 Docker Compose 部署，Cloud 版则把 `LANGFUSE_BASE_URL` 改成 `https://cloud.langfuse.com`。系统会上报一次问答主 trace，并记录 `QuestionResolver`、`QueryRewriter`、RAG 检索、答案生成和 LangChain LLM 调用。`POST /api/declaration-qa/feedback` 会把用户反馈写成 Langfuse score。

## 问答链路

```text
ChatRequest
  -> context_build
  -> query_preprocess
  -> QuestionResolver
  -> prepare_task
  -> ticket selection / ticket context
  -> metadata filter
  -> QueryRewriter
  -> RAG retrieval
  -> evidence gate
  -> answer generation
  -> audit / SSE
```

多问题会拆成多个子任务独立检索，最后共享证据池汇总回答。

`retrieval_plan` 主要字段：

- `need_complete_sections`: 需要完整回填的结构化章节，目前支持 `申请材料`、`办理流程`。
- `metadata_needs`: 需要满足的元数据范围，如 `item_scope`、`ticket_scope`。
- `ticket_fact_type`: `status`、`rejection`、`correction`、`none`。

## RAG 引擎

当前有两个引擎：

- `InMemoryRAGEngine`: 本地内存检索，适合开发和单测。
- `ExternalIndexRAGEngine`: PostgreSQL + Milvus + Elasticsearch，适合真实数据。

检索流程：

1. 标准化 metadata filter。
2. 多 query 向量检索 + BM25 检索。
3. RRF 融合候选。
4. Rerank 重排。
5. 按 `retrieval_plan` 回填 parent 或同章节 sibling。
6. 输出 `sources`、`evidence_text` 和 `rag_status`。

## API

- `GET /api/declaration-qa/health`
- `POST /api/declaration-qa/chat`
- `POST /api/declaration-qa/chat/stream`
- `GET /api/declaration-qa/tickets`
- `GET /api/declaration-qa/sessions`
- `GET /api/declaration-qa/sessions/{session_id}`
- `DELETE /api/declaration-qa/sessions/{session_id}`
- `GET /api/declaration-qa/audit/{audit_id}`
- `POST /api/declaration-qa/feedback`
- `POST /api/declaration-qa/knowledge/upload`
- `POST /api/declaration-qa/knowledge/uploads`
- `GET /api/declaration-qa/knowledge/documents`
- `POST /api/declaration-qa/knowledge/reindex`
- `POST /api/declaration-qa/evaluation/datasets`
- `POST /api/declaration-qa/evaluation/datasets/{dataset_id}/runs`

SSE 常见事件：

- `run_started`
- `context_built`
- `query_preprocessed`
- `question_resolution`
- `ticket_selected`
- `knowledge_filter`
- `query_rewrite`
- `rag_status`
- `sources`
- `token`
- `audit_logged`
- `done`

## 知识入库

单文件入库：

```text
POST /api/declaration-qa/knowledge/upload
```

批量入库：

```text
POST /api/declaration-qa/knowledge/uploads
```

生产建议使用：

```env
DECLARATION_QA_INDEX_QUEUE_BACKEND=rq
```

worker 会执行解析、元数据提取、层级 chunk、Embedding、索引写入和原子发布。问答默认只召回 `doc_status=published` 的文档。

## 测试

```powershell
python -m pytest
```

常用测试：

```powershell
python -m pytest tests/declaration_qa_workflow_test.py -q
python -m pytest tests/declaration_qa_production_test.py -q
python -m pytest tests/declaration_qa_vector_index_test.py -q
```

如果 `.env` 指向真实 PostgreSQL / Milvus / Elasticsearch，测试结果会受真实配置影响。需要纯内存测试时，临时覆盖相关环境变量。

## 目录

```text
src/declaration_qa/
  api/                  FastAPI 路由和请求响应模型
  application/          LangGraph workflow 和节点
  core/                 配置、领域模型、业务适配
  indexing/             知识入库、任务队列、worker
  rag/                  RAG 引擎、QuestionResolver、QueryRewriter、模型适配
  search/               Milvus / Elasticsearch / memory 索引
  storage/              PostgreSQL repository、审计、缓存
frontend/               React 前端
materials/              知识材料
sql/postgresql/         PostgreSQL schema 和种子数据
tests/                  测试
```

常改文件：

- workflow 节点：`src/declaration_qa/application/*nodes.py`
- 问题解析：`src/declaration_qa/rag/question_resolver/`
- query 改写：`src/declaration_qa/rag/query_rewriter/`
- 检索计划：`src/declaration_qa/rag/retrieval_plan.py`
- RAG 引擎：`src/declaration_qa/rag/engine/`
- API 路由：`src/declaration_qa/api/server/routes/`

## 开发边界

- 默认不保留向后兼容。重构、接口改名、参数调整或行为修正时，直接更新所有调用方、测试和文档。
- 不新增旧入口、旧参数、兼容分支或兼容 shim，除非用户明确要求。
- `QuestionResolver` 不生成答案。
- `QueryRewriter` 不生成 metadata filter，也不决定回填策略。
- `biz_type` 不参与知识库过滤。
- 工单状态、退回原因、补正信息必须来自业务数据。
- 最终回答必须基于证据和工单事实。
