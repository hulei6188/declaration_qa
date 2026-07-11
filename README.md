0# Declaration QA

政务申报辅助问答系统，用来回答办事指南、申请材料、办理流程、办理条件、办理时限、咨询电话，以及工单进度、驳回原因、补正建议等问题。

核心设计：

- `Agent`：通过强 Tool Schema 完成知识查询消歧、拆分和改写，观察工具结果后直接生成最终答案。
- `RAGEngine`：按 metadata filter 和 `retrieval_plan` 检索，并做 section-aware backfill。
- `KnowledgeSearchTool`：校验 Agent 查询、注入脱敏冻结问题，并执行多路混合检索、RRF 和 rerank。

## 快速启动

Windows 使用 Conda `base` 环境，完成依赖安装和 `.env` 配置后，先启动 Docker 基础设施：

```cmd
start-docker.bat
```

再启动 API、异步入库 Worker 和前端：

```cmd
start-app.bat
```

应用启动脚本会打开 `http://127.0.0.1:5173`，不会隐式启动 Docker。

关闭 API、异步入库 Worker 和前端（保留 Docker 基础设施运行）：

```cmd
stop-app.bat
```

关闭 Docker 基础设施（保留数据库和索引数据卷）：

```cmd
stop-docker.bat
```

首次在 Conda `base` 环境安装依赖：

```cmd
conda activate base
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
python -m uvicorn declaration_qa.api.server.app:app --app-dir src --host 127.0.0.1 --port 8010 --reload
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
- `DECLARATION_QA_EMBEDDING_PROVIDER=qwen3-vl-embedding`
- `DECLARATION_QA_RERANK_PROVIDER=dashscope`
- `DECLARATION_QA_RERANK_MODEL=qwen3-vl-rerank`

## 关键配置

配置参考 `.env.example`。

常用环境变量：

- `DECLARATION_QA_DATABASE_URL`: PostgreSQL 连接地址。
- `DECLARATION_QA_REDIS_URL`: Redis 连接地址。
- `DECLARATION_QA_QUERY_IMAGE_OCR_ENABLED`: 聊天查询图片是否执行 OCR，默认开启。
- `DECLARATION_QA_QUERY_IMAGE_OCR_MAX_CHARS`: 查询图片 OCR 进入检索的最大字符数，默认 `4000`。
- `DECLARATION_QA_QUERY_IMAGE_OCR_TIMEOUT_SECONDS`: 查询图片 OCR 超时时间，默认 `30` 秒。
- `DECLARATION_QA_VECTOR_BACKEND`: `memory` 或 `milvus`。
- `DECLARATION_QA_MILVUS_URI` / `DECLARATION_QA_MILVUS_COLLECTION`
- `DECLARATION_QA_KEYWORD_BACKEND`: `memory` 或 `elasticsearch`。
- `DECLARATION_QA_ELASTICSEARCH_URL` / `DECLARATION_QA_ELASTICSEARCH_INDEX`
- `DECLARATION_QA_LLM_PROVIDER` / `DECLARATION_QA_LLM_MODEL`
- `DECLARATION_QA_EMBEDDING_PROVIDER` / `DECLARATION_QA_EMBEDDING_MODEL`
- `DECLARATION_QA_EMBEDDING_DIMENSIONS`: 使用 `qwen3-vl-embedding` 时建议设为 `1024`。
- `DECLARATION_QA_EMBEDDING_IMAGE_RES_LEVEL`: 图片向量分辨率等级，`0` 表示不传该参数。
- `DECLARATION_QA_RERANK_PROVIDER` / `DECLARATION_QA_RERANK_MODEL`
- 推荐图片检索场景使用 `DECLARATION_QA_RERANK_MODEL=qwen3-vl-rerank`。
- `DECLARATION_QA_RERANK_SCORE_THRESHOLD`
- `DECLARATION_QA_INDEX_QUEUE_BACKEND`: `inline` 或 `rq`。
- `DECLARATION_QA_ATTACHMENT_INDEXING_ENABLED`: 是否解析附件并生成派生 chunk。
- `DECLARATION_QA_ATTACHMENT_MAX_PER_DOCUMENT`: 单个办事指南最多解析的附件数，默认 `30`。
- `DECLARATION_QA_DOCUMENT_PARSER_PROVIDER`: 文档解析器，支持 `auto`、`docling`、`marker`、`llamaparse`、`native`。
- `DECLARATION_QA_DOCUMENT_PARSER_ORDER`: `auto` 模式下的尝试顺序，默认 `docling,marker,native`。
- `DECLARATION_QA_DOCUMENT_PARSER_FALLBACK_ENABLED`: 指定外部解析器失败时是否回退到 `native`。
- `DECLARATION_QA_DOCLING_DOCUMENT_TIMEOUT_SECONDS`: 单个 Docling PDF 的处理时限，默认 `3600` 秒。
- `DECLARATION_QA_DOCLING_ACCELERATOR_DEVICE`: Docling 加速设备，支持 `auto`、`cpu`、`cuda`、`mps`、`xpu`。
- `DECLARATION_QA_DOCLING_ACCELERATOR_NUM_THREADS`: Docling CPU 线程数，默认 `4`。
- `DECLARATION_QA_DOCLING_LAYOUT_BATCH_SIZE` / `DECLARATION_QA_DOCLING_TABLE_BATCH_SIZE` / `DECLARATION_QA_DOCLING_OCR_BATCH_SIZE`: Docling PDF pipeline 批大小，显存较小时建议从 `4` 或 `8` 起步。
- `DECLARATION_QA_INDEX_JOB_TIMEOUT_SECONDS`: RQ 整体索引任务时限，默认 `7200` 秒，必须大于单文档解析时限。
- `DECLARATION_QA_LLAMAPARSE_API_KEY`: 使用 `llamaparse` 时需要配置。
- `DECLARATION_QA_CHAT_IMAGE_DIR`: 聊天图片查询的临时落盘目录。

## 文档解析

入库解析层已改为可插拔解析器：

- `docling`：本地结构化解析 PDF、Word、PPT、Excel、图片等文档，适合默认生产方案。
- `marker`：本地 PDF/图片解析器，适合复杂版面和表格较多的 PDF。
- `llamaparse`：云端高精度解析器，适合数据允许出网且需要更强版面理解的场景。
- `native`：项目内置解析器，支持文本、Markdown、PDF、DOCX、CSV、XLS/XLSX、图片 OCR。

解析结果统一为 `ParsedDocument` 和 `ParsedBlock`。Docling 解析成功时，主文档不再使用项目内置 `parent_child` / `hierarchical` 切分，而是直接使用 Docling `HybridChunker` 生成主文档 chunk；前端上传时选择的切分策略只影响非 Docling 解析结果和部分附件文本兜底。

表格不会只压成纯文本。系统会为表格保留三种表达：

- `table_text`：自然语言化表格文本，用于 embedding、BM25 和 rerank。
- `table_json`：原生行列、单元格、跨行跨列、页码和 bbox 等结构，存入 metadata。
- `table_html`：前端证据详情展示使用。

入库时会为表格额外生成三类语义 chunk：

- `table_parent`：物理父块，保存整表上下文，不进入向量/关键词索引。
- `table_summary`：整表摘要，适合“这个表讲什么、需要哪些字段”类召回。
- `table_row` / `table_kv`：行级或字段-内容级 chunk，适合精确召回表单字段、材料要求、办理时限等。

命中 `table_summary/table_row/table_kv` 后，父子回填会带回整表上下文。回答生成时，模型接收自然语言化的表格内容和精简后的 `llm_table_context`，不会把完整 HTML/JSON 混进检索文本。

## 上下文工程

系统采用分层上下文记忆：

- 短期记忆：Redis 保存最近 5 轮消息和抽取式摘要，仅用于多轮省略和当前附件理解，不形成工单绑定。
- 长期记忆：公开知识问答审计落库后，通过后台线程异步抽取长期记忆，写入 PostgreSQL `declaration_qa.context_memories`，同时写入独立的 Milvus collection 和 Elasticsearch index；包含 `ticket_*` 工具结果的轮次不会写入长期记忆。
- 长期检索：回答生成前按用户隔离执行语义向量 + BM25 混合检索，并通过 `DECLARATION_QA_LONG_MEMORY_DISTANCE_THRESHOLD` 过滤向量距离。
- Prompt 约束：长期记忆只用于理解历史上下文；如与检索证据或业务数据冲突，以证据和业务数据为准。

首次启用长期记忆表：

```powershell
Get-Content sql\postgresql\007_declaration_qa_context_memories.sql | docker exec -i declaration-qa-postgres-1 psql -U agentbase -d agentbase -v ON_ERROR_STOP=1
```

常用配置：

- `DECLARATION_QA_CONTEXT_MEMORY_ENABLED`
- `DECLARATION_QA_SHORT_MEMORY_TTL_SECONDS`
- `DECLARATION_QA_LONG_MEMORY_TTL_SECONDS`
- `DECLARATION_QA_LONG_MEMORY_ASYNC_WORKERS`
- `DECLARATION_QA_LONG_MEMORY_TOP_K`
- `DECLARATION_QA_LONG_MEMORY_DISTANCE_THRESHOLD`
- `DECLARATION_QA_MEMORY_MILVUS_COLLECTION`
- `DECLARATION_QA_MEMORY_ELASTICSEARCH_INDEX`

## Langfuse

打开 Langfuse 可观测：

```env
DECLARATION_QA_LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=http://localhost:3000
```

本地 Langfuse 可用 Docker Compose 部署，Cloud 版则把 `LANGFUSE_BASE_URL` 改成 `https://cloud.langfuse.com`。系统会上报一次问答主 trace，并记录 Agent 决策、工具调用、知识检索、答案生成和 LangChain LLM 调用。`POST /api/declaration-qa/feedback` 会把用户反馈写成 Langfuse score。

## 问答链路

```text
ChatRequest
  -> context_build
  -> image OCR / query_preprocess
  -> agent_prepare (冻结脱敏原始知识 Query)
  -> agent_decide
  -> Tool Registry
       -> ticket_list_query
       -> ticket_detail_query
       -> knowledge_search
  -> agent_observe
  -> agent_decide (循环，最多 4 轮)
  -> agent_evidence_check
  -> agent final answer
  -> audit / SSE
```

Agent 使用模型原生 Tool Calling；模型不可用时使用确定性规则兜底。相同工具参数不会重复执行，达到最大轮数后进入证据门控。

Tool Registry 默认注册：

- `ticket_list_query`: 按可信 `user_id` 返回用户涉及的 `item_code` 列表。
- `ticket_detail_query`: 按可信 `user_id + item_code` 返回同事项的全部申请；每条申请包含主键 `application_id` 和完整 `status_history`。
- `knowledge_search`: 接收 Agent 规划的结构化 `queries`，调用 Milvus + Elasticsearch 混合检索。

`user_id` 只能由服务端认证上下文注入，不出现在模型可填写的工具参数中。Agent 自主确定 `item_code` 和相关 `application_id`，不要求用户绑定或选择工单。

工单和知识检索强隔离：业务工具结果不会进入知识查询规划、metadata filter、Embedding、BM25、RRF 或 Rerank。`knowledge_search` 必须在 Agent 观察任何业务工具结果前规划；执行器还会自动注入脱敏冻结的用户原始问题。

`knowledge_search` 可以与工单工具真正并发执行，不按工单意图或关键词设置调用限制；隔离的是数据与证据用途，不限制 Agent 的工具选择。若详情调用依赖同轮列表结果，执行器会先等待列表完成，再执行详情。

## RAG 引擎

当前有两个引擎：

- `InMemoryRAGEngine`: 本地内存检索，适合开发和单测。
- `ExternalIndexRAGEngine`: PostgreSQL + Milvus + Elasticsearch，适合真实数据。

检索流程：

1. 标准化 metadata filter。
2. 多 query 向量检索 + BM25 检索。
3. RRF 融合候选。
4. Rerank 重排。
5. 回填 parent，并按父块和视觉资产去重。
6. 输出 `sources`、`evidence_text` 和 `rag_status`。

## 多模态 RAG

入库侧：

- Markdown 正文按层级/父子策略切分。
- PDF、DOCX、图片、表格类附件会解析为文本、表格字段和图片资产。
- PDF 页面会渲染为图片资产，DOCX 内嵌图片和原始图片附件会登记到 `declaration_qa.assets`。
- 图片资产会同时执行 OCR 和 VL 视觉理解：OCR 保留字段、号码等原文，`qwen3-vl-plus` 生成面向检索的 `visual_summary`。
- 每张图片只生成一个 `attachment_image` / `page_image` 派生 chunk，避免 OCR 与摘要形成重复证据；chunk metadata 包含 `ocr_text`、`visual_summary`、`asset_url` 和摘要执行状态。
- 使用 `qwen3-vl-embedding` 时，文本和图片会写入同一个 Milvus collection。

视觉摘要配置：

- `DECLARATION_QA_IMAGE_SUMMARY_ENABLED=true`
- `DECLARATION_QA_IMAGE_SUMMARY_MODEL=qwen3-vl-plus`
- `DECLARATION_QA_IMAGE_SUMMARY_API_URL` 和 `DECLARATION_QA_IMAGE_SUMMARY_API_KEY` 留空时复用 OCR 的 DashScope 地址和密钥。
- VL 摘要失败不会中断文档入库，失败原因会记录在 chunk metadata 和 Langfuse 的 `index.image_summary` 节点中。

检索侧：

```text
文本 query:
  Agent queries -> knowledge_search -> Milvus dense + Elasticsearch BM25

图片 + 文本 query:
  文本部分 -> Milvus dense + Elasticsearch BM25
  图片+文本部分 -> qwen3-vl-embedding -> Milvus dense
  两路结果 -> merge / qwen3-vl-rerank / backfill

纯图片 query:
  图片 embedding -> Milvus dense
  查询图片 OCR -> 文本 embedding + Elasticsearch BM25
  视觉和 OCR 文本候选融合 -> qwen3-vl-rerank 二排
  OCR 失败时自动降级为纯视觉检索
```

最终答案由同一个 Tool-Calling Agent 基于工具观察直接生成。工具返回后先执行 Evidence Gate：证据不足才回到 Agent 继续选工具，证据齐全则直接执行唯一一次最终回答，不先生成候选答案。工具规划轮次保持文本输入；如果配置的 Agent 模型名包含 `vl`（或显式开启 `DECLARATION_QA_AGENT_VL_ENABLED`）且存在有效图片，最终轮会直接接收用户原图和最多 3 张高分知识证据原图。图片数量、分数和体积可通过 `DECLARATION_QA_AGENT_VL_MAX_EVIDENCE_IMAGES`、`DECLARATION_QA_AGENT_VL_MIN_IMAGE_SCORE` 和 `DECLARATION_QA_AGENT_VL_MAX_IMAGE_BYTES` 调整。用户原图只用于理解问题，公开办理结论仍必须由知识库证据支持。

## API

- `GET /api/declaration-qa/health`
- `POST /api/declaration-qa/chat`
- `POST /api/declaration-qa/chat/stream`
- `GET /api/declaration-qa/sessions`
- `GET /api/declaration-qa/sessions/{session_id}`
- `DELETE /api/declaration-qa/sessions/{session_id}`
- `GET /api/declaration-qa/audit/{audit_id}`
- `GET /api/declaration-qa/agent-runs/{run_id}`
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
- `agent_prepared`
- `agent_decision`
- `tool_executed`
- `agent_observation`
- `evidence`
- `sources`
- `token`
- `audit_logged`
- `done`

## 知识入库

Agentic 工单工具上线前先执行数据库迁移：

```powershell
Get-Content sql\postgresql\008_declaration_qa_agentic_ticket_history.sql | docker exec -i declaration-qa-postgres-1 psql -U agentbase -d agentbase -v ON_ERROR_STOP=1
```

迁移会把 `item_code/item_name` 提升为申请主表字段，并为状态流水增加 `sequence_no`、`from_status`、原因、材料、上游事件 ID 和实际发生时间。数据库触发器会自动分配顺序、关闭上一条 current、按申请内 `source_event_id` 去重并同步申请主表当前状态；上游仍必须对每次变更执行 INSERT，不能覆盖历史行。

Agent 消息和工具调用持久化迁移：

```powershell
Get-Content sql\postgresql\009_declaration_qa_agent_message_history.sql | docker exec -i declaration-qa-postgres-1 psql -U agentbase -d agentbase -v ON_ERROR_STOP=1
```

迁移后，聊天响应中的 `run_id` 与 `audit_id` 相同。使用 `GET /api/declaration-qa/agent-runs/{run_id}` 可按当前用户读取完整的 system/user/assistant/tool 消息、工具调用生命周期、多模态消息组成部分和证据引用；普通会话接口仍只返回可见的用户消息和最终回答。

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

重新索引前清空本地 Docker 知识库数据：

```powershell
@'
from declaration_qa.core.config import get_settings
from declaration_qa.search.vector.milvus import MilvusVectorIndex
from declaration_qa.search.keyword.elasticsearch_keyword_index import ElasticsearchKeywordIndex

settings = get_settings()
MilvusVectorIndex(
    uri=settings.milvus_uri,
    token=settings.milvus_token,
    collection_name=settings.milvus_collection,
    dimension=settings.embedding_dimensions,
    allow_fallback=False,
).clear()
ElasticsearchKeywordIndex(
    url=settings.elasticsearch_url,
    api_key=settings.elasticsearch_api_key,
    index_name=settings.elasticsearch_index,
    timeout_seconds=settings.external_request_timeout_seconds,
).clear()
'@ | python -

docker exec declaration-qa-postgres-1 psql -U agentbase -d agentbase -v ON_ERROR_STOP=1 -c "TRUNCATE TABLE declaration_qa.assets, declaration_qa.chunks, declaration_qa.index_jobs, declaration_qa.documents RESTART IDENTITY CASCADE;"
```

该操作只清空知识文档、chunk、附件资产和索引任务，不清空业务表和 `knowledge_bases`。

## 测试

```powershell
python -m pytest
```

常用测试：

```powershell
python -m pytest tests\declaration_qa_workflow_test.py -q
python -m pytest tests\declaration_qa_agentic_rag_test.py -q
python -m pytest tests\declaration_qa_production_test.py -q
python -m pytest tests\declaration_qa_vector_index_test.py -q
python -m pytest tests\declaration_qa_chat_image_test.py tests\declaration_qa_qwen3_vl_embedding_test.py -q
```

如果 `.env` 指向真实 PostgreSQL / Milvus / Elasticsearch，测试结果会受真实配置影响。需要纯内存测试时，临时覆盖相关环境变量。

## 目录

```text
src/declaration_qa/
  api/                  FastAPI 路由和请求响应模型
  application/          LangGraph workflow 和节点
    tools/              Tool Registry、工单工具和知识检索工具
  core/                 配置、领域模型、业务适配
  indexing/             知识入库、任务队列、worker
  rag/                  RAG 引擎和模型适配
  search/               Milvus / Elasticsearch / memory 索引
  storage/              PostgreSQL repository、审计、缓存
frontend/               React 前端
materials/              知识材料
sql/postgresql/         PostgreSQL schema 和种子数据
tests/                  测试
```

常改文件：

- workflow 节点：`src/declaration_qa/application/*nodes.py`
- Agent 工具规划与最终生成：`src/declaration_qa/application/agent.py`
- Knowledge Tool Schema：`src/declaration_qa/application/tools/knowledge.py`
- 检索计划：`src/declaration_qa/rag/retrieval_plan.py`
- RAG 引擎：`src/declaration_qa/rag/engine/`
- API 路由：`src/declaration_qa/api/server/routes/`

## 开发边界

- 默认不保留向后兼容。重构、接口改名、参数调整或行为修正时，直接更新所有调用方、测试和文档。
- 不新增旧入口、旧参数、兼容分支或兼容 shim，除非用户明确要求。
- Agent 只能通过 `knowledge_search.queries` 规划公开知识查询，不能传 metadata filter 或召回数量。
- 工单工具结果不得用于生成或修改知识检索 query。
- `biz_type` 不参与知识库过滤。
- 工单状态、退回原因、补正信息必须来自业务数据。
- 最终回答必须基于证据和工单事实。
