# Declaration QA

政务申报 Agentic RAG 系统，支持办事指南问答、材料与流程检索、多模态证据、个人工单查询和知识库管理。

## 核心能力

- 单 Agent 自主理解问题、规划查询、调用知识和工单工具并直接生成答案。
- 复合问题由 Agent 在 `knowledge_search.queries` 或同轮多工具调用中自主拆分，独立工具并行执行。
- Milvus 向量检索与 Elasticsearch BM25，经 RRF 融合和 Qwen3-VL Rerank 重排。
- 支持 PDF、DOCX、PPTX、XLSX、Markdown 和图片解析，以及表格、OCR、图片摘要与图文检索。
- 工单列表、申请详情及完整状态历史与知识检索隔离。
- SSE 流式回答，前端按 Agent 轮次展示真实工具调用过程。
- PostgreSQL 持久化会话、Agent Run、消息、工具调用、证据和审计记录。

## 技术栈

| 模块 | 技术 |
| --- | --- |
| 后端 | Python 3.10+、FastAPI、Uvicorn、Pydantic、SQLAlchemy、Psycopg |
| Agent | LangGraph、LangChain、原生 Tool Calling、Pydantic Tool Schema |
| 模型 | DashScope Qwen3-VL、Qwen3-VL Embedding、Qwen3-VL Rerank、OCR |
| 检索 | Milvus、Elasticsearch BM25、RRF、Parent Context |
| 文档解析 | MinerU、LlamaParse、PyMuPDF、PyPDF、Calamine |
| 数据与任务 | PostgreSQL、Redis、RQ |
| 前端 | React 18、Vite 5、Lucide React、SSE |
| 可观测 | Langfuse、结构化审计日志 |
| 基础设施 | Docker Compose |

## 检索流程

```text
用户问题
  -> 单 Agent 决策
  -> Tool Registry（独立工具并行）
       -> knowledge_search
       -> ticket_list_query
       -> ticket_detail_query
  -> 工具结果写回同一 Agent 消息链
  -> 继续调用工具或直接多模态回答
```

`knowledge_search` 只接收 Agent 生成的 `queries`。同一轮多条查询经 Milvus、Elasticsearch、RRF 融合后统一 Rerank；多轮证据累计去重后回填同一 Agent。系统不再使用独立问题拆解器、Worker、Evidence Gate 或最终回答模型调用。

## 快速启动

要求：Windows、Conda `base`、Node.js、Docker Desktop。

首次安装：

```cmd
conda activate base
pip install -r requirements-dev.txt
pip install -e .
cd frontend
npm install
cd ..
copy .env.example .env
```

根据实际模型服务修改 `.env`，然后启动：

```cmd
start-docker.bat
start-app.bat
```

- 前端：http://127.0.0.1:5173
- API 文档：http://127.0.0.1:8010/docs

重启或关闭：

```cmd
restart-app.bat
stop-app.bat
stop-docker.bat
```

`stop-app.bat` 只关闭前后端和索引 Worker；`stop-docker.bat` 关闭基础设施但保留数据卷。

## 配置

完整配置见 [.env.example](.env.example)，常用配置包括：

```env
DECLARATION_QA_DATABASE_URL=postgresql+psycopg://agentbase:agentbase@localhost:5433/agentbase
DECLARATION_QA_REDIS_URL=redis://localhost:6380/1
DECLARATION_QA_VECTOR_BACKEND=milvus
DECLARATION_QA_MILVUS_URI=http://localhost:19530
DECLARATION_QA_KEYWORD_BACKEND=elasticsearch
DECLARATION_QA_ELASTICSEARCH_URL=http://localhost:9200
DECLARATION_QA_LLM_PROVIDER=dashscope
DECLARATION_QA_EMBEDDING_PROVIDER=qwen3-vl-embedding
DECLARATION_QA_RERANK_PROVIDER=dashscope
DECLARATION_QA_DOCUMENT_PARSER_PROVIDER=auto
DECLARATION_QA_INDEX_QUEUE_BACKEND=rq
```

本地轻量运行可将向量、关键词、Embedding、Rerank 和模型 Provider 切换为项目提供的内存或确定性实现。

## 目录

```text
src/declaration_qa/
  api/             FastAPI 与 SSE
  application/     LangGraph、单 Agent 和工具
  indexing/        文档解析、切分与异步入库
  rag/             检索、重排和模型适配
  search/          Milvus 与 Elasticsearch
  storage/         PostgreSQL、Redis 与审计
frontend/          React 前端
materials/         示例知识材料
sql/postgresql/    Schema、迁移和模拟数据
tests/             测试
```

## 测试

```cmd
python -m pytest
```

测试会读取当前 `.env`。连接真实外部服务时，结果可能受现有索引和数据影响。
