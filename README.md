# Gov Service Agent

面向政务服务场景的 Agentic RAG 系统。系统可以检索办事指南、材料规则和政策依据，也可以通过工具查询办件、预约、缴费、证照和送达状态，并在写操作前要求人工确认。

## 核心能力

- 单 Agent 工具调用与多轮对话
- Milvus 向量检索 + Elasticsearch 关键词检索 + RRF 融合与重排
- PDF、Office、Markdown、图片等知识入库与结构化切分
- 预约、办件、缴费、回执、证照、送达等业务查询工具
- 预约和取消操作的人机确认
- PostgreSQL 会话、审计、Agent Run 和工具调用记录
- Langfuse Trace、用户反馈、Ragas 和 Agent 回归评测
- 失败样本回流、版本化数据集和发布门禁

## 架构

```text
React Frontend
      |
FastAPI + LangGraph Agent
      |-- Knowledge Search --> Milvus + Elasticsearch + Reranker
      |-- Business Tools  --> PostgreSQL business_* tables
      |-- Memory          --> PostgreSQL Checkpointer / Store
      |-- Observability   --> Langfuse
      `-- Evaluation      --> Eval PostgreSQL + RQ Worker
```

## 技术栈

| 层级 | 技术 |
| --- | --- |
| 前端 | React 18、Vite、Lucide、React Markdown |
| API 与 Agent | FastAPI、Pydantic、LangGraph、LangChain |
| 模型 | DeepSeek、Qwen3-VL、DashScope Embedding / Rerank |
| 检索 | Milvus、Elasticsearch、RRF、父子上下文扩展 |
| 数据与任务 | PostgreSQL、Redis、RQ、SQLAlchemy |
| 文档处理 | MinerU、Chonkie、PyMuPDF、Pypdf、python-calamine |
| 可观测与评测 | Langfuse、Ragas、规则评分、独立模型裁判 |
| 测试 | Pytest、Playwright |

## 快速开始

### 1. 准备环境

- Python 3.10+
- Conda 环境 `langchain-agent`
- Node.js 20+
- Docker Desktop

```powershell
conda activate langchain-agent
pip install -e .

cd frontend
npm install
cd ..

Copy-Item .env.example .env
```

在 `.env` 中至少配置模型、Embedding 和 Rerank 所需的 API Key。完整配置项及默认值见 [.env.example](.env.example)。

### 2. 启动基础设施

```powershell
docker compose up -d postgres postgres-eval redis milvus elasticsearch
```

首次运行时初始化主库和评测库：

```powershell
Get-Content .\sql\postgresql\declaration_qa_postgresql_all.sql -Raw |
  docker compose exec -T postgres psql -v ON_ERROR_STOP=1 -U agentbase -d agentbase

Get-Content .\sql\postgresql\declaration_qa_postgresql_all.sql -Raw |
  docker compose exec -T postgres-eval psql -v ON_ERROR_STOP=1 -U agentbase -d agentbase_eval
```

已经执行过此前迁移的主库，新增评测闭环时执行：

```powershell
Get-Content .\sql\postgresql\013_agent_evaluation.sql -Raw |
  docker compose exec -T postgres psql -v ON_ERROR_STOP=1 -U agentbase -d agentbase

Get-Content .\sql\postgresql\014_multiturn_form_filling.sql -Raw |
  docker compose exec -T postgres psql -v ON_ERROR_STOP=1 -U agentbase -d agentbase
```

### 3. 启动应用

```powershell
.\restart-app.bat
```

启动后可访问：

| 服务 | 地址 |
| --- | --- |
| 前端 | <http://127.0.0.1:5173> |
| API / Swagger | <http://127.0.0.1:8010/docs> |
| 健康检查 | <http://127.0.0.1:8010/api/declaration-qa/health> |
| Langfuse（如已独立部署） | <http://127.0.0.1:3000> |

也可以单独启动组件：

```powershell
.\restart-app.bat api
.\restart-app.bat frontend
.\restart-app.bat worker
.\restart-app.bat eval-worker
.\restart-app.bat mineru
```

## 关键配置

开发环境可以使用内存检索和内联队列；生产环境应使用以下配置：

```env
GOV_SERVICE_AGENT_ENV=production
GOV_SERVICE_AGENT_ALLOW_IN_MEMORY_FALLBACKS=false

GOV_SERVICE_AGENT_VECTOR_BACKEND=milvus
GOV_SERVICE_AGENT_KEYWORD_BACKEND=elasticsearch
GOV_SERVICE_AGENT_INDEX_QUEUE_BACKEND=rq

GOV_SERVICE_AGENT_EVALUATION_ENABLED=true
GOV_SERVICE_AGENT_EVAL_DATABASE_URL=postgresql+psycopg://eval_user:password@eval-db/gov_service_agent_eval
GOV_SERVICE_AGENT_EVAL_QUEUE_BACKEND=rq
GOV_SERVICE_AGENT_EVAL_PII_SALT=replace-with-a-long-random-secret
GOV_SERVICE_AGENT_EVAL_JUDGE_ENABLED=true
GOV_SERVICE_AGENT_EVAL_JUDGE_MODEL=independent-judge-model
GOV_SERVICE_AGENT_EVAL_JUDGE_API_KEY=

GOV_SERVICE_AGENT_ONLINE_EVAL_ENABLED=true
GOV_SERVICE_AGENT_ONLINE_EVAL_INTERVAL_MINUTES=15
GOV_SERVICE_AGENT_ONLINE_EVAL_SAMPLE_RATE=0.05
GOV_SERVICE_AGENT_ONLINE_EVAL_SAMPLE_LIMIT=50
GOV_SERVICE_AGENT_ONLINE_EVAL_LOOKBACK_HOURS=24
GOV_SERVICE_AGENT_ONLINE_EVAL_FAITHFULNESS_THRESHOLD=0.90

GOV_SERVICE_AGENT_LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_BASE_URL=http://localhost:3000
```

生产模式会在启动时检查数据库、Redis、检索后端、模型、鉴权、评测库和队列配置，缺失时直接拒绝启动。

## 知识库

前端“文件库”支持单文件、批量文件和文本入库。默认使用 Chonkie 结构分块，也支持父子分块和层级分块。

重建全部知识索引：

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8010/api/declaration-qa/knowledge/reindex
```

MinerU 用于复杂 PDF 和 Office 文档解析，是可选组件；未启用时会按配置回退到本地解析器。

## 多轮申报和投诉

`application_submit` 与 `complaint_create` 会把用户每轮明确提供的字段合并到会话中的 `dialog_task`。字段未齐时返回下一缺失字段；全部必填项通过服务端校验后才进入人工确认并正式写入。

不同申报事项的字段配置保存在 PostgreSQL 表 `declaration_qa.business_item_form_schemas`，按 `item_code + region_code + version_no` 选择当前发布版本。`fields` 是 JSON 数组，每个字段支持：

```json
{
  "name": "contact_phone",
  "label": "联系电话",
  "type": "phone",
  "required": true,
  "aliases": ["手机号", "电话"]
}
```

新增事项时创建一条 `status='published'` 的 Schema 即可，无需为每个事项新增 Tool 类。修改字段时新增版本，不要直接覆盖已被在途会话引用的版本。

## 评测

评测由两部分组成：

- PostgreSQL 保存数据集、用例、运行结果、评分、候选失败样本和发布门禁
- Langfuse 保存 Agent Trace，并接收 `eval.*` 和 `eval.case_passed` 分数

内置数据集：

| 文件 | 用途 | 用例数 |
| --- | --- | ---: |
| `agent-smoke-v1.json` | PR 冒烟 | 20 |
| `agent-golden-v1.json` | Agent 回归 | 60 |
| `database-retrieval-golden-v1.json` | 正文与DOC/DOCX/PDF/XLS/XLSX/图片附件检索回归 | 150 |
| `full-regression-v1.json` | RAG + Agent 全量回归 | 260 |

导入数据集：

```powershell
python -m gov_service_agent.application.evals.import_dataset `
  materials/evaluation/agent-golden-v1.json
```

生产环境需启动独立评测 Worker：

```powershell
python -m gov_service_agent.application.evals.worker
```

固定检索指标会在同一次运行中同时输出 `retrieval_recall_at_5`、
`retrieval_recall_at_10` 和 `retrieval_mrr_at_10`。启动运行时传入
`"run_retrieval_ablation": true`，会冻结Agent实际生成的检索Query，额外输出
`bm25`、`vector`、`rrf`、`rrf_rerank` 四组消融结果。

线上Trace采样必须使用独立调度进程，避免多个API Worker重复启动定时任务：

```powershell
python -m gov_service_agent.application.evals.scheduler
```

调度器按稳定哈希抽样生产审计Trace，并无条件保留差评、工具错误、失败和高延迟样本；
独立Judge的忠实性等分数写回对应Langfuse Trace，抽样结果自动进入脱敏候选池。
候选仍需人工补充 `doc_id/chunk_id` 金标后才能提升为黄金用例。

GitHub Actions 中配置 `EVAL_BASE_URL`、数据集 ID、基线 Run ID 和 `EVAL_INTERNAL_API_KEY` 后，即可启用 PR 与夜间发布门禁。

## 常用 API

所有接口以 `/api/declaration-qa` 开头，完整参数见 Swagger。

| 功能 | 接口 |
| --- | --- |
| 流式问答 | `POST /chat/stream` |
| 人工确认 | `POST /chat/confirm/stream` |
| 会话 | `GET /sessions` |
| Agent Run | `GET /agent-runs/{run_id}` |
| 文档上传 | `POST /knowledge/documents`、`POST /knowledge/upload` |
| 文档版本 | `POST /knowledge/documents/{id}/versions` |
| 评测数据集 | `GET/POST /evaluation/datasets` |
| 启动评测 | `POST /evaluation/datasets/{id}/runs` |
| 失败样本池 | `GET /evaluation/candidates` |
| 用户反馈 | `POST /feedback` |

## 项目结构

```text
frontend/                         React 前端
materials/                        知识材料与评测数据集
sql/postgresql/                   PostgreSQL 建库和迁移脚本
src/gov_service_agent/api/           FastAPI 接口
src/gov_service_agent/application/   Agent 工作流、工具和评测执行器
src/gov_service_agent/rag/           检索、解析、切分和模型适配
src/gov_service_agent/memory/        会话与长期记忆
src/gov_service_agent/observability/ Langfuse 与性能观测
src/gov_service_agent/storage/       PostgreSQL 仓储
tests/                            自动化测试
```

Python 包名为 `gov_service_agent`，环境变量统一使用 `GOV_SERVICE_AGENT_*`。为兼容生产数据和现有调用方，PostgreSQL schema 及 API 前缀仍保留 `declaration_qa`。

## 开发验证

```powershell
conda run -n langchain-agent python -m pytest -q

cd frontend
npm run build
```

提交前至少确认：

- `git diff --check` 无格式错误
- 后端相关测试通过
- 前端构建和浏览器交互正常
- 数据库迁移可重复执行
- 写操作经过人工确认

## License

当前仓库未声明开源许可证。对外发布或分发前，请补充许可证和第三方依赖合规说明。
