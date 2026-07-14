# Declaration QA

Declaration QA 是一个面向政务申报场景的 Agentic RAG 系统。它将公开办事指南、申请材料、办理流程、政策依据、样表附件和个人工单数据组织成可追溯的问答能力，并提供知识上传、版本管理、异步索引、会话审计、反馈和评测接口。

项目当前采用单 Agent 原生 Tool Calling 架构。Agent 根据用户问题决定是否查询公开知识、个人工单或两者，并在同一消息链中观察工具结果后生成最终回答。公开知识与个人业务数据在工具层隔离，避免用知识库内容推断个人状态，或用个人工单替代公开政策依据。

## 目录

- [核心能力](#核心能力)
- [系统架构](#系统架构)
- [问答与检索流程](#问答与检索流程)
- [文档解析与切分](#文档解析与切分)
- [技术栈](#技术栈)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [知识库使用](#知识库使用)
- [API 概览](#api-概览)
- [数据存储](#数据存储)
- [项目目录](#项目目录)
- [开发与验证](#开发与验证)
- [生产部署检查](#生产部署检查)
- [常见问题](#常见问题)
- [当前已知限制](#当前已知限制)

## 核心能力

### Agent 问答

- 使用支持原生 Tool Calling 的单 Agent 完成问题理解、查询规划、工具调用和最终回答。
- 简单寒暄和不依赖外部事实的问题可以直接回答。
- 政策、材料、流程、条件、费用、时限和办事规范必须调用 `knowledge_search`。
- 个人申请列表、进度、状态历史和退回原因通过独立工单工具查询。
- 混合问题可以在同一轮并行调用知识工具和工单工具。
- 支持最多多轮工具调用，并保留每轮真实调用过程和结果摘要。

### 混合检索

- Milvus 提供向量语义检索。
- Elasticsearch 提供 BM25 关键词检索。
- Agent 可为每个子查询选择 `keyword`、`semantic` 或 `hybrid` 检索偏好。
- 两个物理通道始终保留，通过加权 RRF 融合候选。
- 每条 Agent 子查询分别执行 RRF、Qwen3-VL Rerank 和阈值过滤。
- 多查询结果按最低覆盖配额合并，多轮历史证据累计去重且不使用最新查询重新重排。
- 检索结果保留 chunk、文档、版本、地区、页码、坐标和附件来源信息。

### 文档和多模态知识

- 支持 PDF、DOCX、PPTX、XLSX、CSV、Markdown、文本和常见图片格式。
- MinerU 输出结构化 `content_list`、Markdown 和图片资源。
- Chonkie 根据标题、正文、表格、列表、公式和图片类型生成最终检索块。
- Markdown 中的官方附件链接可自动下载、解析并与主文档建立来源关系。
- 图片可同时保存视觉资产和 OCR 文本，支持图片展示、文本检索和多模态重排。
- 支持批量上传、异步索引、失败重试和全库重建索引。

### 管理与审计

- 知识文档支持发布、下线、灰度比例、版本创建、版本对比和回滚。
- PostgreSQL 持久化会话、消息、Agent Run、工具调用、证据、审计、反馈和评测记录。
- SSE 流式返回回答和工具事件。
- 前端支持暗色模式、可折叠知识库侧栏、Markdown 回答、GFM 表格和证据展开。
- 可选接入 Langfuse 记录模型调用、检索和重排链路。

## 系统架构

```text
┌─────────────────────────────────────────────────────────────┐
│ React + Vite 前端                                            │
│ 对话、工具过程、证据、知识上传、版本管理、会话历史             │
└──────────────────────────────┬──────────────────────────────┘
                               │ HTTP / SSE
┌──────────────────────────────▼──────────────────────────────┐
│ FastAPI                                                      │
│ Chat API、Knowledge API、Session API、Audit API、Evaluation  │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│ LangGraph / Tool-Calling Agent                               │
│ knowledge_search | ticket_list_query | ticket_detail_query  │
└───────────────┬───────────────────────────────┬─────────────┘
                │                               │
┌───────────────▼────────────────┐  ┌──────────▼──────────────┐
│ RAG 检索                        │  │ 个人工单数据              │
│ Milvus + Elasticsearch + RRF   │  │ PostgreSQL business_*    │
│ Embedding + Rerank             │  │ 申请及完整状态历史         │
└───────────────┬────────────────┘  └─────────────────────────┘
                │
┌───────────────▼─────────────────────────────────────────────┐
│ 知识入库                                                     │
│ RQ/Redis -> MinerU/native -> Chonkie -> PostgreSQL/Milvus/ES │
└─────────────────────────────────────────────────────────────┘
```

运行时由四个本地进程和五个 Docker 服务组成：

| 类型 | 组件 | 默认地址或用途 |
| --- | --- | --- |
| 本地进程 | FastAPI | `http://127.0.0.1:8010` |
| 本地进程 | MinerU API | `http://127.0.0.1:8011` |
| 本地进程 | RQ Worker | 异步执行知识索引任务 |
| 本地进程 | Vite Frontend | `http://127.0.0.1:5173` |
| Docker | PostgreSQL 16 | `localhost:5433` |
| Docker | Redis 7 | `localhost:6380` |
| Docker | Milvus Standalone | `http://localhost:19530` |
| Docker | Elasticsearch 8 | `http://localhost:9200` |
| Docker | Kibana | `http://localhost:5601` |

## 问答与检索流程

### Agent 执行流程

```text
用户问题或图片
  -> 加载会话上下文和已验证事实
  -> Agent 选择工具
       -> knowledge_search：公开知识
       -> ticket_list_query：个人申请列表
       -> ticket_detail_query：申请详情和状态历史
  -> 工具可以同轮并行执行
  -> 工具结果写回 Agent 消息链
  -> 继续调用工具或生成最终回答
  -> SSE 返回文本、证据、工具过程和审计 ID
```

`knowledge_search` 只允许 Agent 提交公开知识查询表达。`user_id`、地区、版本、过滤条件、召回数量和本地图片路径由服务端注入，不由模型生成。

### keyword、semantic 和 hybrid

它们是检索偏好，不是互斥的物理检索模式。每条查询都会保留向量和 BM25 两个通道：

| 策略 | Dense 权重 | BM25 权重 | 典型用途 |
| --- | ---: | ---: | --- |
| `hybrid` | 1 | 1 | 默认混合检索 |
| `semantic` | 2 | 1 | 口语表达、近义表达、概念问题 |
| `keyword` | 1 | 2 | 材料名称、法规文号、事项编码、专有术语 |

权重通过重复命中列表进入 RRF，而不是直接相乘原始余弦分数或 BM25 分数。RRF 使用 `k=60`：

```text
RRF contribution = 1 / (60 + rank)
```

Agent 多查询模式按子查询独立重排：

```text
每条子查询的 Dense + BM25
  -> 按策略权重执行局部 RRF
  -> 使用该子查询独立 Rerank
  -> 每路执行分数阈值和 Top N 限制
  -> 按查询最低配额合并和来源去重
  -> 多轮累计裁剪，不使用最新查询重排历史证据
```

## 文档解析与切分

### 总体入库流程

```text
文件或文本上传
  -> 创建 index job
  -> inline 或 RQ Worker 执行任务
  -> 文档解析
  -> 元数据抽取
  -> Chonkie 结构切分
  -> 附件下载与递归解析
  -> 图片资产和 OCR 处理
  -> Embedding
  -> PostgreSQL 持久化，并同步写入 Milvus、Elasticsearch
```

### 不同文件类型的解析路径

| 文件类型 | 首选解析路径 | 切分方式 |
| --- | --- | --- |
| PDF、DOCX、PPTX、XLSX | MinerU API | MinerU `content_list` 转结构元素后交给 Chonkie |
| DOC | LibreOffice headless 转 DOCX 后进入解析链 | 转换后的 DOCX 继续按 MinerU/native 解析 |
| PNG、JPG、WEBP、BMP、TIFF | MinerU 或配置的 OCR | 图片资产 + OCR 文本 + Chonkie |
| Markdown、TXT、JSON、XML | native 文本解码 | Marko GFM 结构识别 + Chonkie |
| CSV | native 表格解析 | 表格元素 + Chonkie TableChunker |
| 解析失败的受支持文件 | 按 `DOCUMENT_PARSER_ORDER` 降级 | 由成功解析器的结构进入 Chonkie |

默认解析顺序：

```env
DECLARATION_QA_DOCUMENT_PARSER_PROVIDER=auto
DECLARATION_QA_DOCUMENT_PARSER_ORDER=mineru,native
DECLARATION_QA_DOCUMENT_PARSER_FALLBACK_ENABLED=true
DECLARATION_QA_LEGACY_DOC_CONVERSION_ENABLED=true
```

Markdown 不发送给 MinerU。`auto` 模式会根据解析器的 `supports()` 结果跳过不支持的解析器，因此 `.md` 会直接由 native 解析器解码，再由 Marko GFM 识别标题、列表、引用、表格、代码块和 HTML 节点。

老版二进制 `.doc` 会先调用 LibreOffice headless 转成 `.docx`，再进入上述解析顺序。生产环境需要安装 LibreOffice，并可通过 `DECLARATION_QA_LEGACY_DOC_CONVERTER_COMMAND` 指定 `soffice.com` / `soffice` 的完整路径。

### MinerU content_list

MinerU API 返回三类主要数据：

- `md_content`：便于阅读和保存的 Markdown 全文。
- `content_list`：按阅读顺序排列的标题、正文、表格、列表、公式、图片和图表元素。
- `images`：解析过程中生成或提取的图片资源。

系统优先使用 `content_list` 构建 Chonkie 输入，不会先把结构压平成 Markdown 再重新猜测结构。每个元素会转换为 `StructuredElement`，并保留：

- 元素类型和原始序号。
- 标题层级 `text_level`。
- 页码 `page_idx`。
- 页面坐标 `bbox`。
- 表格正文、标题和脚注。
- 图片或图表标题。

### Chonkie 切分规则

当前默认参数：

```env
DECLARATION_QA_CHONKIE_TEXT_MAX_CHARS=900
DECLARATION_QA_CHONKIE_TEXT_MIN_CHARS=120
DECLARATION_QA_CHONKIE_TABLE_MAX_CHARS=3500
```

正文由 Chonkie `RecursiveChunker` 按以下优先级寻找边界：

```text
段落空行
  -> 换行
  -> 句号、问号、感叹号、分号
  -> 逗号、顿号、冒号
  -> 空格
  -> 强制截断
```

表格不会和相邻正文混切。未超过表格上限时保持完整，超长表格交给 `TableChunker`。标题不会单独制造父块，而是维护标题栈并回填到最终 chunk 的 `heading_path` 和内容前缀中。

新入库的 Chonkie 结果遵循以下约束：

- 每个最终检索块一对一写入 `declaration_qa.chunks`。
- 数据库层 `chunk_type` 为 `child`。
- 不额外生成物理 parent、summary 或 KV chunk。
- `parent_section_id` 仅表示章节分组，不对应物理父 chunk。
- 页码、坐标、来源元素和附件来源保存在 metadata 中。

### 附件和图片

Markdown 文档中的官方附件链接会被解析为 `AttachmentRef`。启用附件索引后，系统会下载附件、计算 SHA-256、执行文档解析并生成带来源关系的派生 chunk。

图片通常包含两种用途不同的记录：

```text
attachment_image
  保存 asset_id、文件路径、缩略图、OCR 和视觉摘要，支持图片展示和多模态检索

attachment_chonkie_text
  保存图片 OCR 后的文本，支持 BM25、文本向量检索和引用
```

如果图片 OCR 或视觉摘要超过单块上限，还可能生成额外的 `attachment_image_ocr` 或 `attachment_image_summary` 子块。

## 技术栈

| 层次 | 技术 |
| --- | --- |
| API | Python 3.10+、FastAPI、Uvicorn、Pydantic 2 |
| Agent | LangGraph、LangChain Core、原生 Tool Calling |
| 模型 | DashScope/OpenAI Compatible Chat API、Qwen3-VL |
| Embedding | Qwen3-VL Embedding、OpenAI Compatible Embedding、Hashing fallback |
| Rerank | Qwen3-VL Rerank、Heuristic fallback |
| 解析 | MinerU API、native parser、LlamaParse 可选 |
| 切分 | Chonkie RecursiveChunker、TableChunker |
| 向量检索 | Milvus |
| 关键词检索 | Elasticsearch BM25 |
| 持久化 | PostgreSQL、SQLAlchemy、Psycopg |
| 缓存和任务 | Redis、RQ |
| 前端 | React 18、Vite 5、Lucide React、React Markdown |
| 可观测 | Langfuse、数据库审计、Agent Run 回放 |
| 评测 | 内置规则指标、Ragas 0.3 |
| 基础设施 | Docker Compose |

## 快速开始

### 环境要求

- Windows 10/11。
- Conda 或 Miniconda。
- Docker Desktop，且 Linux Container Engine 正常运行。
- Node.js 18+ 和 npm。
- 可访问所配置模型服务的网络。
- 首次运行 MinerU 时需要下载模型，CPU 模式可以运行但解析速度会明显较慢。

### 1. 创建 Python 环境

`restart-app.bat` 固定使用名为 `langchain-agent` 的 Conda 环境：

```powershell
conda create -n langchain-agent python=3.10 -y
conda activate langchain-agent
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
pip install -e .
```

`requirements-dev.txt` 会继续安装 `requirements.txt`，其中包含 MinerU、Chonkie、FastAPI、Milvus、Elasticsearch、RQ 和测试依赖。

### 2. 安装前端依赖

```powershell
cd frontend
npm install
cd ..
```

### 3. 创建环境配置

```powershell
Copy-Item .env.example .env
```

至少需要根据使用模式配置模型 API Key。远程 DashScope 模式示例：

```env
DECLARATION_QA_LLM_PROVIDER=dashscope
DECLARATION_QA_OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
DECLARATION_QA_OPENAI_API_KEY=replace-with-your-api-key
DECLARATION_QA_AGENT_MODEL=qwen3-vl-plus

DECLARATION_QA_EMBEDDING_PROVIDER=qwen3-vl-embedding
DECLARATION_QA_EMBEDDING_MODEL=qwen3-vl-embedding
DECLARATION_QA_EMBEDDING_DIMENSIONS=1024

DECLARATION_QA_RERANK_PROVIDER=dashscope
DECLARATION_QA_RERANK_API_KEY=replace-with-your-api-key
DECLARATION_QA_RERANK_MODEL=qwen3-vl-rerank
```

不要提交包含真实密钥的 `.env`。

### 4. 启动基础设施

```powershell
.\start-docker.bat
```

等价命令：

```powershell
docker compose up -d
```

检查容器：

```powershell
docker compose ps
```

### 5. 初始化 PostgreSQL

首次创建数据库时执行合并后的 schema 文件：

```powershell
Get-Content .\sql\postgresql\declaration_qa_postgresql_all.sql -Raw |
  docker compose exec -T postgres psql -U agentbase -d agentbase
```

该 SQL 使用 `CREATE ... IF NOT EXISTS` 和迁移语句维护 `declaration_qa` schema。对于已有数据的环境，执行前应先备份数据库并检查 SQL 变更。

### 6. 选择索引队列

开发时可同步执行：

```env
DECLARATION_QA_INDEX_QUEUE_BACKEND=inline
```

需要异步入库时使用 RQ：

```env
DECLARATION_QA_INDEX_QUEUE_BACKEND=rq
DECLARATION_QA_REDIS_URL=redis://localhost:6380/1
DECLARATION_QA_INDEX_QUEUE_NAME=declaration-qa-index
```

生产环境必须使用 `rq`。`restart-app.bat` 会启动 Worker 窗口；当配置为 `inline` 时，索引任务由 API 同步执行，Worker 进程会提示需要 `rq` 后退出。

### 7. 启动应用

```powershell
.\restart-app.bat
```

脚本会：

1. 停止现有 API、MinerU、Worker 和前端进程。
2. 清理 `8010`、`8011`、`5173` 端口。
3. 检查 `langchain-agent` 环境和前端依赖。
4. 启动四个应用窗口。
5. 等待 API 和 MinerU 健康检查。
6. 打开前端页面。

启动成功后访问：

- 前端：<http://127.0.0.1:5173>
- API 文档：<http://127.0.0.1:8010/docs>
- API 健康检查：<http://127.0.0.1:8010/api/declaration-qa/health>
- MinerU 文档：<http://127.0.0.1:8011/docs>
- Kibana：<http://127.0.0.1:5601>

### 8. 停止基础设施

关闭应用窗口后，可以停止 Docker 服务：

```powershell
.\stop-docker.bat
```

该命令保留 PostgreSQL、Milvus 和 Elasticsearch 数据卷。不要在包含重要数据的环境中执行 `docker compose down -v`，它会删除卷。

## 配置说明

完整配置模板见 [.env.example](.env.example)。以下按功能说明最常用配置。

### 运行模式和存储

| 环境变量 | 说明 | 开发默认值 |
| --- | --- | --- |
| `DECLARATION_QA_ENV` | `development` 或 `production` | `development` |
| `DECLARATION_QA_ALLOW_IN_MEMORY_FALLBACKS` | 外部服务失败时是否允许内存兜底 | `true` |
| `DECLARATION_QA_DATABASE_URL` | PostgreSQL SQLAlchemy URL | 本地 5433 |
| `DECLARATION_QA_REDIS_URL` | Redis URL | 本地 6380/1 |
| `DECLARATION_QA_REDIS_TTL_SECONDS` | RAG 缓存 TTL | `600` |

### 检索后端

| 环境变量 | 说明 |
| --- | --- |
| `DECLARATION_QA_VECTOR_BACKEND` | `memory` 或 `milvus` |
| `DECLARATION_QA_MILVUS_URI` | Milvus 地址 |
| `DECLARATION_QA_MILVUS_COLLECTION` | 知识 chunk collection |
| `DECLARATION_QA_KEYWORD_BACKEND` | `memory` 或 `elasticsearch` |
| `DECLARATION_QA_ELASTICSEARCH_URL` | Elasticsearch 地址 |
| `DECLARATION_QA_ELASTICSEARCH_INDEX` | 知识 chunk index |

当 PostgreSQL、Milvus 和 Elasticsearch 同时启用时，系统使用外部索引引擎。PostgreSQL 是 chunk 和元数据的事实源，Milvus 与 Elasticsearch 保存检索索引。

### 模型

| 环境变量 | 说明 |
| --- | --- |
| `DECLARATION_QA_LLM_PROVIDER` | `dashscope`、`openai` 或本地确定性模式 |
| `DECLARATION_QA_OPENAI_API_BASE` | OpenAI Compatible API Base |
| `DECLARATION_QA_OPENAI_API_KEY` | Chat 和部分 Embedding 使用的 Key |
| `DECLARATION_QA_AGENT_MODEL` | Agent Tool Calling 模型 |
| `DECLARATION_QA_AGENT_MAX_ITERATIONS` | 最大 Agent 迭代轮数 |
| `DECLARATION_QA_EMBEDDING_PROVIDER` | `qwen3-vl-embedding`、`openai` 或 `hashing` |
| `DECLARATION_QA_EMBEDDING_DIMENSIONS` | 向量维度，必须与 Milvus collection 一致 |
| `DECLARATION_QA_RERANK_PROVIDER` | `dashscope` 或 `heuristic` |
| `DECLARATION_QA_RERANK_SCORE_THRESHOLD` | 重排最低分数 |
| `DECLARATION_QA_RERANK_MAX_SOURCES` | 单次 Rerank 的安全上限 |
| `DECLARATION_QA_RETRIEVAL_CANDIDATES_PER_QUERY` | 每条查询进入 RRF/Rerank 前的候选窗口 |
| `DECLARATION_QA_RERANK_TOP_N_PER_QUERY` | 每条查询局部 Rerank 后的保留数量，默认 `12` |
| `DECLARATION_QA_MIN_SOURCES_PER_QUERY` | 每条有合格证据的查询最低保留数量，默认 `2` |
| `DECLARATION_QA_MAX_ACCUMULATED_SOURCES` | 多查询、多轮累计证据总上限，默认 `48` |

修改 Embedding 模型或维度后，需要重建 Milvus collection 和知识索引，否则会出现维度不匹配。

### MinerU、OCR 和图片摘要

| 环境变量 | 说明 |
| --- | --- |
| `DECLARATION_QA_MINERU_API_URL` | MinerU FastAPI 地址 |
| `DECLARATION_QA_MINERU_BACKEND` | MinerU backend，默认 `pipeline` |
| `DECLARATION_QA_MINERU_METHOD` | `auto`、OCR 等 MinerU 解析方式 |
| `DECLARATION_QA_MINERU_LANGUAGE` | 解析语言，中文使用 `ch` |
| `DECLARATION_QA_MINERU_MODEL_SOURCE` | 模型来源，默认 `modelscope` |
| `DECLARATION_QA_MINERU_TIMEOUT_SECONDS` | 单文件解析超时 |
| `DECLARATION_QA_LEGACY_DOC_CONVERSION_ENABLED` | 是否将老版 `.doc` 转成 `.docx` 后再解析 |
| `DECLARATION_QA_LEGACY_DOC_CONVERTER_COMMAND` | LibreOffice `soffice.com` / `soffice` 路径，留空时自动查找 |
| `DECLARATION_QA_LEGACY_DOC_CONVERSION_TIMEOUT_SECONDS` | 单个 `.doc` 转换超时 |
| `DECLARATION_QA_OCR_API_URL` | 图片 OCR 网关 |
| `DECLARATION_QA_OCR_API_KEY` | OCR Key |
| `DECLARATION_QA_OCR_MODEL` | OCR 模型 |
| `DECLARATION_QA_IMAGE_SUMMARY_ENABLED` | 是否生成图片视觉摘要 |
| `DECLARATION_QA_IMAGE_SUMMARY_MODEL` | 图片摘要模型 |

图片摘要 API URL 或 Key 为空时会尝试复用 OCR 配置。

### 附件索引

| 环境变量 | 说明 |
| --- | --- |
| `DECLARATION_QA_ATTACHMENT_INDEXING_ENABLED` | 是否处理 Markdown 附件 |
| `DECLARATION_QA_ATTACHMENT_CACHE_DIR` | 附件下载缓存目录 |
| `DECLARATION_QA_ATTACHMENT_ASSET_DIR` | 图片资产目录 |
| `DECLARATION_QA_ATTACHMENT_MAX_BYTES` | 单附件最大字节数 |
| `DECLARATION_QA_ATTACHMENT_MAX_PER_DOCUMENT` | 单文档最大附件数 |
| `DECLARATION_QA_ATTACHMENT_MAX_WORKERS` | 附件并行处理数 |
| `DECLARATION_QA_ATTACHMENT_PDF_OCR_FALLBACK_ENABLED` | PDF 文本质量不足时是否 OCR 兜底 |

### 会话、记忆和鉴权

| 环境变量 | 说明 |
| --- | --- |
| `DECLARATION_QA_CONTEXT_MEMORY_ENABLED` | 是否启用上下文记忆 |
| `DECLARATION_QA_SHORT_MEMORY_TTL_SECONDS` | 短期记忆 TTL |
| `DECLARATION_QA_LONG_MEMORY_TTL_SECONDS` | 长期记忆 TTL |
| `DECLARATION_QA_AUTH_REQUIRED` | 是否要求认证 |
| `DECLARATION_QA_JWT_SECRET` | JWT 密钥 |
| `DECLARATION_QA_INTERNAL_API_KEY` | 服务间 API Key |

生产环境会自动强制鉴权，即使 `DECLARATION_QA_AUTH_REQUIRED=false`。

### Ragas 评测

Ragas 默认关闭，不影响在线问答。评测运行可通过请求字段 `use_ragas=true` 单次启用，也可设置全局默认值：

```env
DECLARATION_QA_RAGAS_ENABLED=false
DECLARATION_QA_RAGAS_TIMEOUT_SECONDS=60
DECLARATION_QA_RAGAS_MAX_RETRIES=1
```

系统对有回答和检索上下文的用例计算 `faithfulness`、`answer_relevancy`；用例填写 `reference_answer` 后还会计算 `context_precision`、`context_recall`。结果写入评测明细的 `ragas_scores` 和 `ragas_status`。Ragas 会调用 Agent 模型和 Embedding 服务，批量评测会产生额外延迟和模型费用。

已有数据库升级时只执行本次幂等迁移：

```powershell
Get-Content .\sql\postgresql\004_ragas_evaluation.sql -Raw |
  docker compose exec -T postgres psql -v ON_ERROR_STOP=1 -U agentbase -d agentbase
```

## 知识库使用

### 前端上传

打开前端左侧知识库面板，可以：

1. 选择一个或多个文件。
2. 填写标题和政策版本。
3. 使用 Chonkie 结构分块上传。
4. 查看解析、元数据、切分、Embedding 和发布进度。
5. 在版本管理中发布、下线、创建版本、设置灰度或回滚。

### API 上传示例

上传单个文件：

```bash
curl -X POST http://127.0.0.1:8010/api/declaration-qa/knowledge/upload \
  -F "file=@item_28.md" \
  -F "title=取水许可办事指南" \
  -F "region_code=340000" \
  -F "policy_version=2026" \
  -F "chunking_strategy=chonkie"
```

上传接口返回 `document_id` 和 `job_id`。异步模式下轮询任务状态：

```bash
curl http://127.0.0.1:8010/api/declaration-qa/index-jobs/JOB_ID
```

任务主要状态：

```text
pending -> indexing -> succeeded
                   -> failed
```

主要 ETL 阶段：

```text
extract -> metadata -> transform -> embedding_load -> atomic_publish
```

### 重建索引

```bash
curl -X POST http://127.0.0.1:8010/api/declaration-qa/knowledge/reindex
```

重建索引会根据 PostgreSQL 中的文档和 chunk 重新同步外部检索索引。生产环境执行前应确认 Worker、Redis、Milvus 和 Elasticsearch 均正常。

## API 概览

统一前缀：`/api/declaration-qa`

### 系统和聊天

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health` | 服务、后端和生产就绪状态 |
| POST | `/chat` | 非流式问答 |
| POST | `/chat/stream` | SSE 流式问答 |

### 知识库

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/knowledge/documents` | 纯文本知识入库 |
| POST | `/knowledge/upload` | 单文件上传 |
| POST | `/knowledge/uploads` | 批量上传 |
| GET | `/knowledge/documents` | 文档管理列表 |
| DELETE | `/knowledge/documents/{document_id}` | 删除文档及索引 |
| POST | `/knowledge/reindex` | 重建知识索引 |
| GET | `/index-jobs/{job_id}` | 查询索引任务 |
| POST | `/index-jobs/{job_id}/retry` | 重试失败任务 |
| POST | `/knowledge/documents/{id}/publish` | 发布版本 |
| POST | `/knowledge/documents/{id}/unpublish` | 下线版本 |
| POST | `/knowledge/documents/{id}/rollout` | 设置灰度比例 |
| POST | `/knowledge/documents/{id}/versions` | 创建新版本 |
| GET | `/knowledge/documents/{left}/compare/{right}` | 对比版本 |
| POST | `/knowledge/documents/rollback` | 回滚版本 |

### 会话、审计和评测

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/sessions` | 查询用户会话 |
| GET | `/sessions/{session_id}` | 查询会话详情 |
| DELETE | `/sessions/{session_id}` | 删除会话 |
| GET | `/agent-runs/{run_id}` | 回放 Agent Run |
| GET | `/audit/{audit_id}` | 查询审计记录 |
| POST | `/feedback` | 提交回答反馈 |
| POST/GET | `/evaluation/datasets` | 创建或查询评测集 |
| POST | `/evaluation/datasets/{id}/runs` | 执行评测 |
| GET | `/evaluation/runs/{run_id}` | 查询评测结果 |

请求和响应的完整字段以 Swagger 为准：<http://127.0.0.1:8010/docs>。

## 数据存储

### PostgreSQL

所有业务表位于 `declaration_qa` schema，主要包括：

- `knowledge_bases`：知识库。
- `documents`：文档、状态、版本和原始正文。
- `chunks`：最终检索块及 metadata。
- `assets`：图片和附件视觉资产。
- `index_jobs`：索引任务和 ETL 状态。
- 会话、消息、Agent Run、工具调用、证据和反馈表。
- `business_*`：示例或接入的个人工单数据。
- 评测数据集、用例、运行和人工标注表。

### Milvus 与 Elasticsearch

- PostgreSQL 保存 chunk 事实数据。
- Milvus 保存 chunk 向量和可过滤 metadata。
- Elasticsearch 保存 BM25 文本索引和关键词字段。
- 文档删除、发布状态变化和重建索引需要同步更新三类存储。

### 本地文件

默认运行数据目录：

```text
data/declaration_qa/uploads/       上传原文件
data/declaration_qa/attachments/   下载的附件缓存
data/declaration_qa/assets/        图片和页面资产
data/declaration_qa/query_images/  用户问题图片
```

这些目录可能包含用户上传内容和业务材料。生产环境应配置访问权限、备份、保留周期和清理策略。

## 项目目录

```text
declaration-qa/
├── frontend/                         React + Vite 前端
│   └── src/
│       ├── components/               聊天、工具过程、证据和知识管理组件
│       ├── App.jsx                   前端状态和 API 编排
│       ├── styles.css                主题与响应式样式
│       └── utils.js                  API 客户端和格式化函数
├── materials/                        示例和整理后的知识材料
├── sql/postgresql/                   PostgreSQL schema 与迁移合集
├── src/declaration_qa/
│   ├── api/                          FastAPI、Schema、鉴权和 SSE
│   ├── application/                  Workflow、Agent、工具和节点
│   ├── core/                         配置、领域模型和运行校验
│   ├── indexing/
│   │   ├── attachments/              附件下载、解析、资产和派生 chunk
│   │   ├── chunking/                 Chonkie 结构适配器
│   │   ├── document_parsing/         MinerU、native、OCR、LlamaParse
│   │   └── service/                  入库、删除和重建索引流程
│   ├── memory/                       上下文记忆
│   ├── observability/                Langfuse 和观测辅助
│   ├── rag/                          RAG 引擎、融合、重排和模型 Provider
│   ├── search/                       Milvus 与 Elasticsearch 适配器
│   └── storage/                      PostgreSQL、Redis 和仓储
├── tests/                            单元和集成测试
├── docker-compose.yml                基础设施
├── restart-app.bat                   启动和重启应用
├── start-docker.bat                  启动基础设施
└── stop-docker.bat                   停止基础设施并保留数据卷
```

## 开发与验证

### 手动启动单个组件

API：

```powershell
conda activate langchain-agent
python -m uvicorn declaration_qa.api.server.app:app --app-dir src --host 127.0.0.1 --port 8010 --reload
```

MinerU：

```powershell
conda activate langchain-agent
python -m mineru.cli.fast_api --host 127.0.0.1 --port 8011
```

RQ Worker：

```powershell
conda activate langchain-agent
$env:PYTHONPATH="$PWD\src"
python -m declaration_qa.indexing.worker
```

前端：

```powershell
cd frontend
npm run dev
```

### 后端测试

```powershell
conda activate langchain-agent
python -m pytest
```

部分测试会读取当前 `.env`。连接真实 PostgreSQL、Milvus、Elasticsearch 或远程模型时，结果可能受到已有数据、索引状态和网络影响。

### 前端构建

```powershell
cd frontend
npm run build
```

### 健康检查

```powershell
Invoke-RestMethod http://127.0.0.1:8010/api/declaration-qa/health |
  ConvertTo-Json -Depth 8
```

重点字段：

- `production_ready`：生产配置是否满足硬性要求。
- `production_readiness_errors`：缺失的生产配置。
- `vector_backend`、`keyword_backend`：实际检索后端。
- `embedding_provider`、`rerank_provider`：实际模型 Provider。
- `index_queue_backend`：`inline` 或 `rq`。
- `chunks`：当前 RAG 引擎加载的 chunk 数量。

## 生产部署检查

当 `DECLARATION_QA_ENV=production` 时，服务会在启动阶段检查以下硬性条件：

- PostgreSQL 已配置。
- Redis 已配置。
- `DECLARATION_QA_VECTOR_BACKEND=milvus`。
- `DECLARATION_QA_KEYWORD_BACKEND=elasticsearch`。
- 使用真实 Agent 模型，不使用确定性回答模式。
- 使用真实 Embedding，不使用 `hashing`。
- 使用真实 Rerank，不使用 `heuristic`。
- `DECLARATION_QA_INDEX_QUEUE_BACKEND=rq`。
- 配置 JWT Secret 或 Internal API Key。
- `DECLARATION_QA_SEED_KNOWLEDGE=false`。
- 禁止内存 fallback。

建议同时完成：

1. 为 PostgreSQL、Redis、Milvus、Elasticsearch 和本地资产目录建立备份与恢复流程。
2. 将模型 Key、JWT Secret 和 Internal API Key 放入密钥管理系统。
3. 配置反向代理、TLS、请求大小限制、超时和访问日志。
4. 对上传文件执行格式、大小、恶意内容和来源 URL 校验。
5. 为 RQ 任务设置监控、失败告警和死信处理。
6. 监控解析失败率、空 chunk 比例、重复率、召回率和 Rerank 通过率。
7. 使用固定评测集验证材料、流程、法规、时限、附件和图片问题。
8. 在修改 Embedding 模型、维度或 chunk 规则后执行全量重建索引。

## 常见问题

### `API health check timed out`

依次检查：

```powershell
conda env list
docker compose ps
Get-NetTCPConnection -State Listen |
  Where-Object LocalPort -in 8010,8011,5173
```

确认：

- Conda 环境名为 `langchain-agent`。
- `pip install -r requirements-dev.txt` 和 `pip install -e .` 已完成。
- PostgreSQL 和其他依赖容器已启动。
- `.env` 中的数据库、模型和索引地址可访问。
- 端口没有被无关进程占用。

### MinerU 首次启动或解析很慢

首次运行需要下载和加载模型。当前启动脚本设置 `MINERU_MODEL_SOURCE=modelscope`，可减少 Hugging Face 网络问题。CPU 可以运行，但复杂 PDF 和 OCR 会明显较慢。MinerU API 启动完成前，`8011/docs` 不会通过健康检查。

### Markdown 提示 MinerU 不支持

Markdown 本来就不需要 MinerU。在 `auto` 模式下应跳过 MinerU 并进入 native 解析器，然后由 Chonkie 切分。如果仍出现异常，检查：

```env
DECLARATION_QA_DOCUMENT_PARSER_PROVIDER=auto
DECLARATION_QA_DOCUMENT_PARSER_ORDER=mineru,native
DECLARATION_QA_DOCUMENT_PARSER_FALLBACK_ENABLED=true
```

### 上传后一直显示排队中

如果使用 RQ：

- 检查 Redis `6380` 是否可访问。
- 确认 Worker 窗口没有退出。
- 确认 Worker 与 API 使用同一个 `.env` 和队列名称。
- 查询 `/api/declaration-qa/index-jobs/{job_id}` 的 `error_message` 和 ETL 状态。

如果使用 `inline`，上传请求会同步执行，复杂 PDF 可能持续较长时间。

### 检索不到刚上传的文档

检查：

1. 文档状态是否为 `published`。
2. 索引任务是否为 `succeeded`。
3. PostgreSQL 中 `documents.chunk_count` 与实际 chunk 数是否一致。
4. Milvus 和 Elasticsearch 是否已经写入对应 document ID。
5. 查询中的地区、政策版本和 metadata filter 是否排除了文档。
6. 必要时执行知识库重建索引。

### 图片为什么有多个 chunk

图片资产 chunk 用于保存和展示原图，OCR 文本 chunk 用于 BM25 和文本向量检索。长 OCR 或长视觉摘要还会继续拆分，因此一个图片附件产生两个或更多 chunk 属于预期行为。检索层应按 `asset_id` 和来源附件聚合展示。

### 回答显示原始 `###`、`**` 或表格竖线

当前前端使用 React Markdown 和 GFM 渲染助手回答。确认前端依赖已更新并重新启动：

```powershell
cd frontend
npm install
npm run dev
```

## 当前已知限制

- `chunks.chunk_index` 的顺序写入仍需要完善；当前稳定顺序主要依赖 chunk UID 和生成顺序。
- Chonkie 去重当前主要在单次文档或单个附件切分范围内，两个不同附件中的相同模板文字仍可能重复入库。
- 图片资产和 OCR 文本是有意的双模态记录，召回展示层仍需持续优化聚合策略。
- 当前长度限制使用字符数，不是目标模型 tokenizer 的 token 数；超长 HTML 表格需要额外关注上下文预算。
- 仅含表头、空正文或极短内容的质量过滤仍需加强。
- Chonkie 失败时会回退到 MinerU 原生元素块；生产环境应对该降级建立显式告警。
- 本地 `restart-app.bat` 面向 Windows 开发环境，生产部署应改为服务管理、容器编排或进程守护方案。

## License

当前仓库未声明开源许可证。对外发布或分发前，请补充许可证和第三方依赖合规说明。
