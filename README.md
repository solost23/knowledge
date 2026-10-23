# knowledge

基于 Flask + LangChain + ChromaDB 的私人笔记知识库，支持上传文档并通过 LLM 进行问答。

## 功能

- 支持上传 PDF、DOCX、PPTX、XLSX、Markdown 文档
- 使用 fastembed（ONNX Runtime）+ `BAAI/bge-small-zh-v1.5` 对文档分块并存入 ChromaDB，无需 torch
- 基于 LangChain 1.x / LangGraph，支持任意 OpenAI 兼容接口（含本地 vLLM / Ollama / ChatGLM）
- 服务端自动路由：普通问题走 RAG，多步任务先规划再逐步调用只读工具
- Agent 执行状态落入 SQLite checkpoint，失败后可查询进度并从中断步骤恢复
- Agent 评测覆盖路由、计划、步骤、工具和答案关键词
- 问答结果附带来源文件名和匹配片段
- 笔记管理：列表、删除
- 离线检索自检 + 可复跑的 RAG 评测（召回率 / 证据命中 / 答案覆盖）

## 快速启动

无需 Docker，直接本地运行（Python 3.11 + [uv](https://docs.astral.sh/uv/)）：

```shell
uv sync --locked              # 按 uv.lock 创建 .venv 并安装依赖

# 下载 embedding 模型到 ./huggingface（只需一次，约 100MB）
uv run --locked python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-zh-v1.5', cache_dir='./huggingface')"
# 国内网络可加：HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1

cp .env.example .env   # 填入 OPENAI_API_KEY
uv run --locked python main.py
```

服务启动后 API 地址：`http://localhost:8086`，Swagger UI：`http://localhost:8086/openapi/swagger`

> 依赖由 `pyproject.toml` 声明、`uv.lock` 锁定；`uv sync --locked` 会在锁文件过期时直接报错。

## 配置

所有配置项在 `conf/config.yaml`：

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `host` / `port` | 服务监听地址 | `0.0.0.0:8086` |
| `llm.name` | LLM 后端，可选 `chat_glm` / `chat_gpt` | `chat_gpt` |
| `chat_gpt.base_url` / `chat_gpt.model` | OpenAI 兼容接口地址与模型名 | 见配置文件 |
| `chroma.persist_directory` | ChromaDB 数据目录 | `./chroma_db` |
| `embedding.device` | 推理设备，可选 `mps` / `cpu` / `cuda` | `mps` |

LLM 相关环境变量（可写在 `.env`，已 gitignore）：

| 变量 | 说明 |
|------|------|
| `OPENAI_API_KEY` | OpenAI 兼容接口所需 |

## API

### 上传笔记

```
POST /upload
Content-Type: multipart/form-data

file: <文件>  # 支持 pdf、docx、pptx、xlsx、md
```

### 问答

```
GET /question?question=<问题内容>
```

`GET /question` 为兼容入口，也会由服务端自动选择 RAG 或 Agent。

## 自动路由与 Agent 执行

新客户端统一调用：

```http
POST /ask
Content-Type: application/json

{"task":"比较新版和旧版报销制度，并总结差异及来源"}
```

服务端根据请求选择流程。普通事实问答走一次检索和回答；需要比较多个文档或根据中间结果继续行动的任务，会先生成最多五步的计划，再依次调用知识库搜索和文档列表工具。响应包含 `mode`、`route_reason`、最终 `answer` 和 `sources`；Agent 模式还会返回 `plan` 和每步执行结果 `steps`。

Agent 运行会返回唯一的 `execution_id`。执行失败时，响应仍会包含该 ID；客户端可用它查询状态或恢复失败的步骤：

```http
GET /agent/runs/{execution_id}
```

```http
POST /agent/resume
Content-Type: application/json

{"execution_id":"上次响应中的 UUID"}
```

checkpoint 默认写入 `./agent_checkpoints.db`，可用 `AGENT_CHECKPOINT_DB` 指定路径；Docker Compose 将其保存在项目下的 `agent_state/` 并挂载到容器。数据库保存任务、计划和步骤结果，可能包含知识库内容，应按敏感业务数据保护。当前 API 沿用项目现有的无认证方式，适合本地学习和受信环境；公开部署前需要加入身份认证及执行记录的用户隔离。

工具只允许搜索知识库和列出文档；搜索词最多 500 字、每步最多 6 次工具调用，单次最多返回 4 个片段。模型和工具错误会记入服务端日志，API 返回通用错误及可恢复的执行 ID，不回传异常原文。

路由本身需要一次模型调用；Agent 模式会复用该调用生成的计划。配置的 OpenAI 兼容模型需要支持函数调用式结构化输出和工具调用。

返回：

```json
{
  "code": 0,
  "success": true,
  "message": "成功",
  "data": {
    "mode": "rag",
    "route_reason": "单次检索足以回答",
    "answer": "回答内容",
    "sources": [
      {
        "source": "笔记文件名.md",
        "content": "相关片段..."
      }
    ]
  }
}
```

### 笔记列表

```
GET /docs
```

### 删除笔记

```
DELETE /docs/<文件名>
```

## 评测

```shell
uv run --locked python scripts/smoke_retrieval.py   # 离线链路自检（解析→切分→入库→检索，不调 LLM）
uv run --locked python eval/run.py --no-answer      # 只评检索，不花 token
uv run --locked python eval/run.py                  # 检索 + 答案
uv run --locked python eval/agent.py --limit 4      # Agent 路由与执行快速评测
uv run --locked python eval/agent.py                # 完整 Agent 评测并生成报告
```

Agent 评测会调用配置的模型，并要求将 `eval/corpus/` 文档上传到当前知识库。报告包含路由准确率、计划合法率、步骤完成率、预期工具召回和答案关键词召回。评测集与指标定义见 [eval/README.md](eval/README.md)，报告输出到 `eval/reports/`。

## 部署与打包

```shell
# 模型权重不打进镜像，先确保本地有 ./huggingface（见「快速启动」）
docker compose up -d --build
```

- 镜像基于 `python:3.11-slim`，依赖里不再有 torch（embedding 走 ONNX Runtime），镜像明显更小。
- 构建阶段用 `uv sync --locked --no-dev` 按 `uv.lock` 安装运行依赖，并校验锁文件与声明一致。
- `.dockerignore` 排除了 `.venv`、`.uv-cache`、`.hf-home`、`chroma_db`、`.env` 等，构建上下文不到 1 MB。
- embedding 模型通过 volume 挂载，缺失时容器启动会直接报错。
- `OPENAI_API_KEY` 由 compose 从 `.env` 注入，不会打进镜像。

## 体积说明

| 目录 | 大小 | 说明 |
|---|---|---|
| `.venv` | ~460M | 主要是 chromadb / onnxruntime / grpc，不入库 |
| `huggingface/` | ~100M | fastembed 的 ONNX 模型缓存，不入库，需要单独下载 |
| 其余（源码 / 评测 / 数据） | < 10M | |

embedding 已改为 ONNX 运行时（fastembed）。进一步压缩空间主要来自 `onnxruntime`（约 50M）和 chromadb 的依赖；如果不需要 ChromaDB，可换成纯内存向量索引。

## 测试

```shell
uv run --locked python -m unittest test.doc
uv run --locked python -m unittest test.question
uv run --locked python -m unittest test.agent
uv run --locked python -m unittest test.ask
uv run --locked python -m unittest test.evaluation
```
