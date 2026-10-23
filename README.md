# 知序知识库 Agent

基于 Vue 3、Flask、LangChain 和 LangGraph 的本地优先知识库 Agent。系统把普通问题自动交给 RAG；需要比较资料或按步骤完成的任务交给 Agent 规划和执行。文档索引、文档清单和 Agent checkpoint 默认保存在 `backend/data/`。

## 项目结构

```text
.
├── frontend/                   # Vue 3 + TypeScript + Vite 工作台
├── backend/
│   ├── app/
│   │   ├── api/                # HTTP 路由和请求/响应适配
│   │   ├── agents/             # 路由、LangGraph 工作流、只读工具、状态和 checkpoint
│   │   ├── knowledge/          # RAG、文档摄取和格式解析
│   │   ├── infrastructure/     # LLM、Chroma、embedding、SQLite 适配
│   │   ├── core/               # 配置、服务启动和通用响应
│   │   └── main.py             # API 进程入口
│   ├── conf/                   # 后端配置
│   ├── evals/                  # RAG / Agent 评测集、指标和报告
│   ├── tests/                  # 单元测试和测试 fixture
│   ├── scripts/                # 本地检索链路自检
│   ├── data/                   # 本机运行时数据，不进入版本库
│   ├── pyproject.toml          # Python 依赖声明
│   └── uv.lock                 # Python 依赖锁定
├── docker-compose.yaml         # 前端 Nginx + 后端 API
└── .env.example                # 模型密钥示例
```

## 本地开发

需要 Python 3.11、uv、Node.js 20+ 和 npm。

在仓库根目录配置模型密钥：

```shell
cp .env.example .env
```

启动后端（终端一）：

```shell
uv sync --project backend --locked
uv run --project backend --locked python -m app.main
```

首次运行会下载约 100 MB 的 `BAAI/bge-small-zh-v1.5` embedding 模型到 `backend/data/huggingface/`。

启动前端（终端二）：

```shell
cd frontend
npm install
npm run dev
```

浏览器打开 `http://127.0.0.1:5173`。Vite 会把 `/api` 请求转发给本机后端；后端 Swagger 地址是 `http://127.0.0.1:8086/openapi/swagger`。

## 功能和 API

- 上传 `PDF`、`DOCX`、`PPTX`、`XLSX`、`MD`，查看文档清单并删除文档。
- 前端统一调用 `POST /ask`；后端决定走 RAG 还是 Agent，客户端无需传 `mode`。
- Agent 先规划最多五步，再通过受限的只读知识库工具逐步执行，并返回步骤、来源和最终回答。
- Agent 提供 `GET /agent/runs/{execution_id}` 查询状态，以及 `POST /agent/resume` 恢复失败执行。
- `GET /healthz` 提供服务健康检查。

## 测试和评测

```shell
uv run --project backend --locked python -m unittest discover -s backend/tests -p 'test_agent.py'
uv run --project backend --locked python -m unittest discover -s backend/tests -p 'test_ask.py'
uv run --project backend --locked python -m unittest discover -s backend/tests -p 'test_evaluation.py'
uv run --project backend --locked python backend/scripts/smoke_retrieval.py
uv run --project backend --locked python backend/evals/run.py --no-answer
uv run --project backend --locked python backend/evals/agent.py --limit 4
```

Agent 评测会调用模型并产生费用，需要将 `backend/evals/corpus/` 里的文档导入当前知识库。数据集和指标说明见 [backend/evals/README.md](backend/evals/README.md)。前端类型检查和生产构建：

```shell
cd frontend
npm run typecheck
npm run build
```

## 本地容器部署

将 `.env.example` 复制为 `.env` 并填写 `OPENAI_API_KEY`，再运行：

```shell
docker compose up --build -d
```

网页只绑定到 `127.0.0.1:8080`；API 只在 Compose 内网提供服务。数据持久化到 `backend/data/`。查看状态和日志：

```shell
docker compose ps
docker compose logs -f api web
```

目前部署按单用户、本机访问设计。若要让其他用户从公网访问，需要先为 API 增加认证、按用户隔离文档和执行记录，再配置 TLS、备份和密钥管理。
