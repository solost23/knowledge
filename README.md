# 知序知识库 Agent

基于 Vue 3、Flask、LangChain 和 LangGraph 的本地优先知识库 Agent。系统把普通问题自动交给 RAG；需要比较资料或按步骤完成的任务交给 Agent 规划和执行。文档索引、文档清单、对话历史、会话摘要和 Agent checkpoint 默认保存在 `backend/data/`。

## 项目结构

```text
.
├── frontend/                   # Vue 3 + TypeScript + Vite 工作台
├── backend/
│   ├── app/
│   │   ├── api/                # HTTP 路由和请求/响应适配
│   │   ├── application/        # 可复用的业务用例编排
│   │   ├── agents/             # 路由、LangGraph 工作流、只读工具、状态和 checkpoint
│   │   ├── knowledge/          # RAG、文档摄取和格式解析
│   │   ├── infrastructure/     # LLM、Chroma、embedding、SQLite 适配
│   │   ├── prompts/            # 问答、路由、摘要和 Agent 提示词
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

后端按四层边界组织：`api/` 只负责 HTTP 参数和响应格式，`application/` 编排跨领域用例，`knowledge/` 与 `agents/` 承载业务能力，`infrastructure/` 只提供数据库、模型、向量库和线程池适配。新增接口时先放到对应业务服务，再由控制器做薄适配，避免控制器直接调用 repository 或拼装后台任务细节。

`POST /ask` 的异步链路按阶段拆分：`api/controllers/ask.py` 解析 HTTP 输入；`agents/ask_service.py` 校验任务、解析/创建会话、创建执行记录并提交后台任务；`agents/ask_execution.py` 准备历史和路由上下文，并选择 RAG/Agent 执行分支；`agents/ask_run_lifecycle.py` 统一管理 checkpoint 状态变化、取消/失败收尾和对话结果落库；`agents/service.py` 负责 Agent 图工作流。接口先返回可轮询的执行 ID，客户端通过 Agent runs 状态接口读取进度。同步兼容入口 `AskService.ask()` 保留给内部调用，不属于 HTTP `/ask` 的主路径。

前端的请求代码按职责拆分在 `frontend/src/api/http.ts`（传输与错误处理）和 `frontend/src/api/resources.ts`（资源方法）中，`api/client.ts` 只作为稳定的导出入口，组件无需感知请求实现细节。

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

## 配置

提示词集中在 `backend/app/prompts/`，按 `rag.py`、`routing.py`、`memory.py` 和 `agent.py` 分别维护问答（含 RAG 评测）、请求分流、对话摘要和 Agent 规划执行模板。业务代码在调用时传入资料、历史、任务和摘要字数等动态参数。

运行参数默认集中在 `backend/conf/config.yaml`。在仓库根目录或 `backend/` 下的 `.env` 可以用 `APP_CONFIG__区块__参数` 覆盖任意已有配置项；变量名不区分大小写，值按 YAML 标量解析。例如：

```dotenv
APP_CONFIG__CHAT_GPT__BASE_URL=https://api.openai.com/v1
APP_CONFIG__CHAT_GPT__MODEL=gpt-4.1-mini
APP_CONFIG__DOCUMENTS__CHUNK_SIZE=700
APP_CONFIG__AGENT__MAX_TASK_LENGTH=3000
```

默认开启 `chat_gpt.auto_fallback`：普通/结构化的同步和异步非流式调用遇到模型不存在、无可用渠道或明确不支持工具调用时，最多尝试 3 个备用模型。密钥、额度、限流以及普通服务错误不会触发模型切换。设置 `chat_gpt.fallback_models` 可指定有序备用名单；列表为空时才从同一服务、同一密钥的 `/models` 列表按名称筛选聊天模型并按返回顺序尝试。名称筛选不是能力保证，实际调用仍可能失败；备用模型的质量和费用也可能不同。成功模型会在当前 LLM 实例的后续调用中复用。此机制不覆盖流式调用。

Compose 也支持 `.env` 中的 `APP_CONFIG__CHAT_GPT__BASE_URL`、`APP_CONFIG__CHAT_GPT__MODEL`、`APP_CONFIG__CHAT_GPT__AUTO_FALLBACK` 和 `APP_CONFIG__CHAT_GPT__FALLBACK_MODELS`。修改 `.env` 后执行 `docker compose up -d --force-recreate api`；修改代码或 YAML 后执行 `docker compose up -d --build api`。

`APP_HOST`、`APP_PORT`、`APP_MODE`、`APP_DATA_DIR` 和 `APP_CONFIG_FILE` 是服务级环境变量。Compose 部署的上传限制通过 `UPLOAD_MAX_SIZE_MB` 同时设置 API 和 Nginx；本地开发通过 `APP_CONFIG__UPLOAD__MAX_SIZE_MB` 调整 API 限制。

## 功能和 API

- 上传 `PDF`、`DOCX`、`PPTX`、`XLSX`、`MD`，以及 `TXT`、`CSV`、`TSV`、`JSON`、`HTML`、`XML`、常见代码和配置文件，查看文档清单并删除文档。
- 前端统一调用 `POST /ask`；后端决定走 RAG 还是 Agent，客户端无需传 `mode`。
- `POST /ask` 接收当前 `task` 和可选的 `conversation_id`；新对话首轮完成后返回 ID，后续请求携带该 ID 即可连续追问。
- 原始对话保存在本机 SQLite，刷新后可以继续同一会话；较早消息压缩为摘要供模型使用，原文仍可在会话记录中查看。后端只加载当前会话，不会把其他会话的内容混入。
- 长会话达到压缩阈值后，后端会更新该会话的滚动摘要，并保留最近几轮原文；单次请求的历史上下文限制为约 6,000 字符，摘要只在压缩时额外调用模型。
- `POST /conversations`、`GET /conversations`、`GET /conversations/{conversation_id}` 和 `DELETE /conversations/{conversation_id}` 用于管理会话。
- Agent 根据任务内容动态规划所需步骤，再通过受限的只读知识库工具逐步执行，并返回步骤、来源和最终回答。图执行轮数由 `backend/conf/config.yaml` 中的 `agent.recursion_limit` 保护。
- RAG 和 Agent 知识库搜索都会按 Chroma 相关性分数过滤片段；没有达到阈值时直接返回未找到相关资料，不调用回答模型。阈值和召回数量可在 `backend/conf/config.yaml` 的 `chroma.relevance_threshold`、`chroma.top_k` 中调整。
- Agent 提供 `GET /agent/runs/{execution_id}` 查询状态，以及 `POST /agent/resume` 恢复失败执行。
- `GET /healthz` 提供服务健康检查。

SQLite 业务表统一使用 SQLAlchemy ORM：`backend/app/infrastructure/models.py` 定义表模型，`database.py` 管理连接引擎和事务，repository 模块封装数据操作。`get_database()` 复用引擎，每次 `session()` 创建独立 Session，成功时提交、异常时回滚。首次访问会自动建表，并补齐旧数据库中已有版本需要的字段和索引。

数据库默认保存在 `APP_DATA_DIR` 下的 `knowledge.db`、`conversations.db`、`agent_checkpoints.db`，也可分别通过 `KNOWLEDGE_DB`、`CONVERSATION_DB`、`AGENT_CHECKPOINT_DB` 指定路径。Agent 任务记录使用 ORM，LangGraph 的检查点仍由 `SqliteSaver` 管理。

## 测试和评测

运行完整后端测试：

```shell
uv run --project backend --locked --with pytest pytest -q backend/tests
```

运行后端静态检查和格式校验：

```shell
uv run --project backend --locked --group dev ruff check backend/app backend/tests backend/evals
uv run --project backend --locked --group dev ruff format --check backend/app backend/tests backend/evals
```

也可以按功能单独运行测试：

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

后端构建使用 BuildKit 的 uv 下载缓存。只改业务源码时会复用整个依赖层；修改 `pyproject.toml` 或 `uv.lock` 后，依赖层需要重建，但仍能复用当前 builder 已下载的包。首次构建、换 builder 或清理构建缓存后仍需要下载。不要为了日常更新使用 `--no-cache` 或清理 builder 缓存。需要检查具体耗时阶段时运行：

```shell
docker compose --progress=plain build api
```

日志中的 `Downloading` / `Prepared` 主要对应下载和解包，`Installed` 对应安装，`Bytecode compiled` 对应字节码编译；依赖层显示 `CACHED` 时不会重新安装。当前 `unstructured[pdf]` 会引入 PyTorch，Linux x86_64 构建还会拉取 CUDA 依赖，首次下载量较大；项目的 embedding 使用 ONNX。若要进一步缩小依赖，可以另行评估 PyTorch CPU 源或调整 PDF 解析依赖，并验证文档解析功能。

若首次下载受网络影响，可配置构建代理，或在 `backend/pyproject.toml` 中配置可访问的 `[[tool.uv.index]]`（设置 `default = true`），再执行 `uv lock --project backend` 更新锁文件并重新构建。`uv.lock` 记录了包来源和下载 URL，仅设置 `PIP_INDEX_URL` 不会让 `uv sync --locked` 中已锁定的 PyPI 地址改走镜像。

网页只绑定到 `127.0.0.1:8080`；API 只在 Compose 内网提供服务。文档、会话和 Agent 状态持久化到 `backend/data/`。查看状态和日志：

```shell
docker compose ps
docker compose logs -f api web
```

目前部署按单用户、本机访问设计。若要让其他用户从公网访问，需要先为 API 增加认证、按用户隔离文档和执行记录，再配置 TLS、备份和密钥管理。
