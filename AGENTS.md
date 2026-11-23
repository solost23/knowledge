# AGENTS.md

给后续 AI 编码助手的项目约定。开始修改前先读本文件和相关模块；按任务需要查看 `README.md`、配置及测试，不要为了“顺手整理”扩大改动范围。

## 项目概况

- 本仓库是本地优先的知识库 Agent：Vue 3、TypeScript、Vite 前端；Python 3.11、Flask、LangChain/LangGraph 后端。
- 后端入口在 `backend/app/main.py`。主要模块：`api/` 提供 HTTP 接口，`agents/` 管理分流和 Agent 工作流，`knowledge/` 负责 RAG 与文档解析，`infrastructure/` 封装模型、数据库和向量存储，`core/` 管理配置与服务基础设施，`prompts/` 保存提示词。
- 运行配置集中在 `backend/conf/config.yaml`；环境变量通过 `backend/app/core/settings.py` 读取和覆盖。密钥放在未跟踪的 `.env`，不得写入源码、测试输出或文档示例中的真实凭据。
- 用户数据默认存放于 `backend/data/`，包括 SQLite 数据库、向量索引和模型缓存。不要删除、重置或覆盖这些运行数据来验证代码。

## 修改约定

- 先定位现有实现和相邻测试，沿用项目已有模块边界、命名和响应格式。避免大规模重命名、无关重构、复制已有逻辑或留下整段注释掉的旧实现。
- 依赖必须在 `backend/pyproject.toml` 声明并同步更新 `backend/uv.lock`。不要手工编辑锁文件；使用 `uv lock --project backend` 后检查差异，再以 `--locked` 验证。
- 增删依赖前检查传递依赖和平台影响。特别注意：`unstructured[pdf]` 会引入 `unstructured-inference`、PyTorch；Linux ARM64 和 x86_64 的 PyTorch 依赖还会拉取大型 NVIDIA CUDA wheels，即使应用只在 CPU 上运行也可能下载。PDF 解析依赖或实现变化必须保留并验证 PDF 文档导入行为，不能只为缩小镜像而静默降级功能。
- Docker 后端构建上下文是仓库根目录，Dockerfile 为 `backend/Dockerfile`。依赖安装在复制业务源码之前，使用 BuildKit uv 缓存；不要移除该缓存或建议常规构建使用 `--no-cache`。需要改 Docker 配置时确认 `docker compose config --quiet` 和 `docker build --check`。
- 文档支持类型和解析器映射分别在 `backend/app/knowledge/loaders/dispatcher.py` 及 `backend/app/knowledge/loaders/` 中维护；改解析器时检查对应测试及 `README.md` 的支持格式说明。
- 对话、文档和 Agent 执行记录持久化在本地数据库。数据库结构变化要考虑已有用户数据库和迁移兼容，不要只验证空数据库。
- 提示词集中在 `backend/app/prompts/`。修改提示词时检查路由、RAG、Agent 或摘要对应的行为测试。
- 前端 API 类型在 `frontend/src/types/api.ts`，请求封装在 `frontend/src/api/client.ts`，主要工作台位于 `frontend/src/App.vue`。API 结构变更时同步更新类型、前端调用和后端测试。
- 将用户已有的未提交改动视为用户工作：编辑前检查 `git status` 和相关 diff，只触碰完成任务必需的文件。

## 检查命令

所有 shell 命令按工作环境要求使用 `rtk` 前缀；若需直接调用命令，可使用 `rtk proxy <command>`。

```bash
uv run --project backend --locked --with pytest pytest -q backend/tests
uv run --project backend --locked --group dev ruff check backend/app backend/tests backend/evals
uv run --project backend --locked --group dev ruff format --check backend/app backend/tests backend/evals
cd frontend && npm run typecheck && npm run build
docker compose config --quiet
docker build --check -f backend/Dockerfile .
```

按改动范围运行必要检查；依赖、解析器、API 或持久化行为变更应运行相应回归测试。最终说明改了什么、验证了什么，以及尚未验证的行为。
