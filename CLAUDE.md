# CLAUDE.md

## Project map

- `frontend/`: Vue 3 + TypeScript user interface; use npm here.
- `backend/app/api/`: Flask HTTP endpoints and request validation.
- `backend/app/agents/`: task routing, LangGraph plan-execute workflow, bounded read-only tools, SQLite checkpoints, status and recovery.
- `backend/app/knowledge/`: RAG and document ingestion/parsers.
- `backend/app/infrastructure/`: LLM, embedding, Chroma and SQLite adapters.
- `backend/evals/`: RAG and Agent evaluation data, metrics and reports.
- `backend/tests/`: unit tests and fixtures.
- `backend/data/`: ignored local indexes, metadata, checkpoints and model files.

## Development commands

Run from the repository root. Python dependencies are managed only by uv:

```bash
uv sync --project backend --locked
uv run --project backend --locked python -m app.main
uv run --project backend --locked python -m unittest discover -s backend/tests -p 'test_agent.py'
uv run --project backend --locked python -m unittest discover -s backend/tests -p 'test_ask.py'
uv run --project backend --locked python -m unittest discover -s backend/tests -p 'test_evaluation.py'
uv run --project backend --locked python backend/scripts/smoke_retrieval.py
uv run --project backend --locked python backend/evals/run.py --no-answer
```

Frontend commands run from `frontend/`:

```bash
npm install
npm run dev
npm run typecheck
npm run build
```

## Request flow

```text
Vue workspace → POST /ask → Agent dispatcher
                              ├─ RAG → knowledge retrieval → answer + sources
                              └─ Agent → LangGraph plan/execute/finalize
                                         ├─ bounded read-only tools
                                         └─ SQLite checkpoint and recoverable status
```

`POST /ask` owns request routing; the client must not choose the mode. Keep HTTP concerns in `api/`, task orchestration in `agents/`, knowledge behavior in `knowledge/`, and provider/storage code in `infrastructure/`.

The supported upload types are PDF, DOCX, PPTX, XLSX and Markdown. Agent tools are read-only and bounded. Evaluation fixtures are synthetic and must not be treated as user knowledge data.

The service defaults to loopback for local use. Compose serves the frontend on loopback and keeps the API on the internal network. Public deployment requires authentication and per-user data isolation.
