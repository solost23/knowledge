# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Setup (Python 3.11 + uv):**
```bash
uv sync --locked # deps declared in pyproject.toml, pinned by uv.lock
uv lock          # after changing pyproject.toml
```

**Run the application:**
```bash
uv run --locked python main.py          # http://localhost:8086, Swagger at /openapi/swagger
```

**Run tests:**
```bash
uv run --locked python -m unittest test.doc
uv run --locked python -m unittest test.question   # needs a reachable LLM
uv run --locked python -m unittest test.agent      # local graph test, no LLM
uv run --locked python -m unittest test.ask        # routing tests, no LLM
uv run --locked python -m unittest test.evaluation # metric tests, no LLM
```

**Offline smoke test (no LLM):**
```bash
uv run --locked python scripts/smoke_retrieval.py
```

**RAG evaluation:**
```bash
uv run --locked python eval/run.py --no-answer    # retrieval only, no tokens
uv run --locked python eval/run.py                # retrieval + answers
```
Reports land in `eval/reports/`; dataset and metric definitions in `eval/README.md`.

## Architecture

Flask-based knowledge base API that ingests documents and answers questions using LLM + vector search.

**Endpoints:**
- `POST /upload` — Parse a document and store its embeddings in ChromaDB
- `POST /ask` — Automatically route a question to RAG or plan-execute Agent
- `GET /question?question=...` — Compatibility alias for `/ask` routing
- `GET /agent/runs/<execution_id>` — Read persisted Agent execution status
- `POST /agent/resume` — Resume a failed Agent execution from its SQLite checkpoint
- `GET /docs` / `DELETE /docs/<name>` — List / delete notes (metadata in `knowledge.db`)

**Request flow:**

```
Flask (routers/register.py)
  ├─ /upload → controllers/doc.py → services/doc.py
  │     → services/servants/ (parse PDF/DOCX/PPTX/XLSX/MD)
  │     → RecursiveCharacterTextSplitter (500/50)
  │     → ChromaDB (./chroma_db)
  │
  └─ /ask → services/ask.py → model routing
        ├─ rag → services/question.py → Chroma retrieval → answer
        └─ agent → services/agent.py
              → LangGraph: plan → execute step → advance → finalize
              → LangChain tool agent (bounded read-only tools)
              → SQLite checkpoint, status query, failed-run resume
```

**Key components:**
- `initialize/` — Singletons for config, Chroma client, Flask app; loads `.env` at import
- `universal/embedding.py` — fastembed (ONNX) embeddings adapter, used by both app and eval
- `services/agent.py` — LangGraph plan-execute workflow, SQLite state and bounded read-only tools
- `eval/agent.py` — End-to-end Agent evaluation for routing, plans, tools, steps and answers
- `services/servants/` — Per-format parsers (docx2txt / pypdf / python-pptx / openpyxl / plain text) plus LLM factory (`llm.py`)
- `universal/store.py` — SQLite metadata for uploaded notes
- `conf/config.yaml` — server, LLM, Chroma, embedding settings

**LLM backends** (set `llm.name`):
- `chat_gpt` — OpenAI-compatible endpoint (`chat_gpt.base_url` / `chat_gpt.model`, key from `OPENAI_API_KEY`); default
- `chat_glm` — any OpenAI-compatible local server (`chat_glm.endpoint_url`, `/v1` appended automatically)

**Embedding model:** `BAAI/bge-small-zh-v1.5` via `fastembed` (ONNX Runtime, no torch), cached under `./huggingface/`. Adapter lives in `universal/embedding.py`.

**Storage:** ChromaDB persists to `./chroma_db`, note metadata in `./knowledge.db`. No external services required.
