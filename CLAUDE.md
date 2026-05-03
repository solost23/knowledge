# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Install dependencies:**
```bash
pipenv install
```

**Run the application:**
```bash
python main.py
# or via Docker Compose (recommended — starts Milvus + dependencies)
docker-compose up -d
```

**Run tests:**
```bash
python -m unittest test.doc
python -m unittest test.question
```

**Build Docker image:**
```bash
docker build -t knowledge:v1.0.0 .
```

## Architecture

Flask-based knowledge base API that ingests documents and answers questions using LLM + vector search.

**Two endpoints:**
- `POST /upload` — Parse a document and store its embeddings in Milvus
- `GET /question` — Retrieve similar document chunks from Milvus and generate an answer via LLM

**Request flow:**

```
Flask (routers/register.py)
  ├─ /upload → controllers/ → services/doc_service.py
  │     → services/servants/ (parse PDF/DOCX/PPTX/XLSX)
  │     → RecursiveCharacterTextSplitter
  │     → Milvus (store embeddings)
  │
  └─ /question → controllers/ → services/question_service.py
        → Milvus similarity search
        → GPTCache check
        → LLM (ChatGLM / ChatGPT / Wenxin) via load_qa_chain
```

**Key components:**
- `initialize/` — Singletons for Milvus client, HuggingFace embeddings, and Flask app setup
- `services/servants/` — Per-format document parsers (PDF, DOCX, PPTX, XLSX)
- `caches/gpt.py` — GPTCache with map-based backend; cache hits are controlled by `max_distance` in config
- `conf/config.yaml` — All runtime config: server port, LLM selection, Milvus host, embedding device, cache settings

**LLM backends** (set `llm.name` in `conf/config.yaml`):
- `chat_glm` — Local endpoint at `http://127.0.0.1:8000` (default)
- `chat_gpt` — OpenAI API (requires `api_key`)
- `wen_xin` — Baidu Qianfan API

**Embedding model:** `sentence-transformers/all-mpnet-base-v2`, cached in `./huggingface/`, runs on MPS by default (change `embedding.device` for CPU/CUDA).

**Infrastructure (docker-compose):** Milvus vector DB + etcd + MinIO + Attu UI (port 3000). Milvus listens on `19530`.
