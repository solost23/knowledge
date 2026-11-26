#!/usr/bin/env python
"""RAG 评测脚本：召回、证据命中、答案关键词覆盖。

用法：
    uv run --locked python evals/run.py                     # 使用生产切分和检索配置，含答案生成
    uv run --locked python evals/run.py --no-answer         # 只评检索，不调 LLM
    uv run --locked python evals/run.py --chunk-size 300 --chunk-overlap 30 --k 3
    uv run --locked python evals/run.py --limit 5           # 快速冒烟

产物：
    evals/reports/<tag>.json   完整逐题结果
    evals/reports/<tag>.md     汇总报告
"""

import argparse
import json
import os
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
ROOT = EVAL_DIR.parent
sys.path.insert(0, str(ROOT))

from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.settings import config
from app.infrastructure.embeddings import FastEmbeddings
from app.knowledge.loaders import load_file
from app.knowledge.loaders.dispatcher import SUPPORTED_EXTENSIONS
from app.prompts.rag import RAG_PROMPT


def norm(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def build_index(
    corpus_dir: str | Path,
    persist_dir: str | Path,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
):
    if Path(persist_dir).exists():
        raise FileExistsError(f"评测索引已存在，请使用新的 tag：{persist_dir}")

    embeddings = FastEmbeddings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.documents["chunk_size"] if chunk_size is None else chunk_size,
        chunk_overlap=(
            config.documents["chunk_overlap"]
            if chunk_overlap is None
            else chunk_overlap
        ),
    )

    docs = []
    file_count = 0
    for path in sorted(Path(corpus_dir).iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        loaded = filter_complex_metadata(load_file(path))
        file_count += 1
        for doc in loaded:
            doc.metadata["source"] = path.name
        docs.extend(loaded)

    chunks = splitter.split_documents(docs)
    store = Chroma(
        collection_name="eval",
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )
    store.add_documents(chunks)
    return store, embeddings, file_count, len(chunks)


def load_questions(path: str):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_llm():
    cfg = config.chat_gpt
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit(
            "OPENAI_API_KEY 未设置，无法生成答案；可加 --no-answer 只评检索"
        )
    return ChatOpenAI(
        model=cfg.get("model", "gpt-4o-mini"),
        api_key=key,
        base_url=cfg.get("base_url"),
        temperature=cfg.get("temperature", 0),
        max_tokens=cfg.get("max_token", 4096),
        max_retries=1,
    )


def retrieve(
    store, question: str, k: int | None = None, threshold: float | None = None
):
    return store.similarity_search_with_relevance_scores(
        question,
        k=int(config.chroma["top_k"]) if k is None else k,
        score_threshold=(
            float(config.chroma["relevance_threshold"])
            if threshold is None
            else threshold
        ),
    )


def answer_question(llm, question: str, docs):
    if not docs:
        return "知识库中没有找到与问题直接相关的资料。", 0.0, 0
    context = "\n\n".join(
        f"[{i}] 来源：{d.metadata.get('source', '')}\n{d.page_content}"
        for i, d in enumerate(docs, 1)
    )
    started = time.perf_counter()
    resp = llm.invoke(
        RAG_PROMPT.invoke({"context": context, "question": question, "history": []})
    )
    elapsed = (time.perf_counter() - started) * 1000
    usage = getattr(resp, "usage_metadata", None) or {}
    return resp.content, elapsed, usage.get("total_tokens", 0)


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=EVAL_DIR / "corpus")
    ap.add_argument("--questions", type=Path, default=EVAL_DIR / "questions.jsonl")
    ap.add_argument("--k", type=int, default=int(config.chroma["top_k"]))
    ap.add_argument(
        "--relevance-threshold",
        type=float,
        default=float(config.chroma["relevance_threshold"]),
    )
    ap.add_argument("--chunk-size", type=int, default=config.documents["chunk_size"])
    ap.add_argument(
        "--chunk-overlap", type=int, default=config.documents["chunk_overlap"]
    )
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-answer", action="store_true")
    ap.add_argument("--tag", default="")
    return ap


def main():
    args = build_parser().parse_args()
    tag = re.sub(r"[^A-Za-z0-9_-]", "_", args.tag)[:64] or (
        f"chunk{args.chunk_size}_k{args.k}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    )
    persist_dir = EVAL_DIR / ".chroma" / tag
    report_dir = EVAL_DIR / "reports"
    json_path = report_dir / f"{tag}.json"
    md_path = report_dir / f"{tag}.md"
    if json_path.exists() or md_path.exists():
        raise FileExistsError(f"评测报告已存在，请使用新的 tag：{tag}")

    t0 = time.perf_counter()
    store, _emb, n_docs, n_chunks = build_index(
        args.corpus, persist_dir, args.chunk_size, args.chunk_overlap
    )
    build_ms = (time.perf_counter() - t0) * 1000

    questions = load_questions(args.questions)
    if args.limit:
        questions = questions[: args.limit]

    llm = None

    rows = []
    for q in questions:
        started = time.perf_counter()
        hits = retrieve(store, q["question"], args.k, args.relevance_threshold)
        retrieval_ms = (time.perf_counter() - started) * 1000

        docs = [d for d, _ in hits]
        sources = [d.metadata.get("source", "") for d in docs]
        unanswerable = q["gold_doc"] == "__none__"

        row = {
            "id": q["id"],
            "question": q["question"],
            "category": q["category"],
            "gold_doc": q["gold_doc"],
            "sources": sources,
            "retrieval_ms": round(retrieval_ms, 1),
        }

        if not unanswerable:
            rank = next(
                (i + 1 for i, s in enumerate(sources) if s == q["gold_doc"]), None
            )
            row["rank"] = rank
            row["doc_hit"] = rank is not None
            row["evidence_hit"] = any(
                norm(q["gold_evidence"]) in norm(d.page_content) for d in docs
            )

        if not args.no_answer:
            try:
                if docs and llm is None:
                    llm = build_llm()
                answer, gen_ms, tokens = answer_question(llm, q["question"], docs)
                row["answer"] = answer
                row["gen_ms"] = round(gen_ms, 1)
                row["tokens"] = tokens
                if unanswerable:
                    row["abstained"] = any(
                        m in answer
                        for m in [
                            "不知道",
                            "未找到",
                            "没有找到",
                            "没有相关",
                            "无法回答",
                        ]
                    )
                else:
                    normalized_answer = norm(answer).lower()
                    hits = [norm(k).lower() in normalized_answer for k in q["keywords"]]
                    row["keyword_hit"] = all(hits)
                    row["keyword_recall"] = round(sum(hits) / max(len(hits), 1), 3)
            except Exception as e:  # noqa: BLE001
                row["error"] = str(e)[:300]

        rows.append(row)
        status = "OK" if (unanswerable or row.get("doc_hit")) else "MISS"
        print(f"  [{status}] {row['id']} {q['question'][:40]}")

    answerable = [r for r in rows if r["gold_doc"] != "__none__"]
    unanswerable_rows = [r for r in rows if r["gold_doc"] == "__none__"]
    answered = [r for r in answerable if "answer" in r]

    summary = {
        "tag": tag,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "k": args.k,
        "relevance_threshold": args.relevance_threshold,
        "docs": n_docs,
        "chunks": n_chunks,
        "build_ms": round(build_ms, 1),
        "questions": len(rows),
        "answerable": len(answerable),
        "doc_hit_rate": round(
            sum(bool(r.get("doc_hit")) for r in answerable) / max(len(answerable), 1), 4
        ),
        "evidence_hit_rate": round(
            sum(bool(r.get("evidence_hit")) for r in answerable)
            / max(len(answerable), 1),
            4,
        ),
        "mrr": round(
            sum(1.0 / r["rank"] for r in answerable if r.get("rank"))
            / max(len(answerable), 1),
            4,
        ),
        "avg_retrieval_ms": round(
            statistics.mean([r["retrieval_ms"] for r in rows]), 1
        ),
    }

    if answered:
        summary["keyword_recall"] = round(
            sum(r.get("keyword_recall", 0) for r in answered) / len(answered), 4
        )
        summary["answer_accuracy"] = round(
            sum(bool(r.get("keyword_hit")) for r in answered) / len(answered), 4
        )
        summary["avg_gen_ms"] = round(
            statistics.mean([r["gen_ms"] for r in answered]), 1
        )
        summary["total_tokens"] = sum(r.get("tokens", 0) for r in answered)
    if unanswerable_rows:
        abstain_rows = [r for r in unanswerable_rows if "abstained" in r]
        if abstain_rows:
            summary["abstention_rate"] = round(
                sum(bool(r["abstained"]) for r in abstain_rows) / len(abstain_rows), 4
            )

    report_dir.mkdir(parents=True, exist_ok=True)
    with open(json_path, "x", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows}, f, ensure_ascii=False, indent=2)

    with open(md_path, "x", encoding="utf-8") as f:
        f.write(f"# RAG 评测报告：{tag}\n\n")
        f.write("| 指标 | 值 |\n|---|---|\n")
        for key, value in summary.items():
            f.write(f"| {key} | {value} |\n")
        f.write("\n## 检索失败\n\n")
        for r in answerable:
            if not r.get("doc_hit"):
                f.write(
                    f"- {r['id']} {r['question']} → 期望 {r['gold_doc']}，实际 {r['sources'][:3]}\n"
                )
        f.write("\n## 答案未覆盖关键词\n\n")
        for r in answered:
            if not r.get("keyword_hit"):
                f.write(f"- {r['id']} {r['question']} → {r.get('answer', '')[:120]}\n")

    print("\n" + json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nreport: {json_path}")
    print(f"report: {md_path}")


if __name__ == "__main__":
    main()
