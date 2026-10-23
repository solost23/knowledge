#!/usr/bin/env python
"""RAG 评测脚本：召回、证据命中、答案关键词覆盖。

用法：
    uv run --locked python evals/run.py                     # 默认 chunk=500, k=5，含答案生成
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
import shutil
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import yaml
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.infrastructure.embeddings import FastEmbeddings

load_dotenv(os.path.join(ROOT, ".env"))

with open(os.path.join(ROOT, "conf", "config.yaml"), encoding="utf-8") as f:
    CFG = yaml.safe_load(f)


def norm(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def build_index(corpus_dir: str, persist_dir: str, chunk_size: int, chunk_overlap: int):
    if os.path.isdir(persist_dir):
        shutil.rmtree(persist_dir)

    emb_cfg = CFG["embedding"]
    embeddings = FastEmbeddings(
        model_name=emb_cfg.get("model_name", "BAAI/bge-small-zh-v1.5"),
        cache_dir=os.path.join(ROOT, emb_cfg["cache_path"].lstrip("./")),
        threads=emb_cfg.get("threads"),
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    docs = []
    for name in sorted(os.listdir(corpus_dir)):
        if not name.lower().endswith((".md", ".txt")):
            continue
        with open(os.path.join(corpus_dir, name), encoding="utf-8") as f:
            text = f.read()
        docs.append(Document(page_content=text, metadata={"source": name}))

    chunks = splitter.split_documents(docs)
    store = Chroma(
        collection_name="eval",
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    store.add_documents(chunks)
    return store, embeddings, len(docs), len(chunks)


def load_questions(path: str):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_llm():
    cfg = CFG["chat_gpt"]
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY 未设置，无法生成答案；可加 --no-answer 只评检索")
    return ChatOpenAI(
        model=cfg.get("model", "gpt-4o-mini"),
        api_key=key,
        base_url=cfg.get("base_url"),
        temperature=cfg.get("temperature", 0),
        max_tokens=cfg.get("max_token", 4096),
        max_retries=1,
    )


PROMPT = """你是企业知识库助手。只依据下面的资料回答问题，不要编造。
如果资料不足以回答，直接回复「知识库中没有相关信息」。

资料：
{context}

问题：{question}

要求：答案简洁，给出具体数字和结论；不需要重复问题。"""


def answer_question(llm, question: str, docs):
    context = "\n\n".join(
        f"[{i}] 来源：{d.metadata.get('source', '')}\n{d.page_content}"
        for i, d in enumerate(docs, 1)
    )
    started = time.perf_counter()
    resp = llm.invoke(PROMPT.format(context=context, question=question))
    elapsed = (time.perf_counter() - started) * 1000
    usage = getattr(resp, "usage_metadata", None) or {}
    return resp.content, elapsed, usage.get("total_tokens", 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=os.path.join(ROOT, "eval", "corpus"))
    ap.add_argument("--questions", default=os.path.join(ROOT, "eval", "questions.jsonl"))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--chunk-size", type=int, default=500)
    ap.add_argument("--chunk-overlap", type=int, default=50)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-answer", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    tag = args.tag or f"chunk{args.chunk_size}_k{args.k}"
    persist_dir = os.path.join(ROOT, "eval", ".chroma", tag)

    t0 = time.perf_counter()
    store, _emb, n_docs, n_chunks = build_index(
        args.corpus, persist_dir, args.chunk_size, args.chunk_overlap
    )
    build_ms = (time.perf_counter() - t0) * 1000

    questions = load_questions(args.questions)
    if args.limit:
        questions = questions[: args.limit]

    llm = None if args.no_answer else build_llm()

    rows = []
    for q in questions:
        started = time.perf_counter()
        hits = store.similarity_search_with_score(q["question"], k=args.k)
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

        if llm is not None:
            try:
                answer, gen_ms, tokens = answer_question(llm, q["question"], docs)
                row["answer"] = answer
                row["gen_ms"] = round(gen_ms, 1)
                row["tokens"] = tokens
                if unanswerable:
                    row["abstained"] = any(
                        m in answer for m in ["不知道", "未找到", "没有找到", "没有相关", "无法回答"]
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
        "docs": n_docs,
        "chunks": n_chunks,
        "build_ms": round(build_ms, 1),
        "questions": len(rows),
        "answerable": len(answerable),
        "doc_hit_rate": round(sum(bool(r.get("doc_hit")) for r in answerable) / max(len(answerable), 1), 4),
        "evidence_hit_rate": round(sum(bool(r.get("evidence_hit")) for r in answerable) / max(len(answerable), 1), 4),
        "mrr": round(
            sum(1.0 / r["rank"] for r in answerable if r.get("rank")) / max(len(answerable), 1), 4
        ),
        "avg_retrieval_ms": round(statistics.mean([r["retrieval_ms"] for r in rows]), 1),
    }

    if answered:
        summary["keyword_recall"] = round(
            sum(r.get("keyword_recall", 0) for r in answered) / len(answered), 4
        )
        summary["answer_accuracy"] = round(
            sum(bool(r.get("keyword_hit")) for r in answered) / len(answered), 4
        )
        summary["avg_gen_ms"] = round(statistics.mean([r["gen_ms"] for r in answered]), 1)
        summary["total_tokens"] = sum(r.get("tokens", 0) for r in answered)
    if unanswerable_rows:
        abstain_rows = [r for r in unanswerable_rows if "abstained" in r]
        if abstain_rows:
            summary["abstention_rate"] = round(
                sum(bool(r["abstained"]) for r in abstain_rows) / len(abstain_rows), 4
            )

    os.makedirs(os.path.join(ROOT, "eval", "reports"), exist_ok=True)
    json_path = os.path.join(ROOT, "eval", "reports", f"{tag}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows}, f, ensure_ascii=False, indent=2)

    md_path = os.path.join(ROOT, "eval", "reports", f"{tag}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# RAG 评测报告：{tag}\n\n")
        f.write("| 指标 | 值 |\n|---|---|\n")
        for key, value in summary.items():
            f.write(f"| {key} | {value} |\n")
        f.write("\n## 检索失败\n\n")
        for r in answerable:
            if not r.get("doc_hit"):
                f.write(f"- {r['id']} {r['question']} → 期望 {r['gold_doc']}，实际 {r['sources'][:3]}\n")
        f.write("\n## 答案未覆盖关键词\n\n")
        for r in answered:
            if not r.get("keyword_hit"):
                f.write(f"- {r['id']} {r['question']} → {r.get('answer', '')[:120]}\n")

    print("\n" + json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nreport: {json_path}")
    print(f"report: {md_path}")


if __name__ == "__main__":
    main()
