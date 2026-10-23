#!/usr/bin/env python
"""离线链路自检：解析 -> 切分 -> 向量化 -> 入库 -> 检索，不调用 LLM。

用法：
    uv run --locked python scripts/smoke_retrieval.py
    uv run --locked python scripts/smoke_retrieval.py --query "作业分数大于60的学生账号" --top-k 3

退出码 0 表示链路可用，非 0 表示失败。
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import initialize  # noqa: F401  触发配置与 Chroma 单例初始化
from services.doc import DocService
from universal.chroma import chroma

DEFAULT_FILES = [
    "学生作业成绩.docx",
    "学生作业成绩.pptx",
    "学生作业成绩.xlsx",
    "学生作业成绩.md",
]


def ingest(files: list[str]) -> None:
    for name in files:
        path = os.path.join(ROOT, "test", "data", name)
        ext = os.path.splitext(name)[1].lower()
        result, _ = DocService().doc(path, ext, name)
        code = result.get("code") if isinstance(result, dict) else None
        if code != 0:
            raise SystemExit(f"入库失败 {name}: {result}")
        print(f"  ingested {name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="作业评级为优秀的学生姓名是什么")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--skip-ingest", action="store_true")
    args = parser.parse_args()

    if not args.skip_ingest:
        print("1) 入库测试文档")
        ingest(DEFAULT_FILES)

    print(f"2) 检索: {args.query}")
    results = chroma.db.similarity_search_with_score(args.query, k=args.top_k)
    if not results:
        print("FAIL: 没有检索到任何片段")
        return 1

    for i, (doc, distance) in enumerate(results, 1):
        snippet = doc.page_content.replace("\n", " ")[:90]
        print(f"  [{i}] distance={distance:.4f} source={doc.metadata.get('source')}")
        print(f"      {snippet}")

    print("PASS: 检索链路可用")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
