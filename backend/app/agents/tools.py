import json

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool

from app.agents.state import MAX_SEARCH_RESULTS, MAX_SOURCE_LENGTH, SearchKnowledgeInput
from app.infrastructure.document_repository import list_docs
from app.infrastructure.vector_store import chroma


@tool(args_schema=SearchKnowledgeInput)
def search_knowledge(query: str, top_k: int = MAX_SEARCH_RESULTS) -> str:
    """按语义搜索知识库，返回匹配片段及来源文件名。"""
    args = SearchKnowledgeInput(query=query, top_k=top_k)
    docs = [doc for doc, _score in chroma.search(args.query, k=args.top_k)]
    results = [
        {
            "source": doc.metadata.get("source", ""),
            "content": doc.page_content[:MAX_SOURCE_LENGTH],
        }
        for doc in docs
    ]
    return json.dumps({"results": results}, ensure_ascii=False)


@tool
def list_knowledge_documents() -> str:
    """列出已上传到知识库的文档名称和上传时间。"""
    return json.dumps({"documents": list_docs()[:50]}, ensure_ascii=False, default=str)


def message_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


def extract_sources(messages: list) -> list[dict[str, str]]:
    sources = []
    seen = set()
    for message in messages:
        if not isinstance(message, ToolMessage) or message.name != "search_knowledge":
            continue
        try:
            results = json.loads(message_text(message.content)).get("results", [])
        except (json.JSONDecodeError, AttributeError):
            continue
        for result in results:
            source = result.get("source", "")
            content = result.get("content", "")
            key = (source, content)
            if source and key not in seen:
                seen.add(key)
                sources.append({"source": source, "content": content})
    return sources
