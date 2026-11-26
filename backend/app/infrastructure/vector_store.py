from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma

from app.core.settings import BACKEND_DIR, config
from app.infrastructure.document_repository import document_lock, visible_documents
from app.infrastructure.embeddings import FastEmbeddings


class ChromaDB:
    def __init__(self):
        chroma_config = config.chroma
        persist_directory = Path(
            chroma_config.get("persist_directory", "./data/chroma_db")
        )
        if not persist_directory.is_absolute():
            persist_directory = BACKEND_DIR / persist_directory
        self.top_k = int(chroma_config.get("top_k", 4))
        self.relevance_threshold = float(chroma_config.get("relevance_threshold", 0.35))
        self.db = Chroma(
            collection_name="knowledge",
            embedding_function=FastEmbeddings(),
            persist_directory=str(persist_directory),
        )

    def search(self, query: str, k: int | None = None):
        """只返回 SQLite 已发布版本，兼容没有版本元数据的历史片段。"""
        with document_lock:
            visible = visible_documents()
            if not visible:
                return []
            versions = [version for version in visible.values() if version is not None]
            legacy_names = [
                name for name, version in visible.items() if version is None
            ]
            filters = []
            if versions:
                filters.append({"document_version": {"$in": versions}})
            if legacy_names:
                filters.append(
                    {
                        "$and": [
                            {"source": {"$in": legacy_names}},
                            {"publication_managed": {"$ne": True}},
                        ]
                    }
                )
            where = filters[0] if len(filters) == 1 else {"$or": filters}
            matches = self.db.similarity_search_with_relevance_scores(
                query,
                k=k or self.top_k,
                score_threshold=self.relevance_threshold,
                filter=where,
            )
            # 兼容手动构造的历史数据，带非当前版本的片段不能返回给调用方。
            return [
                (document, score)
                for document, score in matches
                if document.metadata.get("source") in visible
                and document.metadata.get("document_version")
                == visible[document.metadata["source"]]
            ]


@lru_cache(maxsize=1)
def get_chroma() -> ChromaDB:
    return ChromaDB()
