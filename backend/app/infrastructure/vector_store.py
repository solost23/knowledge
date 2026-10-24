from langchain_chroma import Chroma
from pathlib import Path

from app.core.settings import BACKEND_DIR, config
from app.infrastructure.embeddings import embedding


class ChromaDB:
    def __init__(self):
        chroma_config = config.chroma
        persist_directory = Path(
            chroma_config.get('persist_directory', './data/chroma_db')
        )
        if not persist_directory.is_absolute():
            persist_directory = BACKEND_DIR / persist_directory
        self.top_k = int(chroma_config.get('top_k', 4))
        self.relevance_threshold = float(
            chroma_config.get('relevance_threshold', 0.35)
        )
        self.db = Chroma(
            collection_name="knowledge",
            embedding_function=embedding(),
            persist_directory=str(persist_directory),
        )

    def search(self, query: str, k: int | None = None):
        """Return only chunks whose vector relevance clears the configured threshold."""
        return self.db.similarity_search_with_relevance_scores(
            query,
            k=k or self.top_k,
            score_threshold=self.relevance_threshold,
        )


chroma = ChromaDB()
