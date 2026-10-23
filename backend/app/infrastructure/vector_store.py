from langchain_chroma import Chroma
from pathlib import Path

from app.core.settings import BACKEND_DIR, config
from app.infrastructure.embeddings import embedding


class ChromaDB:
    def __init__(self):
        persist_directory = Path(
            config.chroma.get('persist_directory', './data/chroma_db')
        )
        if not persist_directory.is_absolute():
            persist_directory = BACKEND_DIR / persist_directory
        self.db = Chroma(
            collection_name="knowledge",
            embedding_function=embedding(),
            persist_directory=str(persist_directory),
        )


chroma = ChromaDB()
