from langchain_chroma import Chroma

from universal.config import config
from universal.embedding import embedding


class ChromaDB:
    def __init__(self):
        self.db = Chroma(
            collection_name="knowledge",
            embedding_function=embedding(),
            persist_directory=config.chroma.get('persist_directory', './chroma_db'),
        )
