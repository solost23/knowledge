from langchain_community.vectorstores import Chroma

from initialize.embedding import embedding


class ChromaDB:
    def __init__(self):
        self.db = Chroma(
            embedding_function=embedding(),
            persist_directory="./chroma_db",
        )
