from langchain_community.vectorstores import milvus

from initialize.embedding import embedding
from universal.config import config


class Milvus:
    """
    初始化 milvus 客户端
    """
    def __init__(self):
        self.db = milvus.Milvus(
            embedding_function=embedding(),
            connection_args=config.milvus,
        )
