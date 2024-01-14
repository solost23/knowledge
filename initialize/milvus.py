from langchain_community.vectorstores import milvus

from initialize.embedding import embedding
from initialize.config import YamlConfig


class Milvus:
    """
    初始化 milvus 客户端
    """
    def __init__(self, yaml_config: YamlConfig):
        self.db = milvus.Milvus(
            embedding_function=embedding(yaml_config),
            connection_args=yaml_config.milvus,
        )
