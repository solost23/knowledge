from fastembed import TextEmbedding
from langchain_core.embeddings import Embeddings

from universal.config import config

_embedding = None


class FastEmbeddings(Embeddings):
    """fastembed（ONNX Runtime）适配层，无需 torch / sentence-transformers。"""

    def __init__(self, model_name: str, cache_dir: str | None = None, threads: int | None = None):
        self.model = TextEmbedding(model_name=model_name, cache_dir=cache_dir, threads=threads)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self.model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(self.model.query_embed(text)).tolist()


def embedding() -> FastEmbeddings:
    global _embedding
    if _embedding is None:
        _embedding = FastEmbeddings(
            model_name=config.embedding.get('model_name', 'BAAI/bge-small-zh-v1.5'),
            cache_dir=config.embedding.get('cache_path', './huggingface'),
            threads=config.embedding.get('threads'),
        )
    return _embedding
