from fastembed import TextEmbedding
from langchain_core.embeddings import Embeddings
from pathlib import Path

from app.core.settings import BACKEND_DIR, config

_embedding = None


class FastEmbeddings(Embeddings):
    """fastembed（ONNX Runtime）适配层，无需 torch / sentence-transformers。"""

    def __init__(
        self,
        model_name: str,
        cache_dir: str | None = None,
        threads: int | None = None,
        batch_size: int = 256,
    ):
        self.model = TextEmbedding(model_name=model_name, cache_dir=cache_dir, threads=threads)
        self.batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [
            vector.tolist()
            for vector in self.model.embed(texts, batch_size=self.batch_size)
        ]

    def embed_query(self, text: str) -> list[float]:
        return next(self.model.query_embed(text)).tolist()


def embedding() -> FastEmbeddings:
    global _embedding
    if _embedding is None:
        cache_dir = Path(config.embedding.get('cache_path', './data/huggingface'))
        if not cache_dir.is_absolute():
            cache_dir = BACKEND_DIR / cache_dir
        _embedding = FastEmbeddings(
            model_name=config.embedding.get('model_name', 'BAAI/bge-small-zh-v1.5'),
            cache_dir=str(cache_dir),
            threads=config.embedding.get('threads'),
            batch_size=int(config.embedding.get('batch_size', 256)),
        )
    return _embedding
