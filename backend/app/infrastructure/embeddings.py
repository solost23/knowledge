from pathlib import Path

from fastembed import TextEmbedding
from langchain_core.embeddings import Embeddings

from app.core.settings import BACKEND_DIR, config


class FastEmbeddings(Embeddings):
    """fastembed（ONNX Runtime）适配层，无需 torch / sentence-transformers。"""

    def __init__(
        self,
        model_name: str | None = None,
        cache_dir: str | None = None,
        threads: int | None = None,
        batch_size: int | None = None,
    ):
        settings = config.embedding
        if cache_dir is None:
            cache_path = Path(settings.get("cache_path", "./data/huggingface"))
            if not cache_path.is_absolute():
                cache_path = BACKEND_DIR / cache_path
            cache_dir = str(cache_path)
        self.model = TextEmbedding(
            model_name=model_name
            or settings.get("model_name", "BAAI/bge-small-zh-v1.5"),
            cache_dir=cache_dir,
            threads=threads if threads is not None else settings.get("threads"),
        )
        self.batch_size = (
            batch_size
            if batch_size is not None
            else int(settings.get("batch_size", 256))
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [
            vector.tolist()
            for vector in self.model.embed(texts, batch_size=self.batch_size)
        ]

    def embed_query(self, text: str) -> list[float]:
        return next(self.model.query_embed(text)).tolist()
