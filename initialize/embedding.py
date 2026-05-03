from langchain_community.embeddings import HuggingFaceEmbeddings

from universal.config import config

_embedding = None


def embedding() -> HuggingFaceEmbeddings:
    global _embedding
    if _embedding is None:
        _embedding = HuggingFaceEmbeddings(
            model_name=config.embedding.get('model_name'),
            model_kwargs=config.embedding.get('model_kwargs'),
            encode_kwargs=config.embedding.get('encode_kwargs'),
            multi_process=config.embedding.get('multi_process'),
            cache_folder=config.embedding.get('cache_path'),
        )
    return _embedding
