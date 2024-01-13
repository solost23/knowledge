from langchain_community.embeddings import HuggingFaceEmbeddings

from universal.config import config


def embedding():
    hf = HuggingFaceEmbeddings(
        model_kwargs=config.embedding.get('model_kwargs'),
        encode_kwargs=config.embedding.get('encode_kwargs'),
        multi_process=config.embedding.get('multi_process'),
        cache_folder=config.embedding.get('cache_path'),
    )
    return hf
