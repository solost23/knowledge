from langchain_community.embeddings import HuggingFaceEmbeddings

from universal.config import config


def embedding():
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    hf = HuggingFaceEmbeddings(
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        cache_folder=config.embedding.get('cache_path'),
    )
    return hf
