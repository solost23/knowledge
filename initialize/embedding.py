from langchain_community.embeddings import HuggingFaceEmbeddings

from initialize.config import YamlConfig


def embedding(yaml_config: YamlConfig):
    hf = HuggingFaceEmbeddings(
        model_kwargs=yaml_config.embedding.get('model_kwargs'),
        encode_kwargs=yaml_config.embedding.get('encode_kwargs'),
        multi_process=yaml_config.embedding.get('multi_process'),
        cache_folder=yaml_config.embedding.get('cache_path'),
    )
    return hf
