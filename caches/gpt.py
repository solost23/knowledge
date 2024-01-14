import hashlib

import gptcache
from gptcache.adapter.api import init_similar_cache
from langchain.globals import set_llm_cache
from langchain.cache import GPTCache
from gptcache.embedding import LangChain
from gptcache.similarity_evaluation import SearchDistanceEvaluation

from universal.config import config
from initialize.embedding import embedding


def gpt():
    if not config.cache.get('use'):
        return

    # 开启gpt_cache缓存
    set_llm_cache(GPTCache(init_gptcache))


def init_gptcache(cache_obj: gptcache.Cache, llm: str):
    # cache_base = CacheBase('sqlite')
    # vector_base = VectorBase(
    #     'milvus',
    #     host='localhost',
    #     port='19530',
    #     dimension=4,
    # )
    # data_manager = get_data_manager(cache_base, vector_base)
    # cache_obj.init(
    #     pre_embedding_func=get_prompt,
    #     data_manager=manager_factory(
    #         manager="map",
    #         data_dir=f'map_cache_{hash(llm)}'
    #     ),
    #     similarity_evaluation=SearchDistanceEvaluation(),
    # )

    # 相似性缓存
    evaluation_config = config.cache.get('search_distance_evaluation')
    init_similar_cache(
        cache_obj=cache_obj,
        embedding=LangChain(embedding(config)),
        data_dir=f"map_cache_{config.llm.get('name')}",
        evaluation=SearchDistanceEvaluation(
            max_distance=evaluation_config.get('max_distance'),
            positive=evaluation_config.get('positive'),
        ),
    )
