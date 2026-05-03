import gptcache
from gptcache.adapter.api import init_similar_cache
from langchain.globals import set_llm_cache
from langchain.cache import GPTCache
from gptcache.embedding import LangChain
from gptcache.similarity_evaluation import SearchDistanceEvaluation

from universal.config import config
from initialize.embedding import embedding

_cache_initialized = False


def init_cache():
    global _cache_initialized
    if _cache_initialized or not config.cache.get('use'):
        return
    set_llm_cache(GPTCache(_init_gptcache))
    _cache_initialized = True


def _init_gptcache(cache_obj: gptcache.Cache, llm: str):
    evaluation_config = config.cache.get('search_distance_evaluation')
    init_similar_cache(
        cache_obj=cache_obj,
        embedding=LangChain(embedding()),
        data_dir=f"map_cache_{config.llm.get('name')}",
        evaluation=SearchDistanceEvaluation(
            max_distance=evaluation_config.get('max_distance'),
            positive=evaluation_config.get('positive'),
        ),
    )
