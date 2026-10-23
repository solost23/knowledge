import os

from langchain_openai import ChatOpenAI

from app.core.settings import config

ChatGptName = 'chat_gpt'
ChatGLmName = 'chat_glm'


def build_configured_llm() -> ChatOpenAI:
    name = config.llm.get('name')
    if name == ChatGLmName:
        return chat_glm()
    if name == ChatGptName:
        return chat_gpt()
    raise EnvironmentError(f'未知的 LLM 配置: {name}')


def _chat(model: str, base_url: str | None, api_key: str, temperature: float, max_tokens: int) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=1,
    )


def chat_gpt() -> ChatOpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("环境变量 OPENAI_API_KEY 未设置")
    return _chat(
        model=config.chat_gpt.get('model', 'gpt-4o-mini'),
        base_url=config.chat_gpt.get('base_url'),
        api_key=api_key,
        temperature=config.chat_gpt.get('temperature', 0),
        max_tokens=config.chat_gpt.get('max_token', 4096),
    )


def chat_glm() -> ChatOpenAI:
    """任何暴露 OpenAI 兼容接口的本地推理服务（vLLM / Ollama / ChatGLM）都可以走这里。"""
    base_url = config.chat_glm.get('endpoint_url', 'http://127.0.0.1:8000').rstrip('/')
    if not base_url.endswith('/v1'):
        base_url += '/v1'
    return _chat(
        model=config.chat_glm.get('model', 'chatglm3'),
        base_url=base_url,
        api_key=os.environ.get('CHATGLM_API_KEY', 'EMPTY'),
        temperature=config.chat_glm.get('temperature', 0),
        max_tokens=config.chat_glm.get('max_token', 4096),
    )
