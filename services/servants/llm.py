import os

from langchain_community.llms.chatglm import ChatGLM
from langchain_community.llms.openai import OpenAIChat
from langchain_wenxin.llms import Wenxin
from universal.config import config

ChatGLmName = 'chat_glm'
ChatGptName = 'chat_gpt'
WenXinName = "wen_xin"


def chat_glm() -> ChatGLM:
    return ChatGLM(
        endpoint_url=config.chat_glm.get('endpoint_url'),
        max_token=config.chat_glm.get('max_token'),
        timeout=config.chat_glm.get('timeout', 60),
    )


def chat_gpt() -> OpenAIChat:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("环境变量 OPENAI_API_KEY 未设置")
    return OpenAIChat(
        max_token=config.chat_gpt.get('max_token'),
    )


def wen_xin() -> Wenxin:
    api_key = os.environ.get("WENXIN_API_KEY")
    secret_key = os.environ.get("WENXIN_SECRET_KEY")
    if not api_key or not secret_key:
        raise EnvironmentError("环境变量 WENXIN_API_KEY 或 WENXIN_SECRET_KEY 未设置")
    return Wenxin(
        baidu_api_key=api_key,
        baidu_secret_key=secret_key,
    )
