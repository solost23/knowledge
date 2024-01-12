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
    )


def chat_gpt() -> OpenAIChat:
    os.environ["OPENAI_API_KEY"] = config.chat_gpt.get('api_key')
    return OpenAIChat(
        max_token=config.chat_gpt.get('max_token'),
    )


def wen_xin() -> Wenxin:
    return Wenxin(
        baidu_api_key=config.wen_xin.get('api_key'),
        baidu_secret_key=config.wen_xin.get('secret_key')
    )
