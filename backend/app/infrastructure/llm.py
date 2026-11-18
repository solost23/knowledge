import os
from typing import ClassVar

from langchain_openai import ChatOpenAI

from app.core.settings import config


class LLMConfigurationError(RuntimeError):
    """Raised when the selected model cannot be configured from application settings."""


class LLM(ChatOpenAI):
    """从应用配置创建可直接用于 RAG 和 Agent 的聊天模型。"""

    CHAT_GPT: ClassVar[str] = "chat_gpt"
    CHAT_GLM: ClassVar[str] = "chat_glm"

    def __init__(self, name: str | None = None):
        name = config.llm.get("name") if name is None else name

        if name == self.CHAT_GPT:
            settings = config.chat_gpt
            default_model = "gpt-4o-mini"

            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise LLMConfigurationError("环境变量 OPENAI_API_KEY 未设置")

            base_url = settings.get("base_url")

        elif name == self.CHAT_GLM:
            settings = config.chat_glm
            default_model = "chatglm3"
            api_key = os.environ.get("CHATGLM_API_KEY", "EMPTY")

            base_url = settings.get("endpoint_url", "http://127.0.0.1:8000").rstrip("/")
            if not base_url.endswith("/v1"):
                base_url += "/v1"

        else:
            raise LLMConfigurationError(f"未知的 LLM 配置: {name}")

        super().__init__(
            model=settings.get("model", default_model),
            api_key=api_key,
            base_url=base_url,
            temperature=settings.get("temperature", 0),
            max_tokens=settings.get("max_token", 4096),
            timeout=settings.get("timeout", 60),
            max_retries=1,
        )
