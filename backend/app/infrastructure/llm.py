import asyncio
import os
from typing import ClassVar

from langchain_openai import ChatOpenAI
from loguru import logger
from openai import APIStatusError
from pydantic import PrivateAttr

from app.core.settings import config


class LLMConfigurationError(RuntimeError):
    """Raised when the selected model cannot be configured from application settings."""


class LLM(ChatOpenAI):
    """从应用配置创建可直接用于 RAG 和 Agent 的聊天模型。"""

    CHAT_GPT: ClassVar[str] = "chat_gpt"
    CHAT_GLM: ClassVar[str] = "chat_glm"

    _auto_fallback: bool = PrivateAttr(default=False)
    _fallback_models: list[str] = PrivateAttr(default_factory=list)
    _selected_model: str | None = PrivateAttr(default=None)

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

        self._auto_fallback = settings.get("auto_fallback", False)
        candidates = settings.get("fallback_models", [])
        if not isinstance(candidates, list) or any(
            not isinstance(model, str) or not model.strip() for model in candidates
        ):
            raise LLMConfigurationError("fallback_models 必须是非空模型名组成的列表")
        self._fallback_models = list(dict.fromkeys(candidates))

    @staticmethod
    def _can_fallback(error: APIStatusError) -> bool:
        # Only model availability/capability errors warrant changing models.
        # Authentication, quota, context limits and generic 5xx errors must surface.
        if error.status_code in (401, 403, 429):
            return False
        body = error.body if isinstance(error.body, dict) else {}
        detail = body.get("error", body)
        if not isinstance(detail, dict):
            detail = {}
        code = str(detail.get("code", "")).lower()
        message = str(detail.get("message", error.message)).lower()
        if code in {"model_not_found", "model_not_available", "no_available_channel"}:
            return True
        if "no available channel" in message:
            return True
        return (
            error.status_code == 400
            and any(
                field in message
                for field in ("tools", "tool_choice", "function calling")
            )
            and any(word in message for word in ("not support", "unsupported"))
        )

    @staticmethod
    def _is_chat_candidate(model: str) -> bool:
        name = model.lower().split("/")[-1]
        return name.startswith(
            ("gpt-", "claude-", "gemini-", "qwen", "deepseek-", "llama", "glm-")
        ) and not any(
            word in name
            for word in (
                "embedding",
                "rerank",
                "image",
                "audio",
                "tts",
                "realtime",
                "vision",
                "instruct",
                "moderation",
                "transcribe",
            )
        )

    def _candidates(self):
        # An explicit list also acts as an allowlist; never expand it remotely.
        if self._fallback_models:
            return self._fallback_models
        try:
            page = self.root_client.with_options(max_retries=0).models.list()
            return list(
                dict.fromkeys(
                    model.id for model in page.data if self._is_chat_candidate(model.id)
                )
            )
        except Exception:
            logger.warning("无法读取备用模型列表，请配置 chat_gpt.fallback_models")
            return []

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        selected = self._selected_model or self.model_name
        attempted = set()
        candidates = [selected]
        last_error = None
        # Discovery is lazy, and at most three alternatives are attempted.
        for index in range(4):
            if index == 1:
                candidates.extend(
                    model for model in self._candidates() if model != selected
                )
            if index >= len(candidates):
                break
            model = candidates[index]
            if model in attempted:
                continue
            attempted.add(model)
            # Copy so model-specific payload handling uses the candidate's name.
            client = self.model_copy(update={"model_name": model})
            try:
                result = ChatOpenAI._generate(
                    client, messages, stop=stop, run_manager=run_manager, **kwargs
                )
            except APIStatusError as error:
                if not self._auto_fallback or not self._can_fallback(error):
                    raise
                last_error = error
                logger.warning("模型 {} 不可用，尝试备用模型", model)
                continue
            choice = kwargs.get("tool_choice")
            required_name = (
                choice.get("function", {}).get("name")
                if isinstance(choice, dict)
                else None
            )
            if (
                self._auto_fallback
                and required_name
                and not any(
                    call.get("name") == required_name
                    for generation in result.generations
                    for call in getattr(generation.message, "tool_calls", [])
                )
            ):
                last_error = RuntimeError("模型未返回请求的结构化工具结果")
                logger.warning("模型 {} 未返回请求的工具结果，尝试备用模型", model)
                continue
            self._selected_model = model
            if model != selected:
                logger.info("已切换到备用模型 {}", model)
            return result
        # Preserve the provider's actionable error when discovery/all attempts fail.
        raise last_error

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return await asyncio.to_thread(
            self._generate,
            messages,
            stop=stop,
            run_manager=run_manager.get_sync() if run_manager else None,
            **kwargs,
        )
