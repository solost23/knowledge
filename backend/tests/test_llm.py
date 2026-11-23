import asyncio
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from openai import APIStatusError, OpenAI
from pydantic import BaseModel

from app.infrastructure.llm import LLM, LLMConfigurationError


class LLMTestCase(unittest.TestCase):
    def setUp(self):
        self.settings = SimpleNamespace(
            llm={"name": "chat_gpt"},
            chat_gpt={
                "model": "test-gpt",
                "base_url": "http://127.0.0.1:9000/v1",
                "temperature": 0.2,
                "max_token": 1024,
                "timeout": 15,
            },
            chat_glm={},
        )
        config_patch = patch("app.infrastructure.llm.config", self.settings)
        config_patch.start()
        self.addCleanup(config_patch.stop)
        env_patch = patch.dict(
            os.environ, {"OPENAI_API_KEY": "offline-test-key"}, clear=True
        )
        env_patch.start()
        self.addCleanup(env_patch.stop)

    def test_constructs_model_from_application_settings(self):
        model = LLM()

        self.assertEqual("test-gpt", model.model_name)
        self.assertEqual("http://127.0.0.1:9000/v1", model.openai_api_base)
        self.assertEqual("offline-test-key", model.openai_api_key.get_secret_value())
        self.assertEqual(0.2, model.temperature)
        self.assertEqual(1024, model.max_tokens)
        self.assertEqual(15, model.request_timeout)
        self.assertEqual(1, model.max_retries)

    def test_selects_glm_from_application_settings(self):
        self.settings.llm["name"] = "chat_glm"

        model = LLM()

        self.assertEqual("chatglm3", model.model_name)
        self.assertEqual("http://127.0.0.1:8000/v1", model.openai_api_base)
        self.assertEqual("EMPTY", model.openai_api_key.get_secret_value())

    def test_explicit_glm_selection_normalizes_endpoint(self):
        self.settings.llm["name"] = "unknown"
        os.environ["CHATGLM_API_KEY"] = "local-test-key"
        for endpoint in (
            "http://127.0.0.1:8000",
            "http://127.0.0.1:8000/",
            "http://127.0.0.1:8000/v1",
            "http://127.0.0.1:8000/v1/",
        ):
            with self.subTest(endpoint=endpoint):
                self.settings.chat_glm = {"endpoint_url": endpoint}

                model = LLM("chat_glm")

                self.assertEqual("http://127.0.0.1:8000/v1", model.openai_api_base)
                self.assertEqual(
                    "local-test-key", model.openai_api_key.get_secret_value()
                )

    def test_missing_api_key_raises_configuration_error(self):
        os.environ.pop("OPENAI_API_KEY")

        with self.assertRaisesRegex(LLMConfigurationError, "OPENAI_API_KEY 未设置"):
            LLM()

    def test_unknown_provider_raises_configuration_error(self):
        self.settings.llm["name"] = "unknown"

        with self.assertRaisesRegex(LLMConfigurationError, "未知的 LLM 配置: unknown"):
            LLM()

    def test_instance_runs_in_prompt_pipeline(self):
        model = LLM()
        chain = (
            ChatPromptTemplate.from_template("{question}") | model | StrOutputParser()
        )
        result = ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="回答"))]
        )

        with patch.object(LLM, "_generate", return_value=result) as generate:
            answer = chain.invoke({"question": "问题"})

        self.assertEqual("回答", answer)
        self.assertEqual("问题", generate.call_args.args[0][0].content)

    def api_error(self, status=503, code="model_not_found"):
        return APIStatusError(
            "unavailable",
            response=httpx.Response(
                status, request=httpx.Request("POST", "https://example.com/v1")
            ),
            body={"error": {"code": code, "message": "unavailable"}},
        )

    def enable_fallback(self, candidates=None):
        self.settings.chat_gpt.update(
            auto_fallback=True, fallback_models=candidates or []
        )
        return LLM()

    def test_fallback_preserves_tools_and_reuses_successful_model(self):
        model = self.enable_fallback(["backup", "backup"])
        result = ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content="",
                        tool_calls=[{"name": "route", "args": {}, "id": "call-1"}],
                    )
                )
            ]
        )
        names = []

        def generate(client, messages, **kwargs):
            names.append(client.model_name)
            self.assertEqual([{"type": "function"}], kwargs["tools"])
            if client.model_name == "test-gpt":
                raise self.api_error()
            return result

        with patch.object(ChatOpenAI, "_generate", autospec=True, side_effect=generate):
            for _ in range(2):
                model.invoke(
                    "hello",
                    tools=[{"type": "function"}],
                    tool_choice={"function": {"name": "route"}},
                )
        self.assertEqual(["test-gpt", "backup", "backup"], names)
        self.assertEqual("test-gpt", model.model_name)

    def test_discovers_chat_models_only_after_failure(self):
        model = self.enable_fallback()
        root = Mock()
        root.with_options.return_value.models.list.return_value.data = [
            SimpleNamespace(id=name)
            for name in [
                "text-embedding-3-small",
                "gpt-image-1",
                "test-gpt",
                "gpt-backup",
            ]
        ]
        model.root_client = root
        result = ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="ok"))]
        )
        with patch.object(
            ChatOpenAI, "_generate", side_effect=[self.api_error(), result]
        ):
            self.assertEqual("ok", model.invoke("hello").content)
        self.assertEqual("gpt-backup", model._selected_model)
        root.with_options.return_value.models.list.assert_called_once()

    def test_no_discovery_when_primary_succeeds(self):
        model = self.enable_fallback()
        model.root_client = Mock()
        result = ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="ok"))]
        )
        with patch.object(ChatOpenAI, "_generate", return_value=result):
            model.invoke("hello")
        model.root_client.with_options.assert_not_called()

    def test_does_not_switch_for_auth_quota_or_generic_errors(self):
        model = self.enable_fallback(["backup"])
        for status, code in [
            (401, "invalid_api_key"),
            (403, "forbidden"),
            (429, "insufficient_quota"),
            (503, "server_error"),
            (400, "context_length_exceeded"),
        ]:
            error = self.api_error(status, code)
            with (
                self.subTest(status=status),
                patch.object(ChatOpenAI, "_generate", side_effect=error) as generate,
            ):
                with self.assertRaises(APIStatusError) as raised:
                    model.invoke("hello")
                self.assertIs(error, raised.exception)
                self.assertEqual(1, generate.call_count)

    def test_limits_attempts_and_preserves_error(self):
        model = self.enable_fallback(["a", "b", "c", "d"])
        error = self.api_error()
        with patch.object(ChatOpenAI, "_generate", side_effect=error) as generate:
            with self.assertRaises(APIStatusError) as raised:
                model.invoke("hello")
        self.assertEqual(4, generate.call_count)
        self.assertIs(error, raised.exception)

    def test_disabled_fallback_does_not_switch(self):
        model = LLM()
        with patch.object(
            ChatOpenAI, "_generate", side_effect=self.api_error()
        ) as generate:
            with self.assertRaises(APIStatusError):
                model.invoke("hello")
        self.assertEqual(1, generate.call_count)

    def test_discovery_failure_preserves_original_error(self):
        model = self.enable_fallback()
        model.root_client = Mock()
        model.root_client.with_options.side_effect = RuntimeError("discovery failed")
        error = self.api_error()
        with patch.object(ChatOpenAI, "_generate", side_effect=error):
            with self.assertRaises(APIStatusError) as raised:
                model.invoke("hello")
        self.assertIs(error, raised.exception)

    def test_async_invocation_uses_fallback(self):
        model = self.enable_fallback(["backup"])
        result = ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="ok"))]
        )
        with patch.object(
            ChatOpenAI, "_generate", side_effect=[self.api_error(), result]
        ):
            self.assertEqual("ok", asyncio.run(model.ainvoke("hello")).content)

    def test_retries_when_model_ignores_required_tool(self):
        model = self.enable_fallback(["backup"])
        plain = ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="ok"))]
        )
        structured = ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content="",
                        tool_calls=[{"name": "route", "args": {}, "id": "call-1"}],
                    )
                )
            ]
        )
        with patch.object(ChatOpenAI, "_generate", side_effect=[plain, structured]):
            result = model.invoke("hello", tool_choice={"function": {"name": "route"}})
        self.assertEqual("route", result.tool_calls[0]["name"])
        self.assertEqual("backup", model._selected_model)

    def test_structured_output_fallback_through_real_sdk(self):
        class Route(BaseModel):
            mode: str

        model = self.enable_fallback()
        requests = []

        def handle(request):
            if request.method == "GET":
                return httpx.Response(
                    200,
                    json={
                        "object": "list",
                        "data": [
                            {
                                "id": "gpt-backup",
                                "object": "model",
                                "created": 0,
                                "owned_by": "test",
                            }
                        ],
                    },
                )
            payload = json.loads(request.content)
            requests.append(payload)
            if payload["model"] == "test-gpt":
                return httpx.Response(
                    503,
                    json={
                        "error": {
                            "code": "model_not_found",
                            "message": "No available channel",
                        }
                    },
                )
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "gpt-backup",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "tool_calls",
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call-1",
                                        "type": "function",
                                        "function": {
                                            "name": "Route",
                                            "arguments": '{"mode":"rag"}',
                                        },
                                    }
                                ],
                            },
                        }
                    ],
                },
            )

        with httpx.Client(transport=httpx.MockTransport(handle)) as http:
            model.root_client = OpenAI(
                api_key="offline-test-key",
                base_url="https://example.com/v1",
                http_client=http,
                max_retries=0,
            )
            model.client = model.root_client.chat.completions
            router = model.with_structured_output(Route, method="function_calling")
            self.assertEqual("rag", router.invoke("hello").mode)
            self.assertEqual("rag", router.invoke("again").mode)
        self.assertEqual(
            ["test-gpt", "gpt-backup", "gpt-backup"],
            [request["model"] for request in requests],
        )
        self.assertEqual(requests[0]["tools"], requests[1]["tools"])
        self.assertEqual(requests[0]["tool_choice"], requests[1]["tool_choice"])


if __name__ == "__main__":
    unittest.main()
