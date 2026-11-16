import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.prompts import ChatPromptTemplate

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


if __name__ == "__main__":
    unittest.main()
