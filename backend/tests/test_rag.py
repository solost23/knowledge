import unittest
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from app.knowledge.rag import QuestionService


class QuestionServiceTestCase(unittest.TestCase):
    def test_question_returns_answer_and_sources_without_model_call(self):
        docs = [
            Document(
                page_content='报销额度为600元，示例：{"amount": 600}',
                metadata={"source": "报销制度.md"},
            )
        ]
        messages = []

        def answer(prompt):
            messages.extend(prompt.to_messages())
            return AIMessage(content="基于资料的回答")

        with patch("app.knowledge.rag.get_chroma") as get_chroma:
            get_chroma.return_value.search.return_value = [(docs[0], 0.86)]
            result, status_code = QuestionService().question(
                "报销额度是多少？",
                history=[
                    {"role": "user", "content": "我想了解报销规则"},
                    {"role": "assistant", "content": "请问需要了解哪一项？"},
                ],
                llm=RunnableLambda(answer),
            )

        self.assertEqual(200, status_code)
        self.assertEqual(0, result.get("code"))
        self.assertEqual("基于资料的回答", result["data"]["answer"])
        self.assertEqual(
            [
                {
                    "source": "报销制度.md",
                    "content": docs[0].page_content,
                }
            ],
            result["data"]["sources"],
        )
        self.assertEqual(["system", "human", "ai", "human"], [m.type for m in messages])
        self.assertEqual("我想了解报销规则", messages[1].content)
        self.assertIn(docs[0].page_content, messages[-1].content)
        self.assertIn("当前问题：报销额度是多少？", messages[-1].content)

    def test_question_skips_model_when_no_relevant_documents(self):
        with (
            patch("app.knowledge.rag.get_chroma") as get_chroma,
            patch("app.knowledge.rag.LLM") as model,
        ):
            get_chroma.return_value.search.return_value = []
            result, status_code = QuestionService().question("今天适合出门吗？")

        self.assertEqual(200, status_code)
        self.assertEqual("未找到相关资料", result["message"])
        self.assertEqual(
            "知识库中没有找到与问题直接相关的资料。", result["data"]["answer"]
        )
        self.assertEqual([], result["data"]["sources"])
        model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
