import unittest
from unittest.mock import patch

from langchain_core.documents import Document

from app.knowledge.rag import QuestionService


class FakeChain:
    def __or__(self, _parser):
        return self

    def invoke(self, _inputs):
        return "基于资料的回答"


class FakePrompt:
    def __or__(self, _llm):
        return FakeChain()


class QuestionServiceTestCase(unittest.TestCase):
    def test_question_returns_answer_and_sources_without_model_call(self):
        docs = [Document(page_content="报销额度为600元", metadata={"source": "报销制度.md"})]
        with patch("app.knowledge.rag.chroma.db.similarity_search", return_value=docs), \
                patch("app.knowledge.rag.PROMPT", FakePrompt()):
            result, status_code = QuestionService().question("报销额度是多少？", llm=object())

        self.assertEqual(200, status_code)
        self.assertEqual(0, result.get('code'))
        self.assertEqual("基于资料的回答", result["data"]["answer"])
        self.assertEqual([{
            "source": "报销制度.md",
            "content": "报销额度为600元",
        }], result["data"]["sources"])


if __name__ == "__main__":
    unittest.main()
