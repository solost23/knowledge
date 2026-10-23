import os
import tempfile
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage

from app.agents.memory import prepare_history
from app.infrastructure.conversation_repository import (
    append_exchange,
    compression_batch,
    create_conversation,
    get_conversation,
    recent_history,
)


class FakeSummarizer:
    def __init__(self):
        self.calls = 0

    def invoke(self, _messages):
        self.calls += 1
        return AIMessage(content="用户正在比较报销制度，已确认申请期限为30天。")


class ConversationMemoryTestCase(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.database_env = patch.dict(os.environ, {
            "CONVERSATION_DB": os.path.join(self.temp_directory.name, "conversations.sqlite")
        })
        self.database_env.start()
        self.addCleanup(self.database_env.stop)
        self.addCleanup(self.temp_directory.cleanup)

    def test_long_conversation_is_summarized_and_recent_turns_are_kept(self):
        conversation = create_conversation()
        conversation_id = conversation["id"]
        for index in range(10):
            append_exchange(
                conversation_id,
                f"用户问题 {index}",
                f"助手回答 {index}",
                {"answer": f"助手回答 {index}", "mode": "rag", "sources": []},
            )

        summarizer = FakeSummarizer()
        history = prepare_history(conversation_id, summarizer)
        persisted = get_conversation(conversation_id)

        self.assertEqual(1, summarizer.calls)
        self.assertIsNone(compression_batch(conversation_id))
        self.assertIn("30天", history[0]["content"])
        self.assertEqual("assistant", history[0]["role"])
        self.assertEqual(6, len(history) - 1)
        self.assertEqual(20, len(persisted["messages"]))
        self.assertNotIn("用户问题 0", "\n".join(item["content"] for item in history))

    def test_short_conversation_does_not_call_summarizer(self):
        conversation = create_conversation()
        append_exchange(
            conversation["id"], "第一个问题", "第一个回答", {"answer": "第一个回答"}
        )
        summarizer = FakeSummarizer()

        history = prepare_history(conversation["id"], summarizer)

        self.assertEqual(0, summarizer.calls)
        self.assertEqual(2, len(history))

    def test_history_is_isolated_and_bounded_per_conversation(self):
        first = create_conversation()
        second = create_conversation()
        append_exchange(
            first["id"], "第一轮" * 300, "回答" * 2000, {"answer": "回答"}
        )

        first_history = recent_history(first["id"])
        second_history = recent_history(second["id"])

        self.assertLessEqual(sum(len(item["content"]) for item in first_history), 6000)
        self.assertEqual([], second_history)
        self.assertIn("第一轮", "\n".join(item["content"] for item in first_history))


if __name__ == "__main__":
    unittest.main()
