import os
import sqlite3
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from langgraph.checkpoint.base import empty_checkpoint
from sqlalchemy import func, inspect, select
from sqlalchemy.exc import IntegrityError

from app.agents.checkpoints import (
    checkpoint_values,
    clear_run_cancel,
    create_run,
    finish_run,
    get_run,
    open_checkpointer,
    request_run_cancel,
    run_config,
    save_run_result,
    set_run_mode,
    update_run_progress,
)
from app.infrastructure.conversation_repository import (
    append_exchange,
    create_conversation,
    delete_conversation,
    get_conversation,
    recent_history,
    save_summary,
)
from app.infrastructure.database import dispose_databases, get_database
from app.infrastructure.document_repository import (
    delete_doc,
    exists_doc,
    insert_doc,
    list_docs,
)
from app.infrastructure.document_upload_repository import (
    get_upload_job,
    list_active_upload_jobs,
    update_upload_progress,
)
from app.infrastructure.models import ConversationMessage, DocumentRecord


class DatabaseTestCase(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        env = patch.dict(
            os.environ,
            {
                "KNOWLEDGE_DB": str(self.root / "knowledge.db"),
                "CONVERSATION_DB": str(self.root / "conversations.db"),
                "AGENT_CHECKPOINT_DB": str(self.root / "agent.db"),
            },
        )
        env.start()
        self.addCleanup(env.stop)
        self.addCleanup(dispose_databases)

    def _legacy_database(self, filename: str, schema: str) -> None:
        with closing(sqlite3.connect(self.root / filename)) as connection:
            connection.executescript(schema)

    def test_failed_transaction_rolls_back_all_writes(self):
        database = get_database()
        with self.assertRaises(IntegrityError), database.session() as session:
            session.add(DocumentRecord(name="manual.pdf"))
            session.flush()
            session.add(DocumentRecord(name="manual.pdf"))
            session.flush()

        self.assertEqual([], list_docs())
        insert_doc("manual.pdf")
        self.assertTrue(exists_doc("manual.pdf"))

    def test_threads_reuse_engine_and_have_separate_sessions(self):
        barrier = threading.Barrier(4)

        def write_document(index):
            database = get_database()
            with database.session() as session:
                barrier.wait(timeout=5)
                session.add(DocumentRecord(name=f"{index}.pdf"))
            return database.engine, session

        with ThreadPoolExecutor(max_workers=4) as executor:
            resources = list(executor.map(write_document, range(4)))

        self.assertEqual(1, len({id(engine) for engine, _ in resources}))
        self.assertEqual(4, len({id(session) for _, session in resources}))
        self.assertEqual(4, len(list_docs()))

    def test_existing_documents_and_uploads_keep_their_data(self):
        self._legacy_database(
            "knowledge.db",
            """
            CREATE TABLE docs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                created_at DATETIME DEFAULT (datetime('now', 'localtime'))
            );
            INSERT INTO docs VALUES (7, '旧文档.pdf', '2026-01-01 08:00:00');
            CREATE TABLE document_upload_jobs (
                id TEXT PRIMARY KEY, filename TEXT NOT NULL, status TEXT NOT NULL,
                progress INTEGER NOT NULL, stage TEXT NOT NULL, error TEXT,
                created_at DATETIME DEFAULT (datetime('now', 'localtime')),
                updated_at DATETIME DEFAULT (datetime('now', 'localtime'))
            );
            INSERT INTO document_upload_jobs (id, filename, status, progress, stage)
            VALUES ('upload-1', '旧文档.pdf', 'processing', 42, '正在解析');
        """,
        )

        insert_doc("旧文档.pdf")
        self.assertEqual(
            [{"id": 7, "name": "旧文档.pdf", "created_at": "2026-01-01 08:00:00"}],
            list_docs(),
        )
        self.assertEqual(42, get_upload_job("upload-1")["progress"])
        self.assertEqual("upload-1", list_active_upload_jobs()[0]["id"])
        update_upload_progress("upload-1", 60, "开始写入")
        self.assertEqual(60, get_upload_job("upload-1")["progress"])
        self.assertTrue(delete_doc("旧文档.pdf"))
        self.assertFalse(delete_doc("旧文档.pdf"))
        insert_doc("新文档.pdf")
        self.assertGreater(list_docs()[0]["id"], 7)

    def test_legacy_conversations_gain_summary_fields_and_keep_messages(self):
        self._legacy_database(
            "conversations.db",
            """
            CREATE TABLE conversations (
                id TEXT PRIMARY KEY, title TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE TABLE conversation_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content TEXT NOT NULL, result_json TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            INSERT INTO conversations (id, title) VALUES ('conversation-1', '原有标题');
            INSERT INTO conversation_messages (conversation_id, role, content)
            VALUES ('conversation-1', 'user', '原有问题');
        """,
        )

        conversation = get_conversation("conversation-1")
        self.assertEqual("原有问题", conversation["messages"][0]["content"])
        append_exchange("conversation-1", "后续问题", "回答", {"answer": "回答"})
        conversation = get_conversation("conversation-1")
        self.assertEqual("原有标题", conversation["title"])
        self.assertEqual(
            ["user", "user", "assistant"],
            [item["role"] for item in conversation["messages"]],
        )
        self.assertEqual({"answer": "回答"}, conversation["messages"][-1]["result"])
        self.assertEqual(3, len(recent_history("conversation-1")))
        indexes = inspect(get_database("conversations").engine).get_indexes(
            "conversation_messages"
        )
        self.assertIn(
            "idx_conversation_messages_conversation",
            {index["name"] for index in indexes},
        )

        self.assertTrue(delete_conversation("conversation-1"))
        self.assertFalse(delete_conversation("conversation-1"))
        with get_database("conversations").session() as session:
            self.assertEqual(
                0, session.scalar(select(func.count()).select_from(ConversationMessage))
            )

    def test_summary_cannot_be_replaced_by_an_older_batch(self):
        conversation_id = create_conversation()["id"]
        append_exchange(conversation_id, "问题", "回答", {})
        save_summary(conversation_id, "新摘要", 2)
        save_summary(conversation_id, "旧摘要", 1)
        history = recent_history(conversation_id)
        self.assertEqual(1, len(history))
        self.assertIn("新摘要", history[0]["content"])

    def test_missing_conversation_does_not_leave_orphan_messages(self):
        with self.assertRaises(KeyError):
            append_exchange("missing", "问题", "回答", {})
        with get_database("conversations").session() as session:
            self.assertEqual(
                0, session.scalar(select(func.count()).select_from(ConversationMessage))
            )

    def test_legacy_agent_runs_gain_fields_and_keep_progress_after_reopening(self):
        self._legacy_database(
            "agent.db",
            """
            CREATE TABLE agent_runs (
                execution_id TEXT PRIMARY KEY, task TEXT NOT NULL,
                status TEXT NOT NULL, error_code TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            INSERT INTO agent_runs (execution_id, task, status, created_at)
            VALUES ('old-run', '旧任务', 'failed', '2026-01-01 00:00:00');
        """,
        )

        with open_checkpointer() as checkpointer:
            run = get_run(checkpointer, "old-run")
            self.assertEqual("2026-01-01 00:00:00Z", run["started_at"])
            self.assertEqual({}, run["progress"])
            self.assertIsNone(run["result"])
            update_run_progress(checkpointer, "old-run", {"stage": "planning"})
            update_run_progress(checkpointer, "old-run", {"current_step": 1})
            set_run_mode(checkpointer, "old-run", "rag", route_reason="单次检索")
            set_run_mode(checkpointer, "old-run", "rag")
            save_run_result(checkpointer, "old-run", {"answer": "答案"})

        dispose_databases()
        with open_checkpointer() as checkpointer:
            run = get_run(checkpointer, "old-run")
            self.assertEqual({"stage": "planning", "current_step": 1}, run["progress"])
            self.assertEqual("单次检索", run["route_reason"])
            self.assertEqual({"answer": "答案"}, run["result"])
            self.assertFalse(run["cancel_requested"])

    def test_cancellation_prevents_success_and_terminal_runs_cannot_finish_twice(self):
        self.addCleanup(clear_run_cancel, "run-1")
        with open_checkpointer() as writer, open_checkpointer() as reader:
            create_run(writer, "run-1", "任务", {"stage": "planning"})
            self.assertTrue(request_run_cancel(reader, "run-1"))
            self.assertFalse(
                finish_run(writer, "run-1", "completed", respect_cancel=True)
            )
            self.assertEqual("running", get_run(reader, "run-1")["status"])
            self.assertTrue(finish_run(writer, "run-1", "cancelled"))
            self.assertFalse(finish_run(reader, "run-1", "completed"))
            self.assertFalse(request_run_cancel(reader, "run-1"))
            self.assertEqual("cancelled", get_run(reader, "run-1")["status"])

    def test_orm_and_langgraph_can_share_an_in_memory_database(self):
        with patch.dict(os.environ, {"AGENT_CHECKPOINT_DB": ":memory:"}):
            with open_checkpointer() as checkpointer:
                create_run(checkpointer, "memory-run", "内存任务", {})
                checkpoint = empty_checkpoint()
                checkpoint["channel_values"] = {"answer": "已保存"}
                checkpoint["channel_versions"] = {"answer": "1"}
                checkpoint_config = run_config("memory-run")
                checkpoint_config["configurable"]["checkpoint_ns"] = ""
                checkpointer.put(checkpoint_config, checkpoint, {}, {"answer": "1"})
                update_run_progress(checkpointer, "memory-run", {"stage": "completed"})
                self.assertTrue(finish_run(checkpointer, "memory-run", "completed"))
                self.assertEqual(
                    "已保存", checkpoint_values(checkpointer, "memory-run")["answer"]
                )
                self.assertEqual(
                    "completed", get_run(checkpointer, "memory-run")["status"]
                )


if __name__ == "__main__":
    unittest.main()
