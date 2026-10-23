import json
import os
import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path
from uuid import uuid4

from app.core.settings import DATA_DIR

MAX_HISTORY_MESSAGES = 10
MAX_HISTORY_CHARS = 6000
MAX_HISTORY_MESSAGE_CHARS = 2500
KEEP_RECENT_MESSAGES = 6
COMPRESS_AFTER_MESSAGES = 8
COMPRESS_AFTER_CHARS = 8000
MAX_COMPRESSION_BATCH_CHARS = 8000
MAX_COMPRESSION_MESSAGE_CHARS = 2000


def _database_path() -> str:
    path = Path(os.environ.get("CONVERSATION_DB", DATA_DIR / "conversations.db"))
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


@contextmanager
def _connection():
    connection = sqlite3.connect(_database_path(), timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    with closing(connection):
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                summary TEXT NOT NULL DEFAULT '',
                summarized_through INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS conversation_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                result_json TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_conversation_messages_conversation
                ON conversation_messages(conversation_id, id);
            CREATE INDEX IF NOT EXISTS idx_conversations_updated
                ON conversations(updated_at DESC);
        """)
        columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(conversations)").fetchall()
        }
        if "summary" not in columns:
            connection.execute(
                "ALTER TABLE conversations ADD COLUMN summary TEXT NOT NULL DEFAULT ''"
            )
        if "summarized_through" not in columns:
            connection.execute(
                "ALTER TABLE conversations ADD COLUMN summarized_through INTEGER NOT NULL DEFAULT 0"
            )
        connection.commit()
        try:
            yield connection
        finally:
            connection.commit()


def _conversation(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "title": row["title"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def create_conversation() -> dict:
    conversation_id = str(uuid4())
    with _connection() as connection:
        connection.execute(
            "INSERT INTO conversations (id, title) VALUES (?, ?)",
            (conversation_id, "新对话"),
        )
        row = connection.execute(
            "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
    return _conversation(row)


def list_conversations(limit: int = 100) -> list[dict]:
    with _connection() as connection:
        rows = connection.execute(
            "SELECT * FROM conversations ORDER BY updated_at DESC, id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [_conversation(row) for row in rows]


def get_conversation(conversation_id: str) -> dict | None:
    with _connection() as connection:
        conversation = connection.execute(
            "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
        if conversation is None:
            return None
        rows = connection.execute(
            "SELECT id, role, content, result_json, created_at "
            "FROM conversation_messages WHERE conversation_id = ? ORDER BY id",
            (conversation_id,),
        ).fetchall()
    result = _conversation(conversation)
    result["messages"] = [
        {
            "id": row["id"],
            "role": row["role"],
            "content": row["content"],
            "result": json.loads(row["result_json"]) if row["result_json"] else None,
            "created_at": row["created_at"],
        }
        for row in rows
    ]
    return result


def recent_history(conversation_id: str) -> list[dict[str, str]]:
    with _connection() as connection:
        conversation = connection.execute(
            "SELECT summary, summarized_through FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
        if conversation is None:
            return []
        rows = connection.execute(
            "SELECT role, content FROM conversation_messages "
            "WHERE conversation_id = ? AND id > ? ORDER BY id DESC LIMIT ?",
            (conversation_id, conversation["summarized_through"], MAX_HISTORY_MESSAGES),
        ).fetchall()

    summary = conversation["summary"]
    summary_message = None
    if summary:
        summary_message = {
            "role": "assistant",
            "content": f"此前对话摘要（仅用于理解当前问题）：\n{summary}",
        }
    remaining = MAX_HISTORY_CHARS - (len(summary_message["content"]) if summary_message else 0)
    recent = []
    for row in rows:
        content = row["content"]
        if remaining <= 0:
            break
        content = content[:min(remaining, MAX_HISTORY_MESSAGE_CHARS)]
        recent.append({"role": row["role"], "content": content})
        remaining -= len(content)
    return ([summary_message] if summary_message else []) + list(reversed(recent))


def compression_batch(conversation_id: str) -> dict | None:
    with _connection() as connection:
        conversation = connection.execute(
            "SELECT summary, summarized_through FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
        if conversation is None:
            return None
        rows = connection.execute(
            "SELECT id, role, content FROM conversation_messages "
            "WHERE conversation_id = ? AND id > ? ORDER BY id",
            (conversation_id, conversation["summarized_through"]),
        ).fetchall()

    old_messages = rows[:-KEEP_RECENT_MESSAGES]
    old_chars = sum(len(row["content"]) for row in old_messages)
    if len(old_messages) < COMPRESS_AFTER_MESSAGES and old_chars < COMPRESS_AFTER_CHARS:
        return None
    batch = []
    batch_chars = 0
    for row in old_messages:
        content = row["content"][:MAX_COMPRESSION_MESSAGE_CHARS]
        if batch and batch_chars + len(content) > MAX_COMPRESSION_BATCH_CHARS:
            break
        batch.append({"id": row["id"], "role": row["role"], "content": content})
        batch_chars += len(content)
    return {
        "summary": conversation["summary"],
        "messages": batch,
        "through_id": batch[-1]["id"],
    }


def save_summary(conversation_id: str, summary: str, through_id: int) -> None:
    with _connection() as connection:
        connection.execute(
            "UPDATE conversations SET summary = ?, summarized_through = ? "
            "WHERE id = ? AND summarized_through < ?",
            (summary, through_id, conversation_id, through_id),
        )


def append_exchange(
    conversation_id: str,
    question: str,
    answer: str,
    result: dict,
) -> None:
    title = question.strip().replace("\n", " ")[:60] or "新对话"
    with _connection() as connection:
        conversation = connection.execute(
            "SELECT id FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
        if conversation is None:
            raise KeyError(conversation_id)
        connection.execute(
            "INSERT INTO conversation_messages (conversation_id, role, content) "
            "VALUES (?, 'user', ?)",
            (conversation_id, question),
        )
        connection.execute(
            "INSERT INTO conversation_messages "
            "(conversation_id, role, content, result_json) VALUES (?, 'assistant', ?, ?)",
            (conversation_id, answer, json.dumps(result, ensure_ascii=False)),
        )
        connection.execute(
            "UPDATE conversations SET title = CASE WHEN title = '新对话' THEN ? ELSE title END, "
            "updated_at = datetime('now') WHERE id = ?",
            (title, conversation_id),
        )


def delete_conversation(conversation_id: str) -> bool:
    with _connection() as connection:
        cursor = connection.execute(
            "DELETE FROM conversations WHERE id = ?", (conversation_id,)
        )
    return cursor.rowcount > 0
