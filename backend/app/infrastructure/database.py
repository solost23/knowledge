"""Shared engines, per-operation transactions and legacy SQLite schema upgrades."""

import atexit
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Literal

from sqlalchemy import URL, Connection, create_engine, event, inspect, update
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool
from sqlalchemy.schema import CreateColumn

from app.core.settings import DATA_DIR
from app.infrastructure.models import (
    AgentRun,
    Base,
    Conversation,
    ConversationMessage,
    DocumentRecord,
    DocumentUploadJob,
)

DatabaseName = Literal["knowledge", "conversations", "agent"]
_TABLES = {
    "knowledge": (DocumentRecord.__table__, DocumentUploadJob.__table__),
    "conversations": (Conversation.__table__, ConversationMessage.__table__),
    "agent": (AgentRun.__table__,),
}
_LEGACY_COLUMNS = {
    "conversations": ("summary", "summarized_through"),
    "agent_runs": (
        "progress_json",
        "mode",
        "route_reason",
        "conversation_id",
        "cancel_requested",
        "result_json",
        "started_at",
        "finished_at",
        "duration_ms",
    ),
}


class Database:
    def __init__(self, path: str, name: DatabaseName = "knowledge"):
        self.lock = threading.RLock() if path == ":memory:" else None
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(
            URL.create("sqlite+pysqlite", database=path),
            connect_args={"timeout": 30, "check_same_thread": False},
            **({"poolclass": StaticPool} if path == ":memory:" else {}),
        )

        @event.listens_for(self.engine, "connect")
        def enable_foreign_keys(connection, _record):
            cursor = connection.execute("PRAGMA foreign_keys = ON")
            cursor.close()

        with self.engine.begin() as connection:
            tables = _TABLES[name]
            Base.metadata.create_all(connection, tables=tables)
            for table in tables:
                for index in table.indexes:
                    index.create(connection, checkfirst=True)
                if table.name not in _LEGACY_COLUMNS:
                    continue
                columns = {
                    column["name"]
                    for column in inspect(connection).get_columns(table.name)
                }
                for column_name in _LEGACY_COLUMNS[table.name]:
                    if column_name not in columns:
                        definition = CreateColumn(table.c[column_name]).compile(
                            dialect=connection.dialect
                        )
                        connection.exec_driver_sql(
                            f"ALTER TABLE {table.name} ADD COLUMN {definition}"
                        )
            if AgentRun.__table__ in tables:
                connection.execute(
                    update(AgentRun)
                    .where(AgentRun.started_at.is_(None))
                    .values(started_at=AgentRun.created_at + "Z")
                )

    @contextmanager
    def session(self, connection: Connection | None = None) -> Iterator[Session]:
        """Commit on success, roll back on failure, and always close the Session."""
        with (
            self.lock or nullcontext(),
            Session(
                connection if connection is not None else self.engine,
                expire_on_commit=False,
            ) as session,
            session.begin(),
        ):
            yield session


_DATABASES: dict[tuple[str, str], Database] = {}
_DATABASES_LOCK = threading.Lock()


def get_database(name: DatabaseName = "knowledge") -> Database:
    env_name, filename = {
        "knowledge": ("KNOWLEDGE_DB", "knowledge.db"),
        "conversations": ("CONVERSATION_DB", "conversations.db"),
        "agent": ("AGENT_CHECKPOINT_DB", "agent_checkpoints.db"),
    }[name]
    path = os.environ.get(env_name, str(DATA_DIR / filename))
    if path != ":memory:":
        path = str(Path(path).resolve())
    key = (name, path)
    with _DATABASES_LOCK:
        if key not in _DATABASES:
            _DATABASES[key] = Database(path, name)
        return _DATABASES[key]


def dispose_databases() -> None:
    """Release cached engines, for application shutdown or isolated tests."""
    with _DATABASES_LOCK:
        for database in _DATABASES.values():
            database.engine.dispose()
        _DATABASES.clear()


atexit.register(dispose_databases)
