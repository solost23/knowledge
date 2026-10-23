import os
import sqlite3
import threading
from contextlib import closing, contextmanager
from pathlib import Path

from app.agents.state import AgentRunBusy
from app.core.settings import DATA_DIR

_ACTIVE_EXECUTIONS: set[str] = set()
_ACTIVE_EXECUTIONS_LOCK = threading.Lock()


def _checkpoint_path() -> str:
    raw_path = os.environ.get(
        "AGENT_CHECKPOINT_DB", str(DATA_DIR / "agent_checkpoints.db")
    )
    if raw_path == ":memory:":
        return raw_path
    path = Path(raw_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


@contextmanager
def open_checkpointer():
    connection = sqlite3.connect(
        _checkpoint_path(),
        timeout=30,
        check_same_thread=False,
    )
    with closing(connection):
        from langgraph.checkpoint.sqlite import SqliteSaver

        checkpointer = SqliteSaver(connection)
        checkpointer.setup()
        checkpointer.conn.execute("""
            CREATE TABLE IF NOT EXISTS agent_runs (
                execution_id TEXT PRIMARY KEY,
                task TEXT NOT NULL,
                status TEXT NOT NULL,
                error_code TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        checkpointer.conn.commit()
        yield checkpointer


def run_config(execution_id: str) -> dict:
    return {
        "configurable": {"thread_id": execution_id},
        "recursion_limit": 32,
    }


@contextmanager
def claim_execution(execution_id: str):
    with _ACTIVE_EXECUTIONS_LOCK:
        if execution_id in _ACTIVE_EXECUTIONS:
            raise AgentRunBusy(execution_id)
        _ACTIVE_EXECUTIONS.add(execution_id)
    try:
        yield
    finally:
        with _ACTIVE_EXECUTIONS_LOCK:
            _ACTIVE_EXECUTIONS.discard(execution_id)


def get_run(checkpointer, execution_id: str):
    row = checkpointer.conn.execute(
        "SELECT execution_id, task, status, error_code, created_at, updated_at "
        "FROM agent_runs WHERE execution_id = ?",
        (execution_id,),
    ).fetchone()
    if row is None:
        return None
    columns = ("execution_id", "task", "status", "error_code", "created_at", "updated_at")
    return dict(zip(columns, row))


def set_run_status(checkpointer, execution_id: str, status: str, error_code=None):
    checkpointer.conn.execute(
        "UPDATE agent_runs SET status = ?, error_code = ?, updated_at = datetime('now') "
        "WHERE execution_id = ?",
        (status, error_code, execution_id),
    )
    checkpointer.conn.commit()


def checkpoint_values(checkpointer, execution_id: str) -> dict:
    saved = checkpointer.get_tuple(run_config(execution_id))
    if saved is None:
        return {}
    return saved.checkpoint.get("channel_values", {})
