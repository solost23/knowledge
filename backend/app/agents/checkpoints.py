import json
import os
import sqlite3
import threading
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path

from app.agents.state import AgentRunBusy
from app.core.settings import DATA_DIR

_ACTIVE_EXECUTIONS: set[str] = set()
_ACTIVE_EXECUTIONS_LOCK = threading.Lock()
_CANCEL_REQUESTED: set[str] = set()
_CANCEL_REQUESTED_LOCK = threading.Lock()


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
                progress_json TEXT NOT NULL DEFAULT '{}',
                mode TEXT NOT NULL DEFAULT 'agent',
                route_reason TEXT NOT NULL DEFAULT '',
                conversation_id TEXT,
                cancel_requested INTEGER NOT NULL DEFAULT 0,
                result_json TEXT,
                started_at TEXT,
                finished_at TEXT,
                duration_ms INTEGER,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        columns = {
            row[1]
            for row in checkpointer.conn.execute("PRAGMA table_info(agent_runs)").fetchall()
        }
        migrations = {
            "progress_json": "TEXT NOT NULL DEFAULT '{}'",
            "mode": "TEXT NOT NULL DEFAULT 'agent'",
            "route_reason": "TEXT NOT NULL DEFAULT ''",
            "conversation_id": "TEXT",
            "cancel_requested": "INTEGER NOT NULL DEFAULT 0",
            "result_json": "TEXT",
            "started_at": "TEXT",
            "finished_at": "TEXT",
            "duration_ms": "INTEGER",
        }
        for name, definition in migrations.items():
            if name not in columns:
                checkpointer.conn.execute(
                    f"ALTER TABLE agent_runs ADD COLUMN {name} {definition}"
                )
        checkpointer.conn.execute(
            "UPDATE agent_runs SET started_at = created_at || 'Z' WHERE started_at IS NULL"
        )
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
        "SELECT execution_id, task, status, error_code, progress_json, mode, route_reason, "
        "conversation_id, cancel_requested, result_json, started_at, finished_at, duration_ms, "
        "created_at, updated_at "
        "FROM agent_runs WHERE execution_id = ?",
        (execution_id,),
    ).fetchone()
    if row is None:
        return None
    columns = (
        "execution_id", "task", "status", "error_code", "progress_json", "mode",
        "route_reason", "conversation_id", "cancel_requested", "result_json", "started_at",
        "finished_at", "duration_ms", "created_at", "updated_at",
    )
    run = dict(zip(columns, row))
    run["cancel_requested"] = bool(run["cancel_requested"])
    try:
        run["progress"] = json.loads(run.pop("progress_json"))
    except (TypeError, ValueError):
        run["progress"] = {}
        run.pop("progress_json", None)
    try:
        run["result"] = json.loads(run.pop("result_json")) if run["result_json"] else None
    except (TypeError, ValueError):
        run["result"] = None
        run.pop("result_json", None)
    if run["status"] == "running":
        run["duration_ms"] = _duration_ms(run["started_at"])
    return run


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _duration_ms(started_at: str | None, finished_at: str | None = None) -> int:
    if not started_at:
        return 0
    try:
        started = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        finished = datetime.fromisoformat((finished_at or _now()).replace("Z", "+00:00"))
        return max(0, round((finished - started).total_seconds() * 1000))
    except (TypeError, ValueError):
        return 0


def create_run(checkpointer, execution_id: str, task: str, progress: dict, mode="agent", conversation_id=None):
    with _CANCEL_REQUESTED_LOCK:
        _CANCEL_REQUESTED.discard(execution_id)
    checkpointer.conn.execute(
        "INSERT INTO agent_runs "
        "(execution_id, task, status, progress_json, mode, conversation_id, started_at) "
        "VALUES (?, ?, 'running', ?, ?, ?, ?)",
        (execution_id, task, json.dumps(progress, ensure_ascii=False), mode, conversation_id, _now()),
    )
    checkpointer.conn.commit()


def set_run_mode(checkpointer, execution_id: str, mode: str, route_reason=None, conversation_id=None):
    checkpointer.conn.execute(
        "UPDATE agent_runs SET mode = ?, route_reason = COALESCE(?, route_reason), "
        "conversation_id = COALESCE(?, conversation_id), updated_at = datetime('now') "
        "WHERE execution_id = ?",
        (mode, route_reason, conversation_id, execution_id),
    )
    checkpointer.conn.commit()


def set_run_status(checkpointer, execution_id: str, status: str, error_code=None):
    row = checkpointer.conn.execute(
        "SELECT started_at FROM agent_runs WHERE execution_id = ?", (execution_id,)
    ).fetchone()
    finished_at = _now() if status in ("completed", "failed", "cancelled") else None
    duration_ms = _duration_ms(row[0], finished_at) if row and finished_at else None
    checkpointer.conn.execute(
        "UPDATE agent_runs SET status = ?, error_code = ?, finished_at = ?, duration_ms = ?, "
        "updated_at = datetime('now') "
        "WHERE execution_id = ?",
        (status, error_code, finished_at, duration_ms, execution_id),
    )
    checkpointer.conn.commit()


def finish_run(checkpointer, execution_id: str, status: str, error_code=None, respect_cancel=False) -> bool:
    finished_at = _now()
    row = checkpointer.conn.execute(
        "SELECT started_at, cancel_requested FROM agent_runs "
        "WHERE execution_id = ? AND status = 'running'",
        (execution_id,),
    ).fetchone()
    if row is None or (respect_cancel and row[1]):
        return False
    duration_ms = _duration_ms(row[0], finished_at)
    cancel_clause = " AND cancel_requested = 0" if respect_cancel else ""
    cursor = checkpointer.conn.execute(
        "UPDATE agent_runs SET status = ?, error_code = ?, finished_at = ?, duration_ms = ?, "
        f"updated_at = datetime('now') WHERE execution_id = ? AND status = 'running'{cancel_clause}",
        (status, error_code, finished_at, duration_ms, execution_id),
    )
    if cursor.rowcount != 1:
        checkpointer.conn.commit()
        return False
    checkpointer.conn.commit()
    return True


def save_run_result(checkpointer, execution_id: str, result: dict):
    checkpointer.conn.execute(
        "UPDATE agent_runs SET result_json = ?, updated_at = datetime('now') WHERE execution_id = ?",
        (json.dumps(result, ensure_ascii=False), execution_id),
    )
    checkpointer.conn.commit()


def request_run_cancel(checkpointer, execution_id: str) -> bool:
    run = get_run(checkpointer, execution_id)
    if run is None or run["status"] != "running":
        return False
    cursor = checkpointer.conn.execute(
        "UPDATE agent_runs SET cancel_requested = 1, updated_at = datetime('now') "
        "WHERE execution_id = ? AND status = 'running'",
        (execution_id,),
    )
    if cursor.rowcount != 1:
        checkpointer.conn.commit()
        return False
    checkpointer.conn.commit()
    with _CANCEL_REQUESTED_LOCK:
        _CANCEL_REQUESTED.add(execution_id)
    update_run_progress(checkpointer, execution_id, {
        "current_action": "正在停止，等待当前模型或工具调用返回",
    })
    return True


def run_cancel_requested(checkpointer, execution_id: str) -> bool:
    with _CANCEL_REQUESTED_LOCK:
        return execution_id in _CANCEL_REQUESTED


def clear_run_cancel(execution_id: str):
    with _CANCEL_REQUESTED_LOCK:
        _CANCEL_REQUESTED.discard(execution_id)


def update_run_progress(checkpointer, execution_id: str, progress: dict):
    row = checkpointer.conn.execute(
        "SELECT progress_json FROM agent_runs WHERE execution_id = ?",
        (execution_id,),
    ).fetchone()
    if row is None:
        return
    try:
        current = json.loads(row[0])
    except (TypeError, ValueError):
        current = {}
    current.update(progress)
    checkpointer.conn.execute(
        "UPDATE agent_runs SET progress_json = ?, updated_at = datetime('now') "
        "WHERE execution_id = ?",
        (json.dumps(current, ensure_ascii=False), execution_id),
    )
    checkpointer.conn.commit()


def checkpoint_values(checkpointer, execution_id: str) -> dict:
    saved = checkpointer.get_tuple(run_config(execution_id))
    if saved is None:
        return {}
    return saved.checkpoint.get("channel_values", {})
