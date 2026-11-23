import json
import threading
from contextlib import contextmanager
from datetime import UTC, datetime

from sqlalchemy import func, select, update

from app.agents.state import AgentRunBusy, AgentRunCancelled
from app.core.settings import config
from app.infrastructure.database import get_database
from app.infrastructure.models import AgentRun

_ACTIVE_EXECUTIONS: set[str] = set()
_ACTIVE_EXECUTIONS_LOCK = threading.Lock()
_CANCEL_REQUESTED: set[str] = set()
_CANCEL_REQUESTED_LOCK = threading.Lock()


@contextmanager
def open_checkpointer():
    from langgraph.checkpoint.sqlite import SqliteSaver

    database = get_database("agent")
    connection = database.engine.connect()
    checkpointer = SqliteSaver(connection.connection.driver_connection)
    checkpointer._database = database
    checkpointer._connection = connection
    if database.lock is not None:
        checkpointer.lock = database.lock
    try:
        with checkpointer.lock:
            checkpointer.setup()
        yield checkpointer
    finally:
        with checkpointer.lock:
            connection.close()


@contextmanager
def _session(checkpointer):
    # ORM and LangGraph take turns using this checkpointer's connection.
    with (
        checkpointer.lock,
        checkpointer._database.session(checkpointer._connection) as session,
    ):
        yield session


def run_config(execution_id: str, recursion_limit: int | None = None) -> dict:
    configured_limit = config.agent["recursion_limit"]
    return {
        "configurable": {"thread_id": execution_id},
        "recursion_limit": configured_limit
        if recursion_limit is None
        else recursion_limit,
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
    with _session(checkpointer) as session:
        record = session.get(AgentRun, execution_id)
        if record is None:
            return None
        run = record.as_dict()
    run["cancel_requested"] = bool(run["cancel_requested"])
    try:
        run["progress"] = json.loads(run.pop("progress_json"))
    except (TypeError, ValueError):
        run["progress"] = {}
    result_json = run.pop("result_json")
    try:
        run["result"] = json.loads(result_json) if result_json else None
    except (TypeError, ValueError):
        run["result"] = None
    if run["status"] == "running":
        run["duration_ms"] = _duration_ms(run["started_at"])
    return run


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _duration_ms(started_at: str | None, finished_at: str | None = None) -> int:
    if not started_at:
        return 0
    try:
        started = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        finished = datetime.fromisoformat(
            (finished_at or _now()).replace("Z", "+00:00")
        )
        return max(0, round((finished - started).total_seconds() * 1000))
    except (TypeError, ValueError):
        return 0


def create_run(
    checkpointer,
    execution_id: str,
    task: str,
    progress: dict,
    mode="agent",
    conversation_id=None,
):
    with _CANCEL_REQUESTED_LOCK:
        _CANCEL_REQUESTED.discard(execution_id)
    with _session(checkpointer) as session:
        session.add(
            AgentRun(
                execution_id=execution_id,
                task=task,
                status="running",
                progress_json=json.dumps(progress, ensure_ascii=False),
                mode=mode,
                conversation_id=conversation_id,
                started_at=_now(),
            )
        )


def set_run_mode(
    checkpointer, execution_id: str, mode: str, route_reason=None, conversation_id=None
):
    values = {"mode": mode, "updated_at": func.datetime("now")}
    if route_reason is not None:
        values["route_reason"] = route_reason
    if conversation_id is not None:
        values["conversation_id"] = conversation_id
    with _session(checkpointer) as session:
        session.execute(
            update(AgentRun)
            .where(AgentRun.execution_id == execution_id)
            .values(**values)
        )


def set_run_status(checkpointer, execution_id: str, status: str, error_code=None):
    with _session(checkpointer) as session:
        run = session.get(AgentRun, execution_id)
        if run is None:
            return
        run.status = status
        run.error_code = error_code
        run.finished_at = (
            _now() if status in ("completed", "failed", "cancelled") else None
        )
        run.duration_ms = (
            _duration_ms(run.started_at, run.finished_at) if run.finished_at else None
        )
        run.updated_at = func.datetime("now")


def finish_run(
    checkpointer, execution_id: str, status: str, error_code=None, respect_cancel=False
) -> bool:
    finished_at = _now()
    with _session(checkpointer) as session:
        started_at = session.scalar(
            select(AgentRun.started_at).where(AgentRun.execution_id == execution_id)
        )
        statement = update(AgentRun).where(
            AgentRun.execution_id == execution_id, AgentRun.status == "running"
        )
        if respect_cancel:
            statement = statement.where(AgentRun.cancel_requested == 0)
        result = session.execute(
            statement.values(
                status=status,
                error_code=error_code,
                finished_at=finished_at,
                duration_ms=_duration_ms(started_at, finished_at),
                updated_at=func.datetime("now"),
            )
        )
        return result.rowcount == 1


def save_run_result(checkpointer, execution_id: str, result: dict):
    with _session(checkpointer) as session:
        session.execute(
            update(AgentRun)
            .where(AgentRun.execution_id == execution_id)
            .values(
                result_json=json.dumps(result, ensure_ascii=False),
                updated_at=func.datetime("now"),
            )
        )


def request_run_cancel(checkpointer, execution_id: str) -> bool:
    with _session(checkpointer) as session:
        result = session.execute(
            update(AgentRun)
            .where(AgentRun.execution_id == execution_id, AgentRun.status == "running")
            .values(cancel_requested=1, updated_at=func.datetime("now"))
        )
        if result.rowcount != 1:
            return False
    with _CANCEL_REQUESTED_LOCK:
        _CANCEL_REQUESTED.add(execution_id)
    update_run_progress(
        checkpointer,
        execution_id,
        {"current_action": "正在停止，等待当前模型或工具调用返回"},
    )
    return True


def run_cancel_requested(checkpointer, execution_id: str) -> bool:
    with _CANCEL_REQUESTED_LOCK:
        return execution_id in _CANCEL_REQUESTED


def clear_run_cancel(execution_id: str):
    with _CANCEL_REQUESTED_LOCK:
        _CANCEL_REQUESTED.discard(execution_id)


def update_run_progress(
    checkpointer,
    execution_id: str | None,
    progress: dict,
    *,
    check_cancel: bool = False,
):
    if checkpointer is None or execution_id is None:
        return
    if check_cancel and run_cancel_requested(checkpointer, execution_id):
        raise AgentRunCancelled(execution_id)
    with _session(checkpointer) as session:
        run = session.get(AgentRun, execution_id)
        if run is None:
            return
        try:
            current = json.loads(run.progress_json)
        except (TypeError, ValueError):
            current = {}
        current.update(progress)
        run.progress_json = json.dumps(current, ensure_ascii=False)
        run.updated_at = func.datetime("now")


def checkpoint_values(checkpointer, execution_id: str) -> dict:
    saved = checkpointer.get_tuple(run_config(execution_id))
    if saved is None:
        return {}
    return saved.checkpoint.get("channel_values", {})
