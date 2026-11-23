"""Persistence and status transitions for background ask executions."""

from app.agents.checkpoints import (
    clear_run_cancel,
    finish_run,
    get_run,
    open_checkpointer,
    run_cancel_requested,
    save_run_result,
    set_run_mode,
    set_run_status,
    update_run_progress,
)
from app.agents.dispatcher import RequestRoute
from app.agents.state import AgentRunCancelled
from app.infrastructure.conversation_repository import append_exchange


class AskRunLifecycle:
    """Own checkpoint transitions and conversation persistence for an ask run."""

    @staticmethod
    def ensure_not_cancelled(execution_id: str) -> None:
        with open_checkpointer() as checkpointer:
            if run_cancel_requested(checkpointer, execution_id):
                raise AgentRunCancelled(execution_id)

    @staticmethod
    def record_route(
        execution_id: str, conversation_id: str, route: RequestRoute
    ) -> None:
        with open_checkpointer() as checkpointer:
            set_run_mode(
                checkpointer,
                execution_id,
                route.mode,
                route.reason,
                conversation_id,
            )
            update_run_progress(
                checkpointer,
                execution_id,
                {
                    "stage": "executing" if route.mode == "agent" else "retrieving",
                    "plan": route.steps if route.mode == "agent" else [],
                    "current_step": 0,
                    "active_step": None,
                    "current_action": (
                        "执行计划已生成，准备开始"
                        if route.mode == "agent"
                        else "正在检索知识库并生成回答"
                    ),
                },
            )

    @classmethod
    def complete_rag(
        cls,
        execution_id: str,
        task: str,
        conversation_id: str,
        route: RequestRoute,
        answer_data: dict,
    ) -> dict:
        with open_checkpointer() as checkpointer:
            if not finish_run(
                checkpointer, execution_id, "completed", respect_cancel=True
            ):
                raise AgentRunCancelled(execution_id)
            update_run_progress(
                checkpointer,
                execution_id,
                {"stage": "completed", "current_action": "检索与回答已完成"},
            )
            run = get_run(checkpointer, execution_id)
            result = {
                "execution_id": execution_id,
                "status": "completed",
                "mode": "rag",
                "task": route.resolved_task.strip() or task,
                "conversation_id": conversation_id,
                "route_reason": route.reason,
                "plan": [],
                "current_step": 0,
                "steps": [],
                "answer": answer_data.get("answer", ""),
                "sources": answer_data.get("sources", []),
                "started_at": run["started_at"],
                "finished_at": run["finished_at"],
                "duration_ms": run["duration_ms"],
            }
            save_run_result(checkpointer, execution_id, result)
        cls.save_exchange(conversation_id, task, result)
        return result

    @staticmethod
    def save_agent_error_result(
        execution_id: str,
        conversation_id: str,
        route: RequestRoute,
        result_data: dict,
    ) -> None:
        result_data.update(
            {
                "mode": "agent",
                "route_reason": route.reason,
                "conversation_id": conversation_id,
            }
        )
        with open_checkpointer() as checkpointer:
            save_run_result(checkpointer, execution_id, result_data)

    @staticmethod
    def save_exchange(conversation_id: str, task: str, result: dict) -> None:
        append_exchange(
            conversation_id,
            task,
            result.get("answer")
            or ("任务已停止" if result.get("status") == "cancelled" else ""),
            result,
        )

    @classmethod
    def finish_cancelled(
        cls, execution_id: str, task: str, conversation_id: str
    ) -> dict:
        with open_checkpointer() as checkpointer:
            set_run_status(checkpointer, execution_id, "cancelled")
            update_run_progress(
                checkpointer,
                execution_id,
                {
                    "stage": "cancelled",
                    "active_step": None,
                    "current_action": "任务已停止",
                },
            )
            run = get_run(checkpointer, execution_id)
            result = {
                "execution_id": execution_id,
                "status": "cancelled",
                "mode": run["mode"],
                "task": task,
                "conversation_id": conversation_id,
                "route_reason": run["route_reason"],
                "plan": run["progress"].get("plan", []),
                "current_step": run["progress"].get("current_step", 0),
                "steps": run["progress"].get("steps", []),
                "answer": "",
                "sources": [],
                "started_at": run["started_at"],
                "finished_at": run["finished_at"],
                "duration_ms": run["duration_ms"],
            }
            save_run_result(checkpointer, execution_id, result)
        cls.save_exchange(conversation_id, task, result)
        clear_run_cancel(execution_id)
        return result

    @staticmethod
    def fail(execution_id: str, error_code: str, error_message: str) -> dict:
        with open_checkpointer() as checkpointer:
            set_run_status(checkpointer, execution_id, "failed", error_code)
            update_run_progress(
                checkpointer,
                execution_id,
                {
                    "stage": "failed",
                    "active_step": None,
                    "current_action": error_message,
                    "error_message": error_message,
                },
            )
            run = get_run(checkpointer, execution_id)
            result = {
                "execution_id": execution_id,
                "status": "failed",
                "mode": run["mode"],
                "task": run["task"],
                "conversation_id": run["conversation_id"],
                "route_reason": run["route_reason"],
                "error_code": error_code,
                "error_message": error_message,
                "started_at": run["started_at"],
                "finished_at": run["finished_at"],
                "duration_ms": run["duration_ms"],
            }
            save_run_result(checkpointer, execution_id, result)
            return result
