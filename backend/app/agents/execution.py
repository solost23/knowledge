from loguru import logger

from app.agents.checkpoints import (
    checkpoint_values,
    claim_execution,
    clear_run_cancel,
    finish_run,
    get_run,
    open_checkpointer,
    run_config,
    save_run_result,
    terminal_exchange_runs,
)
from app.agents.graph import build_graph
from app.agents.state import AgentRunBusy, AgentRunCancelled, InvalidAgentRequest
from app.core import response
from app.infrastructure.conversation_repository import append_exchange
from app.infrastructure.llm import LLM, LLMConfigurationError


def run_result(execution_id: str, state: dict) -> dict:
    return {
        "execution_id": execution_id,
        "status": "completed",
        "task": state.get("task", ""),
        "plan": state.get("plan", []),
        "current_step": state.get("current_step", 0),
        "steps": state.get("steps", []),
        "answer": state.get("answer", ""),
        "sources": state.get("sources", []),
    }


def run_metadata(run: dict) -> dict:
    return {
        "mode": run.get("mode", "agent"),
        "route_reason": run.get("route_reason", ""),
        "conversation_id": run.get("conversation_id"),
        "started_at": run.get("started_at"),
        "finished_at": run.get("finished_at"),
        "duration_ms": run.get("duration_ms", 0),
    }


def recover_pending_exchanges() -> int:
    """持久化终态结果兼作 outbox，启动时幂等补写跨库会话。"""
    recovered = 0
    for run in terminal_exchange_runs():
        result = run.get("result")
        if not isinstance(result, dict) or not result:
            continue
        try:
            append_exchange(
                run["conversation_id"],
                run["task"],
                result.get("answer")
                or (
                    "任务已停止"
                    if run["status"] == "cancelled"
                    else result.get("error_message", "")
                ),
                {**result, "execution_id": run["execution_id"]},
            )
            recovered += 1
        except KeyError:
            # 用户删除的会话不重建，也不阻止其他待补写记录。
            continue
        except Exception:
            logger.exception("补写执行结果到会话失败，保留结果供下次启动重试")
    return recovered


def persist_completed(checkpointer, execution_id: str, state: dict) -> dict:
    run = get_run(checkpointer, execution_id)
    if run["status"] == "completed":
        result = run.get("result") or {
            **run_result(execution_id, state),
            **run_metadata(run),
        }
        if not run.get("result"):
            save_run_result(checkpointer, execution_id, result)
    else:
        result = {**run_result(execution_id, state), **run_metadata(run)}
        if not finish_run(
            checkpointer,
            execution_id,
            "completed",
            respect_cancel=True,
            result=result,
            progress={
                "stage": "completed",
                "plan": state.get("plan", []),
                "current_step": len(state.get("plan", [])),
                "active_step": None,
                "steps": state.get("steps", []),
                "current_action": "任务执行完成",
            },
        ):
            current = get_run(checkpointer, execution_id)
            if current["status"] == "completed":
                result = current.get("result") or result
            elif current["cancel_requested"] or current["status"] == "cancelled":
                raise AgentRunCancelled(execution_id)
            else:
                raise RuntimeError("执行记录不处于可完成状态")
    if run["conversation_id"]:
        try:
            append_exchange(
                run["conversation_id"], run["task"], result.get("answer", ""), result
            )
        except KeyError:
            pass
        except Exception:
            # 执行结果已提交；恢复请求可以幂等补写另一个数据库中的会话。
            logger.exception("保存已完成任务的会话失败，可通过恢复请求重试")
    return result


def run_created(
    execution_id: str,
    task: str,
    plan: list[str],
    llm=None,
    on_complete=None,
):
    try:
        with open_checkpointer() as checkpointer:
            llm = llm or LLM()
            graph = build_graph(
                llm,
                checkpointer,
                execution_id=execution_id,
            )
            with claim_execution(execution_id):
                recursion_limit = run_config(execution_id)["recursion_limit"]
                state = graph.invoke(
                    {
                        "task": task,
                        "plan": plan,
                        "current_step": 0,
                        "steps": [],
                        "sources": [],
                        "answer": "",
                    },
                    config=run_config(
                        execution_id,
                        recursion_limit=max(recursion_limit, len(plan) * 3 + 10),
                    ),
                )
                result = persist_completed(checkpointer, execution_id, state)
        _notify_completion(on_complete, result, "保存 Agent 执行结果失败")
        return response.success("任务完成", result)
    except AgentRunCancelled:
        result = _save_cancelled_result(execution_id)
        clear_run_cancel(execution_id)
        _notify_completion(on_complete, result, "保存已取消 Agent 任务失败")
        return response.success(
            "任务已停止",
            result or {"execution_id": execution_id, "status": "cancelled"},
        )
    except LLMConfigurationError as error:
        logger.error(f"Agent LLM 配置错误: {error}")
        resumable = _mark_failed(
            execution_id,
            "agent_configuration_error",
            "模型配置错误，任务未能继续",
            "保存 Agent 配置错误状态时出错",
        )
        return response.error(
            500,
            "Agent 模型配置错误",
            {
                "execution_id": execution_id,
                "status": "failed",
                "error_code": "agent_configuration_error",
                "resumable": resumable,
            },
        )
    except InvalidAgentRequest as error:
        logger.warning(f"Agent 请求参数无效: {error}")
        return response.error(400, str(error))
    except AgentRunBusy:
        return response.error(409, "该 Agent 任务正在执行")
    except Exception as error:
        logger.exception(f"Agent 任务执行失败: {error}")
        resumable = _mark_failed(
            execution_id,
            "agent_execution_failed",
            "执行中断，可查询状态并尝试恢复",
            "保存 Agent 失败状态时出错",
            only_if_exists=True,
        )
        return response.error(
            502,
            "Agent 执行失败，可查询状态并尝试恢复",
            {
                "execution_id": execution_id,
                "status": "failed",
                "error_code": "agent_execution_failed",
                "resumable": resumable,
            },
        )


def _notify_completion(callback, result, failure_message):
    if callback and result:
        try:
            callback(result)
        except Exception:
            logger.exception(failure_message)


def _save_cancelled_result(execution_id: str):
    try:
        with open_checkpointer() as checkpointer:
            run = get_run(checkpointer, execution_id)
            if run["status"] in ("completed", "cancelled"):
                return run.get("result")
            values = checkpoint_values(checkpointer, execution_id)
            result = {
                "execution_id": execution_id,
                "status": "cancelled",
                "task": run["task"],
                "plan": values.get("plan") or run["progress"].get("plan", []),
                "current_step": values.get("current_step", 0),
                "steps": values.get("steps", []),
                "answer": "",
                "sources": values.get("sources", []),
                **run_metadata(run),
            }
            progress = {
                **run["progress"],
                "stage": "cancelled",
                "active_step": None,
                "current_action": "任务已停止",
            }
            if not finish_run(
                checkpointer,
                execution_id,
                "cancelled",
                result=result,
                progress=progress,
            ):
                return get_run(checkpointer, execution_id).get("result")
            clear_run_cancel(execution_id)
            if run["conversation_id"]:
                append_exchange(
                    run["conversation_id"], run["task"], "任务已停止", result
                )
            return result
    except Exception:
        logger.exception("保存 Agent 取消状态时出错")
        return None


def _mark_failed(
    execution_id,
    error_code,
    current_action,
    failure_message,
    only_if_exists=False,
):
    resumable = False
    try:
        with open_checkpointer() as checkpointer:
            run = get_run(checkpointer, execution_id)
            if run is None or run["status"] in ("completed", "cancelled"):
                return False
            resumable = bool(checkpoint_values(checkpointer, execution_id))
            finish_run(
                checkpointer,
                execution_id,
                "failed",
                error_code,
                result={
                    "execution_id": execution_id,
                    "task": run["task"],
                    "error_code": error_code,
                    "error_message": current_action,
                    "resumable": resumable,
                    **run_metadata(run),
                },
                progress={
                    **run["progress"],
                    "stage": "failed",
                    "active_step": None,
                    "current_action": current_action,
                },
            )
    except Exception:
        logger.exception(failure_message)
    return resumable
