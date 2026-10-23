from uuid import UUID, uuid4

from loguru import logger

from app.agents.checkpoints import (
    claim_execution,
    checkpoint_values,
    get_run,
    open_checkpointer,
    run_config,
    set_run_status,
)
from app.agents.graph import build_graph
from app.agents.state import (
    MAX_PLAN_STEPS,
    MAX_STEP_LENGTH,
    MAX_TASK_LENGTH,
    AgentRunBusy,
    InvalidAgentRequest,
)
from app.core import response
from app.infrastructure.llm import build_configured_llm


def _validate_execution_id(execution_id: str) -> str:
    try:
        return str(UUID(str(execution_id)))
    except (ValueError, TypeError, AttributeError) as error:
        raise InvalidAgentRequest("execution_id 必须是有效的 UUID") from error


def _validate_task(task: str) -> str:
    task = task.strip() if isinstance(task, str) else ""
    if not task:
        raise InvalidAgentRequest("任务内容不能为空")
    if len(task) > MAX_TASK_LENGTH:
        raise InvalidAgentRequest(f"任务内容不能超过 {MAX_TASK_LENGTH} 个字符")
    return task


def _validate_plan(plan: list[str] | None) -> list[str]:
    if plan is None:
        return []
    if not isinstance(plan, list) or len(plan) > MAX_PLAN_STEPS:
        raise InvalidAgentRequest(f"执行步骤不能超过 {MAX_PLAN_STEPS} 步")
    steps = []
    for step in plan:
        if not isinstance(step, str) or not step.strip():
            raise InvalidAgentRequest("执行步骤不能为空")
        step = step.strip()
        if len(step) > MAX_STEP_LENGTH:
            raise InvalidAgentRequest(f"单个执行步骤不能超过 {MAX_STEP_LENGTH} 个字符")
        steps.append(step)
    return steps


def _run_result(execution_id: str, state: dict) -> dict:
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


def _is_completed(state: dict) -> bool:
    return (
        bool(state.get("answer"))
        and state.get("current_step", 0) >= len(state.get("plan", []))
    )


class AgentTaskService:
    def execute(self, task: str, plan: list[str] | None = None, llm=None):
        execution_id = str(uuid4())
        run_created = False
        resumable = False
        try:
            task = _validate_task(task)
            plan = _validate_plan(plan)
            with open_checkpointer() as checkpointer:
                checkpointer.conn.execute(
                    "INSERT INTO agent_runs (execution_id, task, status) VALUES (?, ?, 'running')",
                    (execution_id, task),
                )
                checkpointer.conn.commit()
                run_created = True
                llm = llm or build_configured_llm()
                graph = build_graph(llm, checkpointer)
                with claim_execution(execution_id):
                    state = graph.invoke(
                        {
                            "task": task,
                            "plan": plan,
                            "current_step": 0,
                            "steps": [],
                            "sources": [],
                            "answer": "",
                        },
                        config=run_config(execution_id),
                    )
                set_run_status(checkpointer, execution_id, "completed")
                return response.success("任务完成", _run_result(execution_id, state))
        except EnvironmentError as error:
            logger.error(f"Agent LLM 配置错误: {error}")
            if not run_created:
                return response.error(500, "Agent 模型配置或状态存储错误")
            try:
                with open_checkpointer() as checkpointer:
                    set_run_status(checkpointer, execution_id, "failed", "agent_configuration_error")
                    resumable = bool(checkpoint_values(checkpointer, execution_id))
            except Exception:
                logger.exception("保存 Agent 配置错误状态时出错")
            return response.error(500, "Agent 模型配置错误", {
                "execution_id": execution_id,
                "status": "failed",
                "error_code": "agent_configuration_error",
                "resumable": resumable,
            })
        except InvalidAgentRequest as error:
            logger.warning(f"Agent 请求参数无效: {error}")
            return response.error(400, str(error))
        except AgentRunBusy:
            return response.error(409, "该 Agent 任务正在执行")
        except Exception as error:
            logger.exception(f"Agent 任务执行失败: {error}")
            if not run_created:
                return response.error(500, "Agent 状态存储暂时不可用")
            try:
                with open_checkpointer() as checkpointer:
                    if get_run(checkpointer, execution_id):
                        set_run_status(checkpointer, execution_id, "failed", "agent_execution_failed")
                        resumable = bool(checkpoint_values(checkpointer, execution_id))
            except Exception:
                logger.exception("保存 Agent 失败状态时出错")
            return response.error(502, "Agent 执行失败，可查询状态并尝试恢复", {
                "execution_id": execution_id,
                "status": "failed",
                "error_code": "agent_execution_failed",
                "resumable": resumable,
            })

    def resume(self, execution_id: str):
        try:
            execution_id = _validate_execution_id(execution_id)
            with open_checkpointer() as checkpointer:
                run = get_run(checkpointer, execution_id)
                if run is None:
                    return response.error(404, "未找到该 Agent 执行记录")
                values = checkpoint_values(checkpointer, execution_id)
                if run["status"] == "completed" or _is_completed(values):
                    set_run_status(checkpointer, execution_id, "completed")
                    return response.success("任务已完成", _run_result(execution_id, values))
                if run["status"] not in ("failed", "running") or not values:
                    return response.error(409, "该执行记录当前不可恢复")

                llm = build_configured_llm()
                set_run_status(checkpointer, execution_id, "running")
                graph = build_graph(llm, checkpointer)
                with claim_execution(execution_id):
                    state = graph.invoke(None, config=run_config(execution_id))
                set_run_status(checkpointer, execution_id, "completed")
                return response.success("任务恢复完成", _run_result(execution_id, state))
        except InvalidAgentRequest as error:
            return response.error(400, str(error))
        except AgentRunBusy:
            return response.error(409, "该 Agent 任务正在执行")
        except EnvironmentError as error:
            logger.error(f"Agent LLM 配置错误: {error}")
            try:
                with open_checkpointer() as checkpointer:
                    if get_run(checkpointer, execution_id):
                        set_run_status(checkpointer, execution_id, "failed", "agent_configuration_error")
            except Exception:
                logger.exception("保存 Agent 恢复配置错误状态时出错")
            return response.error(500, "Agent 模型配置错误", {
                "execution_id": execution_id,
                "status": "failed",
                "error_code": "agent_configuration_error",
                "resumable": True,
            })
        except Exception as error:
            logger.exception(f"Agent 任务恢复失败: {error}")
            try:
                with open_checkpointer() as checkpointer:
                    if get_run(checkpointer, execution_id):
                        set_run_status(checkpointer, execution_id, "failed", "agent_resume_failed")
            except Exception:
                logger.exception("保存 Agent 恢复失败状态时出错")
            return response.error(502, "Agent 恢复失败，可稍后重试", {
                "execution_id": execution_id,
                "status": "failed",
                "error_code": "agent_resume_failed",
                "resumable": True,
            })

    def get_status(self, execution_id: str):
        try:
            execution_id = _validate_execution_id(execution_id)
            with open_checkpointer() as checkpointer:
                run = get_run(checkpointer, execution_id)
                if run is None:
                    return response.error(404, "未找到该 Agent 执行记录")
                values = checkpoint_values(checkpointer, execution_id)
                if _is_completed(values) and run["status"] != "completed":
                    set_run_status(checkpointer, execution_id, "completed")
                    run["status"] = "completed"
                    run["error_code"] = None
                return response.success("执行状态", {
                    **run,
                    "plan": values.get("plan", []),
                    "current_step": values.get("current_step", 0),
                    "steps": values.get("steps", []),
                    "answer": values.get("answer", ""),
                    "sources": values.get("sources", []),
                })
        except InvalidAgentRequest as error:
            return response.error(400, str(error))
        except Exception as error:
            logger.exception(f"读取 Agent 状态失败: {error}")
            return response.error(500, "读取 Agent 执行状态失败")
