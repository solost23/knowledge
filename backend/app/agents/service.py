from uuid import uuid4

from loguru import logger

from app.agents.checkpoints import (
    checkpoint_values,
    claim_execution,
    create_run,
    get_run,
    open_checkpointer,
    request_run_cancel,
    run_config,
    set_run_mode,
    set_run_status,
    update_run_progress,
)
from app.agents.execution import (
    run_created,
    run_metadata,
    run_result,
)
from app.agents.graph import build_graph
from app.agents.state import (
    AgentRunBusy,
    InvalidAgentRequest,
)
from app.agents.validation import validate_execution_id, validate_plan, validate_task
from app.core import response
from app.infrastructure import submit_task
from app.infrastructure.llm import LLM, LLMConfigurationError

# Kept as module aliases for existing integrations that imported these helpers
# before validation was extracted into its own module.
_validate_execution_id = validate_execution_id
_validate_plan = validate_plan
_validate_task = validate_task


def _is_completed(state: dict) -> bool:
    return bool(state.get("answer")) and state.get("current_step", 0) >= len(
        state.get("plan", [])
    )


class AgentTaskService:
    def _create_run(
        self,
        execution_id: str,
        task: str,
        plan: list[str],
        mode="agent",
        conversation_id=None,
    ):
        initial_progress = {
            "stage": "starting",
            "plan": plan,
            "current_step": 0,
            "active_step": None,
            "steps": [],
            "current_action": "Agent 已接收任务，准备执行计划",
        }
        with open_checkpointer() as checkpointer:
            create_run(
                checkpointer,
                execution_id,
                task,
                initial_progress,
                mode,
                conversation_id,
            )

    def _run_created(
        self, execution_id: str, task: str, plan: list[str], llm=None, on_complete=None
    ):
        return run_created(execution_id, task, plan, llm, on_complete)

    def execute(self, task: str, plan: list[str] | None = None, llm=None):
        try:
            task = validate_task(task)
            plan = validate_plan(plan)
            execution_id = str(uuid4())
            self._create_run(execution_id, task, plan)
        except InvalidAgentRequest as error:
            logger.warning(f"Agent 请求参数无效: {error}")
            return response.error(400, str(error))
        except Exception:
            logger.exception("创建 Agent 执行记录失败")
            return response.error(500, "Agent 状态存储暂时不可用")
        return self._run_created(execution_id, task, plan, llm)

    def create_routed_run(self, task: str, conversation_id: str):
        task = validate_task(task)
        execution_id = str(uuid4())
        initial_progress = {
            "stage": "routing",
            "plan": [],
            "current_step": 0,
            "active_step": None,
            "steps": [],
            "current_action": "正在判断处理方式",
        }
        with open_checkpointer() as checkpointer:
            create_run(
                checkpointer,
                execution_id,
                task,
                initial_progress,
                "routing",
                conversation_id,
            )
            run = get_run(checkpointer, execution_id)
        return execution_id, run

    def run_existing(
        self, execution_id: str, task: str, plan: list[str], llm=None, on_complete=None
    ):
        with open_checkpointer() as checkpointer:
            set_run_mode(checkpointer, execution_id, "agent")
            update_run_progress(
                checkpointer,
                execution_id,
                {
                    "stage": "starting",
                    "plan": plan,
                    "current_step": 0,
                    "active_step": None,
                    "steps": [],
                    "current_action": "Agent 已接收任务，准备执行计划",
                },
            )
        return self._run_created(execution_id, task, plan, llm, on_complete)

    def start(
        self, task: str, plan: list[str] | None = None, llm=None, on_complete=None
    ):
        try:
            task = validate_task(task)
            plan = validate_plan(plan)
            if not plan:
                raise InvalidAgentRequest("Agent 执行计划不能为空")
            execution_id = str(uuid4())
            self._create_run(execution_id, task, plan)
            submit_task(
                "agent",
                self._run_created,
                execution_id,
                task,
                plan,
                llm,
                on_complete,
            )
            body, _ = response.success(
                "Agent 任务已启动",
                {
                    "execution_id": execution_id,
                    "status": "running",
                    "task": task,
                    "plan": plan,
                    "current_step": 0,
                    "active_step": None,
                    "current_action": "Agent 已接收任务，准备执行计划",
                    "steps": [],
                    "answer": "",
                    "sources": [],
                    **self._get_run_metadata(execution_id),
                },
            )
            return body, 202
        except InvalidAgentRequest as error:
            logger.warning(f"Agent 请求参数无效: {error}")
            return response.error(400, str(error))
        except Exception:
            logger.exception("启动 Agent 执行失败")
            return response.error(500, "Agent 状态存储暂时不可用")

    @staticmethod
    def _get_run_metadata(execution_id: str) -> dict:
        with open_checkpointer() as checkpointer:
            run = get_run(checkpointer, execution_id)
            return run_metadata(run or {})

    def resume(self, execution_id: str):
        try:
            execution_id = validate_execution_id(execution_id)
            with open_checkpointer() as checkpointer:
                run = get_run(checkpointer, execution_id)
                if run is None:
                    return response.error(404, "未找到该 Agent 执行记录")
                values = checkpoint_values(checkpointer, execution_id)
                if run["status"] == "completed" or _is_completed(values):
                    set_run_status(checkpointer, execution_id, "completed")
                    update_run_progress(
                        checkpointer,
                        execution_id,
                        {
                            "stage": "completed",
                            "current_step": len(values.get("plan", [])),
                            "active_step": None,
                            "steps": values.get("steps", []),
                            "current_action": "任务执行完成",
                        },
                    )
                    return response.success(
                        "任务已完成", run_result(execution_id, values)
                    )
                if run["status"] not in ("failed", "running") or not values:
                    return response.error(409, "该执行记录当前不可恢复")

                llm = LLM()
                set_run_status(checkpointer, execution_id, "running")
                update_run_progress(
                    checkpointer,
                    execution_id,
                    {
                        "stage": "resuming",
                        "active_step": None,
                        "current_action": "正在从上次中断处恢复执行",
                    },
                )
                graph = build_graph(
                    llm,
                    checkpointer,
                    execution_id=execution_id,
                )
                with claim_execution(execution_id):
                    state = graph.invoke(None, config=run_config(execution_id))
                set_run_status(checkpointer, execution_id, "completed")
                update_run_progress(
                    checkpointer,
                    execution_id,
                    {
                        "stage": "completed",
                        "current_step": len(state.get("plan", [])),
                        "active_step": None,
                        "steps": state.get("steps", []),
                        "current_action": "任务执行完成",
                    },
                )
                return response.success("任务恢复完成", run_result(execution_id, state))
        except InvalidAgentRequest as error:
            return response.error(400, str(error))
        except AgentRunBusy:
            return response.error(409, "该 Agent 任务正在执行")
        except LLMConfigurationError as error:
            logger.error(f"Agent LLM 配置错误: {error}")
            try:
                with open_checkpointer() as checkpointer:
                    if get_run(checkpointer, execution_id):
                        set_run_status(
                            checkpointer,
                            execution_id,
                            "failed",
                            "agent_configuration_error",
                        )
                        update_run_progress(
                            checkpointer,
                            execution_id,
                            {
                                "stage": "failed",
                                "active_step": None,
                                "current_action": "恢复任务时发生模型配置错误",
                            },
                        )
            except Exception:
                logger.exception("保存 Agent 恢复配置错误状态时出错")
            return response.error(
                500,
                "Agent 模型配置错误",
                {
                    "execution_id": execution_id,
                    "status": "failed",
                    "error_code": "agent_configuration_error",
                    "resumable": True,
                },
            )
        except Exception as error:
            logger.exception(f"Agent 任务恢复失败: {error}")
            try:
                with open_checkpointer() as checkpointer:
                    if get_run(checkpointer, execution_id):
                        set_run_status(
                            checkpointer, execution_id, "failed", "agent_resume_failed"
                        )
                        update_run_progress(
                            checkpointer,
                            execution_id,
                            {
                                "stage": "failed",
                                "active_step": None,
                                "current_action": "恢复执行失败，可稍后重试",
                            },
                        )
            except Exception:
                logger.exception("保存 Agent 恢复失败状态时出错")
            return response.error(
                502,
                "Agent 恢复失败，可稍后重试",
                {
                    "execution_id": execution_id,
                    "status": "failed",
                    "error_code": "agent_resume_failed",
                    "resumable": True,
                },
            )

    def get_status(self, execution_id: str):
        try:
            execution_id = validate_execution_id(execution_id)
            with open_checkpointer() as checkpointer:
                run = get_run(checkpointer, execution_id)
                if run is None:
                    return response.error(404, "未找到该 Agent 执行记录")
                values = checkpoint_values(checkpointer, execution_id)
                if _is_completed(values) and run["status"] != "completed":
                    set_run_status(checkpointer, execution_id, "completed")
                    run["status"] = "completed"
                    run["error_code"] = None
                progress = run.pop("progress", {})
                result = run.get("result") or {}
                steps = progress.get("steps", values.get("steps", []))
                return response.success(
                    "执行状态",
                    {
                        **run,
                        "plan": values.get("plan") or progress.get("plan", []),
                        "current_step": max(
                            progress.get("current_step", 0),
                            values.get("current_step", 0),
                        ),
                        "active_step": progress.get("active_step"),
                        "current_action": (
                            "正在停止，等待当前模型或工具调用返回"
                            if run["status"] == "running" and run["cancel_requested"]
                            else progress.get("current_action", "")
                        ),
                        "steps": steps,
                        "answer": values.get("answer", result.get("answer", "")),
                        "sources": values.get("sources") or result.get("sources") or [],
                        "resumable": bool(values) and run["status"] != "completed",
                    },
                )
        except InvalidAgentRequest as error:
            return response.error(400, str(error))
        except Exception as error:
            logger.exception(f"读取 Agent 状态失败: {error}")
            return response.error(500, "读取 Agent 执行状态失败")

    def cancel(self, execution_id: str):
        try:
            execution_id = validate_execution_id(execution_id)
            with open_checkpointer() as checkpointer:
                run = get_run(checkpointer, execution_id)
                if run is None:
                    return response.error(404, "未找到该执行记录")
                if not request_run_cancel(checkpointer, execution_id):
                    return response.error(409, "该任务已结束")
            return response.success(
                "已请求停止任务",
                {
                    "execution_id": execution_id,
                    "status": "running",
                    "cancel_requested": True,
                },
            )
        except InvalidAgentRequest as error:
            return response.error(400, str(error))
        except Exception as error:
            logger.exception(f"请求停止任务失败: {error}")
            return response.error(500, "请求停止任务失败")
