from typing import Annotated, Literal
from concurrent.futures import ThreadPoolExecutor

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from loguru import logger
from pydantic import BaseModel, Field

from app.agents.service import AgentTaskService
from app.agents.memory import prepare_history
from app.agents.checkpoints import (
    finish_run,
    clear_run_cancel,
    get_run,
    open_checkpointer,
    run_cancel_requested,
    save_run_result,
    set_run_mode,
    set_run_status,
    update_run_progress,
)
from app.agents.state import (
    MAX_STEP_LENGTH,
    MAX_TASK_LENGTH,
    AgentRunCancelled,
)
from app.core import response
from app.infrastructure.llm import build_configured_llm
from app.knowledge.rag import QuestionService

ROUTER_PROMPT = """判断用户请求应该走普通 RAG 还是计划执行 Agent。
普通事实问答、单一主题查询、基于一次检索即可回答的请求，选择 rag。
需要比较多个文档、分头查找多个信息、根据中间结果继续行动的请求，选择 agent。
选择 agent 时同时给出完成任务所需数量的有序、具体执行步骤；选择 rag 时 steps 留空。
结合对话历史理解当前请求中的指代和省略。resolved_task 应是能脱离历史独立理解的当前请求，不要把历史里已完成的请求当成新任务。
不要因为用户用了多句话或问号就自动选择 agent。"""

_ASK_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ask-run")


class RequestRoute(BaseModel):
    mode: Literal["rag", "agent"]
    reason: str = Field(description="简短说明选择此处理流程的原因")
    resolved_task: str = Field(
        min_length=1,
        max_length=MAX_TASK_LENGTH,
        description="结合对话历史补全后的当前请求，改写成独立完整的任务",
    )
    steps: list[Annotated[str, Field(min_length=1, max_length=MAX_STEP_LENGTH)]] = Field(
        default_factory=list,
        description="Agent 模式下按任务需要生成有序步骤，RAG 模式下留空",
    )


def classify_task(llm, task: str, history: list[dict[str, str]] | None = None) -> RequestRoute:
    router = llm.with_structured_output(RequestRoute, method="function_calling")
    messages = [SystemMessage(content=ROUTER_PROMPT)]
    for message in history or []:
        message_type = HumanMessage if message["role"] == "user" else AIMessage
        messages.append(message_type(content=message["content"]))
    messages.append(HumanMessage(content=task))
    return router.invoke(messages)


def dispatch_request(
    task: str,
    llm=None,
    history: list[dict[str, str]] | None = None,
    on_agent_complete=None,
):
    """Classify an incoming request and invoke the matching application service."""
    task = task.strip() if isinstance(task, str) else ""
    if not task:
        return response.error(400, "任务内容不能为空")
    if len(task) > MAX_TASK_LENGTH:
        return response.error(400, f"任务内容不能超过 {MAX_TASK_LENGTH} 个字符")

    model = llm or build_configured_llm()
    route = classify_task(model, task, history)
    resolved_task = route.resolved_task.strip() or task
    if route.mode == "agent":
        plan = [step.strip() for step in route.steps if step.strip()]
        if not plan:
            raise ValueError("路由模型选择了 Agent，但没有生成执行步骤")

        def on_complete(result):
            if on_agent_complete:
                result.update({"mode": "agent", "route_reason": route.reason})
                on_agent_complete(result)

        result, status = AgentTaskService().start(
            resolved_task,
            plan=plan,
            llm=model,
            on_complete=on_complete,
        )
    else:
        result, status = QuestionService().question(
            resolved_task, history=history, llm=model
        )

    if status >= 400:
        return result, status

    data = result.get("data") or {}
    data["mode"] = route.mode
    data["route_reason"] = route.reason
    result["data"] = data
    if route.mode == "agent":
        result["message"] = "Agent 任务已启动"
    return result, status


class AskService:
    def start(self, task: str, conversation_id=None):
        from app.infrastructure.conversation_repository import create_conversation, get_conversation

        task = task.strip() if isinstance(task, str) else ""
        if not task:
            return response.error(400, "任务内容不能为空")
        if len(task) > MAX_TASK_LENGTH:
            return response.error(400, f"任务内容不能超过 {MAX_TASK_LENGTH} 个字符")

        if conversation_id is None:
            conversation = create_conversation()
            conversation_id = conversation["id"]
        else:
            conversation_id = str(conversation_id)
            conversation = get_conversation(conversation_id)
            if conversation is None:
                return response.error(404, "未找到该对话")

        try:
            execution_id, run = AgentTaskService().create_routed_run(task, conversation_id)
            _ASK_EXECUTOR.submit(self._run, execution_id, task, conversation_id)
            body, _ = response.success("任务已启动", {
                "execution_id": execution_id,
                "status": "running",
                "mode": "routing",
                "task": task,
                "conversation_id": conversation_id,
                "current_action": "正在判断处理方式",
                "started_at": run["started_at"],
                "duration_ms": 0,
                "answer": "",
                "sources": [],
                "steps": [],
            })
            return body, 202
        except Exception:
            logger.exception("创建问答执行记录失败")
            return response.error(500, "任务状态存储暂时不可用")

    @staticmethod
    def _ensure_not_cancelled(execution_id: str):
        with open_checkpointer() as checkpointer:
            if run_cancel_requested(checkpointer, execution_id):
                raise AgentRunCancelled(execution_id)

    @staticmethod
    def _save_exchange(conversation_id: str, task: str, result: dict):
        from app.infrastructure.conversation_repository import append_exchange

        append_exchange(
            conversation_id,
            task,
            result.get("answer") or ("任务已停止" if result.get("status") == "cancelled" else ""),
            result,
        )

    def _finish_cancelled(self, execution_id: str, task: str, conversation_id: str):
        with open_checkpointer() as checkpointer:
            set_run_status(checkpointer, execution_id, "cancelled")
            update_run_progress(checkpointer, execution_id, {
                "stage": "cancelled",
                "active_step": None,
                "current_action": "任务已停止",
            })
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
        self._save_exchange(conversation_id, task, result)
        clear_run_cancel(execution_id)

    def _fail(self, execution_id: str, error_code: str, error_message: str):
        with open_checkpointer() as checkpointer:
            set_run_status(checkpointer, execution_id, "failed", error_code)
            update_run_progress(checkpointer, execution_id, {
                "stage": "failed",
                "active_step": None,
                "current_action": error_message,
                "error_message": error_message,
            })
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

    def _run(self, execution_id: str, task: str, conversation_id: str):
        try:
            self._ensure_not_cancelled(execution_id)
            model = build_configured_llm()
            history = prepare_history(conversation_id, model)
            self._ensure_not_cancelled(execution_id)
            route = classify_task(model, task, history)
            resolved_task = route.resolved_task.strip() or task
            self._ensure_not_cancelled(execution_id)

            with open_checkpointer() as checkpointer:
                set_run_mode(checkpointer, execution_id, route.mode, route.reason, conversation_id)
                update_run_progress(checkpointer, execution_id, {
                    "stage": "executing" if route.mode == "agent" else "retrieving",
                    "plan": route.steps if route.mode == "agent" else [],
                    "current_step": 0,
                    "active_step": None,
                    "current_action": (
                        "执行计划已生成，准备开始"
                        if route.mode == "agent"
                        else "正在检索知识库并生成回答"
                    ),
                })

            if route.mode == "agent":
                plan = [step.strip() for step in route.steps if step.strip()]
                if not plan:
                    raise ValueError("路由模型选择了 Agent，但没有生成执行步骤")

                def on_complete(agent_result):
                    agent_result.update({
                        "mode": "agent",
                        "route_reason": route.reason,
                        "conversation_id": conversation_id,
                    })
                    self._save_exchange(conversation_id, task, agent_result)

                response_body, status = AgentTaskService().run_existing(
                    execution_id,
                    resolved_task,
                    plan,
                    llm=model,
                    on_complete=on_complete,
                )
                if status >= 400:
                    result = response_body.get("data") or {}
                    result.update({
                        "mode": "agent",
                        "route_reason": route.reason,
                        "conversation_id": conversation_id,
                    })
                    with open_checkpointer() as checkpointer:
                        save_run_result(checkpointer, execution_id, result)
                return

            rag_response, status = QuestionService().question(
                resolved_task,
                history=history,
                llm=model,
            )
            if status >= 400:
                self._fail(execution_id, "rag_execution_failed", rag_response.get("message", "问答处理失败"))
                return
            self._ensure_not_cancelled(execution_id)
            data = rag_response.get("data") or {}
            with open_checkpointer() as checkpointer:
                if not finish_run(checkpointer, execution_id, "completed", respect_cancel=True):
                    raise AgentRunCancelled(execution_id)
                update_run_progress(checkpointer, execution_id, {
                    "stage": "completed",
                    "current_action": "检索与回答已完成",
                })
                run = get_run(checkpointer, execution_id)
                result = {
                    "execution_id": execution_id,
                    "status": "completed",
                    "mode": "rag",
                    "task": resolved_task,
                    "conversation_id": conversation_id,
                    "route_reason": route.reason,
                    "plan": [],
                    "current_step": 0,
                    "steps": [],
                    "answer": data.get("answer", ""),
                    "sources": data.get("sources", []),
                    "started_at": run["started_at"],
                    "finished_at": run["finished_at"],
                    "duration_ms": run["duration_ms"],
                }
                save_run_result(checkpointer, execution_id, result)
            self._save_exchange(conversation_id, task, result)
        except AgentRunCancelled:
            try:
                self._finish_cancelled(execution_id, task, conversation_id)
            except Exception:
                logger.exception("保存已停止任务失败")
        except EnvironmentError as error:
            logger.error(f"请求路由配置错误: {error}")
            self._fail(execution_id, "llm_configuration_error", "模型配置错误")
        except Exception as error:
            logger.exception(f"后台问答执行失败: {error}")
            self._fail(execution_id, "ask_execution_failed", "任务执行失败，请稍后重试")

    def ask(self, task: str, conversation_id=None):
        from app.infrastructure.conversation_repository import (
            append_exchange,
            create_conversation,
            get_conversation,
        )

        try:
            task = task.strip() if isinstance(task, str) else ""
            if not task:
                return response.error(400, "任务内容不能为空")
            if len(task) > MAX_TASK_LENGTH:
                return response.error(400, f"任务内容不能超过 {MAX_TASK_LENGTH} 个字符")

            if conversation_id is None:
                conversation = create_conversation()
                conversation_id = conversation["id"]
            else:
                conversation_id = str(conversation_id)
                conversation = get_conversation(conversation_id)
                if conversation is None:
                    return response.error(404, "未找到该对话")

            model = build_configured_llm()
            history = prepare_history(conversation_id, model)
            def save_agent_result(agent_result):
                agent_result.update({
                    "mode": "agent",
                    "conversation_id": conversation_id,
                })
                append_exchange(
                    conversation_id,
                    task,
                    agent_result.get("answer", ""),
                    agent_result,
                )

            result, status = dispatch_request(
                task,
                llm=model,
                history=history,
                on_agent_complete=save_agent_result,
            )
            if status >= 400:
                return result, status

            data = result.get("data") or {}
            data["conversation_id"] = conversation_id
            if data.get("mode") != "agent":
                append_exchange(conversation_id, task, data.get("answer", ""), data)
            return result, status
        except EnvironmentError as error:
            logger.error(f"请求路由配置错误: {error}")
            return response.error(500, "问答模型配置错误")
        except Exception as error:
            logger.exception(f"请求路由失败: {error}")
            return response.error(502, "请求路由失败，请稍后重试")
