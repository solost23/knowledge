"""Coordinate the phases of a background ask execution."""

from collections.abc import Callable
from dataclasses import dataclass

from loguru import logger

from app.agents.ask_run_lifecycle import AskRunLifecycle
from app.agents.dispatcher import RequestRoute
from app.agents.memory import prepare_history
from app.agents.service import AgentTaskService
from app.agents.state import AgentRunCancelled
from app.infrastructure.llm import LLMConfigurationError
from app.knowledge.rag import QuestionService


@dataclass(frozen=True)
class AskRunContext:
    execution_id: str
    task: str
    conversation_id: str
    model: object
    history: list[dict[str, str]]
    route: RequestRoute

    @property
    def resolved_task(self) -> str:
        return self.route.resolved_task.strip() or self.task


class AskExecution:
    """Coordinate context loading, routing, mode execution, and finalization."""

    def __init__(
        self,
        agent_service: AgentTaskService,
        question_service: QuestionService,
        llm_factory: Callable,
        route_classifier: Callable,
        lifecycle: AskRunLifecycle | None = None,
    ):
        self.agent_service = agent_service
        self.question_service = question_service
        self.llm_factory = llm_factory
        self.route_classifier = route_classifier
        self.lifecycle = lifecycle or AskRunLifecycle()

    def run(self, execution_id: str, task: str, conversation_id: str) -> None:
        try:
            context = self._prepare_context(execution_id, task, conversation_id)
            self.lifecycle.record_route(execution_id, conversation_id, context.route)
            self._execute(context)
        except AgentRunCancelled:
            self._finish_cancelled(execution_id, task, conversation_id)
        except LLMConfigurationError as error:
            logger.error(f"请求路由配置错误: {error}")
            self._fail(execution_id, "llm_configuration_error", "模型配置错误")
        except Exception as error:
            logger.exception(f"后台问答执行失败: {error}")
            self._fail(
                execution_id,
                "ask_execution_failed",
                "任务执行失败，请稍后重试",
            )

    def _prepare_context(
        self, execution_id: str, task: str, conversation_id: str
    ) -> AskRunContext:
        self.lifecycle.ensure_not_cancelled(execution_id)
        model = self.llm_factory()
        history = prepare_history(conversation_id, model)
        self.lifecycle.ensure_not_cancelled(execution_id)
        route = self.route_classifier(model, task, history)
        self.lifecycle.ensure_not_cancelled(execution_id)
        return AskRunContext(
            execution_id=execution_id,
            task=task,
            conversation_id=conversation_id,
            model=model,
            history=history,
            route=route,
        )

    def _execute(self, context: AskRunContext) -> None:
        if context.route.mode == "agent":
            self._execute_agent(context)
        else:
            self._execute_rag(context)

    def _execute_agent(self, context: AskRunContext) -> None:
        plan = [step.strip() for step in context.route.steps if step.strip()]
        if not plan:
            raise ValueError("路由模型选择了 Agent，但没有生成执行步骤")

        def on_complete(result: dict) -> None:
            result.update(
                {
                    "mode": "agent",
                    "route_reason": context.route.reason,
                    "conversation_id": context.conversation_id,
                }
            )
            self.lifecycle.save_exchange(context.conversation_id, context.task, result)

        response_body, status = self.agent_service.run_existing(
            context.execution_id,
            context.resolved_task,
            plan,
            llm=context.model,
            on_complete=on_complete,
        )
        if status >= 400:
            self.lifecycle.save_agent_error_result(
                context.execution_id,
                context.conversation_id,
                context.route,
                response_body.get("data") or {},
            )

    def _execute_rag(self, context: AskRunContext) -> None:
        rag_response, status = self.question_service.question(
            context.resolved_task,
            history=context.history,
            llm=context.model,
        )
        if status >= 400:
            self._fail(
                context.execution_id,
                "rag_execution_failed",
                rag_response.get("message", "问答处理失败"),
            )
            return

        self.lifecycle.ensure_not_cancelled(context.execution_id)
        self.lifecycle.complete_rag(
            context.execution_id,
            context.task,
            context.conversation_id,
            context.route,
            rag_response.get("data") or {},
        )

    def _finish_cancelled(
        self, execution_id: str, task: str, conversation_id: str
    ) -> None:
        try:
            self.lifecycle.finish_cancelled(execution_id, task, conversation_id)
        except Exception:
            logger.exception("保存已停止任务失败")

    def _fail(self, execution_id: str, error_code: str, error_message: str) -> None:
        try:
            self.lifecycle.fail(execution_id, error_code, error_message)
        except Exception:
            logger.exception("保存问答执行失败状态时出错")
