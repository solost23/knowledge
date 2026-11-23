"""Application entry point for synchronous and asynchronous ask requests."""

from loguru import logger

from app.agents.ask_execution import AskExecution
from app.agents.dispatcher import classify_task, dispatch_request
from app.agents.memory import prepare_history
from app.agents.service import AgentTaskService
from app.agents.state import InvalidAgentRequest
from app.agents.validation import validate_task
from app.application.conversations import ConversationService
from app.core import response
from app.infrastructure import submit_task
from app.infrastructure.conversation_repository import append_exchange
from app.infrastructure.llm import LLM, LLMConfigurationError
from app.knowledge.rag import QuestionService


class AskService:
    """Accept ask requests and hand accepted runs to the background workflow."""

    def __init__(
        self,
        agent_service: AgentTaskService | None = None,
        question_service: QuestionService | None = None,
        submit_background_task=None,
        llm_factory=None,
    ):
        self.agent_service = agent_service or AgentTaskService()
        self.question_service = question_service or QuestionService()
        self.submit_background_task = submit_background_task or submit_task
        self.llm_factory = llm_factory or LLM
        self.execution = AskExecution(
            agent_service=self.agent_service,
            question_service=self.question_service,
            llm_factory=self.llm_factory,
            route_classifier=classify_task,
        )

    def start(self, task: str, conversation_id=None):
        """Validate, create a run record, enqueue it, and return HTTP acceptance."""
        try:
            task = validate_task(task)
        except InvalidAgentRequest as error:
            return response.error(400, str(error))

        conversation = ConversationService.resolve(conversation_id)
        if conversation is None:
            return response.error(404, "未找到该对话")
        conversation_id, _conversation = conversation

        try:
            execution_id, run = self.agent_service.create_routed_run(
                task, conversation_id
            )
            self.submit_background_task(
                "ask", self.execution.run, execution_id, task, conversation_id
            )
            body, _ = response.success(
                "任务已启动",
                {
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
                },
            )
            return body, 202
        except Exception:
            logger.exception("创建问答执行记录失败")
            return response.error(500, "任务状态存储暂时不可用")

    def ask(self, task: str, conversation_id=None):
        """Run the legacy synchronous flow, retained for internal callers."""
        try:
            task = validate_task(task)
            conversation = ConversationService.resolve(conversation_id)
            if conversation is None:
                return response.error(404, "未找到该对话")
            conversation_id, _conversation = conversation

            model = self.llm_factory()
            history = prepare_history(conversation_id, model)
            result, status = dispatch_request(
                task,
                llm=model,
                history=history,
                on_agent_complete=lambda agent_result: self._save_agent_result(
                    agent_result, conversation_id, task
                ),
            )
            if status >= 400:
                return result, status

            data = result.get("data") or {}
            data["conversation_id"] = conversation_id
            if data.get("mode") != "agent":
                append_exchange(conversation_id, task, data.get("answer", ""), data)
            return result, status
        except InvalidAgentRequest as error:
            return response.error(400, str(error))
        except LLMConfigurationError as error:
            logger.error(f"请求路由配置错误: {error}")
            return response.error(500, "问答模型配置错误")
        except Exception as error:
            logger.exception(f"请求路由失败: {error}")
            return response.error(502, "请求路由失败，请稍后重试")

    @staticmethod
    def _save_agent_result(agent_result: dict, conversation_id: str, task: str):
        agent_result.update({"mode": "agent", "conversation_id": conversation_id})
        append_exchange(
            conversation_id,
            task,
            agent_result.get("answer", ""),
            agent_result,
        )
