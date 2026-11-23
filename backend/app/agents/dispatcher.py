from typing import Annotated, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.agents.service import AgentTaskService
from app.agents.state import (
    MAX_STEP_LENGTH,
    MAX_TASK_LENGTH,
    InvalidAgentRequest,
)
from app.agents.validation import validate_task
from app.core import response
from app.infrastructure.llm import LLM
from app.knowledge.rag import QuestionService
from app.prompts.routing import ROUTER_PROMPT


class RequestRoute(BaseModel):
    mode: Literal["rag", "agent"]
    reason: str = Field(description="简短说明选择此处理流程的原因")
    resolved_task: str = Field(
        min_length=1,
        max_length=MAX_TASK_LENGTH,
        description="结合对话历史补全后的当前请求，改写成独立完整的任务",
    )
    steps: list[Annotated[str, Field(min_length=1, max_length=MAX_STEP_LENGTH)]] = (
        Field(
            default_factory=list,
            description="Agent 模式下按任务需要生成有序步骤，RAG 模式下留空",
        )
    )


def classify_task(
    llm, task: str, history: list[dict[str, str]] | None = None
) -> RequestRoute:
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
    try:
        task = validate_task(task)
    except InvalidAgentRequest as error:
        return response.error(400, str(error))

    model = llm or LLM()
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
