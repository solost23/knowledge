from typing import Annotated, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
from pydantic import BaseModel, Field

from app.agents.service import AgentTaskService
from app.agents.state import MAX_PLAN_STEPS, MAX_STEP_LENGTH, MAX_TASK_LENGTH
from app.core import response
from app.infrastructure.llm import build_configured_llm
from app.knowledge.rag import QuestionService

ROUTER_PROMPT = """判断用户请求应该走普通 RAG 还是计划执行 Agent。
普通事实问答、单一主题查询、基于一次检索即可回答的请求，选择 rag。
需要比较多个文档、分头查找多个信息、根据中间结果继续行动的请求，选择 agent。
选择 agent 时同时给出 1 到 5 个有序、具体的执行步骤；选择 rag 时 steps 留空。
不要因为用户用了多句话或问号就自动选择 agent。"""


class RequestRoute(BaseModel):
    mode: Literal["rag", "agent"]
    reason: str = Field(description="简短说明选择此处理流程的原因")
    steps: list[Annotated[str, Field(min_length=1, max_length=MAX_STEP_LENGTH)]] = Field(
        default_factory=list,
        max_length=MAX_PLAN_STEPS,
        description="Agent 模式下最多五个有序步骤，RAG 模式下留空",
    )


def classify_task(llm, task: str) -> RequestRoute:
    router = llm.with_structured_output(RequestRoute, method="function_calling")
    return router.invoke([
        SystemMessage(content=ROUTER_PROMPT),
        HumanMessage(content=task),
    ])


def dispatch_request(task: str, llm=None):
    """Classify an incoming request and invoke the matching application service."""
    task = task.strip() if isinstance(task, str) else ""
    if not task:
        return response.error(400, "任务内容不能为空")
    if len(task) > MAX_TASK_LENGTH:
        return response.error(400, f"任务内容不能超过 {MAX_TASK_LENGTH} 个字符")

    model = llm or build_configured_llm()
    route = classify_task(model, task)
    if route.mode == "agent":
        plan = [step.strip() for step in route.steps if step.strip()]
        if not plan:
            raise ValueError("路由模型选择了 Agent，但没有生成执行步骤")
        result, status = AgentTaskService().execute(task, plan=plan, llm=model)
    else:
        result, status = QuestionService().question(task, llm=model)

    if status >= 400:
        return result, status

    data = result.get("data") or {}
    data["mode"] = route.mode
    data["route_reason"] = route.reason
    return response.success("任务完成", data)


class AskService:
    def ask(self, task: str):
        try:
            return dispatch_request(task)
        except EnvironmentError as error:
            logger.error(f"请求路由配置错误: {error}")
            return response.error(500, "问答模型配置错误")
        except Exception as error:
            logger.exception(f"请求路由失败: {error}")
            return response.error(502, "请求路由失败，请稍后重试")
