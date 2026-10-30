from typing import Annotated, TypedDict

from pydantic import BaseModel, ConfigDict, Field

from app.core.settings import config

MAX_SEARCH_RESULTS = 4
MAX_SOURCE_LENGTH = 2000
MAX_TASK_LENGTH = config.agent["max_task_length"]
MAX_STEP_LENGTH = config.agent["max_step_length"]

PlanStep = Annotated[str, Field(min_length=1, max_length=MAX_STEP_LENGTH)]


class TaskPlan(BaseModel):
    steps: list[PlanStep] = Field(
        min_length=1,
        description="按执行顺序排列的、足以完成任务的知识库任务步骤",
    )


class SearchKnowledgeInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: str = Field(min_length=1, max_length=500, description="知识库搜索词")
    top_k: int = Field(default=MAX_SEARCH_RESULTS, ge=1, le=MAX_SEARCH_RESULTS)


class InvalidAgentRequest(Exception):
    pass


class AgentRunBusy(Exception):
    pass


class AgentRunCancelled(Exception):
    pass


class AgentState(TypedDict):
    task: str
    plan: list[str]
    current_step: int
    steps: list[dict]
    sources: list[dict[str, str]]
    answer: str
