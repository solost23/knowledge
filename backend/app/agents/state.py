from typing import Annotated, TypedDict

from pydantic import BaseModel, ConfigDict, Field

MAX_PLAN_STEPS = 5
MAX_SEARCH_RESULTS = 4
MAX_SOURCE_LENGTH = 2000
MAX_TASK_LENGTH = 2000
MAX_STEP_LENGTH = 300
MAX_TOOL_CALLS_PER_STEP = 6
MAX_TOOL_RECURSION = 16

PlanStep = Annotated[str, Field(min_length=1, max_length=MAX_STEP_LENGTH)]


class TaskPlan(BaseModel):
    steps: list[PlanStep] = Field(
        min_length=1,
        max_length=MAX_PLAN_STEPS,
        description="按执行顺序排列的知识库任务步骤，最多五步",
    )


class SearchKnowledgeInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: str = Field(min_length=1, max_length=500, description="知识库搜索词")
    top_k: int = Field(default=MAX_SEARCH_RESULTS, ge=1, le=MAX_SEARCH_RESULTS)


class InvalidAgentRequest(Exception):
    pass


class AgentRunBusy(Exception):
    pass


class AgentState(TypedDict):
    task: str
    plan: list[str]
    current_step: int
    steps: list[dict]
    sources: list[dict[str, str]]
    answer: str
