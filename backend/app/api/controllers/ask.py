from uuid import UUID

from flask import Flask
from flask_openapi3 import Tag
from pydantic import BaseModel, ConfigDict, Field

from app.agents.ask_service import AskService
from app.agents.service import AgentTaskService
from app.agents.state import MAX_TASK_LENGTH

tag = Tag(name="问答")


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: str = Field(
        min_length=1,
        max_length=MAX_TASK_LENGTH,
        description="问题或需要完成的知识库任务",
    )
    conversation_id: UUID | None = Field(default=None, description="当前对话 ID")


class AgentResumeRequest(BaseModel):
    execution_id: UUID = Field(description="Agent 执行 ID")


class AgentRunPath(BaseModel):
    execution_id: UUID = Field(description="Agent 执行 ID")


class AskController:
    def __init__(
        self,
        app: Flask,
        ask_service: AskService | None = None,
        agent_service: AgentTaskService | None = None,
    ):
        self.app = app
        self.ask_service = ask_service or AskService()
        self.agent_service = agent_service or AgentTaskService()

    def register(self):
        @self.app.post("/ask", tags=[tag], summary="自动路由问答与任务")
        def ask(body: AskRequest):
            return self.ask_service.start(body.task, body.conversation_id)

        @self.app.post("/agent/resume", tags=[tag], summary="恢复失败的 Agent 任务")
        def resume_agent(body: AgentResumeRequest):
            return self.agent_service.resume(body.execution_id)

        @self.app.get(
            "/agent/runs/<execution_id>", tags=[tag], summary="查询 Agent 执行状态"
        )
        def agent_status(path: AgentRunPath):
            return self.agent_service.get_status(path.execution_id)

        @self.app.post(
            "/agent/runs/<execution_id>/cancel",
            tags=[tag],
            summary="停止问答或 Agent 执行",
        )
        def cancel_agent(path: AgentRunPath):
            return self.agent_service.cancel(path.execution_id)
