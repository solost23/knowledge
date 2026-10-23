from flask_openapi3 import Tag
from flask import Flask
from pydantic import BaseModel, ConfigDict, Field
from uuid import UUID

from app.agents.dispatcher import AskService
from app.agents.service import AgentTaskService
from app.core import response

tag = Tag(name="问答")


class QuestionQuery(BaseModel):
    question: str = Field(description="问题内容")


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: str = Field(min_length=1, max_length=2000, description="问题或需要完成的知识库任务")
    conversation_id: UUID | None = Field(default=None, description="当前对话 ID")


class AgentResumeRequest(BaseModel):
    execution_id: UUID = Field(description="Agent 执行 ID")


class AgentRunPath(BaseModel):
    execution_id: UUID = Field(description="Agent 执行 ID")


class AskController:
    def __init__(self, app: Flask):
        self.app = app

    def register(self):
        @self.app.post("/ask", tags=[tag], summary="自动路由问答与任务")
        def ask(body: AskRequest):
            task = body.task.strip()
            if not task:
                return response.error(400, '参数 task 不能为空')
            return AskService().start(task, body.conversation_id)

        @self.app.get("/question", tags=[tag], summary="问答（自动路由兼容入口）")
        def question(query: QuestionQuery):
            if not query.question.strip():
                return response.error(400, '参数 question 不能为空')
            return AskService().start(query.question.strip())

        @self.app.post("/agent/resume", tags=[tag], summary="恢复失败的 Agent 任务")
        def resume_agent(body: AgentResumeRequest):
            return AgentTaskService().resume(body.execution_id)

        @self.app.get("/agent/runs/<execution_id>", tags=[tag], summary="查询 Agent 执行状态")
        def agent_status(path: AgentRunPath):
            return AgentTaskService().get_status(path.execution_id)

        @self.app.post("/agent/runs/<execution_id>/cancel", tags=[tag], summary="停止问答或 Agent 执行")
        def cancel_agent(path: AgentRunPath):
            return AgentTaskService().cancel(path.execution_id)
