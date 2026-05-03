from flask_openapi3 import Tag
from flask import Flask, request
from pydantic import BaseModel, Field

from initialize import response
from services.question import QuestionService

tag = Tag(name="问答")


class QuestionQuery(BaseModel):
    question: str = Field(description="问题内容")


class QuestionController:
    def __init__(self, app: Flask):
        self.app = app

    def register(self):
        @self.app.get("/question", tags=[tag], summary="问答")
        def question(query: QuestionQuery):
            if not query.question.strip():
                return response.error(400, '参数 question 不能为空')
            return QuestionService().question(query.question.strip())
