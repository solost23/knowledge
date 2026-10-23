from flask_openapi3 import Tag
from flask import Flask
from pydantic import BaseModel
from uuid import UUID

from app.core import response
from app.infrastructure.conversation_repository import (
    create_conversation,
    delete_conversation,
    get_conversation,
    list_conversations,
)

tag = Tag(name="对话管理")


class ConversationPath(BaseModel):
    conversation_id: UUID


class ConversationController:
    def __init__(self, app: Flask):
        self.app = app

    def register(self):
        @self.app.post("/conversations", tags=[tag], summary="新建对话")
        def create():
            return response.success("对话已创建", create_conversation())

        @self.app.get("/conversations", tags=[tag], summary="对话列表")
        def list_all():
            return response.success("对话列表", list_conversations())

        @self.app.get(
            "/conversations/<conversation_id>", tags=[tag], summary="读取对话"
        )
        def get(path: ConversationPath):
            conversation = get_conversation(str(path.conversation_id))
            if conversation is None:
                return response.error(404, "未找到该对话")
            return response.success("对话内容", conversation)

        @self.app.delete(
            "/conversations/<conversation_id>", tags=[tag], summary="删除对话"
        )
        def delete(path: ConversationPath):
            if not delete_conversation(str(path.conversation_id)):
                return response.error(404, "未找到该对话")
            return response.success("对话已删除", None)
