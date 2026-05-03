import os
from typing import Optional

from flask_openapi3 import Tag
from flask import Flask, request
from pydantic import BaseModel, Field

from initialize import response
from services.doc import DocService

tag = Tag(name="笔记管理")

ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.pptx', '.xlsx', '.md'}


class DocPath(BaseModel):
    name: str = Field(description="文件名")


class UploadForm(BaseModel):
    file: object = Field(description="上传的文件，支持 pdf、docx、pptx、xlsx、md")


class DocController:
    def __init__(self, app: Flask):
        self.app = app

    def register(self):
        @self.app.post("/upload", tags=[tag], summary="上传笔记")
        def upload():
            file = request.files.get('file')
            if not file or not file.filename:
                return response.error(400, '请上传文件')

            ext = os.path.splitext(file.filename)[-1].lower()
            if ext not in ALLOWED_EXTENSIONS:
                return response.error(400, f'不支持的文件类型 {ext}，仅支持 pdf、docx、pptx、xlsx、md')

            return DocService().upload(file)

        @self.app.get("/docs", tags=[tag], summary="笔记列表")
        def list_docs():
            return DocService().list()

        @self.app.delete("/docs/<name>", tags=[tag], summary="删除笔记")
        def delete_doc(path: DocPath):
            return DocService().delete(path.name)
