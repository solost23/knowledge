import os

from flask_openapi3 import Tag
from flask import Flask, request
from pydantic import BaseModel, Field

from app.core import response
from app.knowledge.documents import DocumentService
from app.knowledge.document_imports import DocumentImportService
from app.knowledge.loaders.structured import (
    SUPPORTED_EXTENSIONS,
    SUPPORTED_EXTENSIONS_TEXT,
)

tag = Tag(name="笔记管理")

ALLOWED_EXTENSIONS = SUPPORTED_EXTENSIONS


class DocPath(BaseModel):
    name: str = Field(description="文件名")


class UploadTaskPath(BaseModel):
    task_id: str = Field(description="文档导入任务 ID")


class UploadForm(BaseModel):
    file: object = Field(description=f"上传的文件，支持 {SUPPORTED_EXTENSIONS_TEXT}")


class DocumentController:
    def __init__(self, app: Flask):
        self.app = app
        self.import_service = DocumentImportService()

    def register(self):
        @self.app.post("/upload", tags=[tag], summary="上传笔记")
        def upload():
            file = request.files.get('file')
            if not file or not file.filename:
                return response.error(400, '请上传文件')

            ext = os.path.splitext(file.filename)[-1].lower()
            if ext not in ALLOWED_EXTENSIONS:
                return response.error(400, f'不支持的文件类型 {ext}，仅支持 {SUPPORTED_EXTENSIONS_TEXT}')

            filename = os.path.basename(file.filename.replace("\\", "/"))
            try:
                job = self.import_service.submit(file, ext, filename)
            except Exception as error:
                return response.error(500, f"上传文件失败：{error}")
            body, _ = response.success("文件已接收，正在导入", job)
            return body, 202

        @self.app.get("/upload/tasks/<task_id>", tags=[tag], summary="查询文档导入进度")
        def upload_task(path: UploadTaskPath):
            job = self.import_service.get(path.task_id)
            if not job:
                return response.error(404, "文档导入任务不存在")
            return response.success("成功", job)

        @self.app.get("/docs", tags=[tag], summary="笔记列表")
        def list_docs():
            return DocumentService().list()

        @self.app.delete("/docs/<name>", tags=[tag], summary="删除笔记")
        def delete_doc(path: DocPath):
            return DocumentService().delete(path.name)
