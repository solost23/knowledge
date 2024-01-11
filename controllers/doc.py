from flask import Flask, request

from services.doc import DocService


class DocController:
    def __init__(self, app: Flask):
        self.app = app

    def register(self):
        @self.app.post("/upload")
        def upload():
            file = request.files.get('file')
            return DocService().upload(file)





