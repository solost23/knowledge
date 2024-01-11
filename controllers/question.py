from flask import Flask, request

from services.question import QuestionService


class QuestionController:
    def __init__(self, app: Flask):
        self.app = app

    def register(self):
        @self.app.get("/question")
        def question():
            param: str = request.args.get('question')
            return QuestionService().question(param)