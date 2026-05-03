from flask_openapi3 import OpenAPI

from controllers.doc import DocController
from controllers.question import QuestionController


def register(app: OpenAPI):
    DocController(app).register()
    QuestionController(app).register()
