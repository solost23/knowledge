from flask_openapi3 import OpenAPI

from app.api.controllers.ask import AskController
from app.api.controllers.conversations import ConversationController
from app.api.controllers.documents import DocumentController


def register(app: OpenAPI):
    DocumentController(app).register()
    AskController(app).register()
    ConversationController(app).register()
