from flask import Flask

from controllers.doc import DocController
from controllers.question import QuestionController


def register(app: Flask):
    """
    注册路由
    :param app:
    :return:
    """

    # 文档载入
    DocController(app).register()
    # 问答
    QuestionController(app).register()


