from flask_openapi3 import OpenAPI, Info
from werkzeug.exceptions import RequestEntityTooLarge

from app.core import response
from app.core.settings import config

MAX_REQUEST_BYTES = 100 * 1024 * 1024


class Server:
    def __init__(self, register):
        info = Info(title="Knowledge API", version="1.0.0")
        self.app = OpenAPI(config.name, info=info)
        self.app.config["MAX_CONTENT_LENGTH"] = MAX_REQUEST_BYTES
        register(self.app)

        @self.app.errorhandler(RequestEntityTooLarge)
        def request_too_large(_error):
            return response.error(413, "上传文件过大，请将文件控制在 100 MiB 以内")

        self.app.add_url_rule("/healthz", "healthz", lambda: {"status": "ok"})

    def run(self):
        mode = True
        if config.mode == 'release':
            mode = False

        self.app.run(
            host=config.host,
            port=config.port,
            debug=mode,
        )

    def stop(self):
        pass
