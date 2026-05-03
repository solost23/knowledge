from flask_openapi3 import OpenAPI, Info
from universal.config import config


class Server:
    def __init__(self, register):
        info = Info(title="Knowledge API", version="1.0.0")
        self.app = OpenAPI(config.name, info=info)
        register(self.app)

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
