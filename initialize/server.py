from flask import Flask
from universal.config import config


class Server:
    def __init__(self, register):
        self.app = Flask(config.name)
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
