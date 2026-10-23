import signal
import sys

from loguru import logger

from app.api.router import register
from app.core.server import Server


def on_exit(signo, frame):
    """
    信号处理
    :param signo:
    :param frame:
    :return:
    """
    logger.info("进程结束")
    sys.exit(0)


if __name__ == "__main__":
    server = Server(register)
    signal.signal(signal.SIGINT, on_exit)
    signal.signal(signal.SIGTERM, on_exit)
    server.run()
