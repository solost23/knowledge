import signal
import sys

from loguru import logger

from app.agents.checkpoints import recover_interrupted_runs
from app.agents.execution import recover_pending_exchanges
from app.api.router import register
from app.core.server import Server


def on_exit(signo, frame):
    """Stop the process after a termination signal."""
    logger.info("进程结束")
    sys.exit(0)


if __name__ == "__main__":
    recover_interrupted_runs()
    recover_pending_exchanges()
    server = Server(register)
    signal.signal(signal.SIGINT, on_exit)
    signal.signal(signal.SIGTERM, on_exit)
    server.run()
