import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = BACKEND_DIR.parent
load_dotenv(PROJECT_DIR / ".env", override=False)
load_dotenv(BACKEND_DIR / ".env", override=False)
DATA_DIR = Path(os.environ.get("APP_DATA_DIR", BACKEND_DIR / "data")).resolve()


class YamlConfig:
    def __init__(self, config_path: str | Path | None = None):
        path = Path(config_path or os.environ.get(
            "APP_CONFIG_FILE", BACKEND_DIR / "conf" / "config.yaml"
        ))
        with path.open("r", encoding="utf-8") as file:
            self._config = yaml.safe_load(file)

    def __getattr__(self, name):
        try:
            value = self._config[name]
        except KeyError as error:
            raise AttributeError(f"配置项 '{name}' 不存在") from error
        env_name = {
            "host": "APP_HOST",
            "port": "APP_PORT",
            "mode": "APP_MODE",
        }.get(name)
        if env_name and env_name in os.environ:
            return type(value)(os.environ[env_name])
        return value


config = YamlConfig()
