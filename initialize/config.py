import yaml


class YamlConfig:
    def __init__(self, config_path: str = "./conf/config.yaml"):
        with open(config_path, "r") as file:
            self._config = yaml.safe_load(file)

    def __getattr__(self, name):
        try:
            return self._config[name]
        except KeyError:
            raise AttributeError(f"配置项 '{name}' 不存在")
