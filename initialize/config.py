import yaml


class YamlConfig:
    """
    读取 config.yaml 文件
    """
    def __init__(self, config_path: str = "./conf/config.yaml"):
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(
                file,
            )

    def __getattr__(self, name):
        return self.config[name]
