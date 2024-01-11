from initialize.config import YamlConfig
from universal.config import initialize_config_var, config


def initialize():
    if config is None:
        initialize_config_var(YamlConfig())


initialize()
