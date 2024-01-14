from initialize.config import YamlConfig
from initialize.milvus import Milvus
from universal.config import config, initialize_config_var
from universal.milvus import milvus, initialize_milvus_var


def initialize():
    if config is None:
        yaml_config = YamlConfig()
        initialize_config_var(yaml_config)
    if milvus is None:
        initialize_milvus_var(Milvus(yaml_config))


initialize()
