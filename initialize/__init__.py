

def initialize():
    from universal.config import config, initialize_config_var
    if config is None:
        from initialize.config import YamlConfig
        initialize_config_var(YamlConfig())

    from universal.milvus import milvus, initialize_milvus_var
    if milvus is None:
        from initialize.milvus import Milvus
        initialize_milvus_var(Milvus())


initialize()
