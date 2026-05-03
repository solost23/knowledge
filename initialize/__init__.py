

def initialize():
    from universal.config import config, initialize_config_var
    if config is None:
        from initialize.config import YamlConfig
        initialize_config_var(YamlConfig())

    from universal.chroma import chroma, initialize_chroma_var
    if chroma is None:
        from initialize.chroma import ChromaDB
        initialize_chroma_var(ChromaDB())


initialize()
