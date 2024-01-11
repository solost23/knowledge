from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain_core.documents.base import Document


def load(file_path: str) -> list[Document]:
    loader = UnstructuredPowerPointLoader(file_path)
    return loader.load()
