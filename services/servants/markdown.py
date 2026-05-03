from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents.base import Document


def load(file_path: str) -> list[Document]:
    loader = UnstructuredMarkdownLoader(file_path)
    return loader.load()
