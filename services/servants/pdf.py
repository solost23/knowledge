from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents.base import Document


def load(file_path: str) -> list[Document]:
    loader = PyPDFLoader(file_path)
    return loader.load()
