from langchain_community.document_loaders import Docx2txtLoader
from langchain_core.documents.base import Document


def load(file_path: str) -> list[Document]:
    loader = Docx2txtLoader(file_path)
    return loader.load()
