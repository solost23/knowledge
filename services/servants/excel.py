from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_core.documents.base import Document


def load(file_path: str) -> list[Document]:
    loader = UnstructuredExcelLoader(file_path)
    return loader.load()