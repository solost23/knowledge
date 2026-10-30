# from langchain_core.documents import Document


# def load(file_path: str) -> list[Document]:
#     with open(file_path, encoding="utf-8") as file:
#         return [Document(page_content=file.read(), metadata={"source": file_path})]

from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document


def load(file_path: str) -> list[Document]:
    loader = UnstructuredMarkdownLoader(
        file_path=file_path,
        mode="elements",
    )
    return loader.load()
