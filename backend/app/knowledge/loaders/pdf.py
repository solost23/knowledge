from langchain_core.documents import Document
from pypdf import PdfReader


def load(file_path: str) -> list[Document]:
    docs = []
    for number, page in enumerate(PdfReader(file_path).pages, 1):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(Document(page_content=text, metadata={"source": file_path, "page": number}))
    return docs
