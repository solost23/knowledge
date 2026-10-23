from collections.abc import Callable

from langchain_core.documents import Document
from pypdf import PdfReader


def load(
    file_path: str,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[Document]:
    docs = []
    pages = PdfReader(file_path).pages
    total = len(pages)
    interval = max(1, total // 50)
    for number, page in enumerate(pages, 1):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(Document(page_content=text, metadata={"source": file_path, "page": number}))
        if on_progress and (number % interval == 0 or number == total):
            on_progress(number, total)
    return docs
