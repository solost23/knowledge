import docx2txt
from langchain_core.documents import Document


def load(file_path: str) -> list[Document]:
    text = docx2txt.process(file_path)
    return [Document(page_content=text, metadata={"source": file_path})]
