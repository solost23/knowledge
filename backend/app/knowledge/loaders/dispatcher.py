"""Select a document loader by file extension."""

from pathlib import Path

from langchain_core.documents import Document

from app.knowledge.loaders.doc import load as load_doc
from app.knowledge.loaders.excel import load as load_excel
from app.knowledge.loaders.markdown import load as load_markdown
from app.knowledge.loaders.pdf import load as load_pdf
from app.knowledge.loaders.ppt import load as load_ppt
from app.knowledge.loaders.structured import (
    TEXT_EXTENSIONS,
    load_csv,
    load_html,
    load_json,
    load_plain_text,
    load_tsv,
    load_xml,
)

_LOADERS = {
    ".pdf": load_pdf,
    ".docx": load_doc,
    ".pptx": load_ppt,
    ".xlsx": load_excel,
    ".md": load_markdown,
    **dict.fromkeys(sorted(TEXT_EXTENSIONS), load_plain_text),
    ".csv": load_csv,
    ".htm": load_html,
    ".html": load_html,
    ".json": load_json,
    ".tsv": load_tsv,
    ".xml": load_xml,
}

SUPPORTED_EXTENSIONS = frozenset(_LOADERS)
SUPPORTED_EXTENSIONS_TEXT = "、".join(ext[1:].upper() for ext in _LOADERS)


class UnsupportedFormatError(ValueError):
    """The file extension has no registered loader."""


def load_file(file_path: str | Path) -> list[Document]:
    """Load a document using the extension in its file path."""
    ext = Path(file_path).suffix.lower()
    loader = _LOADERS.get(ext)
    if loader is None:
        raise UnsupportedFormatError(f"暂不支持{ext[1:]}类型文件")
    return loader(str(file_path))
