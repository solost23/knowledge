"""Load plain-text and structured files into LangChain documents.

These formats intentionally use only the Python standard library. Keeping the
loader dependency-free makes them available in the API and in local imports
without adding another document-processing runtime.
"""

# import csv
# import io
# import json
# import re
# from html.parser import HTMLParser
# from pathlib import Path
# from xml.etree import ElementTree

from langchain_community.document_loaders import UnstructuredCSVLoader
from langchain_community.document_loaders import UnstructuredTSVLoader
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_community.document_loaders import UnstructuredXMLLoader

from langchain_community.document_loaders import JSONLoader
from langchain_community.document_loaders import TextLoader

from langchain_core.documents import Document


TEXT_EXTENSIONS = frozenset(
    {
        ".txt",
        ".log",
        ".py",
        ".go",
        ".java",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".sql",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".cfg",
        ".conf",
        ".sh",
        ".bash",
        ".zsh",
        ".c",
        ".h",
        ".cc",
        ".cpp",
        ".hpp",
        ".rs",
        ".rb",
        ".php",
        ".css",
        ".scss",
        ".vue",
        ".svelte",
    }
)

STRUCTURED_EXTENSIONS = frozenset(
    {".csv", ".tsv", ".json", ".html", ".htm", ".xml"}
)

# The order is used by API error messages and kept stable for the UI contract.
SUPPORTED_EXTENSION_LIST = (
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".md",
    *sorted(TEXT_EXTENSIONS),
    *sorted(STRUCTURED_EXTENSIONS),
)
LOADER_EXTENSIONS = TEXT_EXTENSIONS | STRUCTURED_EXTENSIONS
SUPPORTED_EXTENSIONS = frozenset(SUPPORTED_EXTENSION_LIST)
SUPPORTED_EXTENSIONS_TEXT = "、".join(ext[1:].upper() for ext in SUPPORTED_EXTENSION_LIST)


# def _read_text(file_path: str) -> str:
#     """Read common local encodings while keeping malformed text searchable."""

#     data = Path(file_path).read_bytes()
#     for encoding in ("utf-8-sig", "utf-8", "gb18030"):
#         try:
#             return data.decode(encoding)
#         except UnicodeDecodeError:
#             continue
#     return data.decode("utf-8", errors="replace")


# def _document(content: str, file_path: str) -> list[Document]:
#     return [Document(page_content=content, metadata={"source": file_path})]

def _load_plain_text(file_path: str) -> list[Document]:
    loader = TextLoader(
        file_path=file_path,
        encoding="utf-8",
        autodetect_encoding=True
    )
    return loader.load()


# def _load_delimited(file_path: str, delimiter: str) -> list[Document]:
#     content = _read_text(file_path)
#     rows = csv.reader(io.StringIO(content, newline=""), delimiter=delimiter)
#     lines = ["\t".join(cell.strip() for cell in row) for row in rows]
#     return _document("\n".join(lines), file_path)

def _load_csv(file_path: str) -> list[Document]:
    loader = UnstructuredCSVLoader(
        file_path=file_path,
        mode="elements",
    )
    return loader.load()


def _load_tsv(file_path: str) -> list[Document]:
    loader = UnstructuredTSVLoader(
        file_path=file_path,
        mode="elements",
    )
    return loader.load()

# def _load_json(file_path: str) -> list[Document]:
#     value = json.loads(_read_text(file_path))
#     content = json.dumps(value, ensure_ascii=False, indent=2)
#     return _document(content, file_path)


def _load_json(file_path: str) -> list[Document]:
    loader = JSONLoader(
        file_path=file_path,
        jq_schema="tojson",
        text_content=True,
    )
    return loader.load()


# class _HTMLTextParser(HTMLParser):
#     _BLOCK_TAGS = {
#         "address",
#         "article",
#         "aside",
#         "blockquote",
#         "br",
#         "div",
#         "dl",
#         "dt",
#         "dd",
#         "footer",
#         "h1",
#         "h2",
#         "h3",
#         "h4",
#         "h5",
#         "h6",
#         "header",
#         "li",
#         "main",
#         "nav",
#         "ol",
#         "p",
#         "pre",
#         "section",
#         "table",
#         "tr",
#         "td",
#         "th",
#         "ul",
#     }

#     def __init__(self) -> None:
#         super().__init__(convert_charrefs=True)
#         self._parts: list[str] = []
#         self._buffer: list[str] = []
#         self._ignored_depth = 0

#     def _flush(self) -> None:
#         text = re.sub(r"\s+", " ", "".join(self._buffer)).strip()
#         if text:
#             self._parts.append(text)
#         self._buffer.clear()

#     def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
#         tag = tag.lower()
#         if tag in {"script", "style"}:
#             self._flush()
#             self._ignored_depth += 1
#             return
#         if self._ignored_depth == 0 and tag in self._BLOCK_TAGS:
#             self._flush()

#     def handle_endtag(self, tag: str) -> None:
#         tag = tag.lower()
#         if tag in {"script", "style"}:
#             if self._ignored_depth:
#                 self._ignored_depth -= 1
#             return
#         if self._ignored_depth == 0 and tag in self._BLOCK_TAGS:
#             self._flush()

#     def handle_data(self, data: str) -> None:
#         if self._ignored_depth == 0:
#             self._buffer.append(data)

#     def text(self) -> str:
#         self._flush()
#         return "\n".join(self._parts)


# def _load_html(file_path: str) -> list[Document]:
#     parser = _HTMLTextParser()
#     parser.feed(_read_text(file_path))
#     parser.close()
#     return _document(parser.text(), file_path)

def _load_html(file_path: str) -> list[Document]:
    loader = UnstructuredHTMLLoader(
        file_path=file_path,
        mode="single",
        strategy="fast",
    )
    return loader.load()


# def _xml_label(tag: str) -> str:
#     return tag.rsplit("}", 1)[-1]


# def _xml_lines(element: ElementTree.Element, depth: int = 0) -> list[str]:
#     indent = "  " * depth
#     label = _xml_label(element.tag)
#     attributes = " ".join(f"{key}={value}" for key, value in element.attrib.items())
#     own_text = " ".join((element.text or "").split())
#     detail = f": {own_text}" if own_text else ""
#     if attributes:
#         detail = f" [{attributes}]" + detail
#     lines = [f"{indent}{label}{detail}"]
#     for child in element:
#         lines.extend(_xml_lines(child, depth + 1))
#     return lines


# def _load_xml(file_path: str) -> list[Document]:
#     root = ElementTree.fromstring(_read_text(file_path))
#     return _document("\n".join(_xml_lines(root)), file_path)

def _load_xml(file_path: str) -> list[Document]:
    loader = UnstructuredXMLLoader(
        file_path=file_path,
        mode="elements",
        strategy="fast",
        # Avoid Unstructured's NLP sentence segmentation and implicit model download.
        chunking_strategy=None,
    )
    return loader.load()


def load(file_path: str, extension: str) -> list[Document]:
    """Load a supported plain-text or structured extension."""

    extension = extension.lower()
    if extension in TEXT_EXTENSIONS:
        # return _document(_read_text(file_path), file_path)
        return _load_plain_text(file_path=file_path)
    if extension == ".csv":
        # return _load_delimited(file_path, ",")
        return _load_csv(file_path=file_path)
    if extension == ".tsv":
        #return _load_delimited(file_path, "\t")
        return _load_tsv(file_path=file_path)
    if extension == ".json":
        return _load_json(file_path=file_path)
    if extension in {".html", ".htm"}:
        return _load_html(file_path=file_path)
    if extension == ".xml":
        return _load_xml(file_path=file_path)
    raise ValueError(f"Unsupported structured extension: {extension}")
