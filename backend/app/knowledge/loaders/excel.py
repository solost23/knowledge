# from collections.abc import Callable

# from langchain_core.documents import Document
# from openpyxl import load_workbook


# def load(
#     file_path: str,
#     on_progress: Callable[[int, int], None] | None = None,
# ) -> list[Document]:
#     workbook = load_workbook(file_path, read_only=True, data_only=True)
#     parts = []
#     total_rows = sum(sheet.max_row or 0 for sheet in workbook.worksheets)
#     interval = max(1, total_rows // 50)
#     completed_rows = 0
#     try:
#         for sheet in workbook.worksheets:
#             rows = []
#             for row in sheet.iter_rows(values_only=True):
#                 completed_rows += 1
#                 cells = ["" if value is None else str(value) for value in row]
#                 if any(cells):
#                     rows.append("\t".join(cells))
#                 if on_progress and (
#                     completed_rows % interval == 0 or completed_rows == total_rows
#                 ):
#                     on_progress(completed_rows, total_rows)
#             if rows:
#                 parts.append(f"# {sheet.title}\n" + "\n".join(rows))
#     finally:
#         workbook.close()
#     return [Document(page_content="\n\n".join(parts), metadata={"source": file_path})]

from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_core.documents import Document


def load(
    file_path: str,
) -> list[Document]:
    loader = UnstructuredExcelLoader(
        file_path=file_path,
        mode="elements",
    )
    return loader.load()
