# from collections.abc import Callable

# from langchain_core.documents import Document
# from pptx import Presentation


# def load(
#     file_path: str,
#     on_progress: Callable[[int, int], None] | None = None,
# ) -> list[Document]:
#     parts = []
#     slides = Presentation(file_path).slides
#     total = len(slides)
#     interval = max(1, total // 50)
#     for number, slide in enumerate(slides, 1):
#         lines = []
#         for shape in slide.shapes:
#             if shape.has_text_frame and shape.text_frame.text.strip():
#                 lines.append(shape.text_frame.text.strip())
#             if getattr(shape, "has_table", False):
#                 for row in shape.table.rows:
#                     lines.append(" | ".join(cell.text.strip() for cell in row.cells))
#         if lines:
#             parts.append(f"# 幻灯片 {number}\n" + "\n".join(lines))
#         if on_progress and (number % interval == 0 or number == total):
#             on_progress(number, total)
#     if not parts:
#         return []
#     return [Document(page_content="\n\n".join(parts), metadata={"source": file_path})]

from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain_core.documents import Document


def load(file_path: str) -> list[Document]:
    loader = UnstructuredPowerPointLoader(
        file_path=file_path,
        mode="elements",
        strategy="fast",
    )
    return loader.load()
