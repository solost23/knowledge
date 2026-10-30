import os
import os.path
import uuid
from collections.abc import Callable

from loguru import logger
from werkzeug.datastructures.file_storage import FileStorage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import filter_complex_metadata

from app.core import response
from app.core.settings import config
from app.infrastructure.document_repository import (
    delete_doc,
    exists_doc,
    insert_doc,
    list_docs,
)
from app.infrastructure.vector_store import chroma
from app.knowledge.loaders import doc, excel, markdown, pdf, ppt, structured

ProgressCallback = Callable[[int, str], None]


class DocumentService:
    def __init__(self):
        pass

    def upload(self, file: FileStorage) -> str:
        ext = os.path.splitext(file.filename)[-1].lower()
        original_name = os.path.basename(file.filename)
        file_path = f'/tmp/{original_name}_{uuid.uuid4()}{ext}'

        logger.info(f'filepath: {file_path}')

        try:
            file.save(file_path)
            return self.doc(file_path, ext, original_name)
        except Exception as e:
            logger.error(f'上传处理失败: {e}')
            return response.error(500, f'文件处理失败: {str(e)}')
        finally:
            file.close()
            if os.path.isfile(file_path):
                os.remove(file_path)

    def doc(
        self,
        file_path: str,
        ext: str,
        original_name: str,
        progress_callback: ProgressCallback | None = None,
        upload_job_id: str | None = None,
    ) -> str:
        last_progress = -1

        def report(progress: int, stage: str) -> None:
            nonlocal last_progress
            progress = max(0, min(99, progress))
            if progress_callback and progress > last_progress:
                last_progress = progress
                progress_callback(progress, stage)

        report(5, "正在提取文档文本")
        if ext == ".pdf":
            docs = pdf.load(file_path)
        elif ext == ".docx":
            docs = doc.load(file_path)
        elif ext == ".pptx":
            docs = ppt.load(file_path)
        elif ext == ".xlsx":
            docs = excel.load(file_path)
        elif ext == ".md":
            docs = markdown.load(file_path)
        elif ext in structured.LOADER_EXTENSIONS:
            docs = structured.load(file_path, ext)
        else:
            return response.error(400, f'暂不支持{ext[1:]}类型文件')

        report(15, "文档已提取，正在清洗数据")
        docs = filter_complex_metadata(docs)

        report(32, "文档已解析，正在切分文本")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )
        texts = splitter.split_documents(docs)
        if not texts:
            return response.error(400, "文档中没有提取到可索引的文本")

        for text in texts:
            text.metadata["source"] = original_name
            if upload_job_id:
                text.metadata["upload_job_id"] = upload_job_id

        report(38, f"文本切分完成，共 {len(texts)} 个片段")
        chroma_config = config.chroma
        existing = exists_doc(original_name)
        old_ids = (
            chroma.db.get(where={"source": original_name}).get("ids", [])
            if existing else []
        )
        added_ids: list[str] = []
        batch_size = max(1, int(chroma_config.get("import_batch_size", 256)))

        try:
            for start in range(0, len(texts), batch_size):
                batch = texts[start:start + batch_size]
                batch_ids = [str(uuid.uuid4()) for _ in batch]
                added_ids.extend(batch_ids)
                chroma.db.add_texts(
                    texts=[item.page_content for item in batch],
                    metadatas=[item.metadata for item in batch],
                    ids=batch_ids,
                )
                completed = start + len(batch)
                report(
                    38 + int(57 * completed / len(texts)),
                    f"正在生成并写入向量（{completed}/{len(texts)} 个片段）",
                )

            if old_ids:
                chroma.db.delete(ids=old_ids)
            if existing:
                delete_doc(original_name)
            insert_doc(original_name)
        except Exception:
            if added_ids:
                try:
                    chroma.db.delete(ids=added_ids)
                except Exception:
                    logger.exception("向量写入失败后的回滚清理也失败")
            raise

        report(98, "正在保存文档索引记录")

        return response.success("成功", None)

    def discard_upload_job(self, upload_job_id: str, original_name: str) -> None:
        ids = chroma.db.get(where={"upload_job_id": upload_job_id}).get("ids", [])
        if ids:
            chroma.db.delete(ids=ids)
        remaining = chroma.db.get(where={"source": original_name}).get("ids", [])
        if not remaining:
            delete_doc(original_name)

    def list(self) -> str:
        return response.success("成功", list_docs())

    def delete(self, name: str) -> str:
        results = chroma.db.get(where={"source": name})
        ids = results.get("ids", [])
        if ids:
            chroma.db.delete(ids=ids)

        if not delete_doc(name):
            return response.error(404, f'笔记 {name} 不存在')

        return response.success("成功", None)
