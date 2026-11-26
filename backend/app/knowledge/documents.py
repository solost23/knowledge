import os
import os.path
import uuid

from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger
from werkzeug.datastructures.file_storage import FileStorage

from app.core import response
from app.core.settings import config
from app.infrastructure import get_chroma
from app.infrastructure.document_repository import (
    begin_document,
    delete_doc,
    document_lock,
    list_docs,
    publish_document,
    visible_documents,
)
from app.infrastructure.document_upload_repository import update_upload_progress
from app.knowledge.loaders import load_file
from app.knowledge.loaders.dispatcher import UnsupportedFormatError


class DocumentService:
    def upload(self, file: FileStorage) -> str:
        ext = os.path.splitext(file.filename)[-1].lower()
        original_name = os.path.basename(file.filename)
        file_path = f"/tmp/{original_name}_{uuid.uuid4()}{ext}"

        logger.info(f"filepath: {file_path}")

        try:
            file.save(file_path)
            return self.doc(file_path, original_name)
        except Exception as e:
            logger.error(f"上传处理失败: {e}")
            return response.error(500, f"文件处理失败: {str(e)}")
        finally:
            file.close()
            if os.path.isfile(file_path):
                os.remove(file_path)

    def doc(
        self,
        file_path: str,
        original_name: str,
        upload_job_id: str | None = None,
        publication: tuple[str, int] | None = None,
    ) -> str:
        version, generation = publication or begin_document(original_name)
        update_upload_progress(upload_job_id, 5, "正在提取文档文本")
        try:
            docs = load_file(file_path)
        except UnsupportedFormatError as error:
            return response.error(400, str(error))

        update_upload_progress(upload_job_id, 15, "文档已提取，正在清洗数据")
        docs = filter_complex_metadata(docs)

        update_upload_progress(upload_job_id, 32, "文档已解析，正在切分文本")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.documents["chunk_size"],
            chunk_overlap=config.documents["chunk_overlap"],
        )
        texts = splitter.split_documents(docs)
        if not texts:
            return response.error(400, "文档中没有提取到可索引的文本")

        for text in texts:
            text.metadata["source"] = original_name
            text.metadata["document_version"] = version
            text.metadata["publication_managed"] = True
            if upload_job_id:
                text.metadata["upload_job_id"] = upload_job_id

        update_upload_progress(
            upload_job_id, 38, f"文本切分完成，共 {len(texts)} 个片段"
        )
        chroma_config = config.chroma
        store = get_chroma()
        with document_lock:
            previous_version = visible_documents().get(original_name)
            previous = store.db.get(where={"source": original_name})
            old_ids = [
                chunk_id
                for chunk_id, metadata in zip(
                    previous.get("ids", []), previous.get("metadatas", []), strict=True
                )
                if (metadata or {}).get("document_version") == previous_version
            ]
        added_ids: list[str] = []
        batch_size = max(1, int(chroma_config.get("import_batch_size", 256)))

        try:
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                batch_ids = [str(uuid.uuid4()) for _ in batch]
                added_ids.extend(batch_ids)
                store.db.add_texts(
                    texts=[item.page_content for item in batch],
                    metadatas=[item.metadata for item in batch],
                    ids=batch_ids,
                )
                completed = start + len(batch)
                update_upload_progress(
                    upload_job_id,
                    38 + int(57 * completed / len(texts)),
                    f"正在生成并写入向量（{completed}/{len(texts)} 个片段）",
                )

            update_upload_progress(upload_job_id, 98, "正在保存文档索引记录")
            publish_document(original_name, version, generation)
        except Exception:
            if added_ids:
                try:
                    store.db.delete(ids=added_ids)
                except Exception:
                    logger.exception("向量写入失败后的回滚清理也失败")
            raise

        # 发布后的旧向量清理失败不回滚已提交的文档，新版本仍然可检索。
        if old_ids:
            try:
                store.db.delete(ids=old_ids)
            except Exception:
                logger.exception("已发布文档的旧向量清理失败")

        return response.success("成功", None)

    def discard_upload_job(self, upload_job_id: str, original_name: str) -> None:
        store = get_chroma()
        with document_lock:
            visible = visible_documents()
            chunks = store.db.get(where={"upload_job_id": upload_job_id})
            ids = [
                chunk_id
                for chunk_id, metadata in zip(
                    chunks.get("ids", []), chunks.get("metadatas", []), strict=True
                )
                if original_name not in visible
                or (metadata or {}).get("document_version") != visible[original_name]
            ]
            if ids:
                store.db.delete(ids=ids)

    def list(self) -> str:
        return response.success("成功", list_docs())

    def delete(self, name: str) -> str:
        with document_lock:
            # 先撤销可见性和待发布令牌，物理清理失败也不会复活文档。
            deleted = delete_doc(name)
            try:
                store = get_chroma()
                results = store.db.get(where={"source": name})
                ids = results.get("ids", [])
                if ids:
                    store.db.delete(ids=ids)
            except Exception:
                logger.exception("已删除文档的向量清理失败")

        if not deleted:
            return response.error(404, f"笔记 {name} 不存在")

        return response.success("成功", None)
