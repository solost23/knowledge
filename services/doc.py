import os
import os.path
import uuid

from loguru import logger
from werkzeug.datastructures.file_storage import FileStorage
from langchain.text_splitter import RecursiveCharacterTextSplitter

from services.servants import (
    doc,
    excel,
    markdown,
    pdf,
    ppt,
)
from initialize import response
from universal.chroma import chroma
from universal.store import insert_doc, list_docs, delete_doc, exists_doc


class DocService:
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

    def doc(self, file_path: str, ext: str, original_name: str) -> str:
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
        else:
            return response.error(400, f'暂不支持{ext[1:]}类型文件')

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )
        texts = splitter.split_documents(docs)

        for text in texts:
            text.metadata["source"] = original_name

        if exists_doc(original_name):
            old_ids = chroma.db.get(where={"source": original_name}).get("ids", [])
            if old_ids:
                chroma.db.delete(ids=old_ids)
            delete_doc(original_name)

        chroma.db.add_documents(texts)
        insert_doc(original_name)

        return response.success("成功", None)

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
