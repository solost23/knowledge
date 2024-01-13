import os.path
import uuid

from loguru import logger
from werkzeug.datastructures.file_storage import FileStorage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma

from services.servants import (
    doc,
    excel,
    pdf,
    ppt,
    embedding
)
from universal.config import config
from initialize import response


class DocService:
    def __init__(self):
        pass

    def upload(self, file: FileStorage) -> str:
        # 后缀 eg: .xlsx
        file_name = os.path.basename(file.filename) + str(uuid.uuid1())
        ext = os.path.splitext(file.filename)[-1]
        file_path = f'/tmp/{file_name}{ext}'

        logger.info(f'filepath: {file_path}')

        if not os.path.isfile(file_path):
            try:
                file.save(file_path)
            finally:
                file.close()
        return self.doc(file_path, ext)

    def doc(self, file_path: str, ext: str) -> str:
        # 数据载入
        if ext == ".pdf":
            docs = pdf.load(file_path)
        elif ext == ".docx":
            docs = doc.load(file_path)
        elif ext == ".pptx":
            docs = ppt.load(file_path)
        elif ext == ".xlsx":
            docs = excel.load(file_path)
        else:
            return response.error(1500, f'暂不支持{ext[1:]}类型文件')

        # 文档切割
        splitter = RecursiveCharacterTextSplitter()
        texts = splitter.split_documents(docs)

        # 数据插入向量数据库 TODO: 后面改成milvus
        chroma = Chroma.from_documents(texts, embedding.embedding(), persist_directory=config.chroma['file_path'])
        chroma.persist()

        return response.success("成功", None)
