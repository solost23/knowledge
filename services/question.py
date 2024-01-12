import time

from loguru import logger
from langchain_community.vectorstores.chroma import Chroma
from langchain.chains.question_answering import load_qa_chain

from universal.config import config
from services.servants import embedding
from services.servants.llm import ChatGLmName, chat_glm, ChatGptName, chat_gpt, WenXinName, wen_xin
from initialize import response


class QuestionService:
    def __init__(self):
        pass

    def question(self, question: str) -> str:
        # 问题向量化
        chroma = Chroma(
            embedding_function=embedding.embedding(),
            persist_directory=config.chroma.get('file_path')
        )

        # 匹配相似文档
        match_docs = chroma.similarity_search(question)

        llm_name = config.llm.get('name')
        if llm_name == ChatGLmName:
            # ChatGlm模型 TODO: 模型过慢
            llm = chat_glm()
        elif llm_name == ChatGptName:
            llm = chat_gpt()
        elif llm_name == WenXinName:
            llm = wen_xin()

        # 搜索
        answer = load_qa_chain(llm, verbose=True).\
            run(input_documents=match_docs, question=question)

        logger.info(f'answer: {answer}')

        return response.success("成功", answer)
