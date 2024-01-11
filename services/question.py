from loguru import logger
from langchain_community.llms.chatglm import ChatGLM
from langchain_community.vectorstores.chroma import Chroma
from langchain.chains.question_answering import load_qa_chain

from universal.config import config
from services.servants import embedding
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

        # ChatGlm模型 TODO: 模型过慢
        llm = ChatGLM(
            endpoint_url=config.llm.get('endpoint_url'),
            max_token=config.llm.get('max_token')
        )

        # 搜索
        answer = load_qa_chain(llm, verbose=True).run(input_documents=match_docs, question=question)

        logger.info(f'answer: {answer}')

        return response.success("成功", answer)
