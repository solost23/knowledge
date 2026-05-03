from loguru import logger
from langchain.chains.question_answering import load_qa_chain

from universal.config import config
from services.servants.llm import ChatGLmName, chat_glm, ChatGptName, chat_gpt, WenXinName, wen_xin
from initialize import response
from caches.gpt import init_cache
from universal.chroma import chroma


class QuestionService:
    def __init__(self):
        pass

    def question(self, question: str) -> str:
        try:
            init_cache()

            match_docs = chroma.db.similarity_search(question)

            llm_name = config.llm.get('name')
            if llm_name == ChatGLmName:
                llm = chat_glm()
            elif llm_name == ChatGptName:
                llm = chat_gpt()
            elif llm_name == WenXinName:
                llm = wen_xin()
            else:
                return response.error(500, f'未知的 LLM 配置: {llm_name}')

            llm.cache = config.cache.get('use')

            answer = load_qa_chain(llm, verbose=False).run(
                input_documents=match_docs,
                question=question,
            )

            sources = [
                {
                    "source": doc.metadata.get("source", ""),
                    "content": doc.page_content,
                }
                for doc in match_docs
            ]

            logger.info(f'answer: {answer}')
            return response.success("成功", {"answer": answer, "sources": sources})

        except EnvironmentError as e:
            logger.error(f'LLM 配置错误: {e}')
            return response.error(500, str(e))
        except Exception as e:
            logger.error(f'问答失败: {e}')
            return response.error(500, f'问答处理失败: {str(e)}')
