from langchain_core.output_parsers import StrOutputParser
from loguru import logger

from app.core import response
from app.infrastructure.llm import LLM, LLMConfigurationError
from app.infrastructure.vector_store import chroma
from app.prompts.rag import RAG_PROMPT


class QuestionService:
    def question(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
        llm=None,
    ) -> str:
        try:
            docs = [doc for doc, _score in chroma.search(question)]
            if not docs:
                logger.info("知识库没有找到相关资料，跳过问答模型调用")
                return response.success(
                    "未找到相关资料",
                    {
                        "answer": "知识库中没有找到与问题直接相关的资料。",
                        "sources": [],
                    },
                )

            context = "\n\n".join(
                f"[{index}] 来源：{doc.metadata.get('source', '')}\n{doc.page_content}"
                for index, doc in enumerate(docs, 1)
            )
            answer = (RAG_PROMPT | (llm or LLM()) | StrOutputParser()).invoke(
                {
                    "context": context,
                    "question": question,
                    "history": [
                        ("human" if item["role"] == "user" else "ai", item["content"])
                        for item in history or []
                    ],
                }
            )

            sources = [
                {
                    "source": doc.metadata.get("source", ""),
                    "content": doc.page_content,
                }
                for doc in docs
            ]

            logger.info("RAG 问答完成")
            return response.success("成功", {"answer": answer, "sources": sources})

        except LLMConfigurationError as e:
            logger.error(f"LLM 配置错误: {e}")
            return response.error(500, "问答模型配置错误")
        except Exception as e:
            logger.exception(f"问答失败: {e}")
            return response.error(502, "问答处理失败，请稍后重试")
