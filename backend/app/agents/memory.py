from loguru import logger

from app.agents.tools import message_text
from app.core.settings import config
from app.infrastructure.conversation_repository import (
    compression_batch,
    recent_history,
    save_summary,
)
from app.prompts.memory import SUMMARY_PROMPT


def prepare_history(conversation_id: str, llm) -> list[dict[str, str]]:
    batch = compression_batch(conversation_id)
    if batch:
        transcript = "\n\n".join(
            f"{'用户' if message['role'] == 'user' else '助手'}：{message['content']}"
            for message in batch["messages"]
        )
        try:
            max_chars = config.conversation["max_summary_chars"]
            result = llm.invoke(
                SUMMARY_PROMPT.format_messages(
                    max_chars=max_chars,
                    summary=batch["summary"] or "无",
                    transcript=transcript,
                )
            )
            summary = message_text(result.content).strip()
            if summary:
                save_summary(conversation_id, summary[:max_chars], batch["through_id"])
        except Exception:
            logger.exception("压缩对话上下文失败，继续使用近期消息")
    return recent_history(conversation_id)
