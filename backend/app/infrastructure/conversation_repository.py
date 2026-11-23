import json
from uuid import uuid4

from sqlalchemy import case, delete, func, select, update

from app.core.settings import config
from app.infrastructure.database import get_database
from app.infrastructure.models import Conversation, ConversationMessage

_CONVERSATION_CONFIG = config.conversation
MAX_HISTORY_MESSAGES = _CONVERSATION_CONFIG["max_history_messages"]
MAX_HISTORY_CHARS = _CONVERSATION_CONFIG["max_history_chars"]
MAX_HISTORY_MESSAGE_CHARS = _CONVERSATION_CONFIG["max_history_message_chars"]
KEEP_RECENT_MESSAGES = _CONVERSATION_CONFIG["keep_recent_messages"]
COMPRESS_AFTER_MESSAGES = _CONVERSATION_CONFIG["compress_after_messages"]
COMPRESS_AFTER_CHARS = _CONVERSATION_CONFIG["compress_after_chars"]
MAX_COMPRESSION_BATCH_CHARS = _CONVERSATION_CONFIG["max_compression_batch_chars"]
MAX_COMPRESSION_MESSAGE_CHARS = _CONVERSATION_CONFIG["max_compression_message_chars"]


def _conversation(conversation: Conversation) -> dict:
    return {
        "id": conversation.id,
        "title": conversation.title,
        "created_at": conversation.created_at,
        "updated_at": conversation.updated_at,
    }


def create_conversation() -> dict:
    with get_database("conversations").session() as session:
        conversation = Conversation(id=str(uuid4()), title="新对话")
        session.add(conversation)
        session.flush()
        return _conversation(conversation)


def list_conversations(limit: int = 100) -> list[dict]:
    with get_database("conversations").session() as session:
        conversations = session.scalars(
            select(Conversation)
            .order_by(Conversation.updated_at.desc(), Conversation.id.desc())
            .limit(limit)
        )
        return [_conversation(conversation) for conversation in conversations]


def get_conversation(conversation_id: str) -> dict | None:
    with get_database("conversations").session() as session:
        conversation = session.get(Conversation, conversation_id)
        if conversation is None:
            return None
        messages = session.scalars(
            select(ConversationMessage)
            .where(ConversationMessage.conversation_id == conversation_id)
            .order_by(ConversationMessage.id)
        )
        result = _conversation(conversation)
        result["messages"] = [
            {
                "id": message.id,
                "role": message.role,
                "content": message.content,
                "result": json.loads(message.result_json)
                if message.result_json
                else None,
                "created_at": message.created_at,
            }
            for message in messages
        ]
        return result


def recent_history(conversation_id: str) -> list[dict[str, str]]:
    with get_database("conversations").session() as session:
        conversation = session.get(Conversation, conversation_id)
        if conversation is None:
            return []
        rows = (
            session.execute(
                select(ConversationMessage.role, ConversationMessage.content)
                .where(
                    ConversationMessage.conversation_id == conversation_id,
                    ConversationMessage.id > conversation.summarized_through,
                )
                .order_by(ConversationMessage.id.desc())
                .limit(MAX_HISTORY_MESSAGES)
            )
            .mappings()
            .all()
        )
        summary = conversation.summary

    summary_message = None
    if summary:
        summary_message = {
            "role": "assistant",
            "content": f"此前对话摘要（仅用于理解当前问题）：\n{summary}",
        }
    remaining = MAX_HISTORY_CHARS - (
        len(summary_message["content"]) if summary_message else 0
    )
    recent = []
    for row in rows:
        content = row["content"]
        if remaining <= 0:
            break
        content = content[: min(remaining, MAX_HISTORY_MESSAGE_CHARS)]
        recent.append({"role": row["role"], "content": content})
        remaining -= len(content)
    return ([summary_message] if summary_message else []) + list(reversed(recent))


def compression_batch(conversation_id: str) -> dict | None:
    with get_database("conversations").session() as session:
        conversation = session.get(Conversation, conversation_id)
        if conversation is None:
            return None
        rows = (
            session.execute(
                select(
                    ConversationMessage.id,
                    ConversationMessage.role,
                    ConversationMessage.content,
                )
                .where(
                    ConversationMessage.conversation_id == conversation_id,
                    ConversationMessage.id > conversation.summarized_through,
                )
                .order_by(ConversationMessage.id)
            )
            .mappings()
            .all()
        )
        summary = conversation.summary

    old_messages = rows[:-KEEP_RECENT_MESSAGES]
    old_chars = sum(len(row["content"]) for row in old_messages)
    if len(old_messages) < COMPRESS_AFTER_MESSAGES and old_chars < COMPRESS_AFTER_CHARS:
        return None
    batch = []
    batch_chars = 0
    for row in old_messages:
        content = row["content"][:MAX_COMPRESSION_MESSAGE_CHARS]
        if batch and batch_chars + len(content) > MAX_COMPRESSION_BATCH_CHARS:
            break
        batch.append({"id": row["id"], "role": row["role"], "content": content})
        batch_chars += len(content)
    return {
        "summary": summary,
        "messages": batch,
        "through_id": batch[-1]["id"],
    }


def save_summary(conversation_id: str, summary: str, through_id: int) -> None:
    with get_database("conversations").session() as session:
        session.execute(
            update(Conversation)
            .where(
                Conversation.id == conversation_id,
                Conversation.summarized_through < through_id,
            )
            .values(summary=summary, summarized_through=through_id)
        )


def append_exchange(
    conversation_id: str,
    question: str,
    answer: str,
    result: dict,
) -> None:
    title = question.strip().replace("\n", " ")[:60] or "新对话"
    with get_database("conversations").session() as session:
        if session.get(Conversation, conversation_id) is None:
            raise KeyError(conversation_id)
        session.add_all(
            [
                ConversationMessage(
                    conversation_id=conversation_id, role="user", content=question
                ),
                ConversationMessage(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=answer,
                    result_json=json.dumps(result, ensure_ascii=False),
                ),
            ]
        )
        session.execute(
            update(Conversation)
            .where(Conversation.id == conversation_id)
            .values(
                title=case(
                    (Conversation.title == "新对话", title), else_=Conversation.title
                ),
                updated_at=func.datetime("now"),
            )
        )


def delete_conversation(conversation_id: str) -> bool:
    with get_database("conversations").session() as session:
        result = session.execute(
            delete(Conversation).where(Conversation.id == conversation_id)
        )
        return result.rowcount > 0
