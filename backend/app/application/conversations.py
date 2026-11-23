"""Conversation use cases kept separate from HTTP and persistence details."""

from uuid import UUID

from app.infrastructure.conversation_repository import (
    create_conversation,
    delete_conversation,
    get_conversation,
    list_conversations,
)


class ConversationService:
    """Coordinate conversation lifecycle operations."""

    @staticmethod
    def create() -> dict:
        return create_conversation()

    @staticmethod
    def list() -> list[dict]:
        return list_conversations()

    @staticmethod
    def get(conversation_id: UUID | str) -> dict | None:
        return get_conversation(str(conversation_id))

    @staticmethod
    def delete(conversation_id: UUID | str) -> bool:
        return delete_conversation(str(conversation_id))

    @staticmethod
    def resolve(conversation_id: UUID | str | None) -> tuple[str, dict] | None:
        """Return an existing conversation or create one for a new request."""
        if conversation_id is None:
            conversation = create_conversation()
            return conversation["id"], conversation

        normalized_id = str(conversation_id)
        conversation = get_conversation(normalized_id)
        return (normalized_id, conversation) if conversation is not None else None
