"""SQLite business tables; LangGraph owns its checkpoint tables."""

from sqlalchemy import CheckConstraint, ForeignKey, Index, Text, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    def as_dict(self) -> dict:
        return {
            column.name: getattr(self, column.name) for column in self.__table__.columns
        }


class DocumentRecord(Base):
    __tablename__ = "docs"
    __table_args__ = {"sqlite_autoincrement": True}

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(Text, unique=True)
    created_at: Mapped[str | None] = mapped_column(
        Text, server_default=text("(datetime('now', 'localtime'))")
    )


class DocumentUploadJob(Base):
    __tablename__ = "document_upload_jobs"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    filename: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text)
    progress: Mapped[int]
    stage: Mapped[str] = mapped_column(Text)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[str | None] = mapped_column(
        Text, server_default=text("(datetime('now', 'localtime'))")
    )
    updated_at: Mapped[str | None] = mapped_column(
        Text, server_default=text("(datetime('now', 'localtime'))")
    )


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    title: Mapped[str] = mapped_column(Text)
    summary: Mapped[str] = mapped_column(Text, server_default="")
    summarized_through: Mapped[int] = mapped_column(server_default="0")
    created_at: Mapped[str] = mapped_column(
        Text, server_default=text("(datetime('now'))")
    )
    updated_at: Mapped[str] = mapped_column(
        Text, server_default=text("(datetime('now'))")
    )

    __table_args__ = (Index("idx_conversations_updated", updated_at.desc()),)


class ConversationMessage(Base):
    __tablename__ = "conversation_messages"
    __table_args__ = (
        CheckConstraint("role IN ('user', 'assistant')"),
        Index("idx_conversation_messages_conversation", "conversation_id", "id"),
        {"sqlite_autoincrement": True},
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        Text, ForeignKey("conversations.id", ondelete="CASCADE")
    )
    role: Mapped[str] = mapped_column(Text)
    content: Mapped[str] = mapped_column(Text)
    result_json: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[str] = mapped_column(
        Text, server_default=text("(datetime('now'))")
    )


class AgentRun(Base):
    __tablename__ = "agent_runs"

    execution_id: Mapped[str] = mapped_column(Text, primary_key=True)
    task: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text)
    error_code: Mapped[str | None] = mapped_column(Text)
    progress_json: Mapped[str] = mapped_column(Text, server_default="{}")
    mode: Mapped[str] = mapped_column(Text, server_default="agent")
    route_reason: Mapped[str] = mapped_column(Text, server_default="")
    conversation_id: Mapped[str | None] = mapped_column(Text)
    cancel_requested: Mapped[int] = mapped_column(server_default="0")
    result_json: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[str | None] = mapped_column(Text)
    finished_at: Mapped[str | None] = mapped_column(Text)
    duration_ms: Mapped[int | None]
    created_at: Mapped[str] = mapped_column(
        Text, server_default=text("(datetime('now'))")
    )
    updated_at: Mapped[str] = mapped_column(
        Text, server_default=text("(datetime('now'))")
    )
