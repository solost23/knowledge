from sqlalchemy import delete, select
from sqlalchemy.dialects.sqlite import insert

from app.infrastructure.database import get_database
from app.infrastructure.models import DocumentRecord


def insert_doc(name: str) -> None:
    with get_database().session() as session:
        session.execute(
            insert(DocumentRecord)
            .values(name=name)
            .on_conflict_do_nothing(index_elements=["name"])
        )


def exists_doc(name: str) -> bool:
    with get_database().session() as session:
        return (
            session.scalar(select(DocumentRecord.id).where(DocumentRecord.name == name))
            is not None
        )


def list_docs() -> list[dict]:
    with get_database().session() as session:
        docs = session.scalars(
            select(DocumentRecord).order_by(DocumentRecord.created_at.desc())
        )
        return [doc.as_dict() for doc in docs]


def delete_doc(name: str) -> bool:
    with get_database().session() as session:
        result = session.execute(
            delete(DocumentRecord).where(DocumentRecord.name == name)
        )
        return result.rowcount > 0
