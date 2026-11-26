import threading
import uuid

from sqlalchemy import Column, Integer, MetaData, String, Table, delete, select, update
from sqlalchemy.dialects.sqlite import insert

from app.infrastructure.database import get_database
from app.infrastructure.models import DocumentRecord

# 单进程内序列化发布、删除和检索，向量生成不持有此锁。
document_lock = threading.RLock()
_publications = Table(
    "document_publications",
    MetaData(),
    Column("name", String, primary_key=True),
    Column("active_version", String, nullable=True),
    Column("pending_version", String, nullable=True),
    Column("generation", Integer, nullable=False, default=0),
)


def _publication_database():
    database = get_database()
    with document_lock:
        _publications.create(database.engine, checkfirst=True)
    return database


def begin_document(name: str) -> tuple[str, int]:
    version = str(uuid.uuid4())
    with document_lock, _publication_database().session() as session:
        session.execute(
            insert(_publications)
            .values(name=name, generation=0)
            .on_conflict_do_nothing(index_elements=["name"])
        )
        generation = session.scalar(
            select(_publications.c.generation).where(_publications.c.name == name)
        )
        session.execute(
            update(_publications)
            .where(_publications.c.name == name)
            .values(pending_version=version)
        )
    return version, generation


def publish_document(name: str, version: str, generation: int) -> None:
    with document_lock, _publication_database().session() as session:
        result = session.execute(
            update(_publications)
            .where(
                _publications.c.name == name,
                _publications.c.generation == generation,
                _publications.c.pending_version == version,
            )
            .values(active_version=version, pending_version=None)
        )
        if not result.rowcount:
            raise RuntimeError("文档已删除或已有更新的导入任务")
        session.execute(
            insert(DocumentRecord)
            .values(name=name)
            .on_conflict_do_nothing(index_elements=["name"])
        )


def visible_documents() -> dict[str, str | None]:
    with document_lock, _publication_database().session() as session:
        rows = session.execute(
            select(DocumentRecord.name, _publications.c.active_version).outerjoin(
                _publications, DocumentRecord.name == _publications.c.name
            )
        )
        return dict(rows.all())


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
    with document_lock, _publication_database().session() as session:
        session.execute(
            insert(_publications)
            .values(name=name, generation=0)
            .on_conflict_do_nothing(index_elements=["name"])
        )
        session.execute(
            update(_publications)
            .where(_publications.c.name == name)
            .values(
                active_version=None,
                pending_version=None,
                generation=_publications.c.generation + 1,
            )
        )
        result = session.execute(
            delete(DocumentRecord).where(DocumentRecord.name == name)
        )
        return result.rowcount > 0
