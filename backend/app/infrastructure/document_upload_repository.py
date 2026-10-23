import sqlite3

from app.core.settings import DATA_DIR


def _connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(DATA_DIR / "knowledge.db"), timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS document_upload_jobs (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            status TEXT NOT NULL,
            progress INTEGER NOT NULL,
            stage TEXT NOT NULL,
            error TEXT,
            created_at DATETIME DEFAULT (datetime('now', 'localtime')),
            updated_at DATETIME DEFAULT (datetime('now', 'localtime'))
        )
        """
    )
    connection.commit()
    return connection


def create_upload_job(job_id: str, filename: str) -> dict:
    connection = _connect()
    try:
        with connection:
            connection.execute(
                "INSERT INTO document_upload_jobs (id, filename, status, progress, stage) "
                "VALUES (?, ?, 'queued', 0, '等待处理')",
                (job_id, filename),
            )
    finally:
        connection.close()
    return get_upload_job(job_id)


def update_upload_job(
    job_id: str,
    *,
    status: str,
    progress: int,
    stage: str,
    error: str | None = None,
) -> None:
    connection = _connect()
    try:
        with connection:
            connection.execute(
                "UPDATE document_upload_jobs "
                "SET status = ?, progress = ?, stage = ?, error = ?, "
                "updated_at = datetime('now', 'localtime') WHERE id = ?",
                (status, progress, stage, error, job_id),
            )
    finally:
        connection.close()


def get_upload_job(job_id: str) -> dict | None:
    connection = _connect()
    try:
        row = connection.execute(
            "SELECT id, filename, status, progress, stage, error, created_at, updated_at "
            "FROM document_upload_jobs WHERE id = ?",
            (job_id,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        connection.close()


def list_active_upload_jobs() -> list[dict]:
    connection = _connect()
    try:
        rows = connection.execute(
            "SELECT id, filename, status, progress, stage, error "
            "FROM document_upload_jobs WHERE status IN ('queued', 'processing')"
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()
