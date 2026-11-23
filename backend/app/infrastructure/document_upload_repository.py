from sqlalchemy import func, select, update

from app.infrastructure.database import get_database
from app.infrastructure.models import DocumentUploadJob


def create_upload_job(job_id: str, filename: str) -> dict:
    with get_database().session() as session:
        job = DocumentUploadJob(
            id=job_id, filename=filename, status="queued", progress=0, stage="等待处理"
        )
        session.add(job)
        session.flush()
        return job.as_dict()


def update_upload_job(
    job_id: str,
    *,
    status: str,
    progress: int | None = None,
    stage: str,
    error: str | None = None,
) -> None:
    values = {
        "status": status,
        "stage": stage,
        "error": error,
        "updated_at": func.datetime("now", "localtime"),
    }
    if progress is not None:
        values["progress"] = progress
    with get_database().session() as session:
        session.execute(
            update(DocumentUploadJob)
            .where(DocumentUploadJob.id == job_id)
            .values(**values)
        )


def update_upload_progress(job_id: str | None, progress: int, stage: str) -> None:
    if job_id is None:
        return
    progress = max(0, min(99, progress))
    with get_database().session() as session:
        session.execute(
            update(DocumentUploadJob)
            .where(
                DocumentUploadJob.id == job_id,
                DocumentUploadJob.status.in_(("queued", "processing")),
                DocumentUploadJob.progress < progress,
            )
            .values(
                status="processing",
                progress=progress,
                stage=stage,
                updated_at=func.datetime("now", "localtime"),
            )
        )


def get_upload_job(job_id: str) -> dict | None:
    with get_database().session() as session:
        job = session.get(DocumentUploadJob, job_id)
        return job.as_dict() if job else None


def list_active_upload_jobs() -> list[dict]:
    with get_database().session() as session:
        rows = session.execute(
            select(
                DocumentUploadJob.id,
                DocumentUploadJob.filename,
                DocumentUploadJob.status,
                DocumentUploadJob.progress,
                DocumentUploadJob.stage,
                DocumentUploadJob.error,
            ).where(DocumentUploadJob.status.in_(("queued", "processing")))
        )
        return [dict(row) for row in rows.mappings()]
