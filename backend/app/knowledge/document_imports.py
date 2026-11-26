import os
import uuid

from loguru import logger
from werkzeug.datastructures import FileStorage

from app.infrastructure import submit_task
from app.infrastructure.document_repository import begin_document
from app.infrastructure.document_upload_repository import (
    create_upload_job,
    get_upload_job,
    list_active_upload_jobs,
    update_upload_job,
)
from app.knowledge.documents import DocumentService


class DocumentImportService:
    def __init__(self, document_service: DocumentService | None = None):
        self.document_service = document_service or DocumentService()
        self._recover_interrupted_jobs()

    def submit(self, file: FileStorage, extension: str, filename: str) -> dict:
        job_id = str(uuid.uuid4())
        file_path = f"/tmp/knowledge-upload-{job_id}{extension}"
        job_created = False
        try:
            file.save(file_path)
            job = create_upload_job(job_id, filename)
            job_created = True
            publication = begin_document(filename)
            submit_task(
                "document_import",
                self._process,
                job_id,
                filename,
                file_path,
                publication,
            )
            return job
        except Exception as error:
            if job_created:
                update_upload_job(
                    job_id,
                    status="failed",
                    progress=0,
                    stage="导入任务启动失败",
                    error=str(error) or "导入任务启动失败",
                )
            if os.path.isfile(file_path):
                os.remove(file_path)
            raise
        finally:
            file.close()

    def get(self, job_id: str) -> dict | None:
        return get_upload_job(job_id)

    def _process(
        self,
        job_id: str,
        filename: str,
        file_path: str,
        publication: tuple[str, int] | None = None,
    ) -> None:
        try:
            update_upload_job(
                job_id,
                status="processing",
                progress=1,
                stage="准备解析文档",
            )
            result, status_code = self.document_service.doc(
                file_path,
                filename,
                upload_job_id=job_id,
                publication=publication,
            )
            if status_code != 200 or not result.get("success"):
                raise RuntimeError(result.get("message", "文档导入失败"))
            update_upload_job(
                job_id,
                status="completed",
                progress=100,
                stage="导入完成",
            )
        except Exception as error:
            logger.exception(f"文档导入任务失败: {job_id}")
            try:
                self.document_service.discard_upload_job(job_id, filename)
            except Exception:
                logger.exception(f"清理文档导入任务失败数据失败: {job_id}")
            update_upload_job(
                job_id,
                status="failed",
                stage="导入失败",
                error=str(error) or "文档导入失败",
            )
        finally:
            if os.path.isfile(file_path):
                os.remove(file_path)

    def _recover_interrupted_jobs(self) -> None:
        for job in list_active_upload_jobs():
            try:
                self.document_service.discard_upload_job(job["id"], job["filename"])
            except Exception:
                logger.exception(f"清理中断的文档导入任务失败: {job['id']}")
            update_upload_job(
                job["id"],
                status="failed",
                progress=job["progress"],
                stage="服务重启，导入已中断",
                error="服务重启导致导入中断，请重新上传。",
            )
