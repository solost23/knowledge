import io
import unittest
from unittest.mock import Mock, patch

from app.api.controllers.documents import DocumentController
from app.core.server import Server


class DocumentUploadApiTestCase(unittest.TestCase):
    def test_upload_returns_task_and_task_endpoint_returns_progress(self):
        job = {
            "id": "job-1",
            "filename": "notes.md",
            "status": "queued",
            "progress": 0,
            "stage": "等待处理",
            "error": None,
            "created_at": "2026-10-23 12:00:00",
            "updated_at": "2026-10-23 12:00:00",
        }
        import_service = Mock()
        import_service.submit.return_value = job
        import_service.get.return_value = job

        with patch(
            "app.api.controllers.documents.DocumentImportService",
            return_value=import_service,
        ):
            server = Server(lambda app: DocumentController(app).register())
            client = server.app.test_client()
            uploaded = client.post(
                "/upload",
                data={"file": (io.BytesIO(b"notes"), "notes.md")},
                content_type="multipart/form-data",
            )
            progress = client.get("/upload/tasks/job-1")

        self.assertEqual(202, uploaded.status_code)
        self.assertEqual("job-1", uploaded.json["data"]["id"])
        self.assertEqual(200, progress.status_code)
        self.assertEqual("queued", progress.json["data"]["status"])
        import_service.submit.assert_called_once()
        import_service.get.assert_called_once_with("job-1")


if __name__ == "__main__":
    unittest.main()
