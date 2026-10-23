import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from app.infrastructure import document_upload_repository as repository


class DocumentUploadRepositoryTestCase(unittest.TestCase):
    def test_job_lifecycle_is_persisted(self):
        with TemporaryDirectory() as data_dir, patch.object(repository, "DATA_DIR", Path(data_dir)):
            created = repository.create_upload_job("job-1", "manual.pdf")
            repository.update_upload_job(
                "job-1",
                status="processing",
                progress=42,
                stage="正在解析 PDF（42/100 页）",
            )

            job = repository.get_upload_job("job-1")

        self.assertEqual("queued", created["status"])
        self.assertEqual("processing", job["status"])
        self.assertEqual(42, job["progress"])
        self.assertIn("42/100", job["stage"])
        self.assertIsNone(job["error"])


if __name__ == "__main__":
    unittest.main()
