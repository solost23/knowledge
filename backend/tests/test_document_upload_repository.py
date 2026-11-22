import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from app.infrastructure import document_upload_repository as repository


class DocumentUploadRepositoryTestCase(unittest.TestCase):
    def test_job_lifecycle_is_persisted(self):
        with (
            TemporaryDirectory() as data_dir,
            patch.object(repository, "DATA_DIR", Path(data_dir)),
        ):
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

    def test_progress_never_regresses_or_changes_a_finished_job(self):
        with (
            TemporaryDirectory() as data_dir,
            patch.object(repository, "DATA_DIR", Path(data_dir)),
        ):
            repository.create_upload_job("job-1", "manual.pdf")
            repository.update_upload_progress("job-1", 42, "正在解析")
            repository.update_upload_progress("job-1", 20, "旧进度")
            repository.update_upload_progress("job-1", 42, "重复进度")
            processing = repository.get_upload_job("job-1")
            repository.update_upload_job(
                "job-1", status="failed", stage="导入失败", error="解析失败"
            )
            repository.update_upload_progress("job-1", 98, "迟到的进度")
            failed = repository.get_upload_job("job-1")

        self.assertEqual(42, processing["progress"])
        self.assertEqual("正在解析", processing["stage"])
        self.assertEqual("failed", failed["status"])
        self.assertEqual(42, failed["progress"])
        self.assertEqual("导入失败", failed["stage"])
        self.assertEqual("解析失败", failed["error"])


if __name__ == "__main__":
    unittest.main()
