import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from app.infrastructure import document_upload_repository as repository
from app.knowledge.document_imports import DocumentImportService


class DocumentImportTestCase(unittest.TestCase):
    def test_failure_preserves_the_last_persisted_progress(self):
        with (
            TemporaryDirectory() as data_dir,
            patch.dict(os.environ, {"KNOWLEDGE_DB": str(Path(data_dir) / "test.db")}),
            patch("app.knowledge.document_imports.DocumentService") as documents,
            patch("app.knowledge.document_imports.logger.exception"),
        ):
            service = DocumentImportService()
            repository.create_upload_job("job-1", "manual.pdf")

            def import_document(*_args, **_kwargs):
                repository.update_upload_progress("job-1", 38, "文本切分完成")
                raise RuntimeError("向量写入失败")

            documents.return_value.doc.side_effect = import_document
            service._process("job-1", "manual.pdf", str(Path(data_dir) / "manual.pdf"))
            job = repository.get_upload_job("job-1")

        self.assertEqual("failed", job["status"])
        self.assertEqual(38, job["progress"])
        self.assertEqual("向量写入失败", job["error"])
        documents.return_value.discard_upload_job.assert_called_once_with(
            "job-1", "manual.pdf"
        )
