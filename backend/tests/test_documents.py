import os
from pathlib import Path
import unittest
from unittest.mock import patch

from langchain_core.documents import Document

from app.knowledge.documents import DocumentService


class DocServiceTestCase(unittest.TestCase):
    def test_doc(self):
        file_paths = [
            str(Path(__file__).parent / "fixtures" / "data" / name)
            for name in (
                "学生作业成绩.docx",
                "学生作业成绩.pptx",
                "学生作业成绩.xlsx",
                "学生作业成绩.md",
            )
        ]

        for file_path in file_paths:
            ext = os.path.splitext(file_path)[-1]
            original_name = os.path.basename(file_path)
            with patch("app.knowledge.documents.chroma.db.add_texts", return_value=["chunk-id"]) as add_texts, \
                    patch("app.knowledge.documents.exists_doc", return_value=False), \
                    patch("app.knowledge.documents.insert_doc"):
                result, status_code = DocumentService().doc(file_path, ext, original_name)
            self.assertEqual(0, result.get('code'))
            add_texts.assert_called_once()

    def test_doc_reports_batched_vector_progress(self):
        chunks = [Document(page_content=f"chunk {index}", metadata={}) for index in range(130)]
        progress = []
        with patch("app.knowledge.documents.pdf.load", return_value=[Document(page_content="text")]), \
                patch("app.knowledge.documents.RecursiveCharacterTextSplitter.split_documents", return_value=chunks), \
                patch("app.knowledge.documents.chroma.db.add_texts", side_effect=lambda texts, metadatas, ids: ids) as add_texts, \
                patch("app.knowledge.documents.exists_doc", return_value=False), \
                patch("app.knowledge.documents.insert_doc"):
            result, status_code = DocumentService().doc(
                "large.pdf",
                ".pdf",
                "large.pdf",
                progress_callback=lambda value, stage: progress.append((value, stage)),
                upload_job_id="job-1",
            )

        self.assertEqual(200, status_code)
        self.assertTrue(result["success"])
        self.assertEqual(3, add_texts.call_count)
        self.assertEqual(98, progress[-1][0])
        self.assertIn("130/130", progress[-2][1])
        self.assertEqual("job-1", add_texts.call_args_list[0].kwargs["metadatas"][0]["upload_job_id"])

    def test_list(self):
        with patch("app.knowledge.documents.list_docs", return_value=[]):
            result, status_code = DocumentService().list()
        self.assertEqual(0, result.get('code'))
        self.assertIsInstance(result.get('data'), list)

    def test_delete_not_found(self):
        with patch("app.knowledge.documents.chroma.db.get", return_value={"ids": []}), \
                patch("app.knowledge.documents.delete_doc", return_value=False):
            result, status_code = DocumentService().delete('不存在的文件.md')
        self.assertEqual(404, result.get('code'))


if __name__ == "__main__":
    unittest.main()
