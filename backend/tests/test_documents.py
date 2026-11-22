import os
import unittest
from pathlib import Path
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
            original_name = os.path.basename(file_path)
            with (
                patch(
                    "app.knowledge.documents.get_chroma"
                ) as get_chroma,
                patch("app.knowledge.documents.exists_doc", return_value=False),
                patch("app.knowledge.documents.insert_doc"),
            ):
                add_texts = get_chroma.return_value.db.add_texts
                result, status_code = DocumentService().doc(
                    file_path, original_name
                )
            self.assertEqual(0, result.get("code"))
            add_texts.assert_called_once()

    def test_doc_reports_batched_vector_progress(self):
        chunks = [
            Document(page_content=f"chunk {index}", metadata={}) for index in range(130)
        ]
        with (
            patch("app.knowledge.documents.update_upload_progress") as update_progress,
            patch(
                "app.knowledge.documents.load_file",
                return_value=[Document(page_content="text")],
            ),
            patch(
                "app.knowledge.documents.RecursiveCharacterTextSplitter.split_documents",
                return_value=chunks,
            ),
            patch("app.knowledge.documents.get_chroma") as get_chroma,
            patch("app.knowledge.documents.exists_doc", return_value=False),
            patch("app.knowledge.documents.insert_doc"),
        ):
            add_texts = get_chroma.return_value.db.add_texts
            add_texts.side_effect = lambda texts, metadatas, ids: ids
            result, status_code = DocumentService().doc(
                "large.pdf",
                "large.pdf",
                upload_job_id="job-1",
            )

        progress = [call.args[1:] for call in update_progress.call_args_list]
        self.assertEqual(200, status_code)
        self.assertTrue(result["success"])
        self.assertEqual(1, add_texts.call_count)
        self.assertEqual(98, progress[-1][0])
        self.assertIn("130/130", progress[-2][1])
        self.assertEqual(
            "job-1", add_texts.call_args_list[0].kwargs["metadatas"][0]["upload_job_id"]
        )

    def test_vector_cleanup_failure_does_not_mask_original_write_error(self):
        write_error = RuntimeError("Chroma upsert failed")
        with (
            patch(
                "app.knowledge.documents.load_file",
                return_value=[Document(page_content="text")],
            ),
            patch("app.knowledge.documents.get_chroma") as get_chroma,
            patch("app.knowledge.documents.exists_doc", return_value=False),
        ):
            get_chroma.return_value.db.add_texts.side_effect = write_error
            get_chroma.return_value.db.delete.side_effect = RuntimeError("cleanup failed")
            with self.assertRaisesRegex(RuntimeError, "Chroma upsert failed"):
                DocumentService().doc("broken.pdf", "broken.pdf")

    def test_list(self):
        with patch("app.knowledge.documents.list_docs", return_value=[]):
            result, status_code = DocumentService().list()
        self.assertEqual(0, result.get("code"))
        self.assertIsInstance(result.get("data"), list)

    def test_delete_not_found(self):
        with (
            patch("app.knowledge.documents.get_chroma") as get_chroma,
            patch("app.knowledge.documents.delete_doc", return_value=False),
        ):
            get_chroma.return_value.db.get.return_value = {"ids": []}
            result, status_code = DocumentService().delete("不存在的文件.md")
        self.assertEqual(404, result.get("code"))


if __name__ == "__main__":
    unittest.main()
