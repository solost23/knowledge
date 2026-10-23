import os
from pathlib import Path
import unittest
from unittest.mock import patch

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
            with patch("app.knowledge.documents.chroma.db.add_documents") as add_documents, \
                    patch("app.knowledge.documents.exists_doc", return_value=False), \
                    patch("app.knowledge.documents.insert_doc"):
                result, status_code = DocumentService().doc(file_path, ext, original_name)
            self.assertEqual(0, result.get('code'))
            add_documents.assert_called_once()

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
