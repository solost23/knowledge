import sys, os
sys.path.append(os.getcwd())

import unittest

import initialize
from services.doc import DocService


class DocServiceTestCase(unittest.TestCase):
    def test_doc(self):
        file_paths = [
            f'{os.getcwd()}/test/data/学生作业成绩.docx',
            f'{os.getcwd()}/test/data/学生作业成绩.pptx',
            f'{os.getcwd()}/test/data/学生作业成绩.xlsx',
            f'{os.getcwd()}/test/data/学生作业成绩.md',
        ]

        for file_path in file_paths:
            ext = os.path.splitext(file_path)[-1]
            original_name = os.path.basename(file_path)
            result, status_code = DocService().doc(file_path, ext, original_name)
            self.assertEqual(0, result.get('code'))

    def test_list(self):
        result, status_code = DocService().list()
        self.assertEqual(0, result.get('code'))
        self.assertIsInstance(result.get('data'), list)

    def test_delete_not_found(self):
        result, status_code = DocService().delete('不存在的文件.md')
        self.assertEqual(404, result.get('code'))


if __name__ == "__main__":
    unittest.main()
