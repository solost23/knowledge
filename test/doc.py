import sys, os
sys.path.append(os.getcwd())

import unittest

import initialize
from services.doc import DocService


class DocServiceTestCase(unittest.TestCase):
    """
    测试 doc 函数
    """
    def test_doc(self):
        # 每个文件中存储一部分虚拟学生信息
        file_paths = [
            f'{os.getcwd()}/test/data/学生作业成绩.docx',
            f'{os.getcwd()}/test/data/学生作业成绩.pptx',
            f'{os.getcwd()}/test/data/学生作业成绩.xlsx',
        ]

        for file_path in file_paths:
            ext = os.path.splitext(file_path)[-1]
            code = DocService().doc(file_path, ext).get('code')
            self.assertEqual(0, code)


if __name__ == "__main__":
    unittest.main()
