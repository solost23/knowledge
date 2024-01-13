import sys, os
sys.path.append(os.getcwd())

import unittest

import initialize
from services.question import QuestionService


class QuestionServiceTestCase(unittest.TestCase):
    """
    测试 question 函数
    """
    def test_question(self):
        questions = [
            '作业评级为优秀的学生姓名都是什么？',
            '作业分数大于60的学生账号都是什么？',
            '提交时间为2024/01/11 15:33:25的学生姓名都是什么？',
        ]

        for question in questions:
            self.assertEqual(0, QuestionService().question(question).get('code'))


if __name__ == "__main__":
    unittest.main()
