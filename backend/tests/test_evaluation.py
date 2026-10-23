import unittest
from unittest.mock import patch

from evals.agent import run
from evals.agent_metrics import score_agent_rows


class AgentEvaluationMetricsTestCase(unittest.TestCase):
    def test_scores_routing_plan_steps_tools_and_answer_terms(self):
        summary = score_agent_rows([
            {
                "expected_mode": "rag",
                "mode": "rag",
                "answer_terms": ["600", "元"],
                "answer": "标准是 600 元",
            },
            {
                "expected_mode": "agent",
                "mode": "agent",
                "success": True,
                "plan": ["查新版制度", "查旧版并比较"],
                "min_plan_steps": 2,
                "plan_terms": ["新版", "旧版"],
                "steps": [{"tools": ["search_knowledge"]}],
                "tools": ["search_knowledge"],
                "expected_tools": ["search_knowledge"],
                "answer_terms": ["新版", "旧版"],
                "answer": "新版和旧版制度有差异",
            },
        ])

        self.assertEqual(1.0, summary["route_accuracy"])
        self.assertEqual(1.0, summary["agent_plan_valid_rate"])
        self.assertEqual(0.5, summary["agent_step_completion_rate"])
        self.assertEqual(1.0, summary["agent_plan_term_recall"])
        self.assertEqual(1.0, summary["expected_tool_recall"])
        self.assertEqual(1.0, summary["answer_term_recall"])

    def test_evaluation_runner_scores_fake_rag_and_agent_responses(self):
        class FakeService:
            def ask(self, task):
                if task == "simple":
                    return ({"success": True, "data": {
                        "mode": "rag", "answer": "答案 600 元",
                    }}, 200)
                return ({"success": True, "data": {
                    "mode": "agent",
                    "answer": "新版与旧版差异",
                    "plan": ["查新版", "查旧版并比较"],
                    "steps": [{"tools": ["search_knowledge"]}],
                }}, 200)

        questions = [
            {"id": "rag", "task": "simple", "expected_mode": "rag", "answer_terms": ["600"]},
            {
                "id": "agent",
                "task": "compare",
                "expected_mode": "agent",
                "min_plan_steps": 2,
                "plan_terms": ["新版", "旧版"],
                "answer_terms": ["新版", "旧版"],
                "expected_tools": ["search_knowledge"],
            },
        ]
        with patch("builtins.print"):
            summary, rows = run(questions, service=FakeService())

        self.assertEqual(1.0, summary["route_accuracy"])
        self.assertEqual(2, summary["passed_cases"])
        self.assertEqual(1.0, summary["case_pass_rate"])
        self.assertTrue(all(row["case_pass"] for row in rows))


if __name__ == "__main__":
    unittest.main()
