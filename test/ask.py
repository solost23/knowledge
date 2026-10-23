import unittest
from unittest.mock import patch
from uuid import uuid4
from pydantic import ValidationError

from controllers.question import AskRequest
from initialize.server import Server
from routers.register import register
from services.ask import AskService, RequestRoute


class AskServiceTestCase(unittest.TestCase):
    def test_client_cannot_choose_routing_mode(self):
        with self.assertRaises(ValidationError):
            AskRequest.model_validate({"task": "比较两份制度", "mode": "rag"})

    def test_routes_single_retrieval_request_to_rag(self):
        llm = object()
        route = RequestRoute(mode="rag", reason="单次检索足以回答")
        rag_result = ({"code": 0, "data": {"answer": "答案", "sources": []}}, 200)
        with patch("services.ask.build_configured_llm", return_value=llm), \
                patch("services.ask.classify_task", return_value=route), \
                patch("services.ask.QuestionService.question", return_value=rag_result) as question, \
                patch("services.ask.AgentTaskService.execute") as agent:
            result, status = AskService().ask("知识库里有哪些报销规则？")

        self.assertEqual(200, status)
        self.assertEqual("rag", result["data"]["mode"])
        self.assertEqual("单次检索足以回答", result["data"]["route_reason"])
        question.assert_called_once_with("知识库里有哪些报销规则？", llm=llm)
        agent.assert_not_called()

    def test_routes_multistep_request_to_agent_with_plan(self):
        llm = object()
        route = RequestRoute(
            mode="agent",
            reason="需要分别查找并比较两个版本",
            steps=["查找新版", "查找旧版并比较"],
        )
        agent_result = ({"code": 0, "data": {"answer": "差异", "steps": []}}, 200)
        with patch("services.ask.build_configured_llm", return_value=llm), \
                patch("services.ask.classify_task", return_value=route), \
                patch("services.ask.AgentTaskService.execute", return_value=agent_result) as agent, \
                patch("services.ask.QuestionService.question") as question:
            result, status = AskService().ask("比较新版和旧版报销制度")

        self.assertEqual(200, status)
        self.assertEqual("agent", result["data"]["mode"])
        self.assertEqual(route.reason, result["data"]["route_reason"])
        agent.assert_called_once_with(
            "比较新版和旧版报销制度", plan=route.steps, llm=llm
        )
        question.assert_not_called()

    def test_model_error_response_does_not_expose_exception_details(self):
        with patch("services.ask.build_configured_llm", return_value=object()), \
                patch("services.ask.classify_task", side_effect=RuntimeError("secret-token-value")), \
                patch("services.ask.logger.exception"):
            result, status = AskService().ask("普通问题")

        self.assertEqual(502, status)
        self.assertNotIn("secret-token-value", result["message"])

    def test_rejects_empty_and_oversized_tasks_before_model_call(self):
        with patch("services.ask.build_configured_llm") as build_llm:
            empty, empty_status = AskService().ask("   ")
            long, long_status = AskService().ask("x" * 2001)

        self.assertEqual(400, empty_status)
        self.assertEqual(400, long_status)
        self.assertFalse(empty["success"])
        self.assertFalse(long["success"])
        build_llm.assert_not_called()

    def test_http_ask_rejects_frontend_mode_field(self):
        client = Server(register).app.test_client()
        with patch("controllers.question.AskService.ask") as ask:
            result = client.post("/ask", json={"task": "比较制度", "mode": "agent"})

        self.assertEqual(422, result.status_code)
        ask.assert_not_called()

    def test_http_resume_and_status_routes_use_execution_id(self):
        client = Server(register).app.test_client()
        execution_id = uuid4()
        with patch(
            "controllers.question.AgentTaskService.resume",
            return_value=({"code": 0, "success": True, "data": {}}, 200),
        ) as resume, patch(
            "controllers.question.AgentTaskService.get_status",
            return_value=({"code": 0, "success": True, "data": {}}, 200),
        ) as get_status:
            resume_response = client.post(
                "/agent/resume", json={"execution_id": str(execution_id)}
            )
            status_response = client.get(f"/agent/runs/{execution_id}")

        self.assertEqual(200, resume_response.status_code)
        self.assertEqual(200, status_response.status_code)
        self.assertEqual(execution_id, resume.call_args.args[0])
        self.assertEqual(execution_id, get_status.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
