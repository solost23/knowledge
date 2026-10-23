import os
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from uuid import uuid4
from pydantic import ValidationError

from app.agents.dispatcher import AskService, RequestRoute, dispatch_request
from app.agents.service import AgentTaskService
from app.api.controllers.ask import AskRequest
from app.api.router import register
from app.core import response
from app.core.server import Server
from app.infrastructure.conversation_repository import create_conversation, get_conversation


class AskServiceTestCase(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.database_env = patch.dict(os.environ, {
            "CONVERSATION_DB": os.path.join(self.temp_directory.name, "conversations.sqlite"),
            "AGENT_CHECKPOINT_DB": os.path.join(self.temp_directory.name, "agent.sqlite"),
        })
        self.database_env.start()
        self.addCleanup(self.database_env.stop)
        self.addCleanup(self.temp_directory.cleanup)

    def test_client_cannot_choose_routing_mode(self):
        with self.assertRaises(ValidationError):
            AskRequest.model_validate({"task": "比较两份制度", "mode": "rag"})

    def test_routes_single_retrieval_request_to_rag(self):
        llm = object()
        route = RequestRoute(
            mode="rag", reason="单次检索足以回答", resolved_task="知识库里有哪些报销规则？"
        )
        rag_result = ({"code": 0, "data": {"answer": "答案", "sources": []}}, 200)
        with patch("app.agents.dispatcher.build_configured_llm", return_value=llm), \
                patch("app.agents.dispatcher.classify_task", return_value=route), \
                patch("app.agents.dispatcher.QuestionService.question", return_value=rag_result) as question, \
                patch("app.agents.dispatcher.AgentTaskService.execute") as agent:
            result, status = AskService().ask("知识库里有哪些报销规则？")

        self.assertEqual(200, status)
        self.assertEqual("rag", result["data"]["mode"])
        self.assertEqual("单次检索足以回答", result["data"]["route_reason"])
        question.assert_called_once_with(
            "知识库里有哪些报销规则？", history=[], llm=llm
        )
        agent.assert_not_called()

    def test_routes_multistep_request_to_agent_with_plan(self):
        llm = object()
        route = RequestRoute(
            mode="agent",
            reason="需要分别查找并比较两个版本",
            resolved_task="比较新版和旧版报销制度",
            steps=["查找新版", "查找旧版并比较"],
        )
        agent_result = ({"code": 0, "data": {
            "execution_id": str(uuid4()), "status": "running", "answer": "", "steps": []
        }}, 202)
        with patch("app.agents.dispatcher.build_configured_llm", return_value=llm), \
                patch("app.agents.dispatcher.classify_task", return_value=route), \
                patch("app.agents.dispatcher.AgentTaskService.start", return_value=agent_result) as agent, \
                patch("app.agents.dispatcher.QuestionService.question") as question:
            result, status = AskService().ask("比较新版和旧版报销制度")

        self.assertEqual(202, status)
        self.assertEqual("agent", result["data"]["mode"])
        self.assertEqual(route.reason, result["data"]["route_reason"])
        self.assertEqual("比较新版和旧版报销制度", agent.call_args.args[0])
        self.assertEqual(route.steps, agent.call_args.kwargs["plan"])
        self.assertEqual(llm, agent.call_args.kwargs["llm"])
        self.assertTrue(callable(agent.call_args.kwargs["on_complete"]))
        question.assert_not_called()

    def test_model_error_response_does_not_expose_exception_details(self):
        with patch("app.agents.dispatcher.build_configured_llm", return_value=object()), \
                patch("app.agents.dispatcher.classify_task", side_effect=RuntimeError("secret-token-value")), \
                patch("app.agents.dispatcher.logger.exception"):
            result, status = AskService().ask("普通问题")

        self.assertEqual(502, status)
        self.assertNotIn("secret-token-value", result["message"])

    def test_rejects_empty_and_oversized_tasks_before_model_call(self):
        with patch("app.agents.dispatcher.build_configured_llm") as build_llm:
            empty, empty_status = AskService().ask("   ")
            long, long_status = AskService().ask("x" * 2001)

        self.assertEqual(400, empty_status)
        self.assertEqual(400, long_status)
        self.assertFalse(empty["success"])
        self.assertFalse(long["success"])
        build_llm.assert_not_called()

    def test_start_returns_a_pollable_run_before_routing_finishes(self):
        conversation_id = create_conversation()["id"]
        with patch("app.agents.dispatcher._ASK_EXECUTOR.submit") as submit:
            result, status = AskService().start("需要判断路由的问题", conversation_id)
            execution_id = result["data"]["execution_id"]
            progress, progress_status = AgentTaskService().get_status(execution_id)

        self.assertEqual(202, status)
        self.assertEqual("routing", result["data"]["mode"])
        self.assertEqual(200, progress_status)
        self.assertEqual("running", progress["data"]["status"])
        self.assertEqual("routing", progress["data"]["mode"])
        self.assertTrue(progress["data"]["started_at"])
        self.assertEqual(conversation_id, progress["data"]["conversation_id"])
        submit.assert_called_once()

    def test_rag_cancellation_is_recorded_after_current_model_call_returns(self):
        started = threading.Event()
        release = threading.Event()
        conversation_id = create_conversation()["id"]
        route = RequestRoute(
            mode="rag",
            reason="一次检索即可回答",
            resolved_task="普通问题",
        )

        def blocking_question(*_args, **_kwargs):
            started.set()
            if not release.wait(timeout=5):
                raise TimeoutError("test did not release RAG execution")
            return response.success("成功", {"answer": "不应写入的回答", "sources": []})

        with patch("app.agents.dispatcher.build_configured_llm", return_value=object()), \
                patch("app.agents.dispatcher.classify_task", return_value=route), \
                patch("app.agents.dispatcher.QuestionService.question", side_effect=blocking_question):
            accepted, accepted_status = AskService().start("普通问题", conversation_id)
            execution_id = accepted["data"]["execution_id"]
            try:
                self.assertTrue(started.wait(timeout=3))
                _, cancel_status = AgentTaskService().cancel(execution_id)
                self.assertEqual(200, cancel_status)
            finally:
                release.set()

            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                progress, status = AgentTaskService().get_status(execution_id)
                if status == 200 and progress["data"]["status"] != "running":
                    break
                time.sleep(0.02)

        self.assertEqual(202, accepted_status)
        self.assertEqual("cancelled", progress["data"]["status"])
        self.assertEqual("rag", progress["data"]["mode"])
        self.assertGreater(progress["data"]["duration_ms"], 0)

    def test_http_ask_uses_async_start_endpoint(self):
        client = Server(register).app.test_client()
        accepted = {
            "code": 0,
            "success": True,
            "message": "任务已启动",
            "data": {"execution_id": str(uuid4()), "status": "running", "mode": "routing"},
        }
        with patch("app.api.controllers.ask.AskService.start", return_value=(accepted, 202)) as start:
            result = client.post("/ask", json={"task": "普通知识库问题"})

        self.assertEqual(202, result.status_code)
        self.assertEqual("routing", result.json["data"]["mode"])
        start.assert_called_once()

    def test_http_ask_rejects_frontend_mode_field(self):
        client = Server(register).app.test_client()
        with patch("app.api.controllers.ask.AskService.ask") as ask:
            result = client.post("/ask", json={"task": "比较制度", "mode": "agent"})

        self.assertEqual(422, result.status_code)
        ask.assert_not_called()

    def test_http_resume_and_status_routes_use_execution_id(self):
        client = Server(register).app.test_client()
        execution_id = uuid4()
        with patch(
            "app.api.controllers.ask.AgentTaskService.resume",
            return_value=({"code": 0, "success": True, "data": {}}, 200),
        ) as resume, patch(
            "app.api.controllers.ask.AgentTaskService.get_status",
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

    def test_http_cancel_route_uses_execution_id(self):
        client = Server(register).app.test_client()
        execution_id = uuid4()
        with patch(
            "app.api.controllers.ask.AgentTaskService.cancel",
            return_value=({"code": 0, "success": True, "data": {"cancel_requested": True}}, 200),
        ) as cancel:
            result = client.post(f"/agent/runs/{execution_id}/cancel")

        self.assertEqual(200, result.status_code)
        self.assertTrue(result.json["data"]["cancel_requested"])
        self.assertEqual(execution_id, cancel.call_args.args[0])

    def test_conversation_http_routes_create_read_and_delete(self):
        client = Server(register).app.test_client()
        created_response = client.post("/conversations")
        self.assertEqual(200, created_response.status_code)
        conversation_id = created_response.json["data"]["id"]

        listed = client.get("/conversations")
        fetched = client.get(f"/conversations/{conversation_id}")
        deleted = client.delete(f"/conversations/{conversation_id}")
        missing = client.get(f"/conversations/{conversation_id}")

        self.assertEqual([conversation_id], [item["id"] for item in listed.json["data"]])
        self.assertEqual([], fetched.json["data"]["messages"])
        self.assertEqual(200, deleted.status_code)
        self.assertEqual(404, missing.status_code)

    def test_follow_up_uses_history_to_resolve_request_for_rag(self):
        class RecordingLLM:
            def with_structured_output(self, _schema, **_kwargs):
                return self

            def invoke(self, messages):
                self.messages = messages
                return RequestRoute(
                    mode="rag",
                    reason="基于上下文的单次检索",
                    resolved_task="解释刚才提到的报销申请期限",
                )

        model = RecordingLLM()
        history = [
            {"role": "user", "content": "报销申请需要在几天内提交？"},
            {"role": "assistant", "content": "需要在费用发生后30天内提交。"},
        ]
        rag_result = ({"code": 0, "data": {"answer": "期限是30天。", "sources": []}}, 200)
        with patch("app.agents.dispatcher.build_configured_llm", return_value=model), \
                patch("app.agents.dispatcher.QuestionService.question", return_value=rag_result) as question:
            result, status = dispatch_request("为什么？", history=history)

        self.assertEqual(200, status)
        self.assertEqual("human", model.messages[1].type)
        self.assertEqual("ai", model.messages[2].type)
        self.assertEqual("为什么？", model.messages[-1].content)
        question.assert_called_once_with(
            "解释刚才提到的报销申请期限", history=history, llm=model
        )

    def test_ask_persists_exchange_and_supplies_it_on_follow_up(self):
        conversation_id = create_conversation()["id"]
        llm = object()

        def dispatched(task, llm=None, history=None, on_agent_complete=None):
            return response.success("任务完成", {
                "answer": f"回答：{task}",
                "mode": "rag",
                "sources": [],
            })

        with patch("app.agents.dispatcher.build_configured_llm", return_value=llm), \
                patch("app.agents.dispatcher.dispatch_request", side_effect=dispatched) as dispatch:
            first, first_status = AskService().ask("第一轮问题", conversation_id)
            second, second_status = AskService().ask("接着解释", conversation_id)

        self.assertEqual((200, 200), (first_status, second_status))
        self.assertEqual(conversation_id, first["data"]["conversation_id"])
        self.assertEqual([], dispatch.call_args_list[0].kwargs["history"])
        self.assertEqual([
            {"role": "user", "content": "第一轮问题"},
            {"role": "assistant", "content": "回答：第一轮问题"},
        ], dispatch.call_args_list[1].kwargs["history"])
        conversation = get_conversation(conversation_id)
        self.assertEqual("第一轮问题", conversation["title"])
        self.assertEqual(4, len(conversation["messages"]))

    def test_agent_completion_callback_persists_result_in_conversation(self):
        conversation_id = create_conversation()["id"]
        route = RequestRoute(
            mode="agent",
            reason="需要分步比较资料",
            resolved_task="比较资料",
            steps=["查找资料", "比较差异"],
        )

        def start_agent(task, plan, llm, on_complete):
            on_complete({
                "execution_id": str(uuid4()),
                "status": "completed",
                "answer": "完成",
                "steps": [],
                "sources": [],
            })
            return {
                "code": 0,
                "success": True,
                "message": "Agent 任务已启动",
                "data": {
                "execution_id": str(uuid4()),
                "status": "running",
                "answer": "",
                "steps": [],
                "sources": [],
                },
            }, 202

        with patch("app.agents.dispatcher.build_configured_llm", return_value=object()), \
                patch("app.agents.dispatcher.classify_task", return_value=route), \
                patch("app.agents.dispatcher.AgentTaskService.start", side_effect=start_agent):
            result, status = AskService().ask("比较资料", conversation_id)

        self.assertEqual(202, status)
        self.assertEqual(conversation_id, result["data"]["conversation_id"])
        conversation = get_conversation(conversation_id)
        self.assertEqual(["user", "assistant"], [item["role"] for item in conversation["messages"]])
        saved = conversation["messages"][1]["result"]
        self.assertEqual("agent", saved["mode"])
        self.assertEqual("需要分步比较资料", saved["route_reason"])


if __name__ == "__main__":
    unittest.main()
