import json
import os
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.documents import Document
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import ValidationError

from app.agents.graph import compile_graph
from app.agents.service import AgentTaskService
from app.agents.state import SearchKnowledgeInput, TaskPlan
from app.agents.tools import search_knowledge


class FakePlanner:
    def invoke(self, messages):
        self.request = messages[-1].content
        return TaskPlan(steps=["查找新制度", "查找旧制度并比较差异"])


class FakeExecutor:
    def __init__(self):
        self.prompts = []

    def invoke(self, state, config):
        prompt = state["messages"][-1].content
        self.prompts.append(prompt)
        index = len(self.prompts)
        source = f"制度{index}.md"
        tool_result = json.dumps({
            "results": [{"source": source, "content": f"依据{index}"}],
        }, ensure_ascii=False)
        return {"messages": [
            HumanMessage(content=prompt),
            ToolMessage(content=tool_result, name="search_knowledge", tool_call_id=f"call-{index}"),
            AIMessage(content=f"完成步骤{index}"),
        ]}


class FakeFinalizer:
    def invoke(self, messages):
        return AIMessage(content="新旧制度差异已整理。")


class FailOnceExecutor:
    def __init__(self):
        self.calls = 0

    def invoke(self, state, config):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("temporary model failure")
        prompt = state["messages"][-1].content
        return {"messages": [HumanMessage(content=prompt), AIMessage(content="步骤恢复完成")]}


class AgentGraphTestCase(unittest.TestCase):
    def test_search_tool_uses_relevance_filtered_results(self):
        docs = [
            Document(page_content="报销额度为600元", metadata={"source": "报销制度.md"})
        ]
        with patch("app.agents.tools.chroma.search", return_value=[(docs[0], 0.86)]):
            result = json.loads(search_knowledge.invoke({
                "query": "报销额度是多少？",
                "top_k": 4,
            }))

        self.assertEqual([{
            "source": "报销制度.md",
            "content": "报销额度为600元",
        }], result["results"])

    def test_plan_steps_execute_in_order_and_collect_sources(self):
        planner = FakePlanner()
        executor = FakeExecutor()
        progress_updates = []
        graph = compile_graph(
            planner,
            executor,
            FakeFinalizer(),
            progress_callback=progress_updates.append,
        )

        result = graph.invoke({
            "task": "比较新旧制度",
            "plan": [],
            "current_step": 0,
            "steps": [],
            "sources": [],
            "answer": "",
        })

        self.assertEqual("比较新旧制度", planner.request)
        self.assertEqual(["查找新制度", "查找旧制度并比较差异"], result["plan"])
        self.assertIn("当前步骤：1/2：查找新制度", executor.prompts[0])
        self.assertIn("当前步骤：2/2：查找旧制度并比较差异", executor.prompts[1])
        self.assertEqual(["完成步骤1", "完成步骤2"], [step["result"] for step in result["steps"]])
        self.assertEqual(["search_knowledge"], result["steps"][0]["tools"])
        self.assertEqual(["制度1.md", "制度2.md"], [item["source"] for item in result["sources"]])
        self.assertEqual("新旧制度差异已整理。", result["answer"])
        self.assertEqual([1, 2], [item["active_step"] for item in progress_updates if item.get("active_step")])
        self.assertIn("finalizing", [item["stage"] for item in progress_updates])

    def test_checkpoint_resumes_from_failed_step_after_reopening_database(self):
        executor = FailOnceExecutor()
        state = {
            "task": "恢复任务",
            "plan": ["执行一步"],
            "current_step": 0,
            "steps": [],
            "sources": [],
            "answer": "",
        }
        config = {"configurable": {"thread_id": "resume-test"}}
        with tempfile.TemporaryDirectory() as directory:
            db_path = os.path.join(directory, "checkpoints.sqlite")
            with SqliteSaver.from_conn_string(db_path) as checkpointer:
                checkpointer.setup()
                graph = compile_graph(None, executor, FakeFinalizer(), checkpointer)
                with self.assertRaisesRegex(RuntimeError, "temporary model failure"):
                    graph.invoke(state, config=config)
                snapshot = graph.get_state(config)
                self.assertEqual(("execute_step",), snapshot.next)

            with SqliteSaver.from_conn_string(db_path) as checkpointer:
                graph = compile_graph(None, executor, FakeFinalizer(), checkpointer)
                result = graph.invoke(None, config=config)

        self.assertEqual(["步骤恢复完成"], [step["result"] for step in result["steps"]])
        self.assertEqual("新旧制度差异已整理。", result["answer"])

    def test_agent_service_persists_failed_run_and_resumes_it(self):
        executor = FailOnceExecutor()

        def build_graph(_llm, checkpointer, progress_callback=None):
            return compile_graph(
                None,
                executor,
                FakeFinalizer(),
                checkpointer,
                progress_callback=progress_callback,
            )

        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {"AGENT_CHECKPOINT_DB": os.path.join(directory, "agent.sqlite")}), \
                patch("app.agents.service.build_graph", side_effect=build_graph), \
                patch("app.agents.service.build_configured_llm", return_value=object()), \
                patch("app.agents.service.logger.exception"):
            service = AgentTaskService()
            failed, status = service.execute("需要恢复的任务", plan=["执行一步"], llm=object())
            self.assertEqual(502, status)
            self.assertNotIn("temporary model failure", failed["message"])
            execution_id = failed["data"]["execution_id"]
            self.assertTrue(failed["data"]["resumable"])

            progress, status = service.get_status(execution_id)
            self.assertEqual(200, status)
            self.assertEqual("failed", progress["data"]["status"])
            self.assertEqual([], progress["data"]["steps"])

            resumed, status = service.resume(execution_id)
            self.assertEqual(200, status)
            self.assertEqual(execution_id, resumed["data"]["execution_id"])
            self.assertEqual("completed", resumed["data"]["status"])
            self.assertEqual("新旧制度差异已整理。", resumed["data"]["answer"])

            completed, status = service.get_status(execution_id)
            self.assertEqual(200, status)
            self.assertEqual("completed", completed["data"]["status"])

    def test_agent_start_returns_pollable_run_before_background_work(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {"AGENT_CHECKPOINT_DB": os.path.join(directory, "agent.sqlite")}), \
                patch("app.agents.service._AGENT_EXECUTOR.submit") as submit:
            result, status = AgentTaskService().start(
                "比较两份制度",
                plan=["查找新制度", "比较差异"],
                llm=object(),
            )
            execution_id = result["data"]["execution_id"]
            progress, progress_status = AgentTaskService().get_status(execution_id)

        self.assertEqual(202, status)
        self.assertEqual(200, progress_status)
        self.assertEqual("running", progress["data"]["status"])
        self.assertEqual(["查找新制度", "比较差异"], progress["data"]["plan"])
        self.assertEqual("Agent 已接收任务，准备执行计划", progress["data"]["current_action"])
        submit.assert_called_once()

    def test_agent_status_reports_active_step_during_background_execution(self):
        started = threading.Event()
        release = threading.Event()

        class BlockingExecutor:
            def invoke(self, state, config):
                started.set()
                if not release.wait(timeout=5):
                    raise TimeoutError("test did not release executor")
                prompt = state["messages"][-1].content
                return {"messages": [HumanMessage(content=prompt), AIMessage(content="已完成") ]}

        def build_graph(_llm, checkpointer, progress_callback=None):
            return compile_graph(
                None,
                BlockingExecutor(),
                FakeFinalizer(),
                checkpointer,
                progress_callback=progress_callback,
            )

        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {"AGENT_CHECKPOINT_DB": os.path.join(directory, "agent.sqlite")}), \
                patch("app.agents.service.build_graph", side_effect=build_graph):
            service = AgentTaskService()
            started_response, status = service.start(
                "执行后台任务",
                plan=["正在执行的步骤"],
                llm=object(),
            )
            execution_id = started_response["data"]["execution_id"]
            try:
                self.assertTrue(started.wait(timeout=3))
                progress, status = service.get_status(execution_id)
                self.assertEqual(200, status)
                self.assertEqual("running", progress["data"]["status"])
                self.assertEqual(1, progress["data"]["active_step"])
                self.assertIn("正在执行第 1 步", progress["data"]["current_action"])
            finally:
                release.set()

            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                progress, status = service.get_status(execution_id)
                if status == 200 and progress["data"]["status"] == "completed":
                    break
                time.sleep(0.02)

        self.assertEqual("completed", progress["data"]["status"])
        self.assertEqual(["已完成"], [step["result"] for step in progress["data"]["steps"]])

    def test_agent_cancellation_finishes_at_step_boundary_and_records_duration(self):
        started = threading.Event()
        release = threading.Event()

        class BlockingExecutor:
            def invoke(self, state, config):
                started.set()
                if not release.wait(timeout=5):
                    raise TimeoutError("test did not release executor")
                prompt = state["messages"][-1].content
                return {"messages": [HumanMessage(content=prompt), AIMessage(content="已完成") ]}

        def build_graph(_llm, checkpointer, progress_callback=None):
            return compile_graph(
                None,
                BlockingExecutor(),
                FakeFinalizer(),
                checkpointer,
                progress_callback=progress_callback,
            )

        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {"AGENT_CHECKPOINT_DB": os.path.join(directory, "agent.sqlite")}), \
                patch("app.agents.service.build_graph", side_effect=build_graph):
            service = AgentTaskService()
            started_response, status = service.start(
                "停止测试任务",
                plan=["执行一个可能耗时的步骤"],
                llm=object(),
            )
            execution_id = started_response["data"]["execution_id"]
            try:
                self.assertTrue(started.wait(timeout=3))
                cancel_response, cancel_status = service.cancel(execution_id)
                self.assertEqual(200, cancel_status)
                self.assertTrue(cancel_response["data"]["cancel_requested"])
            finally:
                release.set()

            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                progress, status = service.get_status(execution_id)
                if status == 200 and progress["data"]["status"] != "running":
                    break
                time.sleep(0.02)

        run = progress["data"]
        self.assertEqual("cancelled", run["status"])
        self.assertEqual("任务已停止", run["current_action"])
        self.assertTrue(run["started_at"])
        self.assertTrue(run["finished_at"])
        self.assertGreater(run["duration_ms"], 0)
        self.assertTrue(run["cancel_requested"])

    def test_search_tool_input_is_bounded(self):
        SearchKnowledgeInput(query="费用报销", top_k=4)
        with self.assertRaises(ValidationError):
            SearchKnowledgeInput(query=" ")
        with self.assertRaises(ValidationError):
            SearchKnowledgeInput(query="x" * 501)
        with self.assertRaises(ValidationError):
            SearchKnowledgeInput(query="报销", top_k=5)
        with self.assertRaises(ValidationError):
            TaskPlan(steps=["x" * 301])
        with self.assertRaises(ValidationError):
            TaskPlan(steps=["步骤"] * 6)


if __name__ == "__main__":
    unittest.main()
