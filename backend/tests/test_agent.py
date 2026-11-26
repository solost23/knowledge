import json
import os
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import ValidationError

from app.agents.checkpoints import (
    get_run,
    open_checkpointer,
    recover_interrupted_runs,
    request_run_cancel,
)
from app.agents.execution import (
    _save_cancelled_result,
    persist_completed,
    recover_pending_exchanges,
)
from app.agents.graph import compile_graph
from app.agents.service import AgentTaskService, _validate_plan
from app.agents.state import SearchKnowledgeInput, TaskPlan
from app.agents.tools import search_knowledge
from app.infrastructure.conversation_repository import (
    create_conversation,
    delete_conversation,
    get_conversation,
)
from app.infrastructure.database import dispose_databases


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
        tool_result = json.dumps(
            {
                "results": [{"source": source, "content": f"依据{index}"}],
            },
            ensure_ascii=False,
        )
        return {
            "messages": [
                HumanMessage(content=prompt),
                ToolMessage(
                    content=tool_result,
                    name="search_knowledge",
                    tool_call_id=f"call-{index}",
                ),
                AIMessage(content=f"完成步骤{index}"),
            ]
        }


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
        return {
            "messages": [
                HumanMessage(content=prompt),
                AIMessage(content="步骤恢复完成"),
            ]
        }


class AgentGraphTestCase(unittest.TestCase):
    def test_search_tool_uses_relevance_filtered_results(self):
        docs = [
            Document(page_content="报销额度为600元", metadata={"source": "报销制度.md"})
        ]
        with patch("app.agents.tools.get_chroma") as get_chroma:
            get_chroma.return_value.search.return_value = [(docs[0], 0.86)]
            result = json.loads(
                search_knowledge.invoke(
                    {
                        "query": "报销额度是多少？",
                        "top_k": 4,
                    }
                )
            )

        self.assertEqual(
            [
                {
                    "source": "报销制度.md",
                    "content": "报销额度为600元",
                }
            ],
            result["results"],
        )

    def test_plan_steps_execute_in_order_and_collect_sources(self):
        planner = FakePlanner()
        executor = FakeExecutor()
        graph = compile_graph(
            planner,
            executor,
            FakeFinalizer(),
        )

        with patch("app.agents.graph.update_run_progress") as update_progress:
            result = graph.invoke(
                {
                    "task": "比较新旧制度",
                    "plan": [],
                    "current_step": 0,
                    "steps": [],
                    "sources": [],
                    "answer": "",
                }
            )
        progress_updates = [call.args[2] for call in update_progress.call_args_list]

        self.assertEqual("比较新旧制度", planner.request)
        self.assertEqual(["查找新制度", "查找旧制度并比较差异"], result["plan"])
        self.assertIn("当前步骤：1/2：查找新制度", executor.prompts[0])
        self.assertIn("当前步骤：2/2：查找旧制度并比较差异", executor.prompts[1])
        self.assertEqual(
            ["完成步骤1", "完成步骤2"], [step["result"] for step in result["steps"]]
        )
        self.assertEqual(["search_knowledge"], result["steps"][0]["tools"])
        self.assertEqual(
            ["制度1.md", "制度2.md"], [item["source"] for item in result["sources"]]
        )
        self.assertEqual("新旧制度差异已整理。", result["answer"])
        self.assertEqual(
            [1, 2],
            [
                item["active_step"]
                for item in progress_updates
                if item.get("active_step")
            ],
        )
        self.assertIn("finalizing", [item["stage"] for item in progress_updates])

    def test_plan_with_more_than_five_steps_executes_without_truncation(self):
        plan = [f"步骤 {index}" for index in range(1, 8)]
        planner = FakePlanner()
        executor = FakeExecutor()
        graph = compile_graph(planner, executor, FakeFinalizer())

        result = graph.invoke(
            {
                "task": "执行七个步骤",
                "plan": _validate_plan(plan),
                "current_step": 0,
                "steps": [],
                "sources": [],
                "answer": "",
            },
            config={"recursion_limit": 40},
        )

        self.assertEqual(plan, result["plan"])
        self.assertEqual(7, len(result["steps"]))
        self.assertEqual(7, len(executor.prompts))
        self.assertIn("当前步骤：7/7：步骤 7", executor.prompts[-1])

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

        def build_graph(_llm, checkpointer, execution_id=None):
            return compile_graph(
                None,
                executor,
                FakeFinalizer(),
                checkpointer,
                execution_id=execution_id,
            )

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(
                os.environ,
                {"AGENT_CHECKPOINT_DB": os.path.join(directory, "agent.sqlite")},
            ),
            patch("app.agents.execution.build_graph", side_effect=build_graph),
            patch("app.agents.service.build_graph", side_effect=build_graph),
            patch("app.agents.execution.LLM", return_value=object()),
            patch("app.agents.service.LLM", return_value=object()),
            patch("app.agents.execution.logger.exception"),
        ):
            service = AgentTaskService()
            failed, status = service.execute(
                "需要恢复的任务", plan=["执行一步"], llm=object()
            )
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
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(
                os.environ,
                {"AGENT_CHECKPOINT_DB": os.path.join(directory, "agent.sqlite")},
            ),
            patch("app.agents.service.submit_task") as submit,
        ):
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
        self.assertEqual(
            "Agent 已接收任务，准备执行计划", progress["data"]["current_action"]
        )
        submit.assert_called_once()
        self.assertEqual("agent", submit.call_args.args[0])

    def test_agent_status_reports_active_step_during_background_execution(self):
        started = threading.Event()
        release = threading.Event()

        class BlockingExecutor:
            def invoke(self, state, config):
                started.set()
                if not release.wait(timeout=5):
                    raise TimeoutError("test did not release executor")
                prompt = state["messages"][-1].content
                return {
                    "messages": [
                        HumanMessage(content=prompt),
                        AIMessage(content="已完成"),
                    ]
                }

        def build_graph(_llm, checkpointer, execution_id=None):
            return compile_graph(
                None,
                BlockingExecutor(),
                FakeFinalizer(),
                checkpointer,
                execution_id=execution_id,
            )

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(
                os.environ,
                {"AGENT_CHECKPOINT_DB": os.path.join(directory, "agent.sqlite")},
            ),
            patch("app.agents.execution.build_graph", side_effect=build_graph),
        ):
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
        self.assertEqual(
            ["已完成"], [step["result"] for step in progress["data"]["steps"]]
        )

    def test_agent_cancellation_finishes_at_step_boundary_and_records_duration(self):
        started = threading.Event()
        release = threading.Event()

        class BlockingExecutor:
            def invoke(self, state, config):
                started.set()
                if not release.wait(timeout=5):
                    raise TimeoutError("test did not release executor")
                prompt = state["messages"][-1].content
                return {
                    "messages": [
                        HumanMessage(content=prompt),
                        AIMessage(content="已完成"),
                    ]
                }

        def build_graph(_llm, checkpointer, execution_id=None):
            return compile_graph(
                None,
                BlockingExecutor(),
                FakeFinalizer(),
                checkpointer,
                execution_id=execution_id,
            )

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(
                os.environ,
                {"AGENT_CHECKPOINT_DB": os.path.join(directory, "agent.sqlite")},
            ),
            patch("app.agents.execution.build_graph", side_effect=build_graph),
        ):
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

    def test_startup_outbox_repairs_completed_exchange_once_and_skips_deleted(self):
        from app.agents.ask_run_lifecycle import AskRunLifecycle
        from app.agents.dispatcher import RequestRoute

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(
                os.environ,
                {
                    "APP_DATA_DIR": directory,
                    "AGENT_CHECKPOINT_DB": os.path.join(directory, "agent.db"),
                    "CONVERSATION_DB": os.path.join(directory, "conversations.db"),
                },
            ),
        ):
            service = AgentTaskService()
            conversations = []
            for mode in ("agent", "rag"):
                conversation_id = create_conversation()["id"]
                conversations.append(conversation_id)
                execution_id, _ = service.create_routed_run("问题", conversation_id)
                if mode == "agent":
                    with (
                        patch(
                            "app.agents.execution.append_exchange",
                            side_effect=RuntimeError("崩溃窗口"),
                        ),
                        patch("app.agents.execution.logger.exception"),
                        open_checkpointer() as checkpointer,
                    ):
                        persist_completed(
                            checkpointer,
                            execution_id,
                            {"answer": "答案", "plan": [], "current_step": 0},
                        )
                else:
                    with (
                        patch(
                            "app.agents.ask_run_lifecycle.append_exchange",
                            side_effect=RuntimeError("崩溃窗口"),
                        ),
                        self.assertRaises(RuntimeError),
                    ):
                        AskRunLifecycle.complete_rag(
                            execution_id,
                            "问题",
                            conversation_id,
                            RequestRoute(
                                mode="rag", reason="检索", resolved_task="问题"
                            ),
                            {"answer": "答案"},
                        )
                self.assertEqual([], get_conversation(conversation_id)["messages"])
            deleted_id = create_conversation()["id"]
            deleted_run, _ = service.create_routed_run("已删除问题", deleted_id)
            with open_checkpointer() as checkpointer:
                persist_completed(checkpointer, deleted_run, {"answer": "答案"})
            delete_conversation(deleted_id)
            dispose_databases()
            for _ in range(2):
                recover_interrupted_runs()
                recover_pending_exchanges()
                dispose_databases()
            for conversation_id in conversations:
                messages = get_conversation(conversation_id)["messages"]
                self.assertEqual(["问题", "答案"], [m["content"] for m in messages])
            self.assertIsNone(get_conversation(deleted_id))

    def test_completed_status_uses_result_and_running_status_hides_stale_result(self):
        from app.agents.checkpoints import save_run_result, set_run_status

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(
                os.environ,
                {
                    "APP_DATA_DIR": directory,
                    "AGENT_CHECKPOINT_DB": os.path.join(directory, "agent.db"),
                },
            ),
        ):
            execution_id = "00000000-0000-0000-0000-000000000005"
            service = AgentTaskService()
            service._create_run(execution_id, "问题", ["一步"])
            with open_checkpointer() as checkpointer:
                save_run_result(
                    checkpointer, execution_id, {"answer": "旧答案", "status": "failed"}
                )
            self.assertIsNone(service.get_status(execution_id)[0]["data"]["result"])
            with open_checkpointer() as checkpointer:
                set_run_status(checkpointer, execution_id, "running")
                self.assertIsNone(get_run(checkpointer, execution_id)["result"])
                persist_completed(
                    checkpointer,
                    execution_id,
                    {
                        "plan": ["最终计划"],
                        "current_step": 1,
                        "steps": [],
                        "sources": [],
                        "answer": "最终答案",
                    },
                )
            with patch(
                "app.agents.service.checkpoint_values",
                return_value={
                    "plan": ["旧计划"],
                    "current_step": 9,
                    "steps": [{"old": True}],
                    "sources": [{"old": True}],
                    "answer": "旧答案",
                },
            ):
                result = service.get_status(execution_id)[0]["data"]
            self.assertEqual(["最终计划"], result["plan"])
            self.assertEqual(1, result["current_step"])
            self.assertEqual([], result["steps"])
            self.assertEqual([], result["sources"])
            self.assertEqual("最终答案", result["answer"])

    def test_inner_agent_checks_cancellation_before_models_and_tools(self):
        from unittest.mock import Mock

        from app.agents.graph import build_graph
        from app.agents.state import AgentRunCancelled

        with (
            patch("app.agents.graph.create_agent") as create_agent,
            patch("app.agents.graph.compile_graph"),
            patch("app.agents.graph.run_cancel_requested", return_value=True),
        ):
            build_graph(Mock(), execution_id="inner-loop")
            model_guard, tool_guard = create_agent.call_args.kwargs["middleware"]
            with self.assertRaises(AgentRunCancelled):
                model_guard.before_model({}, None)
            handler = Mock()
            with self.assertRaises(AgentRunCancelled):
                tool_guard.wrap_tool_call(Mock(), handler)
            handler.assert_not_called()

    def test_status_is_read_only_even_with_finalized_checkpoint(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(
                os.environ, {"AGENT_CHECKPOINT_DB": os.path.join(directory, "agent.db")}
            ),
            patch(
                "app.agents.service.checkpoint_values",
                return_value={
                    "plan": ["一步"],
                    "current_step": 1,
                    "answer": "尚未提交的答案",
                },
            ),
        ):
            service = AgentTaskService()
            service._create_run(
                "00000000-0000-0000-0000-000000000001", "任务", ["一步"]
            )
            with open_checkpointer() as checkpointer:
                before = get_run(checkpointer, "00000000-0000-0000-0000-000000000001")
            for _ in range(2):
                body, status = service.get_status(
                    "00000000-0000-0000-0000-000000000001"
                )
                self.assertEqual(200, status)
                self.assertEqual("running", body["data"]["status"])
                self.assertEqual("", body["data"]["answer"])
            with open_checkpointer() as checkpointer:
                after = get_run(checkpointer, "00000000-0000-0000-0000-000000000001")
            before.pop("duration_ms")
            after.pop("duration_ms")
            self.assertEqual(before, after)

    def test_finalizer_cancel_does_not_publish_answer_or_revive_cancelled_run(self):
        service = AgentTaskService()

        def build_graph(_llm, checkpointer, execution_id=None):
            class CancellingFinalizer:
                def invoke(self, messages):
                    request_run_cancel(checkpointer, execution_id)
                    return AIMessage(content="取消后返回的答案")

            return compile_graph(
                None, FakeExecutor(), CancellingFinalizer(), checkpointer, execution_id
            )

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(
                os.environ, {"AGENT_CHECKPOINT_DB": os.path.join(directory, "agent.db")}
            ),
            patch("app.agents.execution.build_graph", side_effect=build_graph),
        ):
            body, status = service.execute("任务", ["一步"], llm=object())
            execution_id = body["data"]["execution_id"]
            self.assertEqual(200, status)
            self.assertEqual("cancelled", body["data"]["status"])
            progress, _ = service.get_status(execution_id)
            self.assertEqual("cancelled", progress["data"]["status"])
            self.assertEqual("", progress["data"]["answer"])
            self.assertEqual(409, service.resume(execution_id)[1])

    def test_completed_run_cannot_be_rewritten_as_cancelled(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(
                os.environ, {"AGENT_CHECKPOINT_DB": os.path.join(directory, "agent.db")}
            ),
        ):
            AgentTaskService()._create_run(
                "00000000-0000-0000-0000-000000000002", "任务", ["一步"]
            )
            state = {
                "task": "任务",
                "plan": ["一步"],
                "current_step": 1,
                "answer": "答案",
            }
            with open_checkpointer() as checkpointer:
                first = persist_completed(
                    checkpointer, "00000000-0000-0000-0000-000000000002", state
                )
                second = persist_completed(
                    checkpointer, "00000000-0000-0000-0000-000000000002", state
                )
            self.assertEqual(first, second)
            self.assertEqual(
                first, _save_cancelled_result("00000000-0000-0000-0000-000000000002")
            )
            self.assertEqual(
                409,
                AgentTaskService().cancel("00000000-0000-0000-0000-000000000002")[1],
            )

    def test_resume_persists_conversation_once_and_repairs_missing_exchange(self):
        executor = FailOnceExecutor()

        def build_graph(_llm, checkpointer, execution_id=None):
            return compile_graph(
                None, executor, FakeFinalizer(), checkpointer, execution_id
            )

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(
                os.environ,
                {
                    "AGENT_CHECKPOINT_DB": os.path.join(directory, "agent.db"),
                    "CONVERSATION_DB": os.path.join(directory, "conversations.db"),
                },
            ),
            patch("app.agents.execution.build_graph", side_effect=build_graph),
            patch("app.agents.service.build_graph", side_effect=build_graph),
            patch("app.agents.service.LLM", return_value=object()),
            patch("app.agents.execution.logger.exception"),
        ):
            service = AgentTaskService()
            conversation_id = create_conversation()["id"]
            execution_id, _ = service.create_routed_run("原始问题", conversation_id)
            _, status = service.run_existing(
                execution_id, "改写问题", ["一步"], llm=object()
            )
            self.assertEqual(502, status)
            with (
                patch(
                    "app.agents.execution.append_exchange",
                    side_effect=RuntimeError("写入失败"),
                ),
                patch("app.agents.service.logger.exception"),
            ):
                self.assertEqual(200, service.resume(execution_id)[1])
            with open_checkpointer() as checkpointer:
                self.assertEqual(
                    "completed", get_run(checkpointer, execution_id)["status"]
                )
            for _ in range(2):
                body, status = service.resume(execution_id)
                self.assertEqual(200, status)
                self.assertEqual(conversation_id, body["data"]["conversation_id"])
            messages = get_conversation(conversation_id)["messages"]
            self.assertEqual(
                ["原始问题", "新旧制度差异已整理。"], [m["content"] for m in messages]
            )
            self.assertEqual(execution_id, messages[-1]["result"]["execution_id"])

    def test_recovery_only_runs_explicitly_and_submission_failure_is_compensated(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(
                os.environ, {"AGENT_CHECKPOINT_DB": os.path.join(directory, "agent.db")}
            ),
        ):
            service = AgentTaskService()
            service._create_run(
                "00000000-0000-0000-0000-000000000003", "任务", ["一步"]
            )
            self.assertEqual(
                "running",
                AgentTaskService().get_status("00000000-0000-0000-0000-000000000003")[
                    0
                ]["data"]["status"],
            )
            self.assertEqual(1, recover_interrupted_runs())
            progress = service.get_status("00000000-0000-0000-0000-000000000003")[0][
                "data"
            ]
            self.assertEqual("failed", progress["status"])
            self.assertEqual("agent_interrupted", progress["error_code"])
            self.assertEqual(0, recover_interrupted_runs())
            with (
                patch(
                    "app.agents.service.submit_task",
                    side_effect=RuntimeError("提交失败"),
                ),
                patch("app.agents.service.logger.exception"),
            ):
                with patch(
                    "app.agents.service.uuid4",
                    return_value="00000000-0000-0000-0000-000000000004",
                ):
                    self.assertEqual(500, service.start("任务", ["一步"])[1])
            progress = service.get_status("00000000-0000-0000-0000-000000000004")[0][
                "data"
            ]
            self.assertEqual("failed", progress["status"])
            self.assertEqual("agent_submission_failed", progress["error_code"])

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
        self.assertEqual(7, len(TaskPlan(steps=["步骤"] * 7).steps))
        self.assertEqual(7, len(_validate_plan(["步骤"] * 7)))


if __name__ == "__main__":
    unittest.main()
