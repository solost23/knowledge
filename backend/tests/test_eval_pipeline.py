import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain_core.documents import Document

from app.infrastructure.vector_store import ChromaDB
from app.knowledge.rag import QuestionService
from app.prompts.rag import RAG_PROMPT
from evals import agent
from evals import run as rag


class EvaluationPipelineTestCase(unittest.TestCase):
    def test_default_inputs_exist_and_resolve_from_script_directory(self):
        args = rag.build_parser().parse_args([])
        self.assertEqual(Path(rag.__file__).resolve().parent / "corpus", args.corpus)
        self.assertTrue(args.corpus.is_dir())
        self.assertTrue(any(args.corpus.glob("*.md")))
        self.assertTrue(rag.load_questions(args.questions))
        agent_args = agent.build_parser().parse_args([])
        self.assertEqual(
            Path(agent.__file__).resolve().parent / "agent_questions.jsonl",
            agent_args.questions,
        )
        self.assertTrue(agent.load_questions(agent_args.questions))

    def test_parser_uses_live_production_config_and_allows_overrides(self):
        with (
            patch.dict(rag.config.documents, chunk_size=240, chunk_overlap=24),
            patch.dict(rag.config.chroma, top_k=7, relevance_threshold=0.62),
        ):
            args = rag.build_parser().parse_args([])
            self.assertEqual(
                (240, 24, 7, 0.62),
                (
                    args.chunk_size,
                    args.chunk_overlap,
                    args.k,
                    args.relevance_threshold,
                ),
            )
            args = rag.build_parser().parse_args(
                [
                    "--chunk-size",
                    "120",
                    "--chunk-overlap",
                    "0",
                    "--k",
                    "2",
                    "--relevance-threshold",
                    "0",
                ]
            )
            self.assertEqual(
                (120, 0, 2, 0.0),
                (
                    args.chunk_size,
                    args.chunk_overlap,
                    args.k,
                    args.relevance_threshold,
                ),
            )

    def test_build_index_uses_loader_metadata_filter_and_configured_splitter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "corpus"
            corpus.mkdir()
            (corpus / "note.md").write_text("# 原始内容", encoding="utf-8")
            (corpus / "table.csv").write_text("a,b", encoding="utf-8")
            (corpus / "ignored.bin").write_bytes(b"skip")
            (corpus / "directory.md").mkdir()
            loaded = [
                [
                    Document(
                        page_content="解析后的段落", metadata={"page": 1, "nested": []}
                    ),
                    Document(page_content="同一文件的第二段", metadata={"page": 2}),
                ],
                [Document(page_content="解析后的表格", metadata={"row": 2})],
            ]
            with (
                patch.object(rag, "load_file", side_effect=loaded) as loader,
                patch.object(rag, "FastEmbeddings") as embeddings,
                patch.object(rag, "Chroma") as chroma,
                patch.object(
                    rag,
                    "RecursiveCharacterTextSplitter",
                    wraps=rag.RecursiveCharacterTextSplitter,
                ) as splitter,
                patch.dict(rag.config.documents, chunk_size=240, chunk_overlap=24),
            ):
                store, emb, count, chunks = rag.build_index(corpus, root / "index")
            self.assertEqual(
                [corpus / "note.md", corpus / "table.csv"],
                [call.args[0] for call in loader.call_args_list],
            )
            embeddings.assert_called_once_with()
            splitter.assert_called_once_with(chunk_size=240, chunk_overlap=24)
            self.assertIs(store, chroma.return_value)
            self.assertIs(emb, embeddings.return_value)
            self.assertEqual((2, 3), (count, chunks))
            documents = store.add_documents.call_args.args[0]
            self.assertEqual(
                ["解析后的段落", "同一文件的第二段", "解析后的表格"],
                [doc.page_content for doc in documents],
            )
            self.assertEqual({"page": 1, "source": "note.md"}, documents[0].metadata)
            self.assertEqual({"page": 2, "source": "note.md"}, documents[1].metadata)
            self.assertEqual({"row": 2, "source": "table.csv"}, documents[2].metadata)
            self.assertEqual(
                str(root / "index"), chroma.call_args.kwargs["persist_directory"]
            )

    def test_explicit_splitter_parameters_override_production_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch.object(rag, "FastEmbeddings"),
                patch.object(rag, "Chroma"),
                patch.object(rag, "RecursiveCharacterTextSplitter") as splitter,
            ):
                rag.build_index(root, root / "index", 120, 0)
            splitter.assert_called_once_with(chunk_size=120, chunk_overlap=0)

    def test_retrieval_matches_production_relevance_contract(self):
        with (
            patch("app.infrastructure.vector_store.Chroma"),
            patch("app.infrastructure.vector_store.FastEmbeddings"),
        ):
            production = ChromaDB()
        store = Mock()
        rag.retrieve(store, "测试问题")
        store.similarity_search_with_relevance_scores.assert_called_once_with(
            "测试问题",
            k=production.top_k,
            score_threshold=production.relevance_threshold,
        )
        store.similarity_search_with_score.assert_not_called()
        rag.retrieve(store, "实验问题", k=2, threshold=0.0)
        store.similarity_search_with_relevance_scores.assert_called_with(
            "实验问题",
            k=2,
            score_threshold=0.0,
        )

    def test_no_evidence_matches_production_and_never_invokes_llm(self):
        llm = Mock()
        with patch("app.knowledge.rag.get_chroma") as get_chroma:
            get_chroma.return_value.search.return_value = []
            result, status = QuestionService().question("未知问题", llm=llm)
        answer, elapsed, tokens = rag.answer_question(llm, "未知问题", [])
        self.assertEqual(200, status)
        self.assertEqual(result["data"]["answer"], answer)
        self.assertEqual((0.0, 0), (elapsed, tokens))
        llm.invoke.assert_not_called()

    def test_answer_uses_production_prompt_and_reports_token_usage(self):
        llm = Mock()
        llm.invoke.return_value = SimpleNamespace(
            content="答案", usage_metadata={"total_tokens": 12}
        )
        docs = [Document(page_content="证据", metadata={"source": "note.md"})]
        answer, _, tokens = rag.answer_question(llm, "问题", docs)
        self.assertEqual(("答案", 12), (answer, tokens))
        expected = RAG_PROMPT.invoke(
            {
                "context": "[1] 来源：note.md\n证据",
                "question": "问题",
                "history": [],
            }
        )
        self.assertEqual(expected, llm.invoke.call_args.args[0])

    def test_main_abstains_without_constructing_model_and_keeps_old_report(self):
        case = {
            "id": "none",
            "question": "未知问题",
            "category": "未知",
            "gold_doc": "__none__",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = Mock()
            store.similarity_search_with_relevance_scores.return_value = []
            with (
                patch.object(rag, "EVAL_DIR", root),
                patch.object(
                    rag, "build_index", return_value=(store, None, 1, 1)
                ) as build,
                patch.object(rag, "load_questions", return_value=[case]),
                patch.object(rag, "build_llm") as model,
                patch("sys.argv", ["evals/run.py", "--tag", "offline"]),
                patch("builtins.print"),
            ):
                rag.main()
                model.assert_not_called()
                self.assertEqual(root / ".chroma" / "offline", build.call_args.args[1])
                report = root / "reports" / "offline.json"
                original = report.read_text(encoding="utf-8")
                result = json.loads(original)
                self.assertTrue(result["rows"][0]["abstained"])
                self.assertEqual(0, result["rows"][0]["tokens"])
                self.assertEqual(1.0, result["summary"]["abstention_rate"])
                with self.assertRaises(FileExistsError):
                    rag.main()
                self.assertEqual(original, report.read_text(encoding="utf-8"))

    def test_existing_index_is_not_deleted_or_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "keep.txt"
            marker.write_text("保留", encoding="utf-8")
            with patch.object(rag, "FastEmbeddings") as embeddings:
                with self.assertRaises(FileExistsError):
                    rag.build_index(root, root)
            embeddings.assert_not_called()
            self.assertEqual("保留", marker.read_text(encoding="utf-8"))
