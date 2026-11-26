import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from unittest.mock import patch

from langchain_core.documents import Document
from sqlalchemy import event

from app.infrastructure import document_repository as repository
from app.infrastructure import document_upload_repository as uploads
from app.infrastructure.database import get_database
from app.infrastructure.vector_store import ChromaDB
from app.knowledge.document_imports import DocumentImportService
from app.knowledge.documents import DocumentService


class DocServiceTestCase(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        env = patch.dict(
            os.environ, {"KNOWLEDGE_DB": str(Path(directory.name) / "test.db")}
        )
        env.start()
        self.addCleanup(env.stop)

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
            original_name = os.path.basename(file_path)
            with (
                patch("app.knowledge.documents.get_chroma") as get_chroma,
            ):
                add_texts = get_chroma.return_value.db.add_texts
                result, status_code = DocumentService().doc(file_path, original_name)
            self.assertEqual(0, result.get("code"))
            add_texts.assert_called_once()

    def test_doc_reports_batched_vector_progress(self):
        chunks = [
            Document(page_content=f"chunk {index}", metadata={}) for index in range(130)
        ]
        with (
            patch("app.knowledge.documents.update_upload_progress") as update_progress,
            patch(
                "app.knowledge.documents.load_file",
                return_value=[Document(page_content="text")],
            ),
            patch(
                "app.knowledge.documents.RecursiveCharacterTextSplitter.split_documents",
                return_value=chunks,
            ),
            patch("app.knowledge.documents.get_chroma") as get_chroma,
        ):
            add_texts = get_chroma.return_value.db.add_texts
            add_texts.side_effect = lambda texts, metadatas, ids: ids
            result, status_code = DocumentService().doc(
                "large.pdf",
                "large.pdf",
                upload_job_id="job-1",
            )

        progress = [call.args[1:] for call in update_progress.call_args_list]
        self.assertEqual(200, status_code)
        self.assertTrue(result["success"])
        self.assertEqual(1, add_texts.call_count)
        self.assertEqual(98, progress[-1][0])
        self.assertIn("130/130", progress[-2][1])
        self.assertEqual(
            "job-1", add_texts.call_args_list[0].kwargs["metadatas"][0]["upload_job_id"]
        )

    def test_vector_cleanup_failure_does_not_mask_original_write_error(self):
        write_error = RuntimeError("Chroma upsert failed")
        with (
            patch(
                "app.knowledge.documents.load_file",
                return_value=[Document(page_content="text")],
            ),
            patch("app.knowledge.documents.get_chroma") as get_chroma,
        ):
            get_chroma.return_value.db.add_texts.side_effect = write_error
            get_chroma.return_value.db.delete.side_effect = RuntimeError(
                "cleanup failed"
            )
            with self.assertRaisesRegex(RuntimeError, "Chroma upsert failed"):
                DocumentService().doc("broken.pdf", "broken.pdf")

    def test_list(self):
        with patch("app.knowledge.documents.list_docs", return_value=[]):
            result, status_code = DocumentService().list()
        self.assertEqual(0, result.get("code"))
        self.assertIsInstance(result.get("data"), list)

    def test_delete_not_found(self):
        with (
            patch("app.knowledge.documents.get_chroma") as get_chroma,
            patch("app.knowledge.documents.delete_doc", return_value=False),
        ):
            get_chroma.return_value.db.get.return_value = {"ids": []}
            result, status_code = DocumentService().delete("不存在的文件.md")
        self.assertEqual(404, result.get("code"))


class MemoryVectors:
    def __init__(self):
        self.chunks = {}
        self.fail_delete = False

    def add_texts(self, texts, metadatas, ids):
        for chunk_id, text, metadata in zip(ids, texts, metadatas, strict=True):
            self.chunks[chunk_id] = Document(page_content=text, metadata=metadata)

    def get(self, where):
        found = [
            (chunk_id, document)
            for chunk_id, document in self.chunks.items()
            if all(document.metadata.get(key) == value for key, value in where.items())
        ]
        return {
            "ids": [item[0] for item in found],
            "metadatas": [item[1].metadata for item in found],
        }

    def delete(self, ids):
        if self.fail_delete:
            raise RuntimeError("清理失败")
        for chunk_id in ids:
            self.chunks.pop(chunk_id, None)

    def similarity_search_with_relevance_scores(
        self, query, k, score_threshold, filter
    ):
        def matches(metadata, condition):
            if "$or" in condition:
                return any(matches(metadata, item) for item in condition["$or"])
            if "$and" in condition:
                return all(matches(metadata, item) for item in condition["$and"])
            key, operation = next(iter(condition.items()))
            if "$in" in operation:
                return metadata.get(key) in operation["$in"]
            return metadata.get(key) != operation["$ne"]

        return [
            (document, 0.9)
            for document in reversed(list(self.chunks.values()))
            if matches(document.metadata, filter)
        ][:k]


class DocumentPublicationTestCase(unittest.TestCase):
    def setUp(self):
        DocServiceTestCase.setUp(self)
        self.store = ChromaDB.__new__(ChromaDB)
        self.store.db = MemoryVectors()
        self.store.top_k = 1
        self.store.relevance_threshold = 0.35
        chroma = patch("app.knowledge.documents.get_chroma", return_value=self.store)
        chroma.start()
        self.addCleanup(chroma.stop)
        loader = patch(
            "app.knowledge.documents.load_file",
            return_value=[Document(page_content="新内容")],
        )
        loader.start()
        self.addCleanup(loader.stop)
        self.service = DocumentService()
        repository.insert_doc("manual.md")
        self.store.db.add_texts(["旧内容"], [{"source": "manual.md"}], ["legacy"])

    def contents(self):
        return [document.page_content for document, _ in self.store.search("内容")]

    def test_legacy_document_remains_visible_until_publication(self):
        self.assertEqual(["旧内容"], self.contents())
        original = self.store.db.add_texts

        def add_and_search(*args, **kwargs):
            original(*args, **kwargs)
            self.assertEqual(["旧内容"], self.contents())

        with patch.object(self.store.db, "add_texts", side_effect=add_and_search):
            self.service.doc("manual.md", "manual.md")
        self.assertEqual(["新内容"], self.contents())

    def test_search_filters_in_chroma_with_requested_k(self):
        with patch.object(
            self.store.db,
            "similarity_search_with_relevance_scores",
            wraps=self.store.db.similarity_search_with_relevance_scores,
        ) as search:
            self.store.search("内容", k=3)
            self.assertEqual(3, search.call_args.kwargs["k"])
            self.assertEqual(
                {
                    "$and": [
                        {"source": {"$in": ["manual.md"]}},
                        {"publication_managed": {"$ne": True}},
                    ]
                },
                search.call_args.kwargs["filter"],
            )
            self.service.doc("manual.md", "manual.md")
            self.store.search("内容")
            self.assertEqual(1, search.call_args.kwargs["k"])
            self.assertEqual(
                {
                    "document_version": {
                        "$in": list(repository.visible_documents().values())
                    }
                },
                search.call_args.kwargs["filter"],
            )

    def test_mixed_legacy_and_published_filter(self):
        self.service.doc("new.md", "new.md")
        self.assertEqual(2, len(self.store.search("内容", k=2)))
        self.store.db.add_texts(
            ["未发布"],
            [
                {
                    "source": "manual.md",
                    "document_version": "pending",
                    "publication_managed": True,
                }
            ],
            ["pending"],
        )
        self.assertEqual(2, len(self.store.search("内容", k=2)))
        self.assertNotIn("未发布", self.contents())

    def test_manual_version_without_marker_is_not_returned_as_legacy(self):
        self.store.db.add_texts(
            ["未发布"],
            [{"source": "manual.md", "document_version": "pending"}],
            ["pending"],
        )
        self.assertEqual([], self.contents())

    def test_publish_failure_and_cleanup_failure_preserve_old_version(self):
        self.store.db.fail_delete = True
        with patch(
            "app.knowledge.documents.publish_document",
            side_effect=RuntimeError("提交失败"),
        ):
            with self.assertRaisesRegex(RuntimeError, "提交失败"):
                self.service.doc("manual.md", "manual.md", "job-1")
        self.assertEqual(["旧内容"], self.contents())
        self.assertEqual(2, len(self.store.db.chunks))

    def test_sqlite_failure_rolls_back_publication_and_preserves_old_version(self):
        engine = get_database().engine

        def fail_record(
            connection, cursor, statement, parameters, context, executemany
        ):
            if statement.startswith("INSERT INTO docs"):
                raise RuntimeError("文档记录写入失败")

        event.listen(engine, "before_cursor_execute", fail_record)
        try:
            with self.assertRaisesRegex(RuntimeError, "文档记录写入失败"):
                self.service.doc("manual.md", "manual.md")
        finally:
            event.remove(engine, "before_cursor_execute", fail_record)
        self.assertEqual(["旧内容"], self.contents())
        self.assertEqual({"manual.md": None}, repository.visible_documents())

    def test_cleanup_failure_after_publish_keeps_new_version(self):
        self.store.db.fail_delete = True
        self.service.doc("manual.md", "manual.md", "job-1")
        self.assertEqual(["新内容"], self.contents())
        self.service.discard_upload_job("job-1", "manual.md")
        self.assertEqual(["新内容"], self.contents())

    def test_delete_failure_in_sqlite_preserves_vectors(self):
        with patch(
            "app.knowledge.documents.delete_doc",
            side_effect=RuntimeError("删除提交失败"),
        ):
            with self.assertRaisesRegex(RuntimeError, "删除提交失败"):
                self.service.delete("manual.md")
        self.assertEqual(["旧内容"], self.contents())

    def test_deleted_document_stays_hidden_after_vector_cleanup_failure(self):
        self.store.db.fail_delete = True
        self.service.delete("manual.md")
        self.assertEqual([], self.contents())
        self.assertEqual([], repository.list_docs())

    def test_delete_during_import_prevents_resurrection(self):
        writing, resume = Event(), Event()
        original = self.store.db.add_texts

        def paused_write(*args, **kwargs):
            writing.set()
            self.assertTrue(resume.wait(5))
            original(*args, **kwargs)

        with ThreadPoolExecutor(max_workers=1) as executor:
            with patch.object(self.store.db, "add_texts", side_effect=paused_write):
                future = executor.submit(self.service.doc, "manual.md", "manual.md")
                try:
                    self.assertTrue(writing.wait(5))
                    self.service.delete("manual.md")
                finally:
                    resume.set()
                with self.assertRaisesRegex(RuntimeError, "已删除"):
                    future.result(timeout=5)
        self.assertEqual([], self.contents())
        self.assertEqual([], repository.list_docs())

    def test_completed_status_write_failure_does_not_discard_published_version(self):
        service = DocumentImportService(self.service)
        uploads.create_upload_job("job-1", "manual.md")
        original = uploads.update_upload_job

        def fail_completed(job_id, **kwargs):
            if kwargs["status"] == "completed":
                raise RuntimeError("完成状态写入失败")
            original(job_id, **kwargs)

        with patch(
            "app.knowledge.document_imports.update_upload_job",
            side_effect=fail_completed,
        ):
            service._process("job-1", "manual.md", "/nonexistent/manual.md")
        self.assertEqual(["新内容"], self.contents())
        self.assertEqual("failed", uploads.get_upload_job("job-1")["status"])

    def test_restart_recovery_preserves_published_job(self):
        uploads.create_upload_job("job-1", "manual.md")
        self.service.doc("manual.md", "manual.md", "job-1")
        DocumentImportService(self.service)
        self.assertEqual(["新内容"], self.contents())

    def test_partial_vector_write_is_not_visible_after_cleanup_failure(self):
        original = self.store.db.add_texts
        self.store.db.fail_delete = True

        def partial_write(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError("部分向量写入失败")

        with patch.object(self.store.db, "add_texts", side_effect=partial_write):
            with self.assertRaisesRegex(RuntimeError, "部分向量写入失败"):
                self.service.doc("manual.md", "manual.md")
        self.assertEqual(["旧内容"], self.contents())

    def test_queued_import_cannot_publish_after_delete(self):
        publication = repository.begin_document("manual.md")
        self.service.delete("manual.md")
        with self.assertRaisesRegex(RuntimeError, "已删除"):
            self.service.doc("manual.md", "manual.md", publication=publication)
        self.assertEqual([], self.contents())

    def test_superseded_import_cannot_overwrite_latest(self):
        earlier = repository.begin_document("manual.md")
        self.service.doc("manual.md", "manual.md")
        with self.assertRaisesRegex(RuntimeError, "更新的导入"):
            self.service.doc("manual.md", "manual.md", publication=earlier)
        self.assertEqual(["新内容"], self.contents())


if __name__ == "__main__":
    unittest.main()
