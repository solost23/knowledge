import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from uuid import uuid4

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.infrastructure import document_repository as repository
from app.infrastructure.database import Database
from app.infrastructure.vector_store import ChromaDB
from app.knowledge.documents import DocumentService


class DeterministicEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # 待发布片段刻意更接近查询，确保 top_k 测试能发现先截断后过滤的问题。
        return [
            [0.99, 0.01] if text.startswith("pending") else [0.8, 0.2] for text in texts
        ]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


class PublicationIntegrationTestCase(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        database = Database(str(Path(directory.name) / "knowledge.db"))
        self.addCleanup(database.engine.dispose)
        database_patch = patch.object(repository, "get_database", return_value=database)
        database_patch.start()
        self.addCleanup(database_patch.stop)

        model_guard = patch(
            "app.infrastructure.vector_store.FastEmbeddings",
            side_effect=AssertionError("集成测试禁止加载真实 embedding 模型"),
        )
        model_guard.start()
        self.addCleanup(model_guard.stop)
        client = chromadb.EphemeralClient(Settings(anonymized_telemetry=False))
        collection_name = f"publication-test-{uuid4().hex}"
        self.store = ChromaDB.__new__(ChromaDB)
        self.store.db = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=DeterministicEmbeddings(),
        )
        self.addCleanup(client.delete_collection, collection_name)
        self.store.top_k = 2
        self.store.relevance_threshold = 0.35
        chroma_patch = patch(
            "app.knowledge.documents.get_chroma", return_value=self.store
        )
        chroma_patch.start()
        self.addCleanup(chroma_patch.stop)
        self.service = DocumentService()

    def contents(self, k=None):
        return [document.page_content for document, _ in self.store.search("查询", k=k)]

    def seed_legacy(self, name="manual.md", text="legacy"):
        repository.insert_doc(name)
        self.store.db.add_texts([text], [{"source": name}], ids=[name])

    def publish(self, text, name="manual.md"):
        with patch(
            "app.knowledge.documents.load_file",
            return_value=[Document(page_content=text)],
        ):
            result, status = self.service.doc("unused.md", name)
        self.assertEqual(200, status)
        self.assertTrue(result["success"])

    def test_legacy_chunks_without_version_are_searchable(self):
        self.seed_legacy()
        self.assertEqual({"manual.md": None}, repository.visible_documents())
        self.assertEqual(["legacy"], self.contents())

    def test_new_chunks_are_hidden_until_publication(self):
        original_add = self.store.db.add_texts

        def add_and_search(*args, **kwargs):
            ids = original_add(*args, **kwargs)
            self.assertEqual(1, self.store.db._collection.count())
            self.assertEqual([], self.contents())
            return ids

        with patch.object(self.store.db, "add_texts", side_effect=add_and_search):
            self.publish("pending-new")
        self.assertEqual(["pending-new"], self.contents())

    def test_replacement_exposes_only_new_version(self):
        self.publish("old-version")
        old_version = repository.visible_documents()["manual.md"]
        original_add = self.store.db.add_texts

        def add_and_search(*args, **kwargs):
            ids = original_add(*args, **kwargs)
            self.assertEqual(2, self.store.db._collection.count())
            self.assertEqual(["old-version"], self.contents())
            return ids

        with patch.object(self.store.db, "add_texts", side_effect=add_and_search):
            self.publish("pending-replacement")
        self.assertNotEqual(old_version, repository.visible_documents()["manual.md"])
        self.assertEqual(["pending-replacement"], self.contents())
        self.assertEqual(["pending-replacement"], self.store.db.get()["documents"])

    def test_failed_old_chunk_cleanup_does_not_expose_old_version(self):
        self.seed_legacy()
        with patch.object(
            self.store.db, "delete", side_effect=RuntimeError("模拟旧片段清理失败")
        ) as cleanup:
            self.publish("new-version")
        cleanup.assert_called_once()
        self.assertCountEqual(
            ["legacy", "new-version"], self.store.db.get()["documents"]
        )
        self.assertEqual(["new-version"], self.contents())

    def test_deleted_document_is_hidden_even_when_cleanup_fails(self):
        self.publish("published")
        self.seed_legacy("survivor.md", "survivor")
        with patch.object(
            self.store.db, "delete", side_effect=RuntimeError("模拟删除清理失败")
        ) as cleanup:
            result, status = self.service.delete("manual.md")
        self.assertEqual(200, status)
        self.assertTrue(result["success"])
        cleanup.assert_called_once()
        self.assertCountEqual(
            ["published", "survivor"], self.store.db.get()["documents"]
        )
        self.assertEqual(["survivor"], self.contents())
        self.assertNotIn("manual.md", repository.visible_documents())

    def test_pending_chunks_do_not_consume_top_k(self):
        for index in range(3):
            self.seed_legacy(f"visible-{index}.md", f"visible-{index}")
        for index in range(12):
            name = f"visible-{index}.md" if index < 3 else f"new-{index}.md"
            version, _ = repository.begin_document(name)
            self.store.db.add_texts(
                [f"pending-{index}"],
                [
                    {
                        "source": name,
                        "document_version": version,
                        "publication_managed": True,
                    }
                ],
                ids=[f"pending-{index}"],
            )

        nearest = self.store.db.similarity_search("查询", k=self.store.top_k)
        self.assertTrue(all(doc.page_content.startswith("pending") for doc in nearest))
        self.assertEqual(2, len(self.contents()))
        self.assertTrue(all(text.startswith("visible") for text in self.contents()))
        self.assertEqual(1, len(self.contents(k=1)))
        self.assertCountEqual(
            ["visible-0", "visible-1", "visible-2"], self.contents(k=3)
        )


if __name__ == "__main__":
    unittest.main()
