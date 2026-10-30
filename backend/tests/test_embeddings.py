import unittest
from unittest.mock import Mock

from app.infrastructure.embeddings import FastEmbeddings


class EmbeddingBatchTestCase(unittest.TestCase):
    def test_configured_batch_size_is_passed_to_fastembed(self):
        vector = Mock()
        vector.tolist.return_value = [0.1, 0.2]
        model = Mock()
        model.embed.return_value = [vector]
        embeddings = FastEmbeddings.__new__(FastEmbeddings)
        embeddings.model = model
        embeddings.batch_size = 128

        result = embeddings.embed_documents(["sample text"])

        self.assertEqual([[0.1, 0.2]], result)
        model.embed.assert_called_once_with(["sample text"], batch_size=128)


if __name__ == "__main__":
    unittest.main()
