import json
import tempfile
import unittest
from pathlib import Path

from app.knowledge.loaders.structured import load


class StructuredLoaderTestCase(unittest.TestCase):
    def _load_content(self, suffix: str, content: str) -> str:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / f"sample{suffix}"
            path.write_text(content, encoding="utf-8")
            return load(str(path), suffix)[0].page_content

    def test_text_and_code_files_are_read_as_text(self):
        content = self._load_content(".py", "def answer():\n    return 'ok'\n")
        self.assertIn("def answer", content)
        self.assertIn("return 'ok'", content)

    def test_csv_and_tsv_extract_cell_content(self):
        csv_content = self._load_content(".csv", "name,score\nAda,98\n")
        tsv_content = self._load_content(".tsv", "name\tscore\nAda\t98\n")
        for content in (csv_content, tsv_content):
            self.assertIn("name", content)
            self.assertIn("score", content)
            self.assertIn("Ada", content)
            self.assertIn("98", content)

    def test_json_is_loaded_without_jq_runtime_error(self):
        content = self._load_content(".json", json.dumps({"名称": "知序", "items": [1, 2]}))
        self.assertIn("名称", content)
        self.assertIn("知序", content)
        self.assertIn("items", content)
        self.assertIn("1", content)

    def test_html_extracts_heading_and_paragraph(self):
        content = self._load_content(
            ".html",
            "<html><style>.hidden { color: red }</style><body><h1>标题</h1><p>正文内容</p><script>alert('x')</script></body></html>",
        )
        self.assertIn("标题", content)
        self.assertIn("正文内容", content)
        self.assertTrue(content.strip())

    def test_xml_load_does_not_require_downloading_nlp_model(self):
        content = self._load_content(".xml", '<root version="1"><item>内容</item></root>')
        self.assertIn("内容", content)


if __name__ == "__main__":
    unittest.main()
