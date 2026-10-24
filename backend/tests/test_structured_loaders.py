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

    def test_csv_and_tsv_keep_rows_and_columns(self):
        csv_content = self._load_content(".csv", "name,score\nAda,98\n")
        tsv_content = self._load_content(".tsv", "name\tscore\nAda\t98\n")
        self.assertEqual("name\tscore\nAda\t98", csv_content)
        self.assertEqual("name\tscore\nAda\t98", tsv_content)

    def test_json_is_pretty_printed_without_escaping_chinese(self):
        content = self._load_content(".json", json.dumps({"名称": "知序", "items": [1, 2]}))
        self.assertIn('"名称": "知序"', content)
        self.assertIn('"items": [', content)

    def test_html_drops_script_and_style_content(self):
        content = self._load_content(
            ".html",
            "<html><style>.hidden { color: red }</style><body><h1>标题</h1>正文<script>alert('x')</script></body></html>",
        )
        self.assertIn("标题", content)
        self.assertIn("正文", content)
        self.assertNotIn("hidden", content)
        self.assertNotIn("alert", content)

    def test_xml_preserves_element_labels_attributes_and_text(self):
        content = self._load_content(".xml", '<root version="1"><item>内容</item></root>')
        self.assertIn("root [version=1]", content)
        self.assertIn("item: 内容", content)


if __name__ == "__main__":
    unittest.main()
