import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.core.settings import YamlConfig


class YamlConfigTestCase(unittest.TestCase):
    def test_nested_environment_override_preserves_scalar_type(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.yaml"
            config_path.write_text(
                "upload:\n  max_size_mb: 100\nchat_gpt:\n  model: example\n",
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {
                    "APP_CONFIG__UPLOAD__MAX_SIZE_MB": "32",
                    "APP_CONFIG__CHAT_GPT__MODEL": "custom-model",
                },
                clear=False,
            ):
                config = YamlConfig(config_path)

        self.assertEqual(32, config.upload["max_size_mb"])
        self.assertIsInstance(config.upload["max_size_mb"], int)
        self.assertEqual("custom-model", config.chat_gpt["model"])

    def test_environment_override_rejects_unknown_setting(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.yaml"
            config_path.write_text("upload:\n  max_size_mb: 100\n", encoding="utf-8")
            with patch.dict(
                os.environ,
                {
                    "APP_CONFIG__UPLOAD__UNKNOWN": "1",
                },
                clear=False,
            ):
                with self.assertRaisesRegex(ValueError, "不存在的配置项"):
                    YamlConfig(config_path)


if __name__ == "__main__":
    unittest.main()
