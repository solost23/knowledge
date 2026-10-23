import unittest

from flask import request

from app.core.server import MAX_REQUEST_BYTES, Server


class ServerTestCase(unittest.TestCase):
    def test_oversized_request_returns_json_error(self):
        def register(app):
            app.add_url_rule("/upload", "upload", lambda: request.get_data(), methods=["POST"])

        server = Server(register)
        server.app.config["MAX_CONTENT_LENGTH"] = 8

        response = server.app.test_client().post("/upload", data=b"123456789")

        self.assertEqual(413, response.status_code)
        self.assertEqual("application/json", response.mimetype)
        self.assertFalse(response.json["success"])
        self.assertEqual(413, response.json["code"])
        self.assertIn("100 MiB", response.json["message"])

    def test_request_limit_is_100_mib(self):
        self.assertEqual(100 * 1024 * 1024, MAX_REQUEST_BYTES)


if __name__ == "__main__":
    unittest.main()
