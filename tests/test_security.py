import asyncio
import os
import socket
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException
from pydantic import ValidationError

from app import (
    MAX_BODY_BYTES,
    MAX_CLASSIFY_CHARS,
    MAX_MESSAGES,
    AnalyzeRequest,
    ClassifyRequest,
    LimitRequestBodyMiddleware,
    SecurityHeadersMiddleware,
    SuggestionMessage,
    analyze_suggestions,
    classify_intent,
    docs_enabled,
    enforce_api_key,
)
from intent import post_duckling, validated_duckling_url
from suggestions import duckling_parse, get_zone, parse_hhmm, sanitize_timezone


ROOT = Path(__file__).resolve().parents[1]


class DucklingUrlTests(unittest.TestCase):
    def test_localhost_and_private_hosts_are_allowed(self):
        self.assertEqual(
            "http://127.0.0.1:8000/parse",
            validated_duckling_url("http://127.0.0.1:8000/parse"),
        )
        self.assertEqual(
            "http://10.1.2.3:8000/parse",
            validated_duckling_url("http://10.1.2.3:8000/parse"),
        )
    def test_hostnames_must_resolve_to_a_private_address(self):
        private = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.1.2.3", 0))]
        with patch("intent.socket.getaddrinfo", return_value=private):
            self.assertEqual(
                "https://duckling.internal/parse",
                validated_duckling_url("https://duckling.internal/parse"),
            )

        with patch("intent.socket.getaddrinfo", side_effect=socket.gaierror):
            with self.assertRaises(ValueError):
                validated_duckling_url("https://duckling.internal/parse")

        metadata = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", 0))]
        for url in (
            "http://2852039166/latest/meta-data/",
            "http://0xa9fea9fe/latest/meta-data/",
            "http://169.254.169.254.nip.io/latest/meta-data/",
        ):
            with self.subTest(url=url):
                with patch("intent.socket.getaddrinfo", return_value=metadata):
                    with self.assertRaises(ValueError):
                        validated_duckling_url(url)

        public = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 0))]
        with patch("intent.socket.getaddrinfo", return_value=public):
            with self.assertRaises(ValueError):
                validated_duckling_url("https://example.com/parse")

    def test_metadata_and_link_local_targets_are_rejected(self):
        refused = (
            "http://169.254.169.254/latest/meta-data/",
            "http://169.254.1.1:8000/parse",
            "http://[fd00:ec2::254]/latest/meta-data/",
            "http://[::ffff:169.254.169.254]/latest/meta-data/",
            "http://0.0.0.0:8000/parse",
            "http://8.8.8.8:8000/parse",
            "http://user:pass@10.0.0.5:8000/parse",
            "file:///etc/passwd",
            "http://metadata.google.internal/computeMetadata/v1/",
            "http://127.0.0.1:8000/parse\nGET /",
            "gopher://127.0.0.1:8000/",
        )
        for url in refused:
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    validated_duckling_url(url)

    def test_post_does_not_follow_redirects(self):
        with patch("intent.requests.post") as post:
            post_duckling({"text": "tomorrow"}, timeout=3, url="http://10.0.0.8:8000/parse")
        self.assertFalse(post.call_args.kwargs["allow_redirects"])
        self.assertEqual("http://10.0.0.8:8000/parse", post.call_args.args[0])


class TimezoneAndClockTests(unittest.TestCase):
    def test_path_like_timezones_are_dropped(self):
        for name in (
            "../../etc/passwd",
            "/etc/passwd",
            "..",
            "America/Toronto/../../etc",
            "America/Tor\nonto",
            "not a zone",
            "",
            None,
        ):
            with self.subTest(name=name):
                self.assertIsNone(sanitize_timezone(name))

    def test_real_iana_names_pass(self):
        self.assertEqual("America/Toronto", sanitize_timezone(" America/Toronto "))
        self.assertEqual("UTC", sanitize_timezone("UTC"))

    def test_zoneinfo_is_not_called_for_a_path(self):
        with patch("suggestions.ZoneInfo") as zone:
            found = get_zone("../../usr/share/zoneinfo/UTC")
        zone.assert_not_called()
        self.assertEqual(found.tzname(None), "UTC")

    def test_duckling_does_not_receive_an_unsafe_timezone(self):
        with patch("suggestions.post_duckling") as post:
            post.return_value.raise_for_status.return_value = None
            post.return_value.json.return_value = []
            duckling_parse("tomorrow", tz="../../etc/passwd")
        self.assertNotIn("tz", post.call_args.args[0])

    def test_clock_values_stay_in_range(self):
        self.assertEqual((9, 0), parse_hhmm("09:00", 1, 2))
        self.assertEqual((1, 2), parse_hhmm("99:00", 1, 2))
        self.assertEqual((1, 2), parse_hhmm("09:0000", 1, 2))
        self.assertEqual((1, 2), parse_hhmm("-1:00", 1, 2))


class ApiBoundaryTests(unittest.TestCase):
    def test_classify_text_is_capped(self):
        with self.assertRaises(ValidationError):
            ClassifyRequest(text="a" * (MAX_CLASSIFY_CHARS + 1))

    def test_message_batch_is_capped(self):
        message = {
            "speaker": "Ada",
            "timestamp": "2026-07-17T10:00:00-04:00",
            "message": "I'll send the proposal tomorrow.",
        }
        with self.assertRaises(ValidationError):
            AnalyzeRequest(messages=[message] * (MAX_MESSAGES + 1))

    def test_classify_errors_do_not_echo_the_exception(self):
        with patch("app.classify", side_effect=RuntimeError("secret path /opt/nlp")):
            with self.assertRaises(HTTPException) as ctx:
                classify_intent(ClassifyRequest(text="Remind me tomorrow"))
        self.assertEqual(500, ctx.exception.status_code)
        self.assertEqual("Intent classification failed", ctx.exception.detail)
        self.assertNotIn("secret", ctx.exception.detail)

    def test_analyze_errors_do_not_echo_the_exception(self):
        request = AnalyzeRequest(
            messages=[
                SuggestionMessage(
                    speaker="Ada",
                    timestamp="2026-07-17T10:00:00-04:00",
                    message="I'll send the proposal tomorrow.",
                )
            ]
        )
        with patch("app.analyze", side_effect=RuntimeError("boom /etc/passwd")):
            with self.assertRaises(HTTPException) as ctx:
                analyze_suggestions(request)
        self.assertEqual(500, ctx.exception.status_code)
        self.assertEqual("Suggestions analysis failed", ctx.exception.detail)
        self.assertNotIn("passwd", ctx.exception.detail)

    def test_api_key_is_optional_until_configured(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CLASSIFIER_API_KEY", None)
            enforce_api_key(None)

    def test_api_key_rejects_a_missing_or_wrong_token(self):
        with patch.dict(os.environ, {"CLASSIFIER_API_KEY": "correct-horse"}):
            for header in (None, "Bearer ", "Bearer wrong", "correct-horse", "Bearer correct-horse-extra"):
                with self.subTest(header=header):
                    with self.assertRaises(HTTPException) as ctx:
                        enforce_api_key(header)
                    self.assertEqual(401, ctx.exception.status_code)
                    self.assertEqual("Unauthorized", ctx.exception.detail)
            enforce_api_key("Bearer correct-horse")

    def test_chunked_body_over_the_limit_is_rejected(self):
        called = {}

        async def downstream(scope, receive, send):
            called["hit"] = True

        middleware = LimitRequestBodyMiddleware(downstream)

        async def receive():
            return {
                "type": "http.request",
                "body": b"x" * (MAX_BODY_BYTES + 1),
                "more_body": False,
            }

        messages = []

        async def send(message):
            messages.append(message)

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/intents/classify",
            "raw_path": b"/v1/intents/classify",
            "query_string": b"",
            "headers": [],
            "client": ("127.0.0.1", 1234),
            "server": ("127.0.0.1", 80),
        }
        asyncio.run(middleware(scope, receive, send))
        self.assertNotIn("hit", called)
        start = next(message for message in messages if message["type"] == "http.response.start")
        self.assertEqual(413, start["status"])

    def test_absurd_content_length_is_rejected_before_the_body_is_read(self):
        called = {}

        async def downstream(scope, receive, send):
            called["hit"] = True

        async def receive():
            called["recv"] = True
            return {"type": "http.request", "body": b"", "more_body": False}

        messages = []

        async def send(message):
            messages.append(message)

        scope = {
            "type": "http",
            "headers": [(b"content-length", b"9" * 40)],
        }
        middleware = LimitRequestBodyMiddleware(downstream)
        asyncio.run(middleware(scope, receive, send))
        self.assertNotIn("hit", called)
        self.assertNotIn("recv", called)
        start = next(message for message in messages if message["type"] == "http.response.start")
        self.assertEqual(400, start["status"])

    def test_oversized_declared_length_is_rejected(self):
        messages = []

        async def downstream(scope, receive, send):
            raise AssertionError("route should not run")

        async def receive():
            raise AssertionError("body should not be read")

        async def send(message):
            messages.append(message)

        scope = {"type": "http", "headers": [(b"content-length", b"9999999999")]}
        asyncio.run(LimitRequestBodyMiddleware(downstream)(scope, receive, send))
        start = next(message for message in messages if message["type"] == "http.response.start")
        self.assertEqual(413, start["status"])

    def test_limit_response_keeps_security_headers(self):
        messages = []

        async def receive():
            return {
                "type": "http.request",
                "body": b"x" * (MAX_BODY_BYTES + 1),
                "more_body": False,
            }

        async def send(message):
            messages.append(message)

        async def downstream(scope, receive, send):
            raise AssertionError("route should not run")

        app = SecurityHeadersMiddleware(LimitRequestBodyMiddleware(downstream))
        asyncio.run(app({"type": "http", "headers": []}, receive, send))
        start = next(message for message in messages if message["type"] == "http.response.start")
        self.assertEqual(413, start["status"])
        names = {name.lower() for name, _ in start["headers"]}
        self.assertIn(b"x-content-type-options", names)
        self.assertIn(b"cache-control", names)

    def test_client_disconnect_does_not_wait_for_another_chunk(self):
        async def receive():
            return {"type": "http.disconnect"}

        async def send(message):
            raise AssertionError("no response expected")

        async def downstream(scope, receive, send):
            raise AssertionError("route should not run")

        asyncio.run(
            LimitRequestBodyMiddleware(downstream)({"type": "http", "headers": []}, receive, send)
        )

    def test_deployed_environments_hide_docs(self):
        with patch.dict(os.environ, {"ENV": "production"}):
            self.assertFalse(docs_enabled())
        with patch.dict(os.environ, {"ENV": "staging"}):
            self.assertFalse(docs_enabled())
        with patch.dict(os.environ, {"ENV": ""}):
            self.assertTrue(docs_enabled())


class HostIpScriptTests(unittest.TestCase):
    def test_host_ip_file_values(self):
        script = ROOT / "start.sh"
        allowed = ("10.0.0.12", "192.168.1.9", "172.16.4.4", "172.31.255.254")
        refused = (
            "169.254.169.254",
            "169.254.1.1",
            "127.0.0.1",
            "0.0.0.0",
            "8.8.8.8",
            "1.1.1.1",
            "172.15.0.1",
            "172.32.0.1",
            "192.169.0.1",
            "11.0.0.1",
            "10.0.0.01",
            "10.0.0.256",
            "8.8.8.8.1",
            "10.0.0.5/latest/meta-data",
            "metadata",
            "10.0.0.5;curl",
            "",
        )
        for ip in allowed:
            with self.subTest(ip=ip):
                result = subprocess.run(
                    ["bash", str(script), "--check-host-ip", ip],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(0, result.returncode, result.stderr)
        for ip in refused:
            with self.subTest(ip=ip):
                result = subprocess.run(
                    ["bash", str(script), "--check-host-ip", ip],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(0, result.returncode)


if __name__ == "__main__":
    unittest.main()
