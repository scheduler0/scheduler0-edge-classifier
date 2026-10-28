import os
import unittest
from unittest.mock import patch

from fastapi import HTTPException
from pydantic import ValidationError

from app import (
    MAX_BODY_BYTES,
    MAX_MESSAGES,
    MAX_TEXT_CHARS,
    AnalyzeRequest,
    ClassifyRequest,
    SuggestionMessage,
    SuggestionOptions,
    analyze_suggestions,
    api_docs_urls,
    body_limit_status,
    classify_intent,
    healthz,
)


TIME_ENTITY = [
    {
        "body": "Monday",
        "dim": "time",
        "value": {
            "type": "value",
            "value": "2026-09-28T09:00:00.000-04:00",
            "grain": "hour",
        },
    }
]


class ApiTests(unittest.TestCase):
    def test_classify_endpoint_returns_the_engine_decision(self):
        with patch("intent.duckling_parse", return_value=TIME_ENTITY):
            result = classify_intent(ClassifyRequest(text="Remind me every Monday at 9am."))
        self.assertEqual("allow", result["decision"])
        self.assertEqual("Remind me every Monday at 9am.", result["text"])

    def test_classify_endpoint_accepts_empty_and_non_english_text(self):
        with patch("intent.duckling_parse", return_value=[]):
            empty = classify_intent(ClassifyRequest(text=""))
            french = classify_intent(ClassifyRequest(text="Rappelle-moi demain"))
        self.assertEqual("reject", empty["decision"])
        self.assertIn(french["decision"], {"allow", "clarify", "reject"})

    def test_analyze_endpoint_maps_unsupported_locales_to_400(self):
        for locale in (
            "fr",
            "es_MX",
            "zh-CN",
            "de_DE",
            "ja",
            "pt-BR",
            "ar_EG",
            "fr-Latn",
            "zh-Hans-CN",
            "de.UTF-8",
            "pt-419",
            "ja@euro",
            "ar-u-nu-latn",
            "  FR  ",
            "еn",
            "\ufeffen",
            "\ufeffen_US",
            "\u00a0fr\u00a0",
            "fr\ufeff",
            "\u200fen",
            "\u2061en",
            "ang",
            "fra",
            "cmn",
            "sgn-CH-DE",
            "no-bok",
            "zh-yue",
            "\ufffcen_US",
            "\u115fen",
        ):
            with self.subTest(locale=locale):
                request = AnalyzeRequest(
                    messages=[
                        SuggestionMessage(
                            speaker="Victor",
                            timestamp="2026-07-17T10:00:00-04:00",
                            message="I'll send the proposal tomorrow.",
                        )
                    ],
                    options=SuggestionOptions(locale=locale),
                )
                with patch("suggestions.duckling_parse") as parsed:
                    with self.assertRaises(HTTPException) as ctx:
                        analyze_suggestions(request)
                self.assertEqual(400, ctx.exception.status_code)
                self.assertEqual("UNSUPPORTED_LOCALE", ctx.exception.detail["code"])
                self.assertIn(locale, ctx.exception.detail["message"])
                parsed.assert_not_called()

    def test_analyze_endpoint_accepts_each_english_region_spelling(self):
        for locale in (
            "en_US",
            "en-gb",
            "en_IE",
            None,
            "en_UK",
            "\u3000en_IE\u3000",
            "en-US-u-nu-arab",
            "enm",
        ):
            with self.subTest(locale=locale):
                request = AnalyzeRequest(
                    messages=[
                        SuggestionMessage(
                            speaker="Victor",
                            timestamp="2026-07-17T10:00:00-04:00",
                            message="I'll send the proposal tomorrow.",
                        )
                    ],
                    options=SuggestionOptions(locale=locale),
                )
                with patch("suggestions.duckling_parse", return_value=TIME_ENTITY):
                    result = analyze_suggestions(request)
                self.assertEqual("COMMITMENT", result["suggestions"][0]["type"])
                self.assertEqual([], result["warnings"])

    def test_healthz_reports_ok_when_both_backends_respond(self):
        with patch("app.requests.post") as post:
            post.return_value.raise_for_status.return_value = None
            body = healthz()
        self.assertEqual({"status": "ok"}, body)
        self.assertEqual("en_GB", post.call_args.kwargs["data"]["locale"])
        self.assertFalse(post.call_args.kwargs["allow_redirects"])

    def test_healthz_fails_when_duckling_is_down(self):
        with patch("app.requests.post", side_effect=ConnectionError("refused")):
            with self.assertRaises(HTTPException) as ctx:
                healthz()
        self.assertEqual(503, ctx.exception.status_code)
        self.assertIn("Duckling", ctx.exception.detail)
        self.assertNotIn("refused", str(ctx.exception.detail))

    def test_healthz_fails_when_spacy_is_down(self):
        with patch("app.nlp", side_effect=RuntimeError("model missing")):
            with self.assertRaises(HTTPException) as ctx:
                healthz()
        self.assertEqual(503, ctx.exception.status_code)
        self.assertIn("spaCy", ctx.exception.detail)
        self.assertNotIn("model missing", str(ctx.exception.detail))

    def test_healthz_rejects_a_metadata_duckling_url(self):
        with patch.dict(os.environ, {"DUCKLING_URL": "http://169.254.169.254/latest/meta-data/"}):
            with patch("app.requests.post") as post:
                with self.assertRaises(HTTPException) as ctx:
                    healthz()
        self.assertEqual(503, ctx.exception.status_code)
        self.assertNotIn("169.254.169.254", str(ctx.exception.detail))
        post.assert_not_called()

    def test_classify_does_not_return_internal_error_text(self):
        with patch("app.classify", side_effect=RuntimeError("boom /opt/secret")):
            with self.assertRaises(HTTPException) as ctx:
                classify_intent(ClassifyRequest(text="hello"))
        self.assertEqual(500, ctx.exception.status_code)
        self.assertNotIn("secret", str(ctx.exception.detail))
        self.assertNotIn("boom", str(ctx.exception.detail))

    def test_analyze_does_not_return_internal_error_text(self):
        request = AnalyzeRequest(
            messages=[
                SuggestionMessage(
                    speaker="Victor",
                    timestamp="2026-07-17T10:00:00-04:00",
                    message="I'll send the proposal tomorrow.",
                )
            ]
        )
        with patch("app.analyze", side_effect=RuntimeError("parser blew up at /opt/secret")):
            with self.assertRaises(HTTPException) as ctx:
                analyze_suggestions(request)
        self.assertEqual(500, ctx.exception.status_code)
        self.assertNotIn("secret", str(ctx.exception.detail))

    def test_classify_rejects_oversized_text(self):
        with self.assertRaises(ValidationError):
            ClassifyRequest(text="a" * (MAX_TEXT_CHARS + 1))

    def test_analyze_rejects_too_many_messages(self):
        messages = [
            SuggestionMessage(
                speaker="Victor",
                timestamp="2026-07-17T10:00:00-04:00",
                message="hi",
            )
            for _ in range(MAX_MESSAGES + 1)
        ]
        with self.assertRaises(ValidationError):
            AnalyzeRequest(messages=messages)

    def test_body_limit_status(self):
        self.assertIsNone(body_limit_status(None, 0))
        self.assertIsNone(body_limit_status(str(MAX_BODY_BYTES), MAX_BODY_BYTES))
        self.assertEqual(413, body_limit_status(str(MAX_BODY_BYTES + 1), 0))
        self.assertEqual(413, body_limit_status(None, MAX_BODY_BYTES + 1))
        self.assertEqual(400, body_limit_status("nope", 0))
        self.assertEqual(400, body_limit_status("-1", 0))

    def test_docs_are_hidden_for_deployed_environments(self):
        self.assertEqual((None, None, None), api_docs_urls("production"))
        self.assertEqual((None, None, None), api_docs_urls(" staging "))
        self.assertEqual(("/docs", "/redoc", "/openapi.json"), api_docs_urls(""))
        self.assertEqual(("/docs", "/redoc", "/openapi.json"), api_docs_urls("local"))

    def test_duckling_url_rejects_metadata_and_non_http_targets(self):
        from intent import DucklingConfigError, safe_duckling_url

        blocked = (
            "http://169.254.169.254/latest/meta-data/",
            "http://169.254.170.2/",
            "http://metadata.google.internal/computeMetadata/v1/",
            "file:///etc/passwd",
            "http://user:pass@127.0.0.1:8000/parse",
            "gopher://127.0.0.1/",
            "http://2130706433/",
        )
        for url in blocked:
            with self.subTest(url=url):
                with patch.dict(os.environ, {"DUCKLING_URL": url}):
                    with self.assertRaises(DucklingConfigError):
                        safe_duckling_url()

    def test_duckling_url_allows_local_and_private_hosts(self):
        from intent import safe_duckling_url

        for url in (
            "http://127.0.0.1:8000/parse",
            "http://10.1.2.3:8000/parse",
            "http://duckling.internal:8000/parse",
        ):
            with self.subTest(url=url):
                with patch.dict(os.environ, {"DUCKLING_URL": url}):
                    self.assertEqual(url, safe_duckling_url())

    def test_duckling_post_does_not_follow_redirects(self):
        with patch("intent.requests.post") as post:
            post.return_value.raise_for_status.return_value = None
            post.return_value.json.return_value = []
            post.return_value.content = b"[]"
            from intent import duckling_parse

            self.assertEqual([], duckling_parse("tomorrow"))
        self.assertFalse(post.call_args.kwargs["allow_redirects"])

    def test_duckling_post_rejects_an_oversized_response(self):
        from intent import MAX_DUCKLING_RESPONSE_BYTES, DucklingConfigError, post_duckling

        with patch("intent.requests.post") as post:
            post.return_value.raise_for_status.return_value = None
            post.return_value.content = b"x" * (MAX_DUCKLING_RESPONSE_BYTES + 1)
            with self.assertRaises(DucklingConfigError):
                post_duckling({"text": "tomorrow"})


if __name__ == "__main__":
    unittest.main()
