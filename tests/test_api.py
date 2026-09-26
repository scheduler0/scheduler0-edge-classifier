import unittest
from unittest.mock import patch

from fastapi import HTTPException

from app import (
    AnalyzeRequest,
    ClassifyRequest,
    SuggestionMessage,
    SuggestionOptions,
    analyze_suggestions,
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
            "es-419",
            "zh-CN",
            "zh-Hans-CN",
            "de_DE",
            "ja",
            "ja_JP",
            "pt-BR",
            "ar_EG",
            "nb_NO",
            "ko_KR",
            "ka",
            "km",
            "lo",
            "my",
            "sw",
            "еn",
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
        for locale in ("en_US", "en-gb", "en_IE", None):
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

    def test_healthz_fails_when_duckling_is_down(self):
        with patch("app.requests.post", side_effect=ConnectionError("refused")):
            with self.assertRaises(HTTPException) as ctx:
                healthz()
        self.assertEqual(503, ctx.exception.status_code)
        self.assertIn("Duckling", ctx.exception.detail)

    def test_healthz_fails_when_spacy_is_down(self):
        with patch("app.nlp", side_effect=RuntimeError("model missing")):
            with self.assertRaises(HTTPException) as ctx:
                healthz()
        self.assertEqual(503, ctx.exception.status_code)
        self.assertIn("spaCy", ctx.exception.detail)


if __name__ == "__main__":
    unittest.main()
