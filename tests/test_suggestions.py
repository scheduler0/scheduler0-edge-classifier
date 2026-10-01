import unittest
from unittest.mock import patch

from suggestions import (
    DUCKLING_ENGLISH_REGIONS,
    UnsupportedLocaleError,
    analyze,
    build_matching_form,
    is_english_locale,
    next_business_day,
    parse_timestamp,
    resolve_duckling_locale,
)


# Languages and region tags Duckling's Locale module ships. English is the only
# language this service accepts; every other language must be rejected before
# spaCy or Duckling run.
DUCKLING_LANGUAGES = [
    "AF", "AR", "BG", "BN", "CA", "CS", "DA", "DE", "EL", "EN", "ES", "ET",
    "FA", "FI", "FR", "GA", "HE", "HI", "HR", "HU", "ID", "IS", "IT", "JA",
    "KA", "KN", "KM", "KO", "LO", "ML", "MN", "MY", "NB", "NE", "NL", "PL",
    "PT", "RO", "RU", "SK", "SV", "SW", "TA", "TE", "TH", "TR", "UK", "VI",
    "ZH",
]

DUCKLING_REGIONS = {
    "AR": ["EG"],
    "EN": sorted(DUCKLING_ENGLISH_REGIONS),
    "ES": ["AR", "CL", "CO", "ES", "MX", "PE", "VE"],
    "NL": ["BE", "NL"],
    "ZH": ["CN", "HK", "MO", "TW"],
}

TOMORROW = [
    {
        "body": "tomorrow",
        "dim": "time",
        "value": {
            "type": "value",
            "value": "2026-07-18T00:00:00.000-04:00",
            "grain": "day",
        },
    }
]


def _request(messages, locale="en_CA", **options):
    payload = {
        "conversation_id": "conv_test",
        "messages": messages,
        "participants": [
            {"id": "user_123", "display_name": "Victor", "timezone": "America/Toronto"},
            {"id": "user_456", "display_name": "John", "timezone": "America/Toronto"},
        ],
        "options": {"default_timezone": "America/Toronto", "locale": locale},
    }
    payload["options"].update(options)
    return payload


def _msg(text, timestamp="2026-07-17T10:00:00-04:00", speaker="Victor", message_id=None):
    message = {"speaker": speaker, "timestamp": timestamp, "message": text}
    if message_id:
        message["id"] = message_id
    return message


class LocaleResolutionTests(unittest.TestCase):
    def test_every_duckling_english_region_canonicalizes(self):
        self.assertEqual(
            sorted(DUCKLING_ENGLISH_REGIONS),
            ["AU", "BZ", "CA", "GB", "IE", "IN", "JM", "NZ", "PH", "TT", "US", "ZA"],
        )
        for region in sorted(DUCKLING_ENGLISH_REGIONS):
            for raw in (
                f"en_{region}",
                f"en-{region}",
                f"en_{region.lower()}",
                f"EN_{region.lower()}",
                f"  en-{region.lower()}  ",
            ):
                with self.subTest(locale=raw):
                    self.assertTrue(is_english_locale(raw))
                    self.assertEqual(f"en_{region}", resolve_duckling_locale(raw))

    def test_english_without_a_known_region_falls_back_to_en_gb(self):
        for raw in (None, "", "   ", "en", "EN", "english", "en_XX", "en-US.UTF-8"):
            with self.subTest(locale=raw):
                self.assertTrue(is_english_locale(raw))
                self.assertEqual("en_GB", resolve_duckling_locale(raw))

    def test_every_non_english_duckling_language_is_rejected(self):
        for lang in DUCKLING_LANGUAGES:
            if lang == "EN":
                continue
            samples = [lang.lower(), f"{lang.lower()}_{lang}", f"{lang.lower()}-{lang}"]
            for region in DUCKLING_REGIONS.get(lang, []):
                samples.append(f"{lang.lower()}_{region}")
                samples.append(f"{lang.lower()}-{region}")
            for raw in samples:
                with self.subTest(locale=raw):
                    self.assertFalse(is_english_locale(raw))
                    with self.assertRaises(UnsupportedLocaleError) as ctx:
                        resolve_duckling_locale(raw)
                    self.assertEqual(raw, ctx.exception.locale)
                    self.assertIn(raw, str(ctx.exception))

    def test_whitespace_around_a_non_english_locale_is_preserved_on_the_error(self):
        raw = "  fr  "
        with self.assertRaises(UnsupportedLocaleError) as ctx:
            resolve_duckling_locale(raw)
        self.assertEqual(raw, ctx.exception.locale)

    def test_analyze_forwards_each_english_locale_to_duckling(self):
        for region in sorted(DUCKLING_ENGLISH_REGIONS):
            locale = f"en-{region.lower()}"
            with self.subTest(locale=locale):
                with patch("suggestions.duckling_parse", return_value=TOMORROW) as parsed:
                    result = analyze(
                        _request(
                            [_msg("I'll send you the proposal tomorrow.")],
                            locale=locale,
                        )
                    )
                self.assertEqual(f"en_{region}", parsed.call_args.kwargs["locale"])
                self.assertEqual("COMMITMENT", result["suggestions"][0]["type"])

    def test_analyze_rejects_every_non_english_language_before_parsing(self):
        for lang in DUCKLING_LANGUAGES:
            if lang == "EN":
                continue
            locale = lang.lower()
            with self.subTest(locale=locale):
                with patch("suggestions.duckling_parse") as parsed:
                    with patch("suggestions.nlp") as nlp:
                        with self.assertRaises(UnsupportedLocaleError):
                            analyze(
                                _request(
                                    [_msg("I'll send you the proposal tomorrow.")],
                                    locale=locale,
                                )
                            )
                parsed.assert_not_called()
                nlp.assert_not_called()

    def test_english_subtags_keep_a_known_region_or_fall_back(self):
        fallbacks = (
            "en_",
            "en_FR",
            "enn",
            "en.",
            "eng",
            "en-",
            "en_Latn_US",
            "en-001",
            "en_US.utf8",
            "en_GB.UTF-8",
            "en\u200b",
            "en\u00a0",
            "en\n",
            "en_US\u200b",
            "en-GB-u-ca-gregory",
        )
        for raw in fallbacks:
            with self.subTest(locale=raw):
                self.assertTrue(is_english_locale(raw))
                self.assertEqual("en_GB", resolve_duckling_locale(raw))

        kept = {
            "en_US\u00a0": "en_US",
            "en__US": "en_US",
            "EN_us_extra": "en_US",
            "en_US_POSIX": "en_US",
            "en-us-x-private": "en_US",
            "en_CA_x_private": "en_CA",
            "  EN_us  ": "en_US",
        }
        for raw, expected in kept.items():
            with self.subTest(locale=raw):
                self.assertTrue(is_english_locale(raw))
                self.assertEqual(expected, resolve_duckling_locale(raw))

    def test_script_posix_and_lookalike_locales_are_rejected(self):
        suffixes = ("-Latn", "-Hans", ".UTF-8", "-u-nu-latn", "@euro", "-419")
        for lang in DUCKLING_LANGUAGES:
            if lang == "EN":
                continue
            for suffix in suffixes:
                raw = f"{lang.lower()}{suffix}"
                with self.subTest(locale=raw):
                    self.assertFalse(is_english_locale(raw))
                    with self.assertRaises(UnsupportedLocaleError) as ctx:
                        resolve_duckling_locale(raw)
                    self.assertEqual(raw, ctx.exception.locale)

        for raw in ("zh-Hans-CN", "  FR  ", "еn", "еn_US"):
            with self.subTest(locale=raw):
                self.assertFalse(is_english_locale(raw))
                with self.assertRaises(UnsupportedLocaleError) as ctx:
                    resolve_duckling_locale(raw)
                self.assertEqual(raw, ctx.exception.locale)

    def test_analyze_rejects_script_subtags_before_parsing(self):
        for lang in DUCKLING_LANGUAGES:
            if lang == "EN":
                continue
            locale = f"{lang.lower()}-Latn"
            with self.subTest(locale=locale):
                with patch("suggestions.duckling_parse") as parsed:
                    with patch("suggestions.nlp") as nlp:
                        with self.assertRaises(UnsupportedLocaleError):
                            analyze(
                                _request(
                                    [_msg("I'll send you the proposal tomorrow.")],
                                    locale=locale,
                                )
                            )
                parsed.assert_not_called()
                nlp.assert_not_called()

    def test_padding_aliases_and_false_english_tags(self):
        # Known regions survive surrounding whitespace, including NBSP and
        # ideographic space. The first region tag wins when several are present.
        kept = {
            "\ten_US\t": "en_US",
            "\u00a0en_US\u00a0": "en_US",
            "\u3000en_CA\u3000": "en_CA",
            "en_US\n": "en_US",
            "eN-uS": "en_US",
            "en_us_gb": "en_US",
        }
        for raw, expected in kept.items():
            with self.subTest(locale=raw):
                self.assertTrue(is_english_locale(raw))
                self.assertEqual(expected, resolve_duckling_locale(raw))

        # UK is not a Duckling region (GB is). A trailing BOM, NUL, or mark
        # inside the tag is not a separator, so the region is not recognized.
        for raw in (
            "en_UK",
            "en-UK",
            "en_uk",
            "en_GBR",
            "en_USA",
            "en_US\ufeff",
            "en\u3000US",
            "english_US",
            "en___",
            "en\r_US",
            "en_US\x00",
            "en.US",
            "en@euro",
        ):
            with self.subTest(locale=raw):
                self.assertTrue(is_english_locale(raw))
                self.assertEqual("en_GB", resolve_duckling_locale(raw))

        for raw in ("\ufeffen", "\ufeffen_US", "\u00a0fr\u00a0", "fr\ufeff", "de\u200b", 123):
            with self.subTest(locale=raw):
                self.assertFalse(is_english_locale(raw))
                with self.assertRaises(UnsupportedLocaleError) as ctx:
                    resolve_duckling_locale(raw)
                self.assertEqual(raw, ctx.exception.locale)

    def test_analyze_rejects_a_bom_prefixed_english_tag_before_parsing(self):
        locale = "\ufeffen_US"
        with patch("suggestions.duckling_parse") as parsed:
            with patch("suggestions.nlp") as nlp:
                with self.assertRaises(UnsupportedLocaleError) as ctx:
                    analyze(_request([_msg("I'll send you the proposal tomorrow.")], locale=locale))
        self.assertEqual(locale, ctx.exception.locale)
        parsed.assert_not_called()
        nlp.assert_not_called()

    def test_analyze_maps_an_unknown_english_region_to_en_gb(self):
        with patch("suggestions.duckling_parse", return_value=TOMORROW) as parsed:
            result = analyze(
                _request(
                    [_msg("I'll send you the proposal tomorrow.")],
                    locale="\u3000en-uk\u3000",
                )
            )
        self.assertEqual("en_GB", parsed.call_args.kwargs["locale"])
        self.assertEqual("COMMITMENT", result["suggestions"][0]["type"])

    def test_format_marks_fullwidth_and_non_strings(self):
        # A mark inside the tag is not a separator. Fullwidth letters are not
        # the ASCII "en" prefix, and non-strings are not English either.
        kept = {
            "\t\ten_JM\t": "en_JM",
            "en_BZ_extra": "en_BZ",
            "en-ph": "en_PH",
            " EN_IN ": "en_IN",
        }
        for raw, expected in kept.items():
            with self.subTest(locale=raw):
                self.assertTrue(is_english_locale(raw))
                self.assertEqual(expected, resolve_duckling_locale(raw))

        for raw in (
            "en\u200b_US",
            "en_\u200bUS",
            "en_US/",
            "en_CA.",
            "en_US\u200bGB",
            "en+US",
            "en/US",
            "en＿US",
            "en_US\u0301",
            "en\ufeff_US",
            "en_US\u202c",
            "en\u200e",
        ):
            with self.subTest(locale=raw):
                self.assertTrue(is_english_locale(raw))
                self.assertEqual("en_GB", resolve_duckling_locale(raw))

        rejected = [
            "ｅｎ",
            "ｅｎ＿ＵＳ",
            "\u202aen_US\u202c",
            "\u202aen\u202c",
            True,
            False,
            0,
            1.5,
            [],
            {},
        ]
        for lang in DUCKLING_LANGUAGES:
            if lang == "EN":
                continue
            rejected.append(lang.lower() + "\u200e")
            rejected.append("\u202a" + lang.lower() + "\u202c")
        for raw in rejected:
            with self.subTest(locale=raw):
                self.assertFalse(is_english_locale(raw))
                with self.assertRaises(UnsupportedLocaleError) as ctx:
                    resolve_duckling_locale(raw)
                self.assertEqual(raw, ctx.exception.locale)

    def test_analyze_rejects_fullwidth_english_before_parsing(self):
        for locale in ("ｅｎ", "\u202aen_US\u202c", "fr\u200e"):
            with self.subTest(locale=locale):
                with patch("suggestions.duckling_parse") as parsed:
                    with patch("suggestions.nlp") as nlp:
                        with self.assertRaises(UnsupportedLocaleError) as ctx:
                            analyze(
                                _request(
                                    [_msg("I'll send you the proposal tomorrow.")],
                                    locale=locale,
                                )
                            )
                self.assertEqual(locale, ctx.exception.locale)
                parsed.assert_not_called()
                nlp.assert_not_called()

    def test_analyze_maps_a_marked_english_tag_to_en_gb(self):
        with patch("suggestions.duckling_parse", return_value=TOMORROW) as parsed:
            result = analyze(
                _request(
                    [_msg("I'll send you the proposal tomorrow.")],
                    locale="en\u200b_US",
                )
            )
        self.assertEqual("en_GB", parsed.call_args.kwargs["locale"])
        self.assertEqual("COMMITMENT", result["suggestions"][0]["type"])


class SuggestionEdgeCaseTests(unittest.TestCase):
    def analyze(self, request, entities=None):
        payload = entities if entities is not None else TOMORROW
        with patch("suggestions.duckling_parse", return_value=payload):
            return analyze(request)

    def test_smoke_scenarios(self):
        commitment = self.analyze(
            _request([_msg("I'll send you the proposal tomorrow.")])
        )
        suggestion = commitment["suggestions"][0]
        self.assertEqual("COMMITMENT", suggestion["type"])
        self.assertEqual("Victor", suggestion["actor"]["display_name"])
        self.assertEqual("proposal", suggestion["object"]["text"])
        self.assertFalse(suggestion["temporal"]["is_inferred"])
        self.assertEqual("tomorrow", suggestion["temporal"]["source_text"])
        self.assertEqual(0.95, suggestion["confidence"])

        request = self.analyze(
            _request([_msg("John, can you review the pull request by Friday?")])
        )
        suggestion = request["suggestions"][0]
        self.assertEqual("REQUEST", suggestion["type"])
        self.assertEqual("John", suggestion["actor"]["display_name"])
        self.assertEqual("user_456", suggestion["actor"]["id"])
        self.assertEqual("pull request", suggestion["object"]["text"])
        self.assertEqual("17:00:00", suggestion["temporal"]["due_at"].split("T")[1][:8])

        deferred = self.analyze(
            _request([_msg("Let's revisit pricing next week.")]),
            entities=[
                {
                    "body": "next week",
                    "dim": "time",
                    "value": {
                        "type": "interval",
                        "from": {"value": "2026-07-20T00:00:00.000-04:00", "grain": "week"},
                        "to": {"value": "2026-07-27T00:00:00.000-04:00", "grain": "week"},
                    },
                }
            ],
        )
        suggestion = deferred["suggestions"][0]
        self.assertEqual("CHECK_BACK", suggestion["type"])
        self.assertIsNone(suggestion["actor"])
        self.assertEqual("pricing", suggestion["object"]["text"])
        self.assertEqual("WEEK", suggestion["temporal"]["precision"])
        self.assertTrue(suggestion["temporal"]["due_at"].startswith("2026-07-20T09:00:00"))

        waiting = self.analyze(
            _request([_msg("I'm still waiting for the signed agreement.", speaker="John")]),
            entities=[],
        )
        suggestion = waiting["suggestions"][0]
        self.assertEqual("FOLLOW_UP", suggestion["type"])
        self.assertEqual("signed agreement", suggestion["object"]["text"])
        self.assertTrue(suggestion["temporal"]["is_inferred"])
        self.assertEqual("next_business_day", suggestion["temporal"]["inference_policy"])

    def test_unicode_apostrophe_commitment_matches_ascii(self):
        curly = self.analyze(_request([_msg("I\u2019ll send the proposal tomorrow.")]))
        straight = self.analyze(_request([_msg("I'll send the proposal tomorrow.")]))
        self.assertEqual(
            curly["suggestions"][0]["type"], straight["suggestions"][0]["type"]
        )
        self.assertEqual(
            curly["suggestions"][0]["object"]["text"],
            straight["suggestions"][0]["object"]["text"],
        )
        self.assertIn("i will", build_matching_form("I\u2019ll send it"))

    def test_state_changes_hide_open_obligations_unless_requested(self):
        messages = [
            _msg("I'll send the proposal tomorrow.", message_id="m1"),
            _msg("I sent the proposal.", timestamp="2026-07-17T12:00:00-04:00", message_id="m2"),
        ]
        hidden = self.analyze(_request(messages))
        self.assertEqual([], hidden["suggestions"])
        self.assertEqual([], hidden["obligations"])

        shown = self.analyze(_request(messages, include_resolved_obligations=True))
        self.assertEqual("COMPLETED", shown["suggestions"][0]["status"])
        self.assertEqual("COMPLETED", shown["obligations"][0]["status"])

        cancelled = self.analyze(
            _request(
                [
                    _msg("I'll send the proposal tomorrow."),
                    _msg(
                        "Don't worry about it anymore.",
                        speaker="John",
                        timestamp="2026-07-17T10:10:00-04:00",
                    ),
                ],
                include_resolved_obligations=True,
            )
        )
        self.assertEqual("CANCELLED", cancelled["obligations"][0]["status"])

    def test_out_of_order_messages_are_sorted_before_completion(self):
        result = self.analyze(
            _request(
                [
                    _msg(
                        "I sent the proposal.",
                        timestamp="2026-07-17T12:00:00-04:00",
                        message_id="later",
                    ),
                    _msg(
                        "I'll send the proposal tomorrow.",
                        timestamp="2026-07-17T10:00:00-04:00",
                        message_id="earlier",
                    ),
                ],
                include_resolved_obligations=True,
            )
        )
        self.assertEqual("COMPLETED", result["obligations"][0]["status"])

    def test_reschedule_updates_the_open_obligation_in_place(self):
        result = self.analyze(
            _request(
                [
                    _msg("I'll send the proposal tomorrow."),
                    _msg(
                        "Actually, I'll send it Friday instead.",
                        timestamp="2026-07-17T10:05:00-04:00",
                    ),
                ]
            ),
            entities=[
                {
                    "body": "Friday",
                    "dim": "time",
                    "value": {
                        "type": "value",
                        "value": "2026-07-24T00:00:00.000-04:00",
                        "grain": "day",
                    },
                }
            ],
        )
        self.assertEqual(1, len(result["suggestions"]))
        self.assertEqual("Friday", result["suggestions"][0]["temporal"]["source_text"])
        self.assertEqual(2, len(result["suggestions"][0]["evidence"]))

    def test_suppression_reasons(self):
        cases = [
            ("If I have time I will send the proposal tomorrow.", "hypothetical"),
            ("For example, I will send the proposal tomorrow.", "example_text"),
            ("> I will send the proposal tomorrow.", "quoted_text"),
            ("I haven't received the invoice.", "negated_action"),
            ("Don't let me forget to send the proposal tomorrow.", "negated_action"),
        ]
        for text, reason in cases:
            with self.subTest(text=text):
                result = self.analyze(_request([_msg(text)]))
                self.assertEqual([], result["suggestions"])
                self.assertEqual(reason, result["warnings"][0]["reason"])

    def test_uncertain_or_already_negated_commitments_are_ignored(self):
        # "i will" is not a commitment once spaCy marks the clause negated, so
        # the message never becomes a suggestion and there is nothing to suppress.
        for text in (
            "I might send the proposal tomorrow.",
            "I will not send the proposal tomorrow.",
        ):
            with self.subTest(text=text):
                result = self.analyze(_request([_msg(text)]))
                self.assertEqual([], result["suggestions"])
                self.assertEqual([], result["warnings"])

    def test_low_confidence_can_be_included(self):
        suppressed = self.analyze(
            _request([_msg("Let's revisit it.")], minimum_confidence=0.99),
            entities=[],
        )
        self.assertEqual([], suppressed["suggestions"])
        self.assertEqual(
            "below_confidence_threshold", suppressed["warnings"][0]["reason"]
        )

        included = self.analyze(
            _request(
                [_msg("Let's revisit it.")],
                minimum_confidence=0.99,
                include_low_confidence=True,
            ),
            entities=[],
        )
        self.assertEqual("CHECK_BACK", included["suggestions"][0]["type"])
        self.assertLess(included["suggestions"][0]["confidence"], 0.99)

    def test_duplicate_commitment_is_suppressed(self):
        result = self.analyze(
            _request(
                [
                    _msg("I'll send the proposal tomorrow.", message_id="a"),
                    _msg(
                        "I'll send the proposal tomorrow.",
                        timestamp="2026-07-17T10:02:00-04:00",
                        message_id="b",
                    ),
                ]
            )
        )
        self.assertEqual(1, len(result["suggestions"]))
        self.assertEqual("DUPLICATE_SUPPRESSED", result["warnings"][0]["code"])
        self.assertEqual("b", result["warnings"][0]["message_id"])

    def test_blank_message_is_skipped(self):
        result = self.analyze(_request([_msg("   ")]))
        self.assertEqual([], result["suggestions"])
        self.assertEqual([], result["warnings"])

    def test_group_request_without_a_name_stays_unresolved(self):
        request = _request([_msg("Can you review the pull request by Friday?")])
        request["participants"].append(
            {"id": "user_789", "display_name": "Ada", "timezone": "America/Toronto"}
        )
        result = self.analyze(request)
        suggestion = result["suggestions"][0]
        self.assertEqual("UNRESOLVED", suggestion["actor"]["display_name"])
        self.assertEqual(0.65, suggestion["confidence"])

    def test_direct_you_resolves_when_speaker_ids_match(self):
        request = _request(
            [
                _msg(
                    "Can you review the pull request by Friday?",
                    speaker={"id": "user_123", "display_name": "Victor"},
                )
            ]
        )
        result = self.analyze(request)
        self.assertEqual("John", result["suggestions"][0]["actor"]["display_name"])
        self.assertEqual("user_456", result["suggestions"][0]["actor"]["id"])

    def test_string_speaker_does_not_match_participant_ids(self):
        result = self.analyze(
            _request([_msg("Can you review the pull request by Friday?")])
        )
        self.assertEqual("UNRESOLVED", result["suggestions"][0]["actor"]["display_name"])

    def test_pronoun_object_uses_the_previous_obligation(self):
        result = self.analyze(
            _request(
                [
                    _msg("I'll send the proposal tomorrow."),
                    _msg(
                        "Can you review it by Friday?",
                        speaker="John",
                        timestamp="2026-07-17T10:01:00-04:00",
                    ),
                ]
            )
        )
        self.assertEqual("proposal", result["suggestions"][1]["object"]["text"])
        self.assertTrue(result["suggestions"][1]["object"]["resolved"])

    def test_ambiguous_cancellation_does_not_close_either_obligation(self):
        result = self.analyze(
            _request(
                [
                    _msg("I'll send the proposal tomorrow."),
                    _msg(
                        "I'll review the contract tomorrow.",
                        timestamp="2026-07-17T10:01:00-04:00",
                    ),
                    _msg("Never mind.", timestamp="2026-07-17T10:02:00-04:00"),
                ]
            )
        )
        self.assertEqual(2, len(result["suggestions"]))
        self.assertTrue(all(item["status"] == "OPEN" for item in result["obligations"]))

    def test_duration_entity_falls_back_to_next_business_day(self):
        result = self.analyze(
            _request([_msg("I'll send the proposal tomorrow.")]),
            entities=[
                {
                    "body": "2 hours",
                    "dim": "duration",
                    "value": {"type": "value", "value": 7200, "unit": "second"},
                }
            ],
        )
        temporal = result["suggestions"][0]["temporal"]
        self.assertTrue(temporal["is_inferred"])
        self.assertEqual("next_business_day", temporal["inference_policy"])
        self.assertTrue(temporal["due_at"].startswith("2026-07-20T09:00:00"))

    def test_deadline_interval_uses_the_end_boundary(self):
        result = self.analyze(
            _request([_msg("The report is due next week.")]),
            entities=[
                {
                    "body": "next week",
                    "dim": "time",
                    "value": {
                        "type": "interval",
                        "from": {"value": "2026-07-20T00:00:00.000-04:00", "grain": "week"},
                        "to": {"value": "2026-07-27T00:00:00.000-04:00", "grain": "week"},
                    },
                }
            ],
        )
        temporal = result["suggestions"][0]["temporal"]
        self.assertEqual("DEADLINE", result["suggestions"][0]["type"])
        self.assertTrue(temporal["due_at"].startswith("2026-07-27T17:00:00"))

    def test_explicit_hour_is_kept(self):
        result = self.analyze(
            _request([_msg("I'll send the proposal at 3pm.")]),
            entities=[
                {
                    "body": "3pm",
                    "dim": "time",
                    "value": {
                        "type": "value",
                        "value": "2026-07-17T15:00:00.000-04:00",
                        "grain": "hour",
                    },
                }
            ],
        )
        temporal = result["suggestions"][0]["temporal"]
        self.assertEqual("TIME", temporal["type"])
        self.assertEqual("HOUR", temporal["precision"])
        self.assertFalse(temporal["is_inferred"])
        self.assertTrue(temporal["due_at"].startswith("2026-07-17T15:00:00"))

    def test_later_is_inferred_three_hours_ahead(self):
        result = self.analyze(
            _request([_msg("I will send the proposal later.")]),
            entities=[],
        )
        temporal = result["suggestions"][0]["temporal"]
        self.assertEqual("later", temporal["source_text"])
        self.assertTrue(temporal["ambiguous"])
        self.assertEqual("later_today", temporal["inference_policy"])
        self.assertTrue(temporal["due_at"].startswith("2026-07-17T13:00:00"))

    def test_custom_due_time_and_invalid_timezone(self):
        result = self.analyze(
            _request(
                [
                    _msg(
                        "I will send the proposal.",
                        speaker={"id": "user_123", "display_name": "Victor", "timezone": "Not/AZone"},
                        timestamp="2026-07-17T10:00:00Z",
                    )
                ],
                default_due_time="11:30",
            ),
            entities=[],
        )
        temporal = result["suggestions"][0]["temporal"]
        self.assertIn("T11:30:00", temporal["due_at"])

    def test_minimal_shape_infers_the_participant(self):
        result = self.analyze(
            {
                "messages": [_msg("I'll send the proposal tomorrow.")],
                "options": {"locale": "en"},
            }
        )
        self.assertEqual("Victor", result["suggestions"][0]["actor"]["display_name"])
        self.assertEqual("COMMITMENT", result["suggestions"][0]["type"])

    def test_parser_failures_are_warnings(self):
        with patch("suggestions.nlp", side_effect=RuntimeError("spacy down")):
            with patch("suggestions.duckling_parse", return_value=[]):
                failed_nlp = analyze(
                    _request([_msg("I'll send the proposal tomorrow.")])
                )
        self.assertEqual([], failed_nlp["suggestions"])
        self.assertEqual("SPACY_PARSE_FAILED", failed_nlp["warnings"][0]["code"])

        with patch("suggestions.duckling_parse", side_effect=RuntimeError("duckling down")):
            failed_time = analyze(_request([_msg("I'll send the proposal tomorrow.")]))
        self.assertEqual("COMMITMENT", failed_time["suggestions"][0]["type"])
        self.assertTrue(failed_time["suggestions"][0]["temporal"]["is_inferred"])
        self.assertEqual("TEMPORAL_PARSE_FAILED", failed_time["warnings"][0]["code"])

    def test_invalid_timestamp_raises(self):
        with self.assertRaises(ValueError):
            self.analyze(_request([_msg("I'll send the proposal.", timestamp="not-a-date")]))

    def test_timestamp_and_business_day_edges(self):
        self.assertEqual(
            parse_timestamp("2026-07-17T10:00:00Z").tzinfo.utcoffset(None).total_seconds(),
            0,
        )
        saturday = parse_timestamp("2026-07-18T22:00:00-04:00")
        sunday = parse_timestamp("2026-07-19T22:00:00-04:00")
        friday = parse_timestamp("2026-07-17T22:00:00-04:00")
        self.assertEqual(0, next_business_day(saturday).weekday())
        self.assertEqual(0, next_business_day(sunday).weekday())
        self.assertEqual(0, next_business_day(friday).weekday())


if __name__ == "__main__":
    unittest.main()
