"""Exercise every language shipped by the installed NLP packages.

spaCy is the linguistic package. Duckling is the temporal package, and the
service only classifies English, so non-English text is checked here for a
stable decision rather than for a scheduling label.
"""

import pkgutil
import re
import unittest
from unittest.mock import patch

import spacy
import spacy.lang

from spacy.symbols import ORTH

from intent import classify, nlp as english_nlp, text_looks_french


# Tokenizer backends that are optional extras on top of the spaCy wheel.
# Vietnamese can run its built-in tokenizer with use_pyvi = false.
OPTIONAL_TOKENIZER_LANGS = {"ja", "ko", "th"}
NON_LANGUAGE_MODULES = {
    "char_classes",
    "lex_attrs",
    "norm_exceptions",
    "punctuation",
    "tokenizer_exceptions",
}

EDGE_STRINGS = [
    "",
    " ",
    "\n\t",
    "?",
    "...",
    "09:00",
    "2026-09-26",
    "don't",
    "l'heure",
    "a\u2014b",
    "hello \U0001f60a",
    "\u00a9",
    "\u00a0",
    "A" * 200,
]

# Format, bidi, and numeral characters beyond the short list above.
FORMAT_EDGE_STRINGS = [
    "\v",
    "\f",
    "\u0085",
    "\u00ad",
    "\u180e",
    "\u200b",
    "\u200c",
    "\u200d",
    "\u2060",
    "\u202aabc\u202c",
    "\u202f",
    "\u205f",
    "\u3000",
    "\ufeff",
    "\ufffd",
    "\ufffc",
    "\u0640",
    "\u0964",
    "\u061f",
    "\u1362",
    "½",
    "①",
    "٥",
    "٣",
    "１２３",
    "3rd",
    "1.5",
    "1,5",
    "IV",
    "https://example.com",
    "a.b+c@example.co.uk",
    "a\u2060b",
    "e\u0301",
]

# Characters every loaded tokenizer treats as a single space token.
SPACE_CHARS = [
    " ",
    "\n",
    "\t",
    "\v",
    "\f",
    "\u0085",
    "\u00a0",
    "\u2000",
    "\u2001",
    "\u2002",
    "\u2003",
    "\u2004",
    "\u2005",
    "\u2006",
    "\u2007",
    "\u2008",
    "\u2009",
    "\u200a",
    "\u2028",
    "\u2029",
    "\u202f",
    "\u205f",
    "\u3000",
]

# Single-character punctuation that stays one is_punct token.
PUNCT_CHARS = ["?", "!", "。", "¿", "،", "؟", "።", "।", "·"]

CURRENCY_CHARS = ["$", "€", "£", "¥", "₹", "₽", "₩", "₺", "₿", "¢"]

RTL_LANGUAGES = {"ar", "fa", "he", "ur"}
CASELESS_LETTER_LANGUAGES = {"am", "ti"}
CASELESS_UNSEGMENTED_LANGUAGES = {"ja", "ko", "zh"}

# Exception tables whose ORTH strings are not what the tokenizer emits.
ORTH_MISMATCH_LANGUAGES = {"bg", "el", "es", "fa", "id", "mk", "ms", "ru", "vi", "zh"}

# Samples whose letters or "9h" clock times match the French marker.
FRENCH_LOOKING_DUCKLING = {"af", "ca", "es", "fr", "hu", "pt", "ro", "vi"}

TIME_ENTITY = [
    {
        "body": "tomorrow",
        "dim": "time",
        "value": {
            "type": "value",
            "value": "2026-09-27T09:00:00.000Z",
            "grain": "hour",
        },
    }
]

# One short sample per Duckling language. These are not expected to classify
# as scheduling requests; they must not crash the English pipeline.
DUCKLING_LANGUAGE_SAMPLES = {
    "af": "Herinner my môre om 9vm",
    "ar": "ذكرني غداً الساعة 9",
    "bg": "Напомни ми утре в 9",
    "bn": "আমাকে আগামীকাল সকাল ৯টায় মনে করিয়ে দাও",
    "ca": "Recorda'm demà a les 9",
    "cs": "Připomeň mi to zítra v 9",
    "da": "Påmind mig i morgen klokken 9",
    "de": "Erinnere mich morgen um 9 Uhr",
    "el": "Θύμισέ μου αύριο στις 9",
    "en": "Remind me tomorrow at 9am",
    "es": "Recuérdame mañana a las 9",
    "et": "Tuleta mulle homme kell 9 meelde",
    "fa": "فردا ساعت ۹ به من یادآوری کن",
    "fi": "Muistuta minua huomenna klo 9",
    "fr": "Rappelle-moi demain à 9h",
    "ga": "Cuir i gcuimhne dom amárach ag 9",
    "he": "תזכיר לי מחר בשעה 9",
    "hi": "मुझे कल सुबह 9 बजे याद दिलाएं",
    "hr": "Podsjeti me sutra u 9",
    "hu": "Emlékeztess holnap 9-kor",
    "id": "Ingatkan saya besok jam 9",
    "is": "Minntu mig á morgun klukkan 9",
    "it": "Ricordami domani alle 9",
    "ja": "明日の9時にリマインドして",
    "ka": "მომაგონე ხვალ 9 საათზე",
    "km": "រំលឹកខ្ញុំនៅថ្ងៃស្អែកម៉ោង 9",
    "kn": "ನಾಳೆ 9 ಗಂಟೆಗೆ ನೆನಪಿಸು",
    "ko": "내일 오전 9시에 알려줘",
    "lo": "ເຕືອນຂ້ອຍມື້ອື່ນເວລາ 9 ໂມງ",
    "ml": "നാളെ 9 മണിക്ക് ഓർമ്മിപ്പിക്കുക",
    "mn": "Маргааш 9 цагт надад сануул",
    "my": "မနက်ဖြန် ၉ နာရီမှာ သတိပေးပါ",
    "nb": "Påminn meg i morgen klokka 9",
    "ne": "भोलि ९ बजे मलाई सम्झाउनु",
    "nl": "Herinner me morgen om 9 uur",
    "pl": "Przypomnij mi jutro o 9",
    "pt": "Lembre-me amanhã às 9h",
    "ro": "Amintește-mi mâine la 9",
    "ru": "Напомни мне завтра в 9",
    "sk": "Pripomeň mi to zajtra o 9",
    "sv": "Påminn mig i morgon klockan 9",
    "sw": "Nikumbushe kesho saa 9",
    "ta": "நாளை 9 மணிக்கு நினைவூட்டு",
    "te": "రేపు 9 గంటలకు గుర్తు చేయండి",
    "th": "เตือนฉันพรุ่งนี้ตอน 9 โมง",
    "tr": "Yarın saat 9'da hatırlat",
    "uk": "Нагадай мені завтра о 9",
    "vi": "Nhắc tôi vào 9 giờ ngày mai",
    "zh": "明天上午9点提醒我",
}


def spacy_language_codes():
    return sorted(
        module.name
        for module in pkgutil.iter_modules(spacy.lang.__path__)
        if not module.name.startswith("_") and module.name not in NON_LANGUAGE_MODULES
    )


def load_language(code):
    """Return a tokenizer for code, or None when an optional backend is missing."""
    if code == "vi":
        return spacy.blank("vi", config={"nlp": {"tokenizer": {"use_pyvi": False}}})
    if code in OPTIONAL_TOKENIZER_LANGS:
        try:
            return spacy.blank(code)
        except (ImportError, OSError):
            return None
    return spacy.blank(code)


class SpacyLanguageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.codes = spacy_language_codes()
        cls.pipelines = {code: load_language(code) for code in cls.codes}

    def test_language_inventory_covers_the_installed_package(self):
        self.assertIn("en", self.codes)
        self.assertIn("xx", self.codes)
        self.assertGreaterEqual(len(self.codes), 70)
        # Duckling languages are classified with the English model even when
        # spaCy does not ship a matching language class (Georgian, Khmer, Lao,
        # Mongolian, Burmese, Swahili).
        for code in ("en", "de", "fr", "es", "zh", "ja", "ar", "ru", "hi"):
            with self.subTest(code=code):
                self.assertIn(code, self.codes)

    def test_every_language_exposes_lexical_data(self):
        # Faroese, Norwegian Nynorsk, and the multilingual tokenizer ship an
        # empty stop list. Every other language class includes one.
        empty_stop_lists = {"fo", "nn", "xx"}
        for code in self.codes:
            with self.subTest(code=code):
                lang_class = spacy.util.get_lang_class(code)
                self.assertEqual(code, lang_class.lang)
                self.assertIsInstance(lang_class.Defaults.stop_words, set)
                if code not in empty_stop_lists:
                    self.assertGreater(len(lang_class.Defaults.stop_words), 0)

    def test_every_available_tokenizer_survives_edge_strings(self):
        for code, pipeline in self.pipelines.items():
            if pipeline is None:
                self.skip_optional(code)
                continue
            for text in EDGE_STRINGS:
                with self.subTest(code=code, text=text):
                    doc = pipeline(text)
                    self.assertEqual(text, doc.text)
                    list(doc)

            with self.subTest(code=code, text="Hello"):
                self.assertGreaterEqual(len(pipeline("Hello")), 1)

    def skip_optional(self, code):
        lang_class = spacy.util.get_lang_class(code)
        self.assertEqual(code, lang_class.lang)
        self.assertGreater(len(lang_class.Defaults.stop_words), 0)

    def test_production_english_model_is_the_full_pipeline(self):
        self.assertEqual("en", english_nlp.meta["lang"])
        self.assertEqual("core_web_sm", english_nlp.meta["name"])
        self.assertIn("tagger", english_nlp.pipe_names)
        self.assertIn("parser", english_nlp.pipe_names)

        doc = english_nlp("Remind me every Monday at 9am.")
        root = next(token for token in doc if token.dep_ == "ROOT")
        self.assertEqual("remind", root.lemma_)
        self.assertEqual("VERB", root.pos_)
        self.assertFalse(any(token.dep_ == "nsubj" for token in doc))

        doc = english_nlp("I love waking up every Monday at 9am.")
        root = next(token for token in doc if token.dep_ == "ROOT")
        self.assertEqual("love", root.lemma_)
        subjects = [token.text.lower() for token in doc if token.dep_ == "nsubj"]
        self.assertIn("i", subjects)

    def test_writing_systems_affixes_and_stopword_shape(self):
        blank_stop_words = set()
        for code in self.codes:
            with self.subTest(code=code):
                defaults = spacy.util.get_lang_class(code).Defaults
                writing = defaults.writing_system
                self.assertEqual(
                    {"direction", "has_case", "has_letters"},
                    set(writing),
                )
                if code in RTL_LANGUAGES:
                    self.assertEqual("rtl", writing["direction"])
                    self.assertFalse(writing["has_case"])
                    self.assertTrue(writing["has_letters"])
                elif code in CASELESS_LETTER_LANGUAGES:
                    self.assertEqual("ltr", writing["direction"])
                    self.assertFalse(writing["has_case"])
                    self.assertTrue(writing["has_letters"])
                elif code in CASELESS_UNSEGMENTED_LANGUAGES:
                    self.assertEqual("ltr", writing["direction"])
                    self.assertFalse(writing["has_case"])
                    self.assertFalse(writing["has_letters"])
                else:
                    self.assertEqual("ltr", writing["direction"])
                    self.assertTrue(writing["has_case"])
                    self.assertTrue(writing["has_letters"])

                for name in ("prefixes", "suffixes", "infixes"):
                    rules = getattr(defaults, name)
                    self.assertIsInstance(rules, list)
                    self.assertGreater(len(rules), 0)
                    for rule in rules:
                        self.assertIsInstance(rule, str)
                        re.compile(rule)

                if any(not word.strip() for word in defaults.stop_words):
                    blank_stop_words.add(code)

        self.assertEqual({"vi"}, blank_stop_words)

    def test_one_stop_word_stays_a_single_stop_token(self):
        for code, pipeline in self.pipelines.items():
            words = spacy.util.get_lang_class(code).Defaults.stop_words
            if pipeline is None or not words:
                continue
            found = False
            for word in sorted(item for item in words if item.strip() and " " not in item):
                doc = pipeline(word)
                if len(doc) == 1 and doc[0].text == word and doc[0].is_stop:
                    found = True
                    break
            with self.subTest(code=code):
                self.assertTrue(found)

    def test_format_characters_round_trip_on_every_tokenizer(self):
        for code, pipeline in self.pipelines.items():
            if pipeline is None:
                continue
            for text in FORMAT_EDGE_STRINGS:
                with self.subTest(code=code, text=text):
                    doc = pipeline(text)
                    self.assertEqual(text, doc.text)
                    self.assertEqual(code, doc.lang_)
                    self.assertEqual(text, "".join(token.text_with_ws for token in doc))
            for text in SPACE_CHARS:
                with self.subTest(code=code, kind="space", text=text):
                    doc = pipeline(text)
                    self.assertEqual([(text, True)], [(token.text, token.is_space) for token in doc])
            for text in PUNCT_CHARS:
                with self.subTest(code=code, kind="punct", text=text):
                    doc = pipeline(text)
                    self.assertEqual([(text, True)], [(token.text, token.is_punct) for token in doc])
            for text in CURRENCY_CHARS:
                with self.subTest(code=code, kind="currency", text=text):
                    doc = pipeline(text)
                    self.assertEqual(
                        [(text, True)],
                        [(token.text, token.is_currency) for token in doc],
                    )
            # Invisible format characters are kept, and they are not spaces.
            for text in ("\u200b", "\u200c", "\u200d", "\u2060", "\u00ad", "\ufeff", "\u180e"):
                with self.subTest(code=code, kind="format", text=text):
                    token = pipeline(text)[0]
                    self.assertEqual(text, token.text)
                    self.assertFalse(token.is_space)
                    self.assertFalse(token.is_punct)

    def test_tokenizer_exceptions_round_trip(self):
        mismatched = set()
        for code, pipeline in self.pipelines.items():
            exceptions = spacy.util.get_lang_class(code).Defaults.tokenizer_exceptions
            self.assertGreater(len(exceptions), 0)
            for key, entries in exceptions.items():
                self.assertIsInstance(key, str)
                self.assertIsInstance(entries, list)
                self.assertGreater(len(entries), 0)
                orths = []
                for entry in entries:
                    self.assertIsInstance(entry, dict)
                    self.assertIsInstance(entry[ORTH], str)
                    orths.append(entry[ORTH])
                if pipeline is None:
                    continue
                doc = pipeline(key)
                self.assertEqual(key, doc.text)
                self.assertEqual(key, "".join(token.text_with_ws for token in doc))
                if orths != [token.text for token in doc]:
                    mismatched.add(code)

        self.assertEqual(ORTH_MISMATCH_LANGUAGES, mismatched)
        fa = self.pipelines["fa"](".ق ")
        self.assertEqual([".ق"], [token.text for token in fa])
        self.assertEqual(" ", fa[0].whitespace_)
        zh = self.pipelines["zh"]("\\t")
        self.assertEqual(["\\", "t"], [token.text for token in zh])
        vi = self.pipelines["vi"]("°c.")
        self.assertEqual(["°c."], [token.text for token in vi])
        es = self.pipelines["es"]("°C.")
        self.assertEqual(["°C", "."], [token.text for token in es])

    def test_numeral_url_and_ordinal_splits(self):
        for code, pipeline in self.pipelines.items():
            if pipeline is None:
                continue
            with self.subTest(code=code):
                self.assert_token_pattern(pipeline, "½", ["½"], like_num=False)
                number = pipeline("123")
                if code == "zh":
                    self.assertEqual(list("123"), [token.text for token in number])
                    self.assertTrue(all(token.like_num for token in number))
                    self.assertEqual(list("１２３"), [token.text for token in pipeline("１２３")])
                    self.assertEqual(["I", "V"], [token.text for token in pipeline("IV")])
                    self.assertFalse(any(token.like_num for token in pipeline("IV")))
                    self.assertEqual(
                        list("https://example.com"),
                        [token.text for token in pipeline("https://example.com")],
                    )
                    self.assertEqual(["1", ".", "5"], [token.text for token in pipeline("1.5")])
                    self.assertEqual(["1", ",", "5"], [token.text for token in pipeline("1,5")])
                    self.assertEqual(["3", "r", "d"], [token.text for token in pipeline("3rd")])
                    continue

                self.assertEqual(["123"], [token.text for token in number])
                self.assertEqual(code != "grc", number[0].like_num)
                if code == "grc":
                    self.assertTrue(number[0].is_digit)
                    self.assertTrue(pipeline("α")[0].like_num)
                    self.assertTrue(pipeline("ι")[0].like_num)
                    self.assertFalse(pipeline("5")[0].like_num)

                if code == "pl":
                    self.assertEqual(["1", ".", "5"], [token.text for token in pipeline("1.5")])
                elif code in {"grc", "la"}:
                    self.assert_token_pattern(pipeline, "1.5", ["1.5"], like_num=False)
                else:
                    self.assert_token_pattern(pipeline, "1.5", ["1.5"], like_num=True)

                if code in {"grc", "la", "ne"}:
                    self.assert_token_pattern(pipeline, "1,5", ["1,5"], like_num=False)
                else:
                    self.assert_token_pattern(pipeline, "1,5", ["1,5"], like_num=True)

                if code in {"la", "ru"}:
                    self.assert_token_pattern(pipeline, "IV", ["IV"], like_num=True)
                else:
                    self.assert_token_pattern(pipeline, "IV", ["IV"], like_num=False)

                if code == "en":
                    self.assert_token_pattern(pipeline, "3rd", ["3rd"], like_num=True)
                else:
                    self.assert_token_pattern(pipeline, "3rd", ["3rd"], like_num=False)

                self.assertEqual(code != "grc", pipeline("٥")[0].like_num)
                self.assertEqual(code != "grc", pipeline("①")[0].like_num)
                self.assertEqual(code != "grc", pipeline("１２３")[0].like_num)
                url = pipeline("https://example.com")
                self.assertEqual(["https://example.com"], [token.text for token in url])
                self.assertEqual(code == "yo", url[0].like_num)
                self.assertEqual(
                    ["a.b+c@example.co.uk"],
                    [token.text for token in pipeline("a.b+c@example.co.uk")],
                )

    def assert_token_pattern(self, pipeline, text, texts, like_num):
        doc = pipeline(text)
        self.assertEqual(texts, [token.text for token in doc])
        self.assertTrue(all(token.like_num is like_num for token in doc))

    def test_production_model_round_trips_every_script_sample(self):
        samples = list(DUCKLING_LANGUAGE_SAMPLES.values()) + FORMAT_EDGE_STRINGS
        for code in self.codes:
            samples.append(lexical_sample(code))
        for text in samples:
            with self.subTest(text=text):
                doc = english_nlp(text)
                self.assertEqual("en", doc.lang_)
                self.assertEqual(text, doc.text)
                self.assertEqual(text, "".join(token.text_with_ws for token in doc))
                for token in doc:
                    token.lemma_
                    token.pos_
                    token.dep_


def lexical_sample(code):
    """A real lexical item for code, or a plain word when the stop list is empty."""
    words = spacy.util.get_lang_class(code).Defaults.stop_words
    for word in sorted(item for item in words if item.strip()):
        return word
    return "hello"


class MultilingualClassifyTests(unittest.TestCase):
    def test_every_duckling_language_sample_classifies_without_crashing(self):
        for code, text in DUCKLING_LANGUAGE_SAMPLES.items():
            with self.subTest(language=code):
                with patch("intent.duckling_parse", return_value=[]):
                    without_time = classify(text)
                with patch(
                    "intent.duckling_parse",
                    return_value=[
                        {
                            "body": "tomorrow",
                            "dim": "time",
                            "value": {
                                "type": "value",
                                "value": "2026-09-27T09:00:00.000Z",
                                "grain": "hour",
                            },
                        }
                    ],
                ):
                    with_time = classify(text)

                for result in (without_time, with_time):
                    self.assertEqual(text, result["text"])
                    self.assertIn(result["decision"], {"allow", "clarify", "reject"})
                    self.assertTrue(result["reason"])
                    self.assertIn("has_temporal_signal", result["features"])
                    self.assertIsInstance(result["tokens"], list)

                if code == "en":
                    self.assertEqual("clarify", without_time["decision"])
                    self.assertEqual("allow", with_time["decision"])
                    self.assertEqual("request_with_temporal_signal", with_time["reason"])

    def test_every_duckling_sample_has_a_stable_decision(self):
        # en_core_web_sm misreads Khmer as a subject-less verb, same shape as
        # the English imperative. French and Portuguese clock forms ("9h") are
        # temporal with no Duckling entity. A few other scripts become
        # declaratives once a time entity is attached.
        for code, text in DUCKLING_LANGUAGE_SAMPLES.items():
            with self.subTest(language=code):
                with patch("intent.duckling_parse", return_value=[]):
                    without_time = classify(text)
                with patch("intent.duckling_parse", return_value=TIME_ENTITY):
                    with_time = classify(text)

                self.assertEqual(code in FRENCH_LOOKING_DUCKLING, text_looks_french(text))
                self.assertEqual(code in {"fr", "pt"}, without_time["features"]["has_temporal_signal"])
                self.assertEqual(
                    code in {"en", "fr", "km"},
                    with_time["features"]["looks_like_request"],
                )

                if code in {"en", "km"}:
                    self.assertEqual(
                        ("clarify", "request_without_temporal_signal"),
                        (without_time["decision"], without_time["reason"]),
                    )
                    self.assertEqual(
                        ("allow", "request_with_temporal_signal"),
                        (with_time["decision"], with_time["reason"]),
                    )
                elif code == "fr":
                    self.assertEqual(
                        ("allow", "request_with_temporal_signal"),
                        (without_time["decision"], without_time["reason"]),
                    )
                    self.assertEqual(
                        ("allow", "request_with_temporal_signal"),
                        (with_time["decision"], with_time["reason"]),
                    )
                elif code == "pt":
                    self.assertEqual(
                        ("clarify", "temporal_signal_without_clear_request"),
                        (without_time["decision"], without_time["reason"]),
                    )
                    self.assertEqual(
                        ("clarify", "temporal_signal_without_clear_request"),
                        (with_time["decision"], with_time["reason"]),
                    )
                elif code in {"ar", "da", "fa", "ga", "sw", "tr"}:
                    self.assertEqual(
                        ("reject", "not_a_schedule_request"),
                        (without_time["decision"], without_time["reason"]),
                    )
                    self.assertEqual(
                        ("reject", "declarative_schedule_not_request"),
                        (with_time["decision"], with_time["reason"]),
                    )
                else:
                    self.assertEqual(
                        ("reject", "not_a_schedule_request"),
                        (without_time["decision"], without_time["reason"]),
                    )
                    self.assertEqual(
                        ("clarify", "temporal_signal_without_clear_request"),
                        (with_time["decision"], with_time["reason"]),
                    )

    def test_duckling_locale_follows_the_french_marker(self):
        for code, text in DUCKLING_LANGUAGE_SAMPLES.items():
            with self.subTest(language=code):
                with patch("intent.requests.post") as post:
                    post.return_value.json.return_value = []
                    post.return_value.raise_for_status.return_value = None
                    classify(text)
                locale = post.call_args.kwargs["data"]["locale"]
                self.assertEqual(
                    "fr_FR" if code in FRENCH_LOOKING_DUCKLING else "en_GB",
                    locale,
                )
                self.assertEqual(text, post.call_args.kwargs["data"]["text"])

    def test_wrapped_samples_keep_their_text(self):
        for code, text in DUCKLING_LANGUAGE_SAMPLES.items():
            for variant in (f"\ufeff{text}", f" {text} ", f"{text}\u200b", f"{text}\n"):
                with self.subTest(language=code, text=variant):
                    with patch("intent.duckling_parse", return_value=[]):
                        result = classify(variant)
                    self.assertEqual(variant, result["text"])
                    self.assertIn(result["decision"], {"allow", "clarify", "reject"})
                    self.assertIsInstance(result["tokens"], list)

    def test_every_spacy_lexicon_item_classifies(self):
        for code in spacy_language_codes():
            text = lexical_sample(code)
            with self.subTest(language=code, text=text):
                with patch("intent.duckling_parse", return_value=[]):
                    result = classify(text)
                self.assertEqual(text, result["text"])
                self.assertIn(result["decision"], {"allow", "clarify", "reject"})
                self.assertTrue(result["reason"])
                self.assertIsInstance(result["features"]["has_temporal_signal"], bool)

    def test_english_model_mistags_a_few_format_characters_as_verbs(self):
        # The small English model tags these as subject-less verbs, so the
        # request heuristic fires. A word joiner is a proper noun and does not.
        for text in ("\ufeff", "\u0964", "\u00ad", "don't"):
            with self.subTest(text=text):
                with patch("intent.duckling_parse", return_value=[]):
                    result = classify(text)
                self.assertEqual("clarify", result["decision"])
                self.assertEqual("request_without_temporal_signal", result["reason"])
                self.assertEqual("VERB", result["features"]["root"]["pos"])
                self.assertTrue(result["features"]["looks_like_request"])

        with patch("intent.duckling_parse", return_value=[]):
            joiner = classify("\u2060")
        self.assertEqual("reject", joiner["decision"])
        self.assertEqual("not_a_schedule_request", joiner["reason"])
        self.assertEqual("PROPN", joiner["features"]["root"]["pos"])


if __name__ == "__main__":
    unittest.main()
