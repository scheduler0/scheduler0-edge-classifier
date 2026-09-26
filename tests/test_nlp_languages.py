"""Exercise every language shipped by the installed NLP packages.

spaCy is the linguistic package. Duckling is the temporal package, and the
service only classifies English, so non-English text is checked here for a
stable decision rather than for a scheduling label.
"""

import pkgutil
import unittest
from unittest.mock import patch

import spacy
import spacy.lang

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
    # Whitespace and control characters the tokenizer must keep intact.
    "\n",
    "\t",
    "\u2003",
    "\u3000",
    "\u2028",
    "\u2029",
    "\ufeffHello",
    "a\x00b",
    "a\x0cb",
    "hello\u200bworld",
    "a\u200db",
    "a\u200cb",
    # Combining marks, bidi controls, and emoji sequences.
    "e\u0301",
    "a\u0301\u0308",
    "\u200fשלום\u200e",
    "\u202eabc\u202c",
    "\U0001f468\u200d\U0001f469\u200d\U0001f467",
    "\U0001f1e8\U0001f1e6",
    "1️⃣",
    "☺\ufe0f",
    # Digits and punctuation across scripts.
    "123",
    "１２３",
    "١٢٣",
    "१२३",
    "৯",
    "၉",
    "!",
    "。",
    "¿",
    "،",
    "¿Qué?",
    "¡Hola!",
    "$5",
    "5€",
    "https://example.com/a?q=1&x=2",
    "a.b+c@example.co.uk",
    "http://例え.jp/パス",
    # Words the per-language prefix rules split differently.
    "it's",
    "it\u2019s",
    "U.S.A.",
    "Größe",
    "cœur",
    "İstanbul",
    "Cześć",
    "你好世界",
    "مرحبا، كيف حالك؟",
    "שָׁלוֹם",
    "สวัสดี",
    "안녕하세요",
    "Xin chào",
    "გამარჯობა",
    "សួស្តី",
    "ສະບາຍດີ",
    "မင်္ဂလာပါ",
]

# Shared invariants. Chinese is the only installed character tokenizer.
# Ancient Greek replaces ASCII like_num with Greek letter numerals.
PUNCT_TOKENS = ["?", "!", "。", "¿", "،"]
SPACE_TOKENS = [" ", "\n", "\t", "\u00a0", "\u2003", "\u3000", "\u2028"]
CURRENCY_SYMBOLS = ["$", "€", "£", "¥", "₹", "₽", "₩", "₺", "₿", "¢"]

RTL_LANGUAGES = {"ar", "fa", "he", "ur"}
CASELESS_LETTER_LANGUAGES = {"am", "ti"}
CASELESS_UNSEGMENTED_LANGUAGES = {"ja", "ko", "zh"}
NOUN_CHUNK_LANGUAGES = {
    "ca",
    "da",
    "de",
    "el",
    "en",
    "es",
    "fa",
    "fi",
    "fr",
    "id",
    "it",
    "ja",
    "la",
    "ms",
    "nb",
    "nl",
    "nn",
    "pt",
    "sv",
    "tr",
}

# Samples whose letters or "9h" clock times match the French marker, so
# classify() sends them to Duckling as fr_FR.
FRENCH_LOOKING_DUCKLING = {"af", "ca", "es", "fr", "hu", "pt", "ro", "vi"}

# Decisions en_core_web_sm currently produces. Non-French scripts are not
# supported; a few are mis-tagged (Khmer as a subject-less verb, several
# others as declaratives once a time entity is attached).
WITHOUT_TIME = {
    "en": ("clarify", "request_without_temporal_signal"),
    "fr": ("allow", "request_with_temporal_signal"),
    "km": ("clarify", "request_without_temporal_signal"),
    "pt": ("clarify", "temporal_signal_without_clear_request"),
}
WITH_TIME = {
    "en": ("allow", "request_with_temporal_signal"),
    "fr": ("allow", "request_with_temporal_signal"),
    "km": ("allow", "request_with_temporal_signal"),
    "ar": ("reject", "declarative_schedule_not_request"),
    "da": ("reject", "declarative_schedule_not_request"),
    "fa": ("reject", "declarative_schedule_not_request"),
    "ga": ("reject", "declarative_schedule_not_request"),
    "sw": ("reject", "declarative_schedule_not_request"),
    "tr": ("reject", "declarative_schedule_not_request"),
}
DEFAULT_WITHOUT_TIME = ("reject", "not_a_schedule_request")
DEFAULT_WITH_TIME = ("clarify", "temporal_signal_without_clear_request")

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
                    self.assertEqual(code, doc.lang_)
                    self.assertEqual(
                        text,
                        "".join(token.text + token.whitespace_ for token in doc),
                    )
                    for token in doc:
                        token.lemma_
                        token.pos_
                        token.dep_

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

    def test_every_language_publishes_tokenizer_defaults(self):
        noun_chunk_codes = set()
        blank_stop_words = set()
        for code in self.codes:
            with self.subTest(code=code):
                lang_class = spacy.util.get_lang_class(code)
                defaults = lang_class.Defaults
                writing = defaults.writing_system
                self.assertEqual(
                    {"direction", "has_case", "has_letters"},
                    set(writing),
                )
                self.assertIn(writing["direction"], {"ltr", "rtl"})
                self.assertIsInstance(writing["has_case"], bool)
                self.assertIsInstance(writing["has_letters"], bool)
                if code in RTL_LANGUAGES:
                    self.assertEqual("rtl", writing["direction"])
                    self.assertFalse(writing["has_case"])
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
                    self.assertTrue(all(isinstance(rule, str) for rule in rules))
                    self.assertTrue(any(rule for rule in rules))
                self.assertIsInstance(defaults.tokenizer_exceptions, dict)
                self.assertGreater(len(defaults.tokenizer_exceptions), 0)
                self.assertTrue(
                    all(isinstance(key, str) for key in defaults.tokenizer_exceptions)
                )
                self.assertTrue(
                    all(isinstance(word, str) for word in defaults.stop_words)
                )
                if any(not word.strip() for word in defaults.stop_words):
                    blank_stop_words.add(code)
                if "noun_chunks" in defaults.syntax_iterators:
                    noun_chunk_codes.add(code)

        self.assertEqual(NOUN_CHUNK_LANGUAGES, noun_chunk_codes)
        # Vietnamese ships one empty stop-word entry. No other language does.
        self.assertEqual({"vi"}, blank_stop_words)

    def test_shared_token_flags_and_script_splits(self):
        for code, pipeline in self.pipelines.items():
            if pipeline is None:
                continue
            for text in PUNCT_TOKENS:
                with self.subTest(code=code, text=text):
                    doc = pipeline(text)
                    self.assertEqual([(text, True)], [(t.text, t.is_punct) for t in doc])
            for text in SPACE_TOKENS:
                with self.subTest(code=code, text=text):
                    doc = pipeline(text)
                    self.assertEqual([(text, True)], [(t.text, t.is_space) for t in doc])
            for text in CURRENCY_SYMBOLS:
                with self.subTest(code=code, text=text):
                    doc = pipeline(text)
                    self.assertEqual(
                        [(text, True)],
                        [(t.text, t.is_currency) for t in doc],
                    )
            bom = pipeline("\ufeff")
            self.assertEqual(1, len(bom))
            self.assertFalse(bom[0].is_space)
            self.assertFalse(bom[0].is_punct)

            if code == "zh":
                hello = pipeline("Hello world")
                # The character tokenizer keeps the space as trailing whitespace
                # on "o", not as its own token.
                self.assertEqual(
                    ["H", "e", "l", "l", "o", "w", "o", "r", "l", "d"],
                    [token.text for token in hello],
                )
                self.assertEqual(" ", hello[4].whitespace_)
                self.assertEqual(list("123"), [token.text for token in pipeline("123")])
                self.assertEqual(
                    list("你好世界"),
                    [token.text for token in pipeline("你好世界")],
                )
                continue

            hello = pipeline("Hello world")
            self.assertEqual(["Hello", "world"], [token.text for token in hello])
            number = pipeline("123")
            self.assertEqual(["123"], [token.text for token in number])
            if code == "grc":
                # The Greek numeral table replaces ASCII like_num.
                self.assertFalse(number[0].like_num)
                self.assertTrue(pipeline("α")[0].like_num)
                self.assertTrue(pipeline("ι")[0].like_num)
                self.assertFalse(pipeline("5")[0].like_num)
            else:
                self.assertTrue(number[0].like_num)
            cjk = pipeline("你好世界")
            self.assertEqual(["你好世界"], [token.text for token in cjk])

    def test_production_model_round_trips_every_script(self):
        texts = list(DUCKLING_LANGUAGE_SAMPLES.values()) + EDGE_STRINGS
        for text in texts:
            with self.subTest(text=text):
                doc = english_nlp(text)
                self.assertEqual("en", doc.lang_)
                self.assertEqual(text, doc.text)
                self.assertEqual(
                    text,
                    "".join(token.text + token.whitespace_ for token in doc),
                )
                if text.strip():
                    self.assertGreater(len(doc), 0)
                for token in doc:
                    token.lemma_
                    token.pos_
                    token.dep_
                    token.head.text


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
        self.assertTrue(set(WITHOUT_TIME) <= set(DUCKLING_LANGUAGE_SAMPLES))
        self.assertTrue(set(WITH_TIME) <= set(DUCKLING_LANGUAGE_SAMPLES))
        for code, text in DUCKLING_LANGUAGE_SAMPLES.items():
            with self.subTest(language=code):
                with patch("intent.duckling_parse", return_value=[]):
                    without_time = classify(text)
                with patch("intent.duckling_parse", return_value=TIME_ENTITY):
                    with_time = classify(text)
                expected_without = WITHOUT_TIME.get(code, DEFAULT_WITHOUT_TIME)
                expected_with = WITH_TIME.get(code, DEFAULT_WITH_TIME)
                self.assertEqual(
                    expected_without,
                    (without_time["decision"], without_time["reason"]),
                )
                self.assertEqual(
                    expected_with,
                    (with_time["decision"], with_time["reason"]),
                )
                self.assertEqual(text, without_time["text"])
                self.assertEqual(text, with_time["text"])
                self.assertEqual(
                    code == "fr" or code == "pt",
                    without_time["features"]["has_temporal_signal"],
                )
                self.assertEqual(
                    code in {"en", "fr", "km"},
                    with_time["features"]["looks_like_request"],
                )

    def test_duckling_locale_follows_the_french_marker(self):
        for code, text in DUCKLING_LANGUAGE_SAMPLES.items():
            with self.subTest(language=code):
                looks_french = text_looks_french(text)
                self.assertEqual(code in FRENCH_LOOKING_DUCKLING, looks_french)
                with patch("intent.requests.post") as post:
                    post.return_value.json.return_value = []
                    post.return_value.raise_for_status.return_value = None
                    result = classify(text)
                self.assertIn(result["decision"], {"allow", "clarify", "reject"})
                data = post.call_args.kwargs["data"]
                self.assertEqual("fr_FR" if looks_french else "en_GB", data["locale"])
                self.assertEqual(text, data["text"])
                self.assertEqual('["time","duration"]', data["dims"])

    def test_padded_and_marked_samples_keep_their_text(self):
        for code, text in DUCKLING_LANGUAGE_SAMPLES.items():
            variants = [
                f"  {text}  ",
                f"{text}\n",
                f"{text} 😊",
                f"\ufeff{text}",
                f"{text} {text}",
            ]
            for variant in variants:
                with self.subTest(language=code, text=variant):
                    with patch("intent.duckling_parse", return_value=[]):
                        without_time = classify(variant)
                    with patch("intent.duckling_parse", return_value=TIME_ENTITY):
                        with_time = classify(variant)
                    for result in (without_time, with_time):
                        self.assertEqual(variant, result["text"])
                        self.assertIn(result["decision"], {"allow", "clarify", "reject"})
                        self.assertIsInstance(result["features"]["has_temporal_signal"], bool)
                        self.assertIsInstance(result["features"]["looks_like_request"], bool)
                        self.assertIsInstance(result["tokens"], list)


if __name__ == "__main__":
    unittest.main()
