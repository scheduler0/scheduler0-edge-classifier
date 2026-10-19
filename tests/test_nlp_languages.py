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

# Languages whose Defaults ship a noun-chunk iterator. Japanese is included
# even though its tokenizer backend is optional and stays unloaded.
NOUN_CHUNK_LANGUAGES = {
    "ca", "da", "de", "el", "en", "es", "fa", "fi", "fr", "id", "it", "ja",
    "la", "ms", "nb", "nl", "nn", "pt", "sv", "tr",
}
TOKEN_MATCH_LANGUAGES = {"fr", "hu", "tr"}

# Whitespace and controls the earlier space list does not cover. File, group,
# record, and unit separators are Unicode whitespace, same as carriage return
# and the Ogham space mark.
EXTRA_SPACE_CHARS = ["\r", "\u1680", "\x1c", "\x1d", "\x1e", "\x1f"]

# Marks that round-trip as a single token and are neither space nor punct.
INVISIBLE_MARKS = ["\x00", "\x7f", "\u200e", "\u200f", "\u202b", "\u202e", "\u2066", "\u2069", "\ufe0f"]

# Letters with no case pair. Han is a stop word only in the Chinese class.
CASELESS_SCRIPT_LETTERS = ["א", "ا", "क", "你", "ก", "ខ", "ກ", "က", "ሀ"]
# Lowercase letters. A few are stop words or, for Ancient Greek, numerals.
CASED_SCRIPT_LETTERS = ["α", "ქ", "а", "ß", "ł", "ø", "å", "ñ", "ğ", "š", "ą", "ı", "ő", "ă", "ơ", "ư", "ð", "þ"]
CASELESS_STOP_LETTERS = {"你": {"zh"}}
CASED_STOP_LETTERS = {
    "а": {"bg", "mk", "ru", "sr", "uk"},
    "å": {"nb"},
    "š": {"sl"},
    "ő": {"hu"},
    "ơ": {"vi"},
    "ư": {"vi"},
}
# Ancient Greek treats Greek letters as numerals. Latin does the same for dotless i.
LETTER_NUMERALS = {"α": {"grc"}, "ı": {"la"}}
NATIVE_DIGITS = ["०", "০", "๐", "០"]

# Diacritics in the French marker, and lookalikes that must stay on en_GB.
FRENCH_DIACRITICS = "àâäçéèêëîïôùûüœæ"
NON_FRENCH_DIACRITICS = "öáíóúýßłøåñğšąıőăơưðþ"
# The English model tags these non-French letters as subject-less verbs.
NON_FRENCH_VERB_MISTAGS = {"ó", "ð"}

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

    def test_noun_chunk_iterators_and_token_match_hooks(self):
        matched = set()
        chunked = set()
        for code in self.codes:
            defaults = spacy.util.get_lang_class(code).Defaults
            iterators = defaults.syntax_iterators
            self.assertIsInstance(iterators, dict)
            if iterators:
                self.assertEqual(["noun_chunks"], list(iterators))
                self.assertTrue(callable(iterators["noun_chunks"]))
                chunked.add(code)
            token_match = getattr(defaults, "token_match", None)
            if token_match is not None:
                self.assertTrue(callable(token_match))
                matched.add(code)
        self.assertEqual(NOUN_CHUNK_LANGUAGES, chunked)
        self.assertEqual(TOKEN_MATCH_LANGUAGES, matched)

    def test_controls_quotes_scripts_and_native_digits(self):
        for code, pipeline in self.pipelines.items():
            if pipeline is None:
                continue
            for text in EXTRA_SPACE_CHARS:
                with self.subTest(code=code, kind="space", text=text):
                    doc = self.round_trip(pipeline, text)
                    self.assertEqual([(text, True)], [(token.text, token.is_space) for token in doc])
            for text in INVISIBLE_MARKS:
                with self.subTest(code=code, kind="mark", text=text):
                    token = self.round_trip(pipeline, text)[0]
                    self.assertEqual(text, token.text)
                    self.assertFalse(token.is_space)
                    self.assertFalse(token.is_punct)
            for text, left, right in (
                ("«", True, False),
                ("»", False, True),
                ("‹", True, False),
            ):
                with self.subTest(code=code, kind="quote", text=text):
                    token = self.round_trip(pipeline, text)[0]
                    self.assertTrue(token.is_punct)
                    self.assertTrue(token.is_quote)
                    self.assertEqual(left, token.is_left_punct)
                    self.assertEqual(right, token.is_right_punct)
                    self.assertFalse(token.is_bracket)
            token = self.round_trip(pipeline, "「")[0]
            self.assertTrue(token.is_punct)
            self.assertFalse(token.is_quote)
            self.assertFalse(token.is_bracket)
            brace = self.round_trip(pipeline, "{")[0]
            self.assertTrue(brace.is_bracket and brace.is_left_punct and brace.is_punct)
            self.assertTrue(self.round_trip(pipeline, "–")[0].is_punct)
            self.assertTrue(self.round_trip(pipeline, "\u2011")[0].is_punct)
            for text in ("−", "°", "\U0001f600"):
                token = self.round_trip(pipeline, text)[0]
                self.assertEqual(text, token.text)
                self.assertFalse(token.is_punct)
                self.assertFalse(token.is_space)

            for text in CASELESS_SCRIPT_LETTERS:
                with self.subTest(code=code, kind="caseless", text=text):
                    token = self.round_trip(pipeline, text)[0]
                    self.assertTrue(token.is_alpha)
                    self.assertFalse(token.is_lower)
                    self.assertFalse(token.like_num)
                    self.assertEqual(code in CASELESS_STOP_LETTERS.get(text, ()), token.is_stop)
            for text in CASED_SCRIPT_LETTERS:
                with self.subTest(code=code, kind="cased", text=text):
                    token = self.round_trip(pipeline, text)[0]
                    self.assertTrue(token.is_alpha)
                    self.assertTrue(token.is_lower)
                    self.assertEqual(code in LETTER_NUMERALS.get(text, ()), token.like_num)
                    self.assertEqual(code in CASED_STOP_LETTERS.get(text, ()), token.is_stop)
            for text in NATIVE_DIGITS:
                with self.subTest(code=code, kind="digit", text=text):
                    token = self.round_trip(pipeline, text)[0]
                    self.assertTrue(token.is_digit)
                    self.assertEqual(code != "grc", token.like_num)
                    self.assertFalse(token.is_alpha)

    def test_urls_emails_clocks_and_emoji_clusters(self):
        urls = (
            "www.example.com",
            "http://example.com",
            "ftp://files.example.com",
            "https://ex.com/a?b=1&c=2",
            "file.txt",
        )
        emails = ("a+b@example.com", "a.b@example.com")
        for code, pipeline in self.pipelines.items():
            if pipeline is None:
                continue
            for text in urls:
                with self.subTest(code=code, kind="url", text=text):
                    doc = self.round_trip(pipeline, text)
                    if code == "zh":
                        self.assertGreater(len(doc), 1)
                        self.assertFalse(any(token.like_url for token in doc))
                    else:
                        self.assertEqual([text], [token.text for token in doc])
                        self.assertTrue(doc[0].like_url)
                        # Yoruba's numeral rules also match a bare URL. A query
                        # string stops that, while like_url still holds.
                        self.assertEqual(code == "yo" and "?" not in text, doc[0].like_num)
            for text in emails:
                with self.subTest(code=code, kind="email", text=text):
                    doc = self.round_trip(pipeline, text)
                    if code == "zh":
                        self.assertGreater(len(doc), 1)
                        self.assertFalse(any(token.like_email for token in doc))
                    else:
                        self.assertEqual([text], [token.text for token in doc])
                        self.assertTrue(doc[0].like_email)
                        self.assertEqual(code == "yo", doc[0].like_num)
            for text in ("mailto:a@b.co", "user@localhost", "10:30", "9h"):
                with self.subTest(code=code, kind="plain", text=text):
                    doc = self.round_trip(pipeline, text)
                    if code == "zh":
                        self.assertGreater(len(doc), 1)
                    else:
                        self.assertEqual([text], [token.text for token in doc])
                        self.assertFalse(doc[0].like_url)
                        self.assertFalse(doc[0].like_email)
                        self.assertFalse(doc[0].like_num)

            family = "👨\u200d👩\u200d👧"
            flag = "🇺🇸"
            with self.subTest(code=code, kind="emoji"):
                # Vietnamese without Pyvi splits on whitespace only. Hungarian's
                # rules also keep these clusters intact.
                family_doc = self.round_trip(pipeline, family)
                flag_doc = self.round_trip(pipeline, flag)
                if code in {"hu", "vi"}:
                    self.assertEqual([family], [token.text for token in family_doc])
                    self.assertEqual([flag], [token.text for token in flag_doc])
                else:
                    self.assertEqual(["👨", "\u200d", "👩", "\u200d", "👧"], [token.text for token in family_doc])
                    self.assertEqual(["🇺", "🇸"], [token.text for token in flag_doc])
                keycap = self.round_trip(pipeline, "1️⃣")
                if code == "zh":
                    self.assertEqual(["1", "\ufe0f", "\u20e3"], [token.text for token in keycap])
                    self.assertTrue(keycap[0].like_num)
                else:
                    self.assertEqual(["1️⃣"], [token.text for token in keycap])
                    self.assertFalse(keycap[0].like_num)

    def test_signed_numbers_dates_and_apostrophes(self):
        not_like_signed = {"fa", "grc", "la", "lb", "lt", "nl", "pl", "si", "ta", "te", "tl", "uk", "yo"}
        split_negative = {"ca", "sl", "zh"}
        split_positive = {"bn", "sl", "zh"}
        glued_range = {"da", "el", "es", "fi", "hu", "lt", "nb", "nl", "nn", "pt", "ro", "sv", "vi"}
        slash_split = {"de", "id", "ms"}
        slash_not_num = {"grc", "la", "yo"}
        percent_glued = {"ca", "el", "es", "grc", "hu", "nl", "ro", "vi"}
        dollar_glued = {"bn", "el", "hu", "vi"}
        hash_glued = {"id", "ms", "nb", "nn", "vi"}
        dotted_split = {"pl"}
        dotted_not_num = {"grc", "la"}
        apostrophe_parts = {"da", "de", "fi", "hu", "ky", "nb", "nn", "pl", "sv", "tt"}
        apostrophe_elision = {"ca", "fr", "it", "lij"}
        heure_parts = apostrophe_parts | {"nl"}
        heure_elision = {"ca", "it", "lij"}

        for code, pipeline in self.pipelines.items():
            if pipeline is None:
                continue
            with self.subTest(code=code):
                negative = self.round_trip(pipeline, "-5")
                positive = self.round_trip(pipeline, "+5")
                if code in split_negative:
                    self.assertEqual(["-", "5"], [token.text for token in negative])
                    self.assertTrue(negative[-1].like_num)
                elif code in not_like_signed:
                    self.assertEqual(["-5"], [token.text for token in negative])
                    self.assertFalse(negative[0].like_num)
                else:
                    self.assertEqual(["-5"], [token.text for token in negative])
                    self.assertTrue(negative[0].like_num)

                if code in split_positive:
                    self.assertEqual(["+", "5"], [token.text for token in positive])
                    self.assertTrue(positive[-1].like_num)
                elif code in not_like_signed:
                    self.assertEqual(["+5"], [token.text for token in positive])
                    self.assertFalse(positive[0].like_num)
                else:
                    self.assertEqual(["+5"], [token.text for token in positive])
                    self.assertTrue(positive[0].like_num)

                span = self.round_trip(pipeline, "1-2")
                iso = self.round_trip(pipeline, "2026-09-30")
                if code in glued_range:
                    self.assertEqual(["1-2"], [token.text for token in span])
                    self.assertFalse(span[0].like_num)
                    self.assertEqual(["2026-09-30"], [token.text for token in iso])
                    self.assertFalse(iso[0].like_num)
                elif code == "grc":
                    self.assertEqual(["1", "-", "2"], [token.text for token in span])
                    self.assertFalse(any(token.like_num for token in span))
                    self.assertEqual(["2026", "-", "09", "-", "30"], [token.text for token in iso])
                    self.assertFalse(any(token.like_num for token in iso))
                elif code == "zh":
                    self.assertEqual(["1", "-", "2"], [token.text for token in span])
                    self.assertEqual(["1", "2"], [token.text for token in span if token.like_num])
                    self.assertEqual(list("2026-09-30"), [token.text for token in iso])
                else:
                    self.assertEqual(["1", "-", "2"], [token.text for token in span])
                    self.assertEqual([True, False, True], [token.like_num for token in span])
                    self.assertEqual(
                        ["2026", "-", "09", "-", "30"],
                        [token.text for token in iso],
                    )
                    self.assertEqual(
                        [True, False, True, False, True],
                        [token.like_num for token in iso],
                    )

                slash = self.round_trip(pipeline, "9/30")
                if code == "zh":
                    self.assertEqual(list("9/30"), [token.text for token in slash])
                elif code in slash_split:
                    self.assertEqual(["9", "/", "30"], [token.text for token in slash])
                    self.assertEqual([True, False, True], [token.like_num for token in slash])
                else:
                    self.assertEqual(["9/30"], [token.text for token in slash])
                    self.assertEqual(code not in slash_not_num, slash[0].like_num)

                percent = self.round_trip(pipeline, "50%")
                if code == "zh":
                    self.assertEqual(["5", "0", "%"], [token.text for token in percent])
                elif code in percent_glued:
                    self.assertEqual(["50%"], [token.text for token in percent])
                    self.assertFalse(percent[0].like_num)
                else:
                    self.assertEqual(["50", "%"], [token.text for token in percent])
                    self.assertTrue(percent[0].like_num)
                    self.assertTrue(percent[1].is_punct)

                dollar = self.round_trip(pipeline, "$5")
                if code in dollar_glued:
                    self.assertEqual(["$5"], [token.text for token in dollar])
                    self.assertFalse(dollar[0].like_num)
                else:
                    self.assertEqual(["$", "5"], [token.text for token in dollar])
                    self.assertTrue(dollar[0].is_currency)
                    self.assertEqual(code != "grc", dollar[1].like_num)

                euro = self.round_trip(pipeline, "5€")
                if code in {"grc", "vi"}:
                    self.assertEqual(["5€"], [token.text for token in euro])
                    self.assertFalse(euro[0].like_num)
                else:
                    self.assertEqual(["5", "€"], [token.text for token in euro])
                    self.assertTrue(euro[0].like_num)
                    self.assertTrue(euro[1].is_currency)

                tagged = self.round_trip(pipeline, "#monday")
                if code == "zh":
                    self.assertEqual(list("#monday"), [token.text for token in tagged])
                elif code in hash_glued:
                    self.assertEqual(["#monday"], [token.text for token in tagged])
                else:
                    self.assertEqual(["#", "monday"], [token.text for token in tagged])
                    self.assertTrue(tagged[0].is_punct)

                for text in ("192.168.0.1", "1.2.3"):
                    dotted = self.round_trip(pipeline, text)
                    if code == "zh":
                        self.assertEqual(list(text), [token.text for token in dotted])
                    elif code in dotted_split:
                        self.assertGreater(len(dotted), 1)
                        self.assertIn(".", [token.text for token in dotted])
                    else:
                        self.assertEqual([text], [token.text for token in dotted])
                        self.assertEqual(code not in dotted_not_num, dotted[0].like_num)

                curly = self.round_trip(pipeline, "don\u2019t")
                if code == "zh":
                    self.assertEqual(list("don\u2019t"), [token.text for token in curly])
                elif code == "en":
                    self.assertEqual(["do", "n\u2019t"], [token.text for token in curly])
                    self.assertTrue(all(token.is_stop for token in curly))
                elif code == "nl":
                    self.assertEqual(["don", "\u2019t"], [token.text for token in curly])
                elif code in apostrophe_elision:
                    self.assertEqual(["don\u2019", "t"], [token.text for token in curly])
                elif code in apostrophe_parts:
                    self.assertEqual(["don", "\u2019", "t"], [token.text for token in curly])
                    self.assertTrue(curly[1].is_quote)
                elif code == "yo":
                    self.assertEqual(["don\u2019t"], [token.text for token in curly])
                    self.assertTrue(curly[0].like_num)
                else:
                    self.assertEqual(["don\u2019t"], [token.text for token in curly])
                    self.assertFalse(curly[0].like_num)

                heure = self.round_trip(pipeline, "l\u2019heure")
                if code == "zh":
                    self.assertEqual(list("l\u2019heure"), [token.text for token in heure])
                elif code == "fr":
                    self.assertEqual(["l\u2019", "heure"], [token.text for token in heure])
                    self.assertTrue(heure[0].is_stop)
                elif code in heure_elision:
                    self.assertEqual(["l\u2019", "heure"], [token.text for token in heure])
                    self.assertFalse(heure[0].is_stop)
                elif code in heure_parts:
                    self.assertEqual(["l", "\u2019", "heure"], [token.text for token in heure])
                else:
                    self.assertEqual(["l\u2019heure"], [token.text for token in heure])

    def test_abbreviations_elisions_and_compounds(self):
        # Final-dot abbreviations, apostrophes, and hyphenated compounds that
        # the earlier edges do not lock. Chinese splits to characters.
        dot_keep = {
            "Feb.": {"am", "ar", "da", "de", "en", "fa", "grc", "id", "la", "ms", "sv", "ti", "vi"},
            "Mar.": {"am", "ar", "da", "en", "fa", "grc", "id", "la", "ms", "ti", "vi"},
            "Aug.": {"am", "ar", "da", "de", "en", "fa", "fo", "grc", "la", "ms", "ro", "sv", "ti", "vi"},
            "Sept.": {"am", "ar", "da", "de", "en", "fa", "fo", "grc", "la", "ms", "sv", "ti", "vi"},
            "Dec.": {"am", "ar", "da", "en", "fa", "grc", "la", "ms", "nl", "sv", "ti", "vi"},
            "Mon.": {"am", "ar", "fa", "grc", "nl", "ti", "vi"},
            "Tue.": {"am", "ar", "fa", "grc", "ti", "vi"},
            "Esq.": {"am", "ar", "fa", "grc", "ti", "vi"},
            "viz.": {"am", "ar", "fa", "grc", "sl", "ti", "vi"},
            "fl.": {"am", "ar", "da", "fa", "grc", "nl", "ti", "vi"},
            "Mx.": {"am", "ar", "fa", "grc", "ti", "vi"},
        }
        stem_stops = {
            ("Mar.", "ga"),
            ("Mon.", "ca"),
            ("Mon.", "fr"),
            ("Tue.", "it"),
            ("Sept.", "fr"),
        }
        n_t_words = {
            "aren't": "are",
            "weren't": "were",
            "wasn't": "was",
            "haven't": "have",
            "wouldn't": "would",
            "couldn't": "could",
        }
        d_words = {"who'd": "who", "what'd": "what", "where'd": "where"}
        hyphen_glue = {"ca", "da", "de", "el", "es", "fi", "hu", "ky", "lb", "nb", "nl", "nn", "pt", "ro", "sr", "sv", "tt", "vi"}
        a_grave_glue = hyphen_glue - {"el"} | {"fr"}
        three = {"ky", "tt"}
        elide = {"ca", "fr", "it", "lij"}
        a_stop = {"ca", "cs", "de", "dsb", "en", "es", "fr", "ga", "hr", "hsb", "hu", "it", "lb", "lij", "lt", "pl", "pt", "sk"}
        both_stop = {"ro", "sl", "sq", "yo"}

        for code, pipeline in self.pipelines.items():
            if pipeline is None:
                continue
            with self.subTest(code=code):
                for text, keep in dot_keep.items():
                    doc = self.round_trip(pipeline, text)
                    stem = text[:-1]
                    if code == "zh":
                        self.assertEqual(list(text), [token.text for token in doc])
                        self.assertTrue(doc[-1].is_stop and doc[-1].is_punct)
                    elif code in keep:
                        self.assertEqual([text], [token.text for token in doc])
                        self.assertEqual(text[:1].isupper(), doc[0].is_title)
                        self.assertEqual(text[:1].islower(), doc[0].is_lower)
                        self.assertFalse(doc[0].like_num)
                    else:
                        self.assertEqual([stem, "."], [token.text for token in doc])
                        self.assertTrue(doc[1].is_punct)
                        self.assertEqual((text, code) in stem_stops, doc[0].is_stop)
                        expected_num = (text == "Sept." and code == "fr") or (
                            text == "Mx." and code == "la"
                        )
                        self.assertEqual(expected_num, doc[0].like_num)

                wed = self.round_trip(pipeline, "Wed.")
                if code == "zh":
                    self.assertEqual(list("Wed."), [token.text for token in wed])
                elif code == "en":
                    self.assertEqual(["We", "d."], [token.text for token in wed])
                    self.assertTrue(wed[0].is_stop)
                    self.assertFalse(wed[1].is_alpha)
                elif code in {"am", "ar", "fa", "grc", "ti", "vi"}:
                    self.assertEqual(["Wed."], [token.text for token in wed])
                else:
                    self.assertEqual(["Wed", "."], [token.text for token in wed])

                for text, pl_stops in (("M.A.", {"A"}), ("J.D.", set()), ("S.A.", {"A"})):
                    doc = self.round_trip(pipeline, text)
                    letters = [ch for ch in text if ch != "."]
                    if code == "zh":
                        self.assertEqual(list(text), [token.text for token in doc])
                        self.assertTrue(all(token.is_stop for token in doc if token.text == "."))
                    elif code in {"lt", "sr"}:
                        self.assertEqual([text[:-1], "."], [token.text for token in doc])
                    elif code == "sl":
                        self.assertEqual([f"{ch}." for ch in letters], [token.text for token in doc])
                    elif code == "pl":
                        expected = []
                        for ch in letters:
                            expected.extend([ch, "."])
                        self.assertEqual(expected, [token.text for token in doc])
                        for token in doc:
                            if token.text.isalpha():
                                self.assertEqual(token.text in pl_stops, token.is_stop)
                    else:
                        self.assertEqual([text], [token.text for token in doc])
                        self.assertTrue(doc[0].is_title)
                        self.assertFalse(doc[0].is_alpha)

                for text, keep in (
                    ("M.Sc.", {"am", "ar", "de", "fa", "grc", "hu", "id", "ti", "vi"}),
                    ("B.Ed.", {"am", "ar", "fa", "grc", "ti", "vi"}),
                ):
                    doc = self.round_trip(pipeline, text)
                    if code == "zh":
                        self.assertEqual(list(text), [token.text for token in doc])
                    elif code == "sl":
                        self.assertEqual([text[:2], text[2:-1], "."], [token.text for token in doc])
                    elif code == "pl":
                        self.assertEqual([text[0], ".", text[2:-1], "."], [token.text for token in doc])
                    elif code in keep:
                        self.assertEqual([text], [token.text for token in doc])
                    else:
                        self.assertEqual([text[:-1], "."], [token.text for token in doc])

                for text, stem in n_t_words.items():
                    doc = self.round_trip(pipeline, text)
                    if code == "zh":
                        self.assertEqual(list(text), [token.text for token in doc])
                    elif code == "en":
                        self.assertEqual([stem, "n't"], [token.text for token in doc])
                        self.assertTrue(all(token.is_stop for token in doc))
                    elif code == "ca":
                        self.assertEqual([stem + "n", "'t"], [token.text for token in doc])
                    elif code in {"fr", "it", "lij"}:
                        self.assertEqual([stem + "n'", "t"], [token.text for token in doc])
                    elif code in three:
                        self.assertEqual([stem + "n", "'", "t"], [token.text for token in doc])
                    else:
                        self.assertEqual([text], [token.text for token in doc])
                        self.assertFalse(doc[0].like_num)

                aint = self.round_trip(pipeline, "ain't")
                if code == "zh":
                    self.assertEqual(list("ain't"), [token.text for token in aint])
                elif code == "en":
                    self.assertEqual(["ai", "n't"], [token.text for token in aint])
                    self.assertFalse(aint[0].is_stop)
                    self.assertTrue(aint[1].is_stop)
                elif code == "ca":
                    self.assertEqual(["ain", "'t"], [token.text for token in aint])
                elif code in {"fr", "it", "lij"}:
                    self.assertEqual(["ain'", "t"], [token.text for token in aint])
                elif code in three:
                    self.assertEqual(["ain", "'", "t"], [token.text for token in aint])
                else:
                    self.assertEqual(["ain't"], [token.text for token in aint])

                for text, stem in d_words.items():
                    doc = self.round_trip(pipeline, text)
                    if code == "zh":
                        self.assertEqual(list(text), [token.text for token in doc])
                    elif code == "en":
                        self.assertEqual([stem, "'d"], [token.text for token in doc])
                        self.assertTrue(all(token.is_stop for token in doc))
                    elif code in elide:
                        self.assertEqual([stem + "'", "d"], [token.text for token in doc])
                        self.assertFalse(any(token.is_stop for token in doc))
                    elif code in three:
                        self.assertEqual([stem, "'", "d"], [token.text for token in doc])
                    else:
                        self.assertEqual([text], [token.text for token in doc])

                wholl = self.round_trip(pipeline, "who'll")
                if code == "zh":
                    self.assertEqual(list("who'll"), [token.text for token in wholl])
                elif code == "en":
                    self.assertEqual(["who", "'ll"], [token.text for token in wholl])
                    self.assertTrue(all(token.is_stop for token in wholl))
                elif code in elide:
                    self.assertEqual(["who'", "ll"], [token.text for token in wholl])
                elif code in three:
                    self.assertEqual(["who", "'", "ll"], [token.text for token in wholl])
                else:
                    self.assertEqual(["who'll"], [token.text for token in wholl])

                whatre = self.round_trip(pipeline, "what're")
                if code == "zh":
                    self.assertEqual(list("what're"), [token.text for token in whatre])
                elif code == "en":
                    self.assertEqual(["what", "'re"], [token.text for token in whatre])
                    self.assertTrue(all(token.is_stop for token in whatre))
                elif code in elide:
                    self.assertEqual(["what'", "re"], [token.text for token in whatre])
                elif code in three:
                    self.assertEqual(["what", "'", "re"], [token.text for token in whatre])
                else:
                    self.assertEqual(["what're"], [token.text for token in whatre])

                idve = self.round_trip(pipeline, "I'd've")
                if code == "zh":
                    self.assertEqual(list("I'd've"), [token.text for token in idve])
                elif code == "en":
                    self.assertEqual(["I", "'d", "'ve"], [token.text for token in idve])
                    self.assertTrue(all(token.is_stop for token in idve))
                elif code == "ca":
                    self.assertEqual(["I'", "d'", "ve"], [token.text for token in idve])
                    self.assertFalse(idve[1].is_stop)
                elif code in {"fr", "it", "lij"}:
                    self.assertEqual(["I'", "d'", "ve"], [token.text for token in idve])
                    self.assertTrue(idve[1].is_stop)
                elif code in three:
                    self.assertEqual(["I", "'", "d", "'", "ve"], [token.text for token in idve])
                else:
                    self.assertEqual(["I'd've"], [token.text for token in idve])
                    self.assertFalse(idve[0].like_num)

                yall = self.round_trip(pipeline, "y'all'd")
                if code == "zh":
                    self.assertEqual(list("y'all'd"), [token.text for token in yall])
                elif code in elide:
                    self.assertEqual(["y'", "all'", "d"], [token.text for token in yall])
                elif code in three:
                    self.assertEqual(["y", "'", "all", "'", "d"], [token.text for token in yall])
                else:
                    self.assertEqual(["y'all'd"], [token.text for token in yall])

                for text, glued in (
                    ("'til", {"tr", "vi"}),
                    ("'twas", {"tr", "vi"}),
                ):
                    doc = self.round_trip(pipeline, text)
                    if code == "zh":
                        self.assertEqual(list(text), [token.text for token in doc])
                    elif code in glued:
                        self.assertEqual([text], [token.text for token in doc])
                    else:
                        self.assertEqual(["'", text[1:]], [token.text for token in doc])
                        self.assertTrue(doc[0].is_quote)
                        self.assertEqual(text == "'til" and code in {"da", "is", "nb"}, doc[1].is_stop)

                oer = self.round_trip(pipeline, "o'er")
                if code == "zh":
                    self.assertEqual(list("o'er"), [token.text for token in oer])
                elif code in elide:
                    self.assertEqual(["o'", "er"], [token.text for token in oer])
                elif code in three:
                    self.assertEqual(["o", "'", "er"], [token.text for token in oer])
                else:
                    self.assertEqual(["o'er"], [token.text for token in oer])

                savoir = self.round_trip(pipeline, "savoir-faire")
                if code == "zh":
                    self.assertEqual(list("savoir-faire"), [token.text for token in savoir])
                elif code in hyphen_glue:
                    self.assertEqual(["savoir-faire"], [token.text for token in savoir])
                else:
                    self.assertEqual(["savoir", "-", "faire"], [token.text for token in savoir])

                tete = self.round_trip(pipeline, "t\u00eate-\u00e0-t\u00eate")
                if code == "zh":
                    self.assertEqual(list("t\u00eate-\u00e0-t\u00eate"), [token.text for token in tete])
                elif code in a_grave_glue:
                    self.assertEqual(["t\u00eate-\u00e0-t\u00eate"], [token.text for token in tete])
                else:
                    self.assertEqual(
                        ["t\u00eate", "-", "\u00e0", "-", "t\u00eate"],
                        [token.text for token in tete],
                    )
                    self.assertEqual(code in {"lij", "yo"}, tete[2].is_stop)

                dessus = self.round_trip(pipeline, "ci-dessus")
                if code == "zh":
                    self.assertEqual(list("ci-dessus"), [token.text for token in dessus])
                elif code in hyphen_glue:
                    self.assertEqual(["ci-dessus"], [token.text for token in dessus])
                else:
                    self.assertEqual(["ci", "-", "dessus"], [token.text for token in dessus])
                    self.assertEqual(code in {"fr", "it", "pl"}, dessus[0].is_stop)
                    self.assertEqual(code == "fr", dessus[2].is_stop)
                    self.assertEqual(code == "la", dessus[0].like_num)

                etait = self.round_trip(pipeline, "c'\u00e9tait")
                if code == "zh":
                    self.assertEqual(list("c'\u00e9tait"), [token.text for token in etait])
                elif code == "fr":
                    self.assertEqual(["c'", "\u00e9tait"], [token.text for token in etait])
                    self.assertTrue(etait[0].is_stop and etait[1].is_stop)
                elif code == "it":
                    self.assertEqual(["c'", "\u00e9tait"], [token.text for token in etait])
                    self.assertTrue(etait[0].is_stop)
                    self.assertFalse(etait[1].is_stop)
                elif code in {"ca", "lij"}:
                    self.assertEqual(["c'", "\u00e9tait"], [token.text for token in etait])
                    self.assertFalse(etait[0].is_stop or etait[1].is_stop)
                elif code in three:
                    self.assertEqual(["c", "'", "\u00e9tait"], [token.text for token in etait])
                else:
                    self.assertEqual(["c'\u00e9tait"], [token.text for token in etait])

                etais = self.round_trip(pipeline, "j'\u00e9tais")
                if code == "zh":
                    self.assertEqual(list("j'\u00e9tais"), [token.text for token in etais])
                elif code == "fr":
                    self.assertEqual(["j'", "\u00e9tais"], [token.text for token in etais])
                    self.assertTrue(etais[0].is_stop and etais[1].is_stop)
                elif code in {"ca", "it", "lij"}:
                    self.assertEqual(["j'", "\u00e9tais"], [token.text for token in etais])
                    self.assertFalse(etais[0].is_stop or etais[1].is_stop)
                elif code in three:
                    self.assertEqual(["j", "'", "\u00e9tais"], [token.text for token in etais])
                else:
                    self.assertEqual(["j'\u00e9tais"], [token.text for token in etais])

                ailleurs = self.round_trip(pipeline, "d'ailleurs")
                if code == "zh":
                    self.assertEqual(list("d'ailleurs"), [token.text for token in ailleurs])
                elif code in {"fr", "it", "lb", "lij"}:
                    self.assertEqual(["d'", "ailleurs"], [token.text for token in ailleurs])
                    self.assertTrue(ailleurs[0].is_stop)
                elif code == "ca":
                    self.assertEqual(["d'", "ailleurs"], [token.text for token in ailleurs])
                    self.assertFalse(ailleurs[0].is_stop)
                elif code in three:
                    self.assertEqual(["d", "'", "ailleurs"], [token.text for token in ailleurs])
                elif code == "yo":
                    self.assertEqual(["d'ailleurs"], [token.text for token in ailleurs])
                    self.assertTrue(ailleurs[0].like_num)
                else:
                    self.assertEqual(["d'ailleurs"], [token.text for token in ailleurs])
                    self.assertFalse(ailleurs[0].like_num)

                est = self.round_trip(pipeline, "est-il")
                if code == "zh":
                    self.assertEqual(list("est-il"), [token.text for token in est])
                elif code in hyphen_glue:
                    self.assertEqual(["est-il"], [token.text for token in est])
                elif code == "fr":
                    self.assertEqual(["est", "-il"], [token.text for token in est])
                    self.assertTrue(est[0].is_stop)
                    self.assertFalse(est[1].is_stop)
                elif code == "la":
                    self.assertEqual(["est", "-", "il"], [token.text for token in est])
                    self.assertTrue(est[0].is_stop)
                elif code in {"az", "it"}:
                    self.assertEqual(["est", "-", "il"], [token.text for token in est])
                    self.assertTrue(est[2].is_stop)
                    self.assertFalse(est[0].is_stop)
                else:
                    self.assertEqual(["est", "-", "il"], [token.text for token in est])
                    self.assertFalse(est[0].is_stop or est[2].is_stop)

                va = self.round_trip(pipeline, "va-t-il")
                if code == "zh":
                    self.assertEqual(list("va-t-il"), [token.text for token in va])
                elif code in hyphen_glue:
                    self.assertEqual(["va-t-il"], [token.text for token in va])
                elif code == "fr":
                    self.assertEqual(["va", "-t", "-il"], [token.text for token in va])
                    self.assertTrue(va[0].is_stop)
                elif code == "it":
                    self.assertEqual(["va", "-", "t", "-", "il"], [token.text for token in va])
                    self.assertTrue(va[0].is_stop and va[4].is_stop)
                elif code == "lt":
                    self.assertEqual(["va", "-", "t", "-", "il"], [token.text for token in va])
                    self.assertTrue(va[0].is_stop)
                    self.assertFalse(va[4].is_stop)
                elif code in {"sl", "sq", "yo"}:
                    self.assertEqual(["va", "-", "t", "-", "il"], [token.text for token in va])
                    self.assertTrue(va[2].is_stop)
                elif code == "az":
                    self.assertEqual(["va", "-", "t", "-", "il"], [token.text for token in va])
                    self.assertTrue(va[4].is_stop)
                    self.assertFalse(va[0].is_stop or va[2].is_stop)
                else:
                    self.assertEqual(["va", "-", "t", "-", "il"], [token.text for token in va])
                    self.assertFalse(any(token.is_stop for token in va))

                une = self.round_trip(pipeline, "quelqu'une")
                if code == "zh":
                    self.assertEqual(list("quelqu'une"), [token.text for token in une])
                elif code == "fr":
                    self.assertEqual(["quelqu'", "une"], [token.text for token in une])
                    self.assertTrue(une[1].is_stop and une[1].like_num)
                elif code in {"ca", "it", "lij"}:
                    self.assertEqual(["quelqu'", "une"], [token.text for token in une])
                    self.assertFalse(une[1].is_stop or une[1].like_num)
                elif code in three:
                    self.assertEqual(["quelqu", "'", "une"], [token.text for token in une])
                else:
                    self.assertEqual(["quelqu'une"], [token.text for token in une])

                ny = self.round_trip(pipeline, "n'y")
                if code == "zh":
                    self.assertEqual(list("n'y"), [token.text for token in ny])
                elif code == "fr":
                    self.assertEqual(["n'", "y"], [token.text for token in ny])
                    self.assertTrue(ny[0].is_stop and ny[1].is_stop)
                elif code == "lij":
                    self.assertEqual(["n'", "y"], [token.text for token in ny])
                    self.assertTrue(ny[0].is_stop)
                    self.assertFalse(ny[1].is_stop)
                elif code in {"ca", "it"}:
                    self.assertEqual(["n'", "y"], [token.text for token in ny])
                    self.assertFalse(ny[0].is_stop or ny[1].is_stop)
                elif code in three:
                    self.assertEqual(["n", "'", "y"], [token.text for token in ny])
                else:
                    self.assertEqual(["n'y"], [token.text for token in ny])

                sen = self.round_trip(pipeline, "s'en")
                if code == "zh":
                    self.assertEqual(list("s'en"), [token.text for token in sen])
                elif code in {"fr", "lij"}:
                    self.assertEqual(["s'", "en"], [token.text for token in sen])
                    self.assertTrue(sen[0].is_stop and sen[1].is_stop)
                elif code == "ca":
                    self.assertEqual(["s'", "en"], [token.text for token in sen])
                    self.assertFalse(sen[0].is_stop)
                    self.assertTrue(sen[1].is_stop)
                elif code == "it":
                    self.assertEqual(["s'", "en"], [token.text for token in sen])
                    self.assertTrue(sen[0].is_stop)
                    self.assertFalse(sen[1].is_stop)
                elif code in three:
                    self.assertEqual(["s", "'", "en"], [token.text for token in sen])
                else:
                    self.assertEqual(["s'en"], [token.text for token in sen])

                ma = self.round_trip(pipeline, "m'a")
                if code == "zh":
                    self.assertEqual(list("m'a"), [token.text for token in ma])
                elif code in {"fr", "it", "lij"}:
                    self.assertEqual(["m'", "a"], [token.text for token in ma])
                    self.assertTrue(ma[0].is_stop and ma[1].is_stop)
                elif code == "ca":
                    self.assertEqual(["m'", "a"], [token.text for token in ma])
                    self.assertFalse(ma[0].is_stop)
                    self.assertTrue(ma[1].is_stop)
                elif code in three:
                    self.assertEqual(["m", "'", "a"], [token.text for token in ma])
                else:
                    self.assertEqual(["m'a"], [token.text for token in ma])

                hors = self.round_trip(pipeline, "hors-d'\u0153uvre")
                if code == "zh":
                    self.assertEqual(list("hors-d'\u0153uvre"), [token.text for token in hors])
                elif code in {"da", "de", "el", "es", "fi", "fr", "hu", "lb", "nb", "nl", "nn", "pt", "ro", "sr", "sv", "vi"}:
                    self.assertEqual(["hors-d'\u0153uvre"], [token.text for token in hors])
                elif code == "ca":
                    self.assertEqual(["hors-d'", "\u0153uvre"], [token.text for token in hors])
                elif code in three:
                    self.assertEqual(["hors-d", "'", "\u0153uvre"], [token.text for token in hors])
                elif code in {"it", "lij"}:
                    self.assertEqual(["hors", "-", "d'", "\u0153uvre"], [token.text for token in hors])
                    self.assertTrue(hors[2].is_stop)
                else:
                    self.assertEqual(["hors", "-", "d'\u0153uvre"], [token.text for token in hors])

                oeil = self.round_trip(pipeline, "trompe-l'\u0153il")
                if code == "zh":
                    self.assertEqual(list("trompe-l'\u0153il"), [token.text for token in oeil])
                elif code in {"da", "de", "el", "es", "fi", "hu", "lb", "nb", "nl", "nn", "pt", "ro", "sr", "sv", "vi"}:
                    self.assertEqual(["trompe-l'\u0153il"], [token.text for token in oeil])
                elif code in {"fr", "it", "lij"}:
                    self.assertEqual(["trompe", "-", "l'", "\u0153il"], [token.text for token in oeil])
                    self.assertTrue(oeil[2].is_stop)
                elif code == "ca":
                    self.assertEqual(["trompe", "-l'", "\u0153il"], [token.text for token in oeil])
                elif code in three:
                    self.assertEqual(["trompe-l", "'", "\u0153il"], [token.text for token in oeil])
                else:
                    self.assertEqual(["trompe", "-", "l'\u0153il"], [token.text for token in oeil])

                for text, separator, own_space in (
                    ("a\r\nb", "\r\n", True),
                    ("a\u00a0\u00a0b", "\u00a0\u00a0", True),
                    ("a   b", "  ", False),
                ):
                    doc = self.round_trip(pipeline, text)
                    self.assertEqual(["a", separator, "b"], [token.text for token in doc])
                    self.assertTrue(doc[1].is_space)
                    if own_space:
                        self.assertEqual("", doc[0].whitespace_)
                    else:
                        self.assertEqual(" ", doc[0].whitespace_)
                    self.assertEqual(code in a_stop or code in both_stop, doc[0].is_stop)
                    self.assertEqual(code in both_stop, doc[2].is_stop)

    def test_numerals_symbols_urls_and_number_words(self):
        no_flags = (
            "\u200f",
            "\u2061",
            "\u2062",
            "\u2063",
            "\u2064",
            "\u206a",
            "\ufff9",
            "\ufffc",
            "\u070f",
            "\u180c",
            "\u17b4",
            "\u2601",
            "\u2602",
        )
        punct = ("\u1361", "\u1363", "\u055e", "\u055d", "\u301a", "\u301b")
        week_glue = {"da", "es", "fi", "hu", "lt", "nb", "nl", "nn", "pt", "ro", "sv", "vi"}
        week_split = {"en", "id", "ky", "lg", "ms", "pl", "sl", "tn"}
        ordinal_glue = {"da", "el", "es", "fi", "hu", "lt", "nb", "nl", "nn", "pt", "ro", "sv", "vi"}
        wifi_glue = {
            "ca", "da", "de", "el", "es", "fi", "hu", "it", "ky", "lb", "nb", "nl", "nn", "pt", "ro", "sr", "sv", "tt", "vi",
        }
        slash_glue = {"el", "hu", "lb", "nl", "ro", "vi"}

        for code, pipeline in self.pipelines.items():
            if pipeline is None:
                continue
            with self.subTest(code=code):
                for text in no_flags:
                    token = self.round_trip(pipeline, text)[0]
                    self.assertEqual(text, token.text)
                    self.assertFalse(
                        any(
                            (
                                token.is_alpha,
                                token.is_digit,
                                token.is_punct,
                                token.is_space,
                                token.is_stop,
                                token.is_currency,
                                token.like_num,
                            )
                        )
                    )
                for text in punct:
                    token = self.round_trip(pipeline, text)[0]
                    self.assertTrue(token.is_punct)
                    self.assertFalse(token.is_stop or token.is_quote or token.is_bracket)

                for text, lower, upper, title in (
                    ("\u014a", False, True, True),
                    ("\u018f", False, True, True),
                    ("\u0292", True, False, False),
                    ("\uac00", False, False, False),
                    ("\u3042", False, False, False),
                    ("\u30a2", False, False, False),
                    ("\u115f", False, False, False),
                    ("\u1160", False, False, False),
                    ("\uffa0", False, False, False),
                ):
                    token = self.round_trip(pipeline, text)[0]
                    self.assertTrue(token.is_alpha)
                    self.assertEqual(lower, token.is_lower)
                    self.assertEqual(upper, token.is_upper)
                    self.assertEqual(title, token.is_title)
                    self.assertFalse(token.like_num)

                for text in ("n\u0303", "c\u0327"):
                    doc = self.round_trip(pipeline, text)
                    if code == "zh":
                        self.assertEqual(list(text), [token.text for token in doc])
                        self.assertTrue(doc[0].is_alpha and doc[0].is_lower)
                    else:
                        self.assertEqual([text], [token.text for token in doc])
                        self.assertTrue(doc[0].is_lower)
                        self.assertFalse(doc[0].is_alpha)

                five = self.round_trip(pipeline, "\uff15")[0]
                self.assertTrue(five.is_digit)
                self.assertEqual(code != "grc", five.like_num)
                self.assertEqual(code == "zh", five.is_stop)
                for text in ("\u096b", "\u09eb", "\u0e55"):
                    token = self.round_trip(pipeline, text)[0]
                    self.assertTrue(token.is_digit)
                    self.assertEqual(code != "grc", token.like_num)
                    self.assertFalse(token.is_stop)

                ampm = self.round_trip(pipeline, "9.00am")
                if code == "zh":
                    self.assertEqual(list("9.00am"), [token.text for token in ampm])
                elif code == "pl":
                    self.assertEqual(["9", ".", "00am"], [token.text for token in ampm])
                    self.assertTrue(ampm[0].like_num)
                else:
                    self.assertEqual(["9.00am"], [token.text for token in ampm])
                    self.assertFalse(ampm[0].like_num)

                clock = self.round_trip(pipeline, "9 A.M.")
                self.assertEqual(" ", clock[0].whitespace_)
                if code == "zh":
                    self.assertEqual(["9", "A", ".", "M", "."], [token.text for token in clock])
                    self.assertTrue(clock[0].like_num and clock[0].is_stop)
                elif code == "grc":
                    self.assertEqual(["9", "A.M."], [token.text for token in clock])
                    self.assertTrue(clock[0].is_digit)
                    self.assertFalse(clock[0].like_num)
                elif code in {"lt", "sr"}:
                    self.assertEqual(["9", "A.M", "."], [token.text for token in clock])
                    self.assertTrue(clock[0].like_num)
                elif code == "sl":
                    self.assertEqual(["9", "A.", "M."], [token.text for token in clock])
                elif code == "pl":
                    self.assertEqual(["9", "A", ".", "M", "."], [token.text for token in clock])
                    self.assertTrue(clock[1].is_stop)
                    self.assertFalse(clock[3].is_stop)
                else:
                    self.assertEqual(["9", "A.M."], [token.text for token in clock])
                    self.assertTrue(clock[0].like_num)
                    self.assertTrue(clock[1].is_title)
                    self.assertFalse(clock[1].is_alpha)

                week = self.round_trip(pipeline, "2026-W41-7")
                if code == "zh":
                    self.assertEqual(list("2026-W41-7"), [token.text for token in week])
                elif code in week_glue:
                    self.assertEqual(["2026-W41-7"], [token.text for token in week])
                    self.assertTrue(week[0].is_title)
                    self.assertFalse(week[0].like_num)
                elif code in week_split:
                    self.assertEqual(
                        ["2026", "-", "W41", "-", "7"],
                        [token.text for token in week],
                    )
                    self.assertTrue(week[0].like_num and week[4].like_num)
                elif code == "el":
                    self.assertEqual(["2026-W", "41-7"], [token.text for token in week])
                    self.assertFalse(week[1].like_num)
                elif code == "grc":
                    self.assertEqual(
                        ["2026", "-", "W41", "-", "7"],
                        [token.text for token in week],
                    )
                    self.assertFalse(week[0].like_num or week[4].like_num)
                else:
                    self.assertEqual(["2026-W41", "-", "7"], [token.text for token in week])
                    self.assertTrue(week[2].like_num)
                    self.assertFalse(week[0].like_num)

                ordinal = self.round_trip(pipeline, "2026-284")
                if code == "zh":
                    self.assertEqual(list("2026-284"), [token.text for token in ordinal])
                elif code in ordinal_glue:
                    self.assertEqual(["2026-284"], [token.text for token in ordinal])
                    self.assertFalse(ordinal[0].like_num)
                elif code == "grc":
                    self.assertEqual(["2026", "-", "284"], [token.text for token in ordinal])
                    self.assertFalse(ordinal[0].like_num or ordinal[2].like_num)
                else:
                    self.assertEqual(["2026", "-", "284"], [token.text for token in ordinal])
                    self.assertTrue(ordinal[0].like_num and ordinal[2].like_num)

                degree = self.round_trip(pipeline, "5\u00b0C")
                if code == "zh":
                    self.assertEqual(["5", "\u00b0", "C"], [token.text for token in degree])
                    self.assertTrue(degree[0].like_num and degree[0].is_stop)
                elif code in {"hu", "nb", "nn", "ro", "vi"}:
                    self.assertEqual(["5\u00b0C"], [token.text for token in degree])
                    self.assertFalse(degree[0].like_num)
                elif code == "es":
                    self.assertEqual(["5", "\u00b0C"], [token.text for token in degree])
                    self.assertTrue(degree[0].like_num)
                elif code == "it":
                    self.assertEqual(["5\u00b0", "C"], [token.text for token in degree])
                elif code in {"sl", "sq"}:
                    self.assertEqual(["5", "\u00b0", "C"], [token.text for token in degree])
                    self.assertTrue(degree[2].is_stop)
                elif code == "la":
                    self.assertEqual(["5", "\u00b0", "C"], [token.text for token in degree])
                    self.assertTrue(degree[2].like_num)
                elif code == "grc":
                    self.assertEqual(["5", "\u00b0", "C"], [token.text for token in degree])
                    self.assertFalse(degree[0].like_num)
                else:
                    self.assertEqual(["5", "\u00b0", "C"], [token.text for token in degree])
                    self.assertTrue(degree[0].like_num)
                    self.assertFalse(degree[2].is_stop or degree[2].like_num)

                minus = self.round_trip(pipeline, "\u22125")
                if code == "zh":
                    self.assertEqual(["\u2212", "5"], [token.text for token in minus])
                    self.assertFalse(minus[0].is_punct)
                    self.assertTrue(minus[1].like_num)
                else:
                    self.assertEqual(["\u22125"], [token.text for token in minus])
                    self.assertFalse(minus[0].like_num or minus[0].is_punct)

                tilde = self.round_trip(pipeline, "9~10")
                if code == "zh":
                    self.assertEqual(["9", "~", "1", "0"], [token.text for token in tilde])
                    self.assertTrue(tilde[1].is_stop)
                    self.assertFalse(tilde[1].is_punct)
                elif code == "pl":
                    self.assertEqual(["9", "~", "10"], [token.text for token in tilde])
                    self.assertTrue(tilde[0].like_num and tilde[2].like_num)
                    self.assertFalse(tilde[1].is_punct)
                else:
                    self.assertEqual(["9~10"], [token.text for token in tilde])
                    self.assertFalse(tilde[0].like_num)

                plus = self.round_trip(pipeline, "+.5")
                if code == "zh":
                    self.assertEqual(["+", ".", "5"], [token.text for token in plus])
                    self.assertTrue(plus[0].is_stop and plus[1].is_punct and plus[2].like_num)
                elif code in {"el", "hu", "vi"}:
                    self.assertEqual(["+.5"], [token.text for token in plus])
                    self.assertTrue(plus[0].like_num)
                elif code == "grc":
                    self.assertEqual(["+.5"], [token.text for token in plus])
                    self.assertFalse(plus[0].like_num)
                elif code == "la":
                    self.assertEqual(["+", ".5"], [token.text for token in plus])
                    self.assertFalse(plus[1].like_num)
                else:
                    self.assertEqual(["+", ".5"], [token.text for token in plus])
                    self.assertFalse(plus[0].is_punct)
                    self.assertTrue(plus[1].like_num)

                euro = self.round_trip(pipeline, "\u20ac5.00")
                if code == "zh":
                    self.assertEqual(["\u20ac", "5", ".", "0", "0"], [token.text for token in euro])
                    self.assertTrue(euro[0].is_currency)
                elif code in {"bn", "hu", "vi"}:
                    self.assertEqual(["\u20ac5.00"], [token.text for token in euro])
                    self.assertFalse(euro[0].like_num)
                elif code == "pl":
                    self.assertEqual(["\u20ac", "5", ".", "00"], [token.text for token in euro])
                    self.assertTrue(euro[0].is_currency and euro[1].like_num and euro[3].like_num)
                elif code in {"grc", "la"}:
                    self.assertEqual(["\u20ac", "5.00"], [token.text for token in euro])
                    self.assertTrue(euro[0].is_currency)
                    self.assertFalse(euro[1].like_num)
                else:
                    self.assertEqual(["\u20ac", "5.00"], [token.text for token in euro])
                    self.assertTrue(euro[0].is_currency and euro[1].like_num)

                for text in ("\uffe5100", "\uff045"):
                    doc = self.round_trip(pipeline, text)
                    if code == "zh":
                        self.assertEqual(list(text), [token.text for token in doc])
                        self.assertTrue(doc[0].is_currency and doc[0].is_stop)
                    else:
                        self.assertEqual([text], [token.text for token in doc])
                        self.assertFalse(doc[0].is_currency or doc[0].like_num)

                eur = self.round_trip(pipeline, "EUR5")
                if code == "zh":
                    self.assertEqual(list("EUR5"), [token.text for token in eur])
                else:
                    self.assertEqual(["EUR5"], [token.text for token in eur])
                    self.assertTrue(eur[0].is_upper)
                    self.assertFalse(eur[0].is_title or eur[0].is_alpha or eur[0].is_currency)

                spaced = self.round_trip(pipeline, "50\u00a0%")
                if code == "zh":
                    self.assertEqual(["5", "0", "\u00a0", "%"], [token.text for token in spaced])
                    self.assertTrue(spaced[2].is_space and spaced[3].is_stop)
                else:
                    self.assertEqual(["50", "\u00a0", "%"], [token.text for token in spaced])
                    self.assertEqual(code != "grc", spaced[0].like_num)
                    self.assertTrue(spaced[1].is_space and spaced[2].is_punct)

                arabic = self.round_trip(pipeline, "100\u066a")
                if code == "zh":
                    self.assertEqual(["1", "0", "0", "\u066a"], [token.text for token in arabic])
                    self.assertTrue(arabic[-1].is_punct)
                    self.assertFalse(arabic[-1].is_stop)
                elif code == "vi":
                    self.assertEqual(["100\u066a"], [token.text for token in arabic])
                    self.assertFalse(arabic[0].like_num)
                else:
                    self.assertEqual(["100", "\u066a"], [token.text for token in arabic])
                    self.assertEqual(code != "grc", arabic[0].like_num)
                    self.assertTrue(arabic[1].is_punct)

                curly = self.round_trip(pipeline, "1\u2019000")
                if code == "zh":
                    self.assertEqual(list("1\u2019000"), [token.text for token in curly])
                else:
                    self.assertEqual(["1\u2019000"], [token.text for token in curly])
                    self.assertFalse(curly[0].like_num)

                sci = self.round_trip(pipeline, "2.5E-3")
                if code == "zh":
                    self.assertEqual(list("2.5E-3"), [token.text for token in sci])
                elif code in {"id", "ms"}:
                    self.assertEqual(["2.5E", "-", "3"], [token.text for token in sci])
                    self.assertTrue(sci[0].is_title and sci[2].like_num)
                    self.assertFalse(sci[0].like_num)
                elif code == "pt":
                    self.assertEqual(["2.", "5E-3"], [token.text for token in sci])
                    self.assertTrue(sci[0].like_num)
                elif code == "pl":
                    self.assertEqual(["2", ".", "5E", "-", "3"], [token.text for token in sci])
                    self.assertTrue(sci[0].like_num and sci[4].like_num)
                else:
                    self.assertEqual(["2.5E-3"], [token.text for token in sci])
                    self.assertTrue(sci[0].is_title)
                    self.assertFalse(sci[0].like_num)

                for text in ("1E10", "0X10"):
                    doc = self.round_trip(pipeline, text)
                    if code == "zh":
                        self.assertEqual(list(text), [token.text for token in doc])
                    else:
                        self.assertEqual([text], [token.text for token in doc])
                        self.assertTrue(doc[0].is_upper and doc[0].is_title)
                        self.assertFalse(doc[0].like_num)

                period = self.round_trip(pipeline, "PT15M")
                if code == "zh":
                    self.assertEqual(list("PT15M"), [token.text for token in period])
                elif code in {"grc", "ro", "vi"}:
                    self.assertEqual(["PT15M"], [token.text for token in period])
                    self.assertTrue(period[0].is_upper)
                    self.assertFalse(period[0].like_num)
                elif code in {"sl", "sq", "yo"}:
                    self.assertEqual(["PT15", "M"], [token.text for token in period])
                    self.assertTrue(period[1].is_stop)
                elif code == "la":
                    self.assertEqual(["PT15", "M"], [token.text for token in period])
                    self.assertTrue(period[1].like_num)
                else:
                    self.assertEqual(["PT15", "M"], [token.text for token in period])
                    self.assertFalse(period[1].is_stop or period[1].like_num)

                p2w = self.round_trip(pipeline, "P2W")
                if code == "zh":
                    self.assertEqual(list("P2W"), [token.text for token in p2w])
                else:
                    self.assertEqual(["P2W"], [token.text for token in p2w])
                    self.assertTrue(p2w[0].is_upper and p2w[0].is_title)
                    self.assertFalse(p2w[0].is_alpha)

                for text, like_num, stop in (
                    ("two", code == "en", code == "en"),
                    ("hundred", code == "en", code == "en"),
                    ("ten", code == "en", code in {"en", "cs", "nl", "pl", "sk"}),
                    ("zero", code in {"ca", "en", "fr", "pl", "pt", "ro"}, code in {"pt", "ro"}),
                    ("uno", code == "es", code in {"es", "it"}),
                    ("deux", code == "fr", code == "fr"),
                    ("eins", False, code == "is"),
                    ("\u043e\u0434\u0438\u043d", code in {"ru", "uk"}, code in {"ru", "uk"}),
                ):
                    doc = self.round_trip(pipeline, text)
                    if code == "zh":
                        self.assertEqual(list(text), [token.text for token in doc])
                        self.assertFalse(any(token.like_num or token.is_stop for token in doc))
                    else:
                        self.assertEqual([text], [token.text for token in doc])
                        self.assertEqual(like_num, doc[0].like_num)
                        self.assertEqual(stop, doc[0].is_stop)
                yi = self.round_trip(pipeline, "\u4e00")[0]
                self.assertTrue(yi.is_alpha)
                self.assertEqual(code == "zh", yi.like_num)
                self.assertEqual(code == "zh", yi.is_stop)
                self.assertFalse(yi.is_lower)

                path = self.round_trip(pipeline, "https://example.com/path?q=1#frag")
                mail = self.round_trip(pipeline, "user.name+tag@sub.example.com")
                if code == "zh":
                    self.assertGreater(len(path), 1)
                    self.assertGreater(len(mail), 1)
                else:
                    self.assertEqual([path.text], [token.text for token in path])
                    self.assertTrue(path[0].like_url)
                    self.assertEqual(code == "yo", path[0].like_num)
                    self.assertEqual([mail.text], [token.text for token in mail])
                    self.assertTrue(mail[0].like_email)
                    self.assertEqual(code == "yo", mail[0].like_num)

                ipv6 = self.round_trip(pipeline, "http://[::1]/")
                if code == "zh":
                    self.assertEqual(list("http://[::1]/"), [token.text for token in ipv6])
                elif code in {"de", "sl"}:
                    self.assertEqual(["http://[::1", "]", "/"], [token.text for token in ipv6])
                    self.assertTrue(ipv6[0].like_url and ipv6[1].is_bracket)
                else:
                    self.assertEqual(["http://[::1]/"], [token.text for token in ipv6])
                    self.assertTrue(ipv6[0].like_url)
                    self.assertFalse(ipv6[0].like_num)

                dotted = self.round_trip(pipeline, "example.com.")
                if code == "zh":
                    self.assertEqual(list("example.com."), [token.text for token in dotted])
                elif code in {"am", "ar", "fa", "grc", "ti", "vi"}:
                    self.assertEqual(["example.com."], [token.text for token in dotted])
                    self.assertFalse(dotted[0].like_url or dotted[0].like_num)
                else:
                    self.assertEqual(["example.com", "."], [token.text for token in dotted])
                    self.assertTrue(dotted[0].like_url)
                    self.assertEqual(code == "yo", dotted[0].like_num)

                ssh = self.round_trip(pipeline, "ssh://git@example.com/org/repo.git")
                if code == "zh":
                    self.assertEqual(list("ssh://git@example.com/org/repo.git"), [token.text for token in ssh])
                else:
                    self.assertEqual([ssh.text], [token.text for token in ssh])
                    self.assertFalse(ssh[0].like_url or ssh[0].like_email)
                    self.assertEqual(code == "yo", ssh[0].like_num)

                wifi = self.round_trip(pipeline, "Wi-Fi")
                if code == "zh":
                    self.assertEqual(list("Wi-Fi"), [token.text for token in wifi])
                elif code in wifi_glue:
                    self.assertEqual(["Wi-Fi"], [token.text for token in wifi])
                    self.assertTrue(wifi[0].is_title)
                else:
                    self.assertEqual(["Wi", "-", "Fi"], [token.text for token in wifi])

                for text in ("macOS", "eBay", "GmbH"):
                    doc = self.round_trip(pipeline, text)
                    if code == "zh":
                        self.assertEqual(list(text), [token.text for token in doc])
                    else:
                        self.assertEqual([text], [token.text for token in doc])
                        self.assertTrue(doc[0].is_alpha)
                        self.assertFalse(doc[0].is_lower or doc[0].is_upper or doc[0].is_title)

                null = self.round_trip(pipeline, "NULL")
                if code == "zh":
                    self.assertEqual(list("NULL"), [token.text for token in null])
                    self.assertTrue(all(token.is_upper and token.is_title for token in null))
                else:
                    self.assertEqual(["NULL"], [token.text for token in null])
                    self.assertTrue(null[0].is_alpha and null[0].is_upper)
                    self.assertFalse(null[0].is_title)

                quot = self.round_trip(pipeline, "&quot;")
                if code == "vi":
                    self.assertEqual(["&quot;"], [token.text for token in quot])
                elif code == "zh":
                    self.assertEqual(list("&quot;"), [token.text for token in quot])
                    self.assertTrue(quot[0].is_stop)
                else:
                    self.assertEqual(["&", "quot", ";"], [token.text for token in quot])
                    self.assertTrue(quot[0].is_punct and quot[2].is_punct)

                home = self.round_trip(pipeline, "~/docs")
                if code == "zh":
                    self.assertEqual(list("~/docs"), [token.text for token in home])
                elif code == "sl":
                    self.assertEqual(["~", "/", "docs"], [token.text for token in home])
                else:
                    self.assertEqual(["~/docs"], [token.text for token in home])
                    self.assertEqual(code == "yo", home[0].like_num)

                parts = self.round_trip(pipeline, "foo/bar/baz")
                if code == "zh":
                    self.assertEqual(list("foo/bar/baz"), [token.text for token in parts])
                elif code in slash_glue:
                    self.assertEqual(["foo/bar/baz"], [token.text for token in parts])
                else:
                    self.assertEqual(["foo", "/", "bar", "/", "baz"], [token.text for token in parts])
                    self.assertEqual(code == "hr", parts[2].is_stop)
                    self.assertFalse(parts[0].is_stop or parts[4].is_stop)

                lemme = self.round_trip(pipeline, "lemme")
                if code == "zh":
                    self.assertEqual(list("lemme"), [token.text for token in lemme])
                else:
                    self.assertEqual(["lemme"], [token.text for token in lemme])
                    self.assertEqual(code == "yo", lemme[0].like_num)

    def test_october_nineteen_clocks_scripts_and_compounds(self):
        # Clocks, ordinal dates, and compounds for 19 October 2026 (ISO week
        # 43, ordinal day 292). Chinese splits to characters unless noted.
        iso_glue = {"da", "el", "es", "fi", "hu", "lt", "nb", "nl", "nn", "pt", "ro", "sv", "vi"}
        dmy_glue = iso_glue | {"tr"}
        week_split = {"en", "id", "ky", "lg", "ms", "pl", "sl", "tn"}
        week_day_glue = iso_glue - {"el"}
        hyphen18 = {
            "ca", "da", "de", "el", "es", "fi", "hu", "ky", "lb", "nb", "nl", "nn", "pt", "ro", "sr", "sv", "tt", "vi",
        }
        grave_glue = (hyphen18 - {"el"}) | {"fr"}
        va_glue = hyphen18 | {"fr"}
        apostrophe_glue = {"da", "de", "es", "fi", "hu", "lb", "nb", "nl", "nn", "ro", "sr", "sv", "vi"}
        in_stop = {"af", "az", "en", "ga", "it", "la", "lij", "sl"}
        et_stop = {"az", "et", "la"}
        a_stop = {"fr", "lb", "lij", "pt", "vi", "yo"}
        de_stop = {"ca", "da", "fr", "ga", "hr", "hu", "la", "lb", "nb", "nl", "pt", "ro", "tr"}
        fin_and_de = {"es", "lij", "sv"}
        debut_de = de_stop | {"es", "lij", "sv"}
        lets_glue = {
            "am", "ar", "bn", "da", "de", "el", "es", "fa", "fi", "grc", "hu", "nb", "nl", "nn", "pl", "ro", "sl", "sr",
            "sv", "ti", "vi",
        }
        yo_days = {"lundi matin", "mercredi matin", "jeudi soir"}
        clocks = ("19h", "19h19", "19h30", "19h00", "09h19")
        script_digits = ("²", "₁", "𝟐", "𝟙", "߁", "༡", "၁", "១", "๑", "᱑")
        currencies = ("₱", "₵", "৳")
        plain_punct = ("‰", "′", "〖", "†")
        unflagged = ("±", "∞", "√", "✓", "🔔", "☕")
        lower_letters = ("ə", "ɛ", "œ", "æ", "ÿ", "µ")

        for code, pipeline in self.pipelines.items():
            if pipeline is None:
                continue
            with self.subTest(code=code):
                for text in currencies:
                    token = self.round_trip(pipeline, text)[0]
                    self.assertTrue(token.is_currency)
                    self.assertFalse(token.is_punct or token.is_stop or token.is_alpha or token.like_num)

                for text in plain_punct:
                    token = self.round_trip(pipeline, text)[0]
                    self.assertTrue(token.is_punct)
                    self.assertFalse(
                        token.is_stop or token.is_quote or token.is_bracket or token.is_currency
                    )

                quote = self.round_trip(pipeline, "„")[0]
                self.assertTrue(quote.is_punct and quote.is_quote and quote.is_left_punct)
                self.assertFalse(quote.is_right_punct or quote.is_stop or quote.is_bracket)

                for text in unflagged:
                    token = self.round_trip(pipeline, text)[0]
                    self.assertFalse(
                        any(
                            (
                                token.is_alpha,
                                token.is_digit,
                                token.is_punct,
                                token.is_space,
                                token.is_stop,
                                token.is_currency,
                                token.like_num,
                            )
                        )
                    )

                times = self.round_trip(pipeline, "×")[0]
                self.assertEqual(code == "zh", times.is_stop)
                self.assertFalse(times.is_punct or times.is_alpha or times.like_num)

                for text in lower_letters:
                    token = self.round_trip(pipeline, text)[0]
                    self.assertTrue(token.is_alpha and token.is_lower)
                    self.assertFalse(token.is_stop or token.like_num)

                caron = self.round_trip(pipeline, "č")[0]
                self.assertTrue(caron.is_alpha and caron.is_lower)
                self.assertEqual(code == "sl", caron.is_stop)
                self.assertFalse(caron.like_num)

                glottal = self.round_trip(pipeline, "ʔ")[0]
                self.assertTrue(glottal.is_alpha)
                self.assertFalse(glottal.is_lower or glottal.is_upper or glottal.is_title)
                self.assertFalse(glottal.is_stop or glottal.like_num)

                pi = self.round_trip(pipeline, "π")[0]
                self.assertTrue(pi.is_alpha and pi.is_lower)
                self.assertEqual(code == "grc", pi.like_num)
                self.assertFalse(pi.is_stop)

                omega = self.round_trip(pipeline, "Ω")[0]
                self.assertTrue(omega.is_alpha and omega.is_upper and omega.is_title)
                self.assertEqual(code == "el", omega.is_stop)
                self.assertFalse(omega.like_num)

                for text in script_digits:
                    token = self.round_trip(pipeline, text)[0]
                    self.assertTrue(token.is_digit)
                    self.assertEqual(code != "grc", token.like_num)
                    self.assertFalse(token.is_stop)

                for text in ("⅓", "⅛"):
                    token = self.round_trip(pipeline, text)[0]
                    self.assertFalse(
                        token.is_digit or token.like_num or token.is_punct or token.is_alpha or token.is_stop
                    )

                fullwidth = self.round_trip(pipeline, "１９")
                if code == "zh":
                    self.assertEqual(["１", "９"], [token.text for token in fullwidth])
                    self.assertTrue(all(token.is_digit and token.like_num and token.is_stop for token in fullwidth))
                else:
                    self.assertEqual(["１９"], [token.text for token in fullwidth])
                    self.assertTrue(fullwidth[0].is_digit)
                    self.assertEqual(code != "grc", fullwidth[0].like_num)

                for text in clocks:
                    doc = self.round_trip(pipeline, text)
                    if code == "zh":
                        self.assertEqual(list(text), [token.text for token in doc])
                        self.assertTrue(all(token.like_num for token in doc if token.is_digit))
                    else:
                        self.assertEqual([text], [token.text for token in doc])
                        self.assertTrue(doc[0].is_lower)
                        self.assertFalse(doc[0].like_num or doc[0].is_alpha)

                clock = self.round_trip(pipeline, "19:19")
                if code == "zh":
                    self.assertEqual(list("19:19"), [token.text for token in clock])
                    self.assertTrue(clock[2].is_punct and clock[2].is_stop)
                else:
                    self.assertEqual(["19:19"], [token.text for token in clock])
                    self.assertFalse(clock[0].like_num or clock[0].is_alpha)

                pm = self.round_trip(pipeline, "7:19pm")
                if code == "zh":
                    self.assertEqual(list("7:19pm"), [token.text for token in pm])
                else:
                    self.assertEqual(["7:19pm"], [token.text for token in pm])
                    self.assertTrue(pm[0].is_lower)
                    self.assertFalse(pm[0].like_num or pm[0].is_alpha)

                spaced = self.round_trip(pipeline, "19 h 19")
                self.assertEqual(" ", spaced[0].whitespace_ if code != "zh" else spaced[1].whitespace_)
                if code == "zh":
                    self.assertEqual(["1", "9", "h", "1", "9"], [token.text for token in spaced])
                    self.assertEqual(" ", spaced[1].whitespace_)
                    self.assertEqual(" ", spaced[2].whitespace_)
                elif code == "grc":
                    self.assertEqual(["19", "h", "19"], [token.text for token in spaced])
                    self.assertTrue(spaced[0].is_digit and spaced[2].is_digit)
                    self.assertFalse(spaced[0].like_num or spaced[1].is_stop)
                elif code in {"ro", "sl"}:
                    self.assertEqual(["19", "h", "19"], [token.text for token in spaced])
                    self.assertTrue(spaced[0].like_num and spaced[1].is_stop and spaced[2].like_num)
                else:
                    self.assertEqual(["19", "h", "19"], [token.text for token in spaced])
                    self.assertTrue(spaced[0].like_num and spaced[2].like_num)
                    self.assertFalse(spaced[1].is_stop)

                self._assert_iso_date(pipeline, code, "2026-10-19", iso_glue)
                self._assert_iso_date(pipeline, code, "2026-292", iso_glue)
                self._assert_iso_date(pipeline, code, "19-10-2026", dmy_glue)

                slash = self.round_trip(pipeline, "19/10/2026")
                if code == "zh":
                    self.assertEqual(list("19/10/2026"), [token.text for token in slash])
                    self.assertTrue(slash[2].is_punct and slash[2].is_stop)
                elif code in {"de", "id", "ms"}:
                    self.assertEqual(["19", "/", "10", "/", "2026"], [token.text for token in slash])
                    self.assertTrue(slash[0].like_num and slash[2].like_num and slash[4].like_num)
                elif code == "fa":
                    self.assertEqual(["19/10/2026"], [token.text for token in slash])
                    self.assertTrue(slash[0].like_num)
                else:
                    self.assertEqual(["19/10/2026"], [token.text for token in slash])
                    self.assertFalse(slash[0].like_num)

                dotted = self.round_trip(pipeline, "19.10.2026")
                if code == "zh":
                    self.assertEqual(list("19.10.2026"), [token.text for token in dotted])
                elif code == "pl":
                    self.assertEqual(["19", ".", "10", ".", "2026"], [token.text for token in dotted])
                    self.assertTrue(all(token.like_num for token in dotted if token.text != "."))
                elif code in {"grc", "la"}:
                    self.assertEqual(["19.10.2026"], [token.text for token in dotted])
                    self.assertFalse(dotted[0].like_num)
                else:
                    self.assertEqual(["19.10.2026"], [token.text for token in dotted])
                    self.assertTrue(dotted[0].like_num)

                stamp = self.round_trip(pipeline, "2026-10-19T19:00:00Z")
                if code == "zh":
                    self.assertGreater(len(stamp), 1)
                    self.assertTrue(stamp[0].like_num and stamp[0].is_stop)
                elif code in iso_glue:
                    self.assertEqual(["2026-10-19T19:00:00Z"], [token.text for token in stamp])
                    self.assertTrue(stamp[0].is_upper and stamp[0].is_title)
                    self.assertFalse(stamp[0].is_alpha or stamp[0].like_num)
                elif code == "grc":
                    self.assertEqual(
                        ["2026", "-", "10", "-", "19T19:00:00Z"],
                        [token.text for token in stamp],
                    )
                    self.assertTrue(stamp[0].is_digit and stamp[2].is_digit)
                    self.assertFalse(stamp[0].like_num or stamp[4].like_num)
                else:
                    self.assertEqual(
                        ["2026", "-", "10", "-", "19T19:00:00Z"],
                        [token.text for token in stamp],
                    )
                    self.assertTrue(stamp[0].like_num and stamp[2].like_num)
                    self.assertTrue(stamp[4].is_upper and stamp[4].is_title)
                    self.assertFalse(stamp[4].like_num)

                week = self.round_trip(pipeline, "2026-W43")
                if code == "zh":
                    self.assertEqual(list("2026-W43"), [token.text for token in week])
                elif code in week_split:
                    self.assertEqual(["2026", "-", "W43"], [token.text for token in week])
                    self.assertTrue(week[0].like_num)
                    self.assertTrue(week[2].is_upper and week[2].is_title)
                    self.assertFalse(week[2].like_num)
                elif code == "grc":
                    self.assertEqual(["2026", "-", "W43"], [token.text for token in week])
                    self.assertTrue(week[0].is_digit)
                    self.assertFalse(week[0].like_num)
                else:
                    self.assertEqual(["2026-W43"], [token.text for token in week])
                    self.assertTrue(week[0].is_upper and week[0].is_title)
                    self.assertFalse(week[0].is_alpha or week[0].like_num)

                week_day = self.round_trip(pipeline, "2026-W43-1")
                if code == "zh":
                    self.assertEqual(list("2026-W43-1"), [token.text for token in week_day])
                elif code in week_day_glue:
                    self.assertEqual(["2026-W43-1"], [token.text for token in week_day])
                    self.assertTrue(week_day[0].is_upper and week_day[0].is_title)
                    self.assertFalse(week_day[0].like_num)
                elif code in week_split:
                    self.assertEqual(["2026", "-", "W43", "-", "1"], [token.text for token in week_day])
                    self.assertTrue(week_day[0].like_num and week_day[4].like_num)
                elif code == "el":
                    self.assertEqual(["2026-W", "43-1"], [token.text for token in week_day])
                    self.assertFalse(week_day[0].like_num or week_day[1].like_num)
                elif code == "grc":
                    self.assertEqual(["2026", "-", "W43", "-", "1"], [token.text for token in week_day])
                    self.assertTrue(week_day[0].is_digit and week_day[4].is_digit)
                    self.assertFalse(week_day[0].like_num or week_day[4].like_num)
                else:
                    self.assertEqual(["2026-W43", "-", "1"], [token.text for token in week_day])
                    self.assertTrue(week_day[2].like_num)
                    self.assertFalse(week_day[0].like_num)

                self._assert_hyphen_trio(
                    pipeline, code, "mother-in-law", hyphen18, in_stop, "in", 2,
                )
                self._assert_grave_compound(pipeline, code, "vis-à-vis", grave_glue)
                self._assert_hyphen_pair(pipeline, code, "peut-être", grave_glue)
                self._assert_va_et_vient(pipeline, code, va_glue, et_stop)
                self._assert_cest_a_dire(pipeline, code, apostrophe_glue)
                self._assert_sil_vous_plait(pipeline, code)
                self._assert_celui_ci(pipeline, code, hyphen18)
                self._assert_nest_ce(pipeline, code, apostrophe_glue)
                self._assert_elision(pipeline, code, "qu'à", "qu'", "à", {"ca", "it"}, code == "fr", code in {"fr", "lij"})
                self._assert_elision(
                    pipeline, code, "n'a", "n'", "a", {"ca", "it"}, code in {"fr", "lij"}, True if code in {"ca", "fr", "it", "lij"} else False,
                )
                self._assert_elision(pipeline, code, "j'y", "j'", "y", {"ca", "it", "lij"}, code == "fr", code == "fr")
                self._assert_men(pipeline, code)
                self._assert_lon(pipeline, code)
                self._assert_dabord(pipeline, code)
                self._assert_nt(pipeline, code, "doesn't", "does", True)
                self._assert_lets(pipeline, code, lets_glue)
                self._assert_mustve(pipeline, code)
                self._assert_whodve(pipeline, code)
                self._assert_title_dot(
                    pipeline, code, "Gen.", {"am", "ar", "en", "fa", "grc", "ms", "nl", "pt", "ti", "vi"},
                )
                self._assert_title_dot(
                    pipeline, code, "Mme.", {"am", "ar", "fa", "fr", "grc", "hu", "lb", "ti", "vi"},
                )
                self._assert_title_dot(
                    pipeline, code, "Ltd.", {"am", "ar", "en", "fa", "grc", "hu", "nl", "pt", "ti", "tr", "vi"},
                )
                self._assert_initials(pipeline, code, "U.K.")
                aka = self.round_trip(pipeline, "a.k.a.")
                if code == "zh":
                    self.assertEqual(list("a.k.a."), [token.text for token in aka])
                    self.assertTrue(all(token.is_stop for token in aka if token.text == "."))
                elif code in {"am", "ar", "fa", "grc", "ti", "vi"}:
                    self.assertEqual(["a.k.a."], [token.text for token in aka])
                    self.assertTrue(aka[0].is_lower)
                    self.assertFalse(aka[0].is_alpha or aka[0].like_num)
                else:
                    self.assertEqual(["a.k.a", "."], [token.text for token in aka])
                    self.assertTrue(aka[1].is_punct)

                for text in yo_days:
                    day, part = text.split(" ")
                    doc = self.round_trip(pipeline, text)
                    if code == "zh":
                        self.assertEqual([ch for ch in text if ch != " "], [token.text for token in doc])
                    else:
                        self.assertEqual([day, part], [token.text for token in doc])
                        self.assertEqual(" ", doc[0].whitespace_)
                        self.assertEqual(code == "yo", doc[0].like_num)
                        self.assertFalse(doc[0].is_stop or doc[1].is_stop or doc[1].like_num)

                vendredi = self.round_trip(pipeline, "vendredi midi")
                if code == "zh":
                    self.assertEqual(
                        [ch for ch in "vendredi midi" if ch != " "],
                        [token.text for token in vendredi],
                    )
                else:
                    self.assertEqual(["vendredi", "midi"], [token.text for token in vendredi])
                    self.assertEqual(code == "yo", vendredi[0].like_num and vendredi[1].like_num)
                    self.assertFalse(vendredi[0].is_stop or vendredi[1].is_stop)

                self._assert_apres_demain_soir(pipeline, code, grave_glue)
                self._assert_cet_apres_midi(pipeline, code, hyphen18 - {"el"})
                self._assert_dans_phrase(
                    pipeline, code, "dans une heure", "une", une_stop=code in {"fr", "sq"}, une_num=code == "fr",
                )
                self._assert_dans_phrase(
                    pipeline, code, "dans cinq jours", "cinq", une_stop=code == "fr", une_num=code == "fr",
                )
                self._assert_clock_phrase(pipeline, code, "à 19h", "à", code in a_stop)
                self._assert_clock_phrase(pipeline, code, "vers 19h", "vers", code == "fr")
                self._assert_clock_phrase(pipeline, code, "dès 19h", "dès", code == "fr")

                heures = self.round_trip(pipeline, "19 heures")
                if code == "zh":
                    self.assertEqual([ch for ch in "19 heures" if ch != " "], [token.text for token in heures])
                    self.assertEqual(" ", heures[1].whitespace_)
                else:
                    self.assertEqual(["19", "heures"], [token.text for token in heures])
                    self.assertEqual(code != "grc", heures[0].like_num)
                    self.assertEqual(code == "grc", heures[0].is_digit and not heures[0].like_num)
                    self.assertFalse(heures[1].is_stop)

                self._assert_dix_neuf(pipeline, code, hyphen18)
                self._assert_deux_mots(pipeline, code, "chaque midi", "chaque", code == "fr", False, code == "yo")
                self._assert_deux_mots(pipeline, code, "ce midi", "ce", code in {"fr", "ro"}, False, code == "yo")
                self._assert_journee(pipeline, code, "fin de journée", de_stop, fin_and_de, {"it"})
                self._assert_journee(pipeline, code, "début de journée", debut_de, set(), set())

                flag = self.round_trip(pipeline, "🇫🇷")
                watch = self.round_trip(pipeline, "⌚️")
                if code in {"hu", "vi"}:
                    self.assertEqual(["🇫🇷"], [token.text for token in flag])
                    self.assertEqual(["⌚️"], [token.text for token in watch])
                else:
                    self.assertEqual(["🇫", "🇷"], [token.text for token in flag])
                    self.assertEqual(["⌚", "️"], [token.text for token in watch])
                self.assertFalse(any(token.is_punct or token.like_num or token.is_alpha for token in flag))
                self.assertFalse(any(token.is_punct or token.like_num or token.is_alpha for token in watch))

    def _assert_iso_date(self, pipeline, code, text, glue):
        doc = self.round_trip(pipeline, text)
        pieces = text.split("-")
        split = []
        for index, piece in enumerate(pieces):
            if index:
                split.append("-")
            split.append(piece)
        if code == "zh":
            self.assertEqual(list(text), [token.text for token in doc])
            self.assertTrue(all(token.is_stop for token in doc if token.text == "-"))
        elif code in glue:
            self.assertEqual([text], [token.text for token in doc])
            self.assertFalse(doc[0].like_num or doc[0].is_alpha)
        elif code == "grc":
            self.assertEqual(split, [token.text for token in doc])
            self.assertTrue(all(token.is_digit and not token.like_num for token in doc if token.text.isdigit()))
        else:
            self.assertEqual(split, [token.text for token in doc])
            self.assertTrue(all(token.like_num for token in doc if token.text.isdigit()))

    def _assert_hyphen_trio(self, pipeline, code, text, glue, stop_middle, middle, middle_index):
        doc = self.round_trip(pipeline, text)
        parts = text.split("-")
        if code == "zh":
            self.assertEqual(list(text), [token.text for token in doc])
        elif code in glue:
            self.assertEqual([text], [token.text for token in doc])
            self.assertTrue(doc[0].is_lower)
            self.assertFalse(doc[0].is_stop)
        else:
            self.assertEqual([parts[0], "-", parts[1], "-", parts[2]], [token.text for token in doc])
            self.assertEqual(code in stop_middle, doc[middle_index].is_stop)
            self.assertEqual(middle, doc[middle_index].text)

    def _assert_grave_compound(self, pipeline, code, text, glue):
        doc = self.round_trip(pipeline, text)
        if code == "zh":
            self.assertEqual(list(text), [token.text for token in doc])
        elif code in glue:
            self.assertEqual([text], [token.text for token in doc])
            self.assertFalse(doc[0].is_stop)
        elif code in {"lij", "yo"}:
            self.assertEqual(["vis", "-", "à", "-", "vis"], [token.text for token in doc])
            self.assertTrue(doc[2].is_stop)
            self.assertFalse(doc[0].is_stop or doc[4].is_stop)
        elif code in {"lt", "lv"}:
            self.assertEqual(["vis", "-", "à", "-", "vis"], [token.text for token in doc])
            self.assertTrue(doc[0].is_stop and doc[4].is_stop)
            self.assertFalse(doc[2].is_stop)
        else:
            self.assertEqual(["vis", "-", "à", "-", "vis"], [token.text for token in doc])
            self.assertFalse(any(token.is_stop for token in doc))

    def _assert_hyphen_pair(self, pipeline, code, text, glue):
        doc = self.round_trip(pipeline, text)
        left, right = text.split("-")
        if code == "zh":
            self.assertEqual(list(text), [token.text for token in doc])
        elif code in glue:
            self.assertEqual([text], [token.text for token in doc])
        else:
            self.assertEqual([left, "-", right], [token.text for token in doc])
            self.assertFalse(any(token.is_stop for token in doc))

    def _assert_va_et_vient(self, pipeline, code, glue, et_stop):
        doc = self.round_trip(pipeline, "va-et-vient")
        if code == "zh":
            self.assertEqual(list("va-et-vient"), [token.text for token in doc])
        elif code in glue:
            self.assertEqual(["va-et-vient"], [token.text for token in doc])
        elif code == "lt":
            self.assertEqual(["va", "-", "et", "-", "vient"], [token.text for token in doc])
            self.assertTrue(doc[0].is_stop and doc[2].is_stop)
        elif code == "it":
            self.assertEqual(["va", "-", "et", "-", "vient"], [token.text for token in doc])
            self.assertTrue(doc[0].is_stop)
            self.assertFalse(doc[2].is_stop)
        elif code in et_stop:
            self.assertEqual(["va", "-", "et", "-", "vient"], [token.text for token in doc])
            self.assertTrue(doc[2].is_stop)
            self.assertFalse(doc[0].is_stop)
        else:
            self.assertEqual(["va", "-", "et", "-", "vient"], [token.text for token in doc])
            self.assertFalse(any(token.is_stop for token in doc))

    def _assert_cest_a_dire(self, pipeline, code, glue):
        doc = self.round_trip(pipeline, "c'est-à-dire")
        if code == "zh":
            self.assertEqual(list("c'est-à-dire"), [token.text for token in doc])
        elif code in glue:
            self.assertEqual(["c'est-à-dire"], [token.text for token in doc])
        elif code in {"ca", "pt"}:
            self.assertEqual(["c'", "est-à-dire"], [token.text for token in doc])
            self.assertFalse(doc[0].is_stop or doc[1].is_stop)
        elif code in {"ky", "tt"}:
            self.assertEqual(["c", "'", "est-à-dire"], [token.text for token in doc])
            self.assertTrue(doc[1].is_quote)
        elif code == "fr":
            self.assertEqual(["c'", "est", "-", "à", "-", "dire"], [token.text for token in doc])
            self.assertTrue(all(token.is_stop for token in doc if token.text != "-"))
        elif code == "it":
            self.assertEqual(["c'", "est", "-", "à", "-", "dire"], [token.text for token in doc])
            self.assertTrue(doc[0].is_stop and doc[5].is_stop)
            self.assertFalse(doc[1].is_stop or doc[3].is_stop)
        elif code == "lij":
            self.assertEqual(["c'", "est", "-", "à", "-", "dire"], [token.text for token in doc])
            self.assertTrue(doc[3].is_stop)
            self.assertFalse(doc[0].is_stop or doc[5].is_stop)
        elif code == "yo":
            self.assertEqual(["c'est", "-", "à", "-", "dire"], [token.text for token in doc])
            self.assertTrue(doc[2].is_stop and doc[4].like_num)
        else:
            self.assertEqual(["c'est", "-", "à", "-", "dire"], [token.text for token in doc])
            self.assertFalse(any(token.is_stop or token.like_num for token in doc))

    def _assert_sil_vous_plait(self, pipeline, code):
        doc = self.round_trip(pipeline, "s'il vous plaît")
        if code == "zh":
            self.assertEqual([ch for ch in "s'il vous plaît" if ch != " "], [token.text for token in doc])
        elif code in {"ky", "tt"}:
            self.assertEqual(["s", "'", "il", "vous", "plaît"], [token.text for token in doc])
            self.assertTrue(doc[1].is_quote)
            self.assertEqual(" ", doc[2].whitespace_)
        elif code == "ca":
            self.assertEqual(["s'", "il", "vous", "plaît"], [token.text for token in doc])
            self.assertFalse(any(token.is_stop for token in doc))
            self.assertEqual(" ", doc[1].whitespace_)
        elif code == "fr":
            self.assertEqual(["s'", "il", "vous", "plaît"], [token.text for token in doc])
            self.assertTrue(doc[0].is_stop and doc[1].is_stop and doc[2].is_stop)
            self.assertFalse(doc[3].is_stop)
        elif code == "it":
            self.assertEqual(["s'", "il", "vous", "plaît"], [token.text for token in doc])
            self.assertTrue(doc[0].is_stop and doc[1].is_stop)
            self.assertFalse(doc[2].is_stop)
        elif code == "lij":
            self.assertEqual(["s'", "il", "vous", "plaît"], [token.text for token in doc])
            self.assertTrue(doc[0].is_stop)
            self.assertFalse(doc[1].is_stop or doc[2].is_stop)
        else:
            self.assertEqual(["s'il", "vous", "plaît"], [token.text for token in doc])
            self.assertEqual(" ", doc[0].whitespace_)
            self.assertFalse(any(token.is_stop for token in doc))

    def _assert_celui_ci(self, pipeline, code, glue):
        doc = self.round_trip(pipeline, "celui-ci")
        if code == "zh":
            self.assertEqual(list("celui-ci"), [token.text for token in doc])
        elif code == "fr":
            self.assertEqual(["celui-ci"], [token.text for token in doc])
            self.assertTrue(doc[0].is_stop)
        elif code in glue:
            self.assertEqual(["celui-ci"], [token.text for token in doc])
            self.assertFalse(doc[0].is_stop)
        elif code in {"it", "pl"}:
            self.assertEqual(["celui", "-", "ci"], [token.text for token in doc])
            self.assertTrue(doc[2].is_stop)
            self.assertFalse(doc[2].like_num)
        elif code == "la":
            self.assertEqual(["celui", "-", "ci"], [token.text for token in doc])
            self.assertTrue(doc[2].like_num)
            self.assertFalse(doc[2].is_stop)
        else:
            self.assertEqual(["celui", "-", "ci"], [token.text for token in doc])
            self.assertFalse(doc[2].is_stop or doc[2].like_num)

    def _assert_nest_ce(self, pipeline, code, glue):
        doc = self.round_trip(pipeline, "n'est-ce")
        if code == "zh":
            self.assertEqual(list("n'est-ce"), [token.text for token in doc])
        elif code in glue:
            self.assertEqual(["n'est-ce"], [token.text for token in doc])
        elif code in {"ca", "el", "pt"}:
            self.assertEqual(["n'", "est-ce"], [token.text for token in doc])
            self.assertFalse(doc[0].is_stop)
        elif code in {"ky", "tt"}:
            self.assertEqual(["n", "'", "est-ce"], [token.text for token in doc])
            self.assertTrue(doc[1].is_quote)
        elif code == "fr":
            self.assertEqual(["n'", "est", "-ce"], [token.text for token in doc])
            self.assertTrue(doc[0].is_stop and doc[1].is_stop)
            self.assertFalse(doc[2].is_stop)
        elif code == "it":
            self.assertEqual(["n'", "est", "-", "ce"], [token.text for token in doc])
            self.assertFalse(any(token.is_stop for token in doc))
        elif code == "lij":
            self.assertEqual(["n'", "est", "-", "ce"], [token.text for token in doc])
            self.assertTrue(doc[0].is_stop)
            self.assertFalse(doc[1].is_stop or doc[3].is_stop)
        else:
            self.assertEqual(["n'est", "-", "ce"], [token.text for token in doc])
            self.assertFalse(any(token.is_stop for token in doc))

    def _assert_elision(self, pipeline, code, text, head, tail, split, head_stop, tail_stop):
        doc = self.round_trip(pipeline, text)
        if code == "zh":
            self.assertEqual(list(text), [token.text for token in doc])
        elif code in {"ky", "tt"}:
            stem = head[:-1]
            self.assertEqual([stem, "'", tail], [token.text for token in doc])
            self.assertTrue(doc[1].is_quote)
            self.assertFalse(doc[0].is_stop or doc[2].is_stop)
        elif code in split or code in {"fr", "lij"}:
            self.assertEqual([head, tail], [token.text for token in doc])
            self.assertEqual(head_stop, doc[0].is_stop)
            self.assertEqual(tail_stop, doc[1].is_stop)
        else:
            self.assertEqual([text], [token.text for token in doc])
            self.assertFalse(doc[0].is_stop)

    def _assert_men(self, pipeline, code):
        doc = self.round_trip(pipeline, "m'en")
        if code == "zh":
            self.assertEqual(list("m'en"), [token.text for token in doc])
        elif code in {"ky", "tt"}:
            self.assertEqual(["m", "'", "en"], [token.text for token in doc])
            self.assertFalse(doc[0].is_stop or doc[2].is_stop)
        elif code in {"fr", "lij"}:
            self.assertEqual(["m'", "en"], [token.text for token in doc])
            self.assertTrue(doc[0].is_stop and doc[1].is_stop)
        elif code == "ca":
            self.assertEqual(["m'", "en"], [token.text for token in doc])
            self.assertFalse(doc[0].is_stop)
            self.assertTrue(doc[1].is_stop)
        elif code == "it":
            self.assertEqual(["m'", "en"], [token.text for token in doc])
            self.assertTrue(doc[0].is_stop)
            self.assertFalse(doc[1].is_stop)
        else:
            self.assertEqual(["m'en"], [token.text for token in doc])

    def _assert_lon(self, pipeline, code):
        doc = self.round_trip(pipeline, "l'on")
        if code == "zh":
            self.assertEqual(list("l'on"), [token.text for token in doc])
        elif code in {"ky", "tt"}:
            self.assertEqual(["l", "'", "on"], [token.text for token in doc])
            self.assertFalse(any(token.is_stop for token in doc))
        elif code in {"it", "lij"}:
            self.assertEqual(["l'", "on"], [token.text for token in doc])
            self.assertTrue(doc[0].is_stop)
            self.assertFalse(doc[1].is_stop)
        elif code == "ca":
            self.assertEqual(["l'", "on"], [token.text for token in doc])
            self.assertFalse(doc[0].is_stop)
            self.assertTrue(doc[1].is_stop)
        elif code == "fr":
            self.assertEqual(["l'", "on"], [token.text for token in doc])
            self.assertTrue(doc[0].is_stop and doc[1].is_stop)
        else:
            self.assertEqual(["l'on"], [token.text for token in doc])

    def _assert_dabord(self, pipeline, code):
        doc = self.round_trip(pipeline, "d'abord")
        if code == "zh":
            self.assertEqual(list("d'abord"), [token.text for token in doc])
        elif code in {"ky", "tt"}:
            self.assertEqual(["d", "'", "abord"], [token.text for token in doc])
            self.assertFalse(doc[0].is_stop or doc[2].is_stop)
        elif code in {"it", "lb", "lij"}:
            self.assertEqual(["d'", "abord"], [token.text for token in doc])
            self.assertTrue(doc[0].is_stop)
            self.assertFalse(doc[1].is_stop)
        elif code == "ca":
            self.assertEqual(["d'", "abord"], [token.text for token in doc])
            self.assertFalse(doc[0].is_stop or doc[1].is_stop)
        elif code == "fr":
            self.assertEqual(["d'", "abord"], [token.text for token in doc])
            self.assertTrue(doc[0].is_stop and doc[1].is_stop)
        else:
            self.assertEqual(["d'abord"], [token.text for token in doc])

    def _assert_nt(self, pipeline, code, text, stem, stem_stop):
        doc = self.round_trip(pipeline, text)
        if code == "zh":
            self.assertEqual(list(text), [token.text for token in doc])
        elif code == "en":
            self.assertEqual([stem, "n't"], [token.text for token in doc])
            self.assertEqual(stem_stop, doc[0].is_stop)
            self.assertTrue(doc[1].is_stop)
        elif code == "ca":
            self.assertEqual([stem + "n", "'t"], [token.text for token in doc])
            self.assertFalse(doc[0].is_stop or doc[1].is_stop)
        elif code in {"fr", "it", "lij"}:
            self.assertEqual([stem + "n'", "t"], [token.text for token in doc])
            self.assertFalse(doc[0].is_stop or doc[1].is_stop)
        elif code in {"ky", "tt"}:
            self.assertEqual([stem + "n", "'", "t"], [token.text for token in doc])
            self.assertTrue(doc[1].is_quote)
        elif code == "yo":
            self.assertEqual([text], [token.text for token in doc])
            self.assertTrue(doc[0].like_num)
        else:
            self.assertEqual([text], [token.text for token in doc])
            self.assertFalse(doc[0].is_stop or doc[0].like_num)

    def _assert_lets(self, pipeline, code, glue):
        doc = self.round_trip(pipeline, "let's")
        if code == "zh":
            self.assertEqual(list("let's"), [token.text for token in doc])
        elif code in glue:
            self.assertEqual(["let's"], [token.text for token in doc])
            self.assertFalse(doc[0].is_stop or doc[0].like_num)
        elif code in {"ca", "fr"}:
            self.assertEqual(["let'", "s"], [token.text for token in doc])
            self.assertFalse(doc[0].is_stop or doc[1].is_stop)
        elif code == "en":
            self.assertEqual(["let", "'s"], [token.text for token in doc])
            self.assertFalse(doc[0].is_stop)
            self.assertTrue(doc[1].is_stop)
        elif code == "yo":
            self.assertEqual(["let", "'s"], [token.text for token in doc])
            self.assertTrue(doc[0].like_num)
            self.assertFalse(doc[1].is_stop)
        else:
            self.assertEqual(["let", "'s"], [token.text for token in doc])
            self.assertFalse(doc[0].is_stop or doc[1].is_stop or doc[0].like_num)

    def _assert_mustve(self, pipeline, code):
        doc = self.round_trip(pipeline, "must've")
        if code == "zh":
            self.assertEqual(list("must've"), [token.text for token in doc])
        elif code == "en":
            self.assertEqual(["must", "'ve"], [token.text for token in doc])
            self.assertTrue(doc[0].is_stop and doc[1].is_stop)
        elif code in {"ca", "fr", "it", "lij"}:
            self.assertEqual(["must'", "ve"], [token.text for token in doc])
            self.assertFalse(doc[0].is_stop or doc[1].is_stop)
        elif code in {"ky", "tt"}:
            self.assertEqual(["must", "'", "ve"], [token.text for token in doc])
            self.assertTrue(doc[1].is_quote)
        else:
            self.assertEqual(["must've"], [token.text for token in doc])
            self.assertFalse(doc[0].is_stop)

    def _assert_whodve(self, pipeline, code):
        doc = self.round_trip(pipeline, "who'd've")
        if code == "zh":
            self.assertEqual(list("who'd've"), [token.text for token in doc])
        elif code == "en":
            self.assertEqual(["who", "'d", "'ve"], [token.text for token in doc])
            self.assertTrue(all(token.is_stop for token in doc))
        elif code == "ca":
            self.assertEqual(["who'", "d'", "ve"], [token.text for token in doc])
            self.assertFalse(any(token.is_stop for token in doc))
        elif code in {"fr", "it", "lij"}:
            self.assertEqual(["who'", "d'", "ve"], [token.text for token in doc])
            self.assertTrue(doc[1].is_stop)
            self.assertFalse(doc[0].is_stop or doc[2].is_stop)
        elif code in {"ky", "tt"}:
            self.assertEqual(["who", "'", "d", "'", "ve"], [token.text for token in doc])
            self.assertTrue(doc[1].is_quote and doc[3].is_quote)
        else:
            self.assertEqual(["who'd've"], [token.text for token in doc])

    def _assert_title_dot(self, pipeline, code, text, kept):
        doc = self.round_trip(pipeline, text)
        if code == "zh":
            self.assertEqual(list(text), [token.text for token in doc])
            self.assertTrue(doc[-1].is_punct and doc[-1].is_stop)
        elif code in kept:
            self.assertEqual([text], [token.text for token in doc])
            self.assertTrue(doc[0].is_title)
            self.assertFalse(doc[0].is_alpha or doc[0].like_num or doc[0].is_stop)
        else:
            self.assertEqual([text[:-1], "."], [token.text for token in doc])
            self.assertTrue(doc[0].is_title and doc[1].is_punct)
            self.assertFalse(doc[0].is_stop)

    def _assert_initials(self, pipeline, code, text):
        doc = self.round_trip(pipeline, text)
        if code == "zh":
            self.assertEqual(list(text), [token.text for token in doc])
            self.assertTrue(all(token.is_stop for token in doc if token.text == "."))
        elif code in {"lt", "sr"}:
            self.assertEqual([text[:-1], "."], [token.text for token in doc])
            self.assertTrue(doc[0].is_upper and doc[0].is_title)
        elif code == "pl":
            self.assertEqual(["U", ".", "K", "."], [token.text for token in doc])
            self.assertTrue(doc[0].is_stop)
            self.assertFalse(doc[2].is_stop)
        elif code == "sl":
            self.assertEqual(["U.", "K."], [token.text for token in doc])
            self.assertTrue(doc[0].is_upper and doc[1].is_upper)
        else:
            self.assertEqual([text], [token.text for token in doc])
            self.assertTrue(doc[0].is_upper and doc[0].is_title)
            self.assertFalse(doc[0].is_alpha)

    def _assert_apres_demain_soir(self, pipeline, code, glue):
        doc = self.round_trip(pipeline, "après-demain soir")
        if code == "zh":
            self.assertEqual([ch for ch in "après-demain soir" if ch != " "], [token.text for token in doc])
        elif code in glue:
            self.assertEqual(["après-demain", "soir"], [token.text for token in doc])
            self.assertEqual(" ", doc[0].whitespace_)
            self.assertFalse(doc[0].is_stop or doc[1].is_stop)
        elif code == "el":
            self.assertEqual(["aprè", "s-demain", "soir"], [token.text for token in doc])
            self.assertEqual(" ", doc[1].whitespace_)
        else:
            self.assertEqual(["après", "-", "demain", "soir"], [token.text for token in doc])
            self.assertEqual(" ", doc[2].whitespace_)
            self.assertFalse(any(token.is_stop for token in doc))

    def _assert_cet_apres_midi(self, pipeline, code, glue):
        doc = self.round_trip(pipeline, "cet après-midi")
        if code == "zh":
            self.assertEqual([ch for ch in "cet après-midi" if ch != " "], [token.text for token in doc])
        elif code == "fr":
            self.assertEqual(["cet", "après-midi"], [token.text for token in doc])
            self.assertTrue(doc[0].is_stop)
            self.assertFalse(doc[1].is_stop)
        elif code in glue:
            self.assertEqual(["cet", "après-midi"], [token.text for token in doc])
            self.assertFalse(doc[0].is_stop or doc[1].is_stop)
        elif code == "el":
            self.assertEqual(["cet", "aprè", "s-midi"], [token.text for token in doc])
        elif code == "yo":
            self.assertEqual(["cet", "après", "-", "midi"], [token.text for token in doc])
            self.assertTrue(doc[3].like_num)
        else:
            self.assertEqual(["cet", "après", "-", "midi"], [token.text for token in doc])
            self.assertFalse(any(token.is_stop or token.like_num for token in doc))

    def _assert_dans_phrase(self, pipeline, code, text, middle, une_stop, une_num):
        doc = self.round_trip(pipeline, text)
        words = text.split(" ")
        if code == "zh":
            self.assertEqual([ch for ch in text if ch != " "], [token.text for token in doc])
        else:
            self.assertEqual(words, [token.text for token in doc])
            self.assertEqual(" ", doc[0].whitespace_)
            self.assertEqual(" ", doc[1].whitespace_)
            self.assertEqual(code == "fr", doc[0].is_stop)
            self.assertEqual(une_stop, doc[1].is_stop)
            self.assertEqual(une_num, doc[1].like_num)
            self.assertEqual(middle, doc[1].text)
            self.assertFalse(doc[2].is_stop)

    def _assert_clock_phrase(self, pipeline, code, text, head, head_stop):
        doc = self.round_trip(pipeline, text)
        if code == "zh":
            self.assertEqual([ch for ch in text if ch != " "], [token.text for token in doc])
            self.assertTrue(doc[0].text != "19h")
        else:
            self.assertEqual([head, "19h"], [token.text for token in doc])
            self.assertEqual(" ", doc[0].whitespace_)
            self.assertEqual(head_stop, doc[0].is_stop)
            self.assertTrue(doc[1].is_lower)
            self.assertFalse(doc[1].like_num or doc[1].is_alpha)

    def _assert_dix_neuf(self, pipeline, code, glue):
        doc = self.round_trip(pipeline, "dix-neuf heures")
        if code == "zh":
            self.assertEqual([ch for ch in "dix-neuf heures" if ch != " "], [token.text for token in doc])
        elif code == "fr":
            self.assertEqual(["dix-neuf", "heures"], [token.text for token in doc])
            self.assertTrue(doc[0].is_stop and doc[0].like_num)
        elif code in glue:
            self.assertEqual(["dix-neuf", "heures"], [token.text for token in doc])
            self.assertFalse(doc[0].is_stop or doc[0].like_num)
        elif code in {"la", "yo"}:
            self.assertEqual(["dix", "-", "neuf", "heures"], [token.text for token in doc])
            self.assertTrue(doc[0].like_num)
            self.assertFalse(doc[0].is_stop)
        else:
            self.assertEqual(["dix", "-", "neuf", "heures"], [token.text for token in doc])
            self.assertFalse(doc[0].like_num or doc[0].is_stop or doc[2].is_stop)

    def _assert_deux_mots(self, pipeline, code, text, head, head_stop, head_num, tail_num):
        doc = self.round_trip(pipeline, text)
        words = text.split(" ")
        if code == "zh":
            self.assertEqual([ch for ch in text if ch != " "], [token.text for token in doc])
        else:
            self.assertEqual(words, [token.text for token in doc])
            self.assertEqual(head, doc[0].text)
            self.assertEqual(head_stop, doc[0].is_stop)
            self.assertEqual(head_num, doc[0].like_num)
            self.assertEqual(tail_num, doc[1].like_num)
            self.assertFalse(doc[1].is_stop)

    def _assert_journee(self, pipeline, code, text, de_stop, both_stop, fin_only):
        doc = self.round_trip(pipeline, text)
        words = text.split(" ")
        if code == "zh":
            self.assertEqual([ch for ch in text if ch != " "], [token.text for token in doc])
        else:
            self.assertEqual(words, [token.text for token in doc])
            self.assertEqual(" ", doc[0].whitespace_)
            self.assertEqual(" ", doc[1].whitespace_)
            if code in both_stop:
                self.assertTrue(doc[0].is_stop and doc[1].is_stop)
            elif code in fin_only:
                self.assertTrue(doc[0].is_stop)
                self.assertFalse(doc[1].is_stop)
            else:
                self.assertFalse(doc[0].is_stop)
                self.assertEqual(code in de_stop, doc[1].is_stop)
            self.assertFalse(doc[2].is_stop)

    def round_trip(self, pipeline, text):
        doc = pipeline(text)
        self.assertEqual(text, doc.text)
        self.assertEqual(text, "".join(token.text_with_ws for token in doc))
        return doc


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

    def test_diacritics_outside_the_french_class_stay_on_english(self):
        # ä and û are tagged as verbs, but the French marker classifies them
        # with the lexicon, which does not treat a bare letter as a request.
        # ó and ð are not in that class, so the English heuristic clarifies.
        for text in FRENCH_DIACRITICS:
            with self.subTest(text=text):
                self.assertTrue(text_looks_french(text))
                with patch("intent.duckling_parse", return_value=[]):
                    result = classify(text)
                self.assertEqual(
                    ("reject", "not_a_schedule_request"),
                    (result["decision"], result["reason"]),
                )
                self.assertFalse(result["features"]["looks_like_request"])

        for text in NON_FRENCH_DIACRITICS:
            with self.subTest(text=text):
                self.assertFalse(text_looks_french(text))
                with patch("intent.duckling_parse", return_value=[]):
                    result = classify(text)
                if text in NON_FRENCH_VERB_MISTAGS:
                    self.assertEqual(
                        ("clarify", "request_without_temporal_signal"),
                        (result["decision"], result["reason"]),
                    )
                    self.assertEqual("VERB", result["features"]["root"]["pos"])
                    self.assertTrue(result["features"]["looks_like_request"])
                else:
                    self.assertEqual(
                        ("reject", "not_a_schedule_request"),
                        (result["decision"], result["reason"]),
                    )
                    self.assertFalse(result["features"]["looks_like_request"])

        # Two English function words still leave a loanword on the French path.
        # A third ("I", "our", "meeting") switches classification back to English.
        for text in ("The café is open", "My café is open", "\ufeffé", "«bonjour»"):
            with self.subTest(text=text):
                self.assertTrue(text_looks_french(text))
                with patch("intent.requests.post") as post:
                    post.return_value.json.return_value = []
                    post.return_value.raise_for_status.return_value = None
                    result = classify(text)
                self.assertEqual("fr_FR", post.call_args.kwargs["data"]["locale"])
                self.assertEqual("reject", result["decision"])
                self.assertEqual("not_a_schedule_request", result["reason"])

        for text in ("I think the café is open", "The café is our meeting"):
            with self.subTest(text=text):
                self.assertFalse(text_looks_french(text))
                with patch("intent.requests.post") as post:
                    post.return_value.json.return_value = []
                    post.return_value.raise_for_status.return_value = None
                    result = classify(text)
                self.assertEqual("en_GB", post.call_args.kwargs["data"]["locale"])
                self.assertEqual("reject", result["decision"])

    def test_script_and_symbol_mistags_are_stable(self):
        # Subject-less verb tags, same shape as the Khmer sample. The closing
        # corner bracket is a number, and a family emoji has a subject.
        expected = {
            "ሀሎ": ("clarify", "request_without_temporal_signal", "VERB", False),
            "\U0001f600": ("clarify", "request_without_temporal_signal", "VERB", False),
            "「": ("clarify", "request_without_temporal_signal", "VERB", False),
            "」": ("reject", "not_a_schedule_request", "NUM", False),
            "👨\u200d👩\u200d👧": ("reject", "not_a_schedule_request", "VERB", True),
            "don’t": ("clarify", "request_without_temporal_signal", "VERB", False),
            "\r": ("reject", "not_a_schedule_request", None, False),
            "\u1680": ("reject", "not_a_schedule_request", None, False),
            "\x1c": ("reject", "not_a_schedule_request", None, False),
            "\x00": ("reject", "not_a_schedule_request", "NOUN", False),
            "\u200e": ("reject", "not_a_schedule_request", "NOUN", False),
            "\u202e": ("reject", "not_a_schedule_request", "NOUN", False),
            "\ufe0f": ("reject", "not_a_schedule_request", "PROPN", False),
        }
        for text, (decision, reason, pos, subject) in expected.items():
            with self.subTest(text=text):
                self.assertFalse(text_looks_french(text))
                with patch("intent.duckling_parse", return_value=[]):
                    result = classify(text)
                self.assertEqual((decision, reason), (result["decision"], result["reason"]))
                self.assertEqual(pos, result["features"]["root"]["pos"])
                self.assertEqual(subject, result["features"]["has_subject"])
                self.assertEqual(text, result["text"])

    def test_new_script_and_clock_edges_keep_their_decisions(self):
        # en_core_web_sm tags on strings the earlier cases do not lock.
        # decision, reason, root text, lemma, pos, subject, french.
        expected = {
            "wasn't": ("reject", "not_a_schedule_request", "was", "be", "AUX", False, False),
            "who'd": ("reject", "informational_question_not_schedule_request", "'d", "would", "VERB", True, False),
            "who'll": ("reject", "informational_question_not_schedule_request", "'ll", "will", "AUX", True, False),
            "what'd": ("reject", "informational_question_not_schedule_request", "'d", "have", "VERB", True, False),
            "what're": ("reject", "informational_question_not_schedule_request", "'re", "be", "AUX", False, False),
            "where'd": ("reject", "informational_question_not_schedule_request", "where", "where", "SCONJ", False, False),
            "I'd've": ("reject", "not_a_schedule_request", "'ve", "'ve", "AUX", True, False),
            "y'all'd": ("reject", "not_a_schedule_request", "y'all'd", "y'all'd", "PROPN", False, False),
            "outta": ("clarify", "request_without_temporal_signal", "outta", "outta", "VERB", False, False),
            "'til": ("reject", "not_a_schedule_request", "til", "til", "ADV", False, False),
            "'twas": ("clarify", "request_without_temporal_signal", "twas", "twa", "VERB", False, False),
            "o'er": ("reject", "not_a_schedule_request", "o'er", "o'er", "NOUN", False, False),
            "Feb.": ("reject", "not_a_schedule_request", "Feb.", "Feb.", "PROPN", False, False),
            "Wed.": ("reject", "not_a_schedule_request", "d.", "d.", "VERB", True, False),
            "midnight": ("reject", "not_a_schedule_request", "midnight", "midnight", "NOUN", False, False),
            "5\u00b0C": ("reject", "not_a_schedule_request", "C", "c", "NOUN", False, False),
            "6h00": ("clarify", "temporal_signal_without_clear_request", "6h00", "6h00", "NUM", False, True),
            "21h": ("clarify", "temporal_signal_without_clear_request", "21h", "21h", "X", False, True),
            "t\u00eate-\u00e0-t\u00eate": ("reject", "not_a_schedule_request", "t\u00eate", "t\u00eate", "PROPN", False, True),
            "savoir-faire": ("reject", "not_a_schedule_request", "faire", "faire", "PROPN", False, False),
            "c'\u00e9tait": ("reject", "not_a_schedule_request", "c'\u00e9tait", "c'\u00e9tait", "PROPN", False, True),
            "est-il": ("reject", "not_a_schedule_request", "est", "est", "X", False, False),
            "va-t-il": ("reject", "not_a_schedule_request", "il", "il", "PROPN", False, False),
            "quelqu'une": ("clarify", "request_without_temporal_signal", "quelqu'une", "quelqu'une", "VERB", False, False),
            "n'y": ("reject", "not_a_schedule_request", "n'y", "n'y", "ADV", False, False),
            "\uffe5100": ("reject", "not_a_schedule_request", "\uffe5100", "\uffe5100", "NOUN", False, False),
            "\uff045": ("reject", "not_a_schedule_request", "\uff045", "\uff045", "PROPN", False, False),
            "EUR5": ("reject", "not_a_schedule_request", "EUR5", "EUR5", "PROPN", False, False),
            "\u22125": ("reject", "not_a_schedule_request", "\u22125", "\u22125", "PUNCT", False, False),
            "9~10": ("reject", "not_a_schedule_request", "9~10", "9~10", "NUM", False, False),
            "+.5": ("reject", "not_a_schedule_request", "+", "+", "NUM", False, False),
            "PT15M": ("reject", "not_a_schedule_request", "M", "M", "PROPN", False, False),
            "2026-284": ("reject", "not_a_schedule_request", "2026", "2026", "NUM", False, False),
            "2026-W41-7": ("reject", "not_a_schedule_request", "7", "7", "NUM", False, False),
            "\U0001f1ec\U0001f1e7": ("reject", "not_a_schedule_request", "\U0001f1e7", "\U0001f1e7", "NOUN", False, False),
            "\U0001f44d\U0001f3fb": ("clarify", "request_without_temporal_signal", "\U0001f3fb", "\U0001f3fb", "VERB", False, False),
            "*\ufe0f\u20e3": ("reject", "not_a_schedule_request", "*", "*", "PUNCT", False, False),
            "\u263a\ufe0f": ("reject", "not_a_schedule_request", "\ufe0f", "\ufe0f", "PROPN", False, False),
            "\u2601": ("reject", "not_a_schedule_request", "\u2601", "\u2601", "NOUN", False, False),
            "\u2602": ("reject", "not_a_schedule_request", "\u2602", "\u2602", "PUNCT", False, False),
            "\u014a": ("clarify", "request_without_temporal_signal", "\u014a", "\u014b", "VERB", False, False),
            "\u018f": ("clarify", "request_without_temporal_signal", "\u018f", "\u018f", "VERB", False, False),
            "\u0292": ("reject", "not_a_schedule_request", "\u0292", "\u0292", "X", False, False),
            "\u4e00": ("reject", "not_a_schedule_request", "\u4e00", "\u4e00", "PUNCT", False, False),
            "zero": ("reject", "not_a_schedule_request", "zero", "zero", "NUM", False, False),
            "deux": ("reject", "not_a_schedule_request", "deux", "deux", "X", False, False),
            "two": ("reject", "not_a_schedule_request", "two", "two", "NUM", False, False),
            "\u200f": ("reject", "not_a_schedule_request", "\u200f", "\u200f", "PUNCT", False, False),
            "\u2061": ("reject", "not_a_schedule_request", "\u2061", "\u2061", "PRON", False, False),
            "\u2064": ("reject", "not_a_schedule_request", "\u2064", "\u2064", "NOUN", False, False),
            "\u1361": ("reject", "not_a_schedule_request", "\u1361", "\u1361", "PUNCT", False, False),
            "\u301a": ("clarify", "request_without_temporal_signal", "\u301a", "\u301a", "VERB", False, False),
            "Wi-Fi": ("reject", "not_a_schedule_request", "Fi", "fi", "NOUN", False, False),
            "macOS": ("reject", "not_a_schedule_request", "macOS", "macos", "NOUN", False, False),
            "eBay": ("reject", "not_a_schedule_request", "eBay", "eBay", "PROPN", False, False),
            "undefined": ("clarify", "request_without_temporal_signal", "undefined", "undefine", "VERB", False, False),
            "nil": ("reject", "not_a_schedule_request", "nil", "nil", "NOUN", False, False),
            "9 A.M.": ("reject", "not_a_schedule_request", "A.M.", "A.M.", "PROPN", False, False),
            "12:00:00": ("reject", "not_a_schedule_request", "12:00:00", "12:00:00", "NUM", False, False),
            "trompe-l'\u0153il": ("reject", "not_a_schedule_request", "l'\u0153il", "l'\u0153il", "PROPN", False, True),
            "hors-d'\u0153uvre": ("reject", "not_a_schedule_request", "d'\u0153uvre", "d'\u0153uvre", "PROPN", False, True),
            "pied-\u00e0-terre": ("reject", "not_a_schedule_request", "pied", "pie", "VERB", False, True),
            "j'\u00e9tais": ("reject", "not_a_schedule_request", "j'\u00e9tais", "j'\u00e9tais", "PROPN", False, True),
            "d'ailleurs": ("reject", "not_a_schedule_request", "d'ailleurs", "d'ailleurs", "X", False, False),
            "s'en": ("reject", "not_a_schedule_request", "s'en", "s'en", "PUNCT", False, False),
            "m'a": ("clarify", "request_without_temporal_signal", "m'a", "m'a", "VERB", False, False),
            "\uff15": ("reject", "not_a_schedule_request", "\uff15", "\uff15", "NUM", False, False),
            "\u096b": ("reject", "not_a_schedule_request", "\u096b", "\u096b", "X", False, False),
            "\uac00": ("reject", "not_a_schedule_request", "\uac00", "\uac00", "PROPN", False, False),
            "\u3042": ("reject", "not_a_schedule_request", "\u3042", "\u3042", "X", False, False),
            "\u30a2": ("clarify", "request_without_temporal_signal", "\u30a2", "\u30a2", "VERB", False, False),
            "n\u0303": ("reject", "not_a_schedule_request", "n\u0303", "n\u0303", "PROPN", False, False),
            "100\u066a": ("reject", "not_a_schedule_request", "\u066a", "\u066a", "NOUN", False, False),
            "1\u2019000": ("reject", "not_a_schedule_request", "1\u2019000", "1\u2019000", "NUM", False, False),
            "P2W": ("reject", "not_a_schedule_request", "P2W", "p2w", "NOUN", False, False),
            "lemme": ("reject", "not_a_schedule_request", "lemme", "lemme", "PROPN", False, False),
            "GmbH": ("reject", "not_a_schedule_request", "GmbH", "GmbH", "PROPN", False, False),
        }
        for text, (decision, reason, root_text, lemma, pos, subject, french) in expected.items():
            with self.subTest(text=text):
                self.assertEqual(french, text_looks_french(text))
                with patch("intent.duckling_parse", return_value=[]):
                    result = classify(text)
                root = result["features"]["root"]
                self.assertEqual(text, result["text"])
                self.assertEqual((decision, reason), (result["decision"], result["reason"]))
                self.assertEqual(root_text, root["text"])
                self.assertEqual(lemma, root["lemma"])
                self.assertEqual(pos, root["pos"])
                self.assertEqual(subject, result["features"]["has_subject"])
                self.assertEqual(
                    reason == "temporal_signal_without_clear_request",
                    result["features"]["has_temporal_signal"],
                )
                self.assertEqual(
                    reason == "request_without_temporal_signal",
                    result["features"]["looks_like_request"],
                )

    def test_october_nineteen_edges_keep_their_decisions(self):
        # en_core_web_sm tags for strings the earlier cases do not lock.
        # decision, reason, root text, lemma, pos, subject, french.
        expected = {
            "19h": ("clarify", "temporal_signal_without_clear_request", "19h", "19h", "NOUN", False, True),
            "19h19": ("clarify", "temporal_signal_without_clear_request", "19h19", "19h19", "NUM", False, True),
            "19h30": ("clarify", "temporal_signal_without_clear_request", "19h30", "19h30", "NUM", False, True),
            "19h00": ("clarify", "temporal_signal_without_clear_request", "19h00", "19h00", "NUM", False, True),
            "09h19": ("clarify", "temporal_signal_without_clear_request", "09h19", "09h19", "NOUN", False, True),
            "19:19": ("reject", "not_a_schedule_request", "19:19", "19:19", "NUM", False, False),
            "7:19pm": ("reject", "not_a_schedule_request", "7:19pm", "7:19pm", "NUM", False, False),
            "19 h 19": ("clarify", "temporal_signal_without_clear_request", "h", "h", "NOUN", False, True),
            "2026-10-19": ("reject", "not_a_schedule_request", "2026", "2026", "NUM", False, False),
            "19-10-2026": ("reject", "not_a_schedule_request", "10", "10", "NUM", False, False),
            "19/10/2026": ("reject", "not_a_schedule_request", "19/10/2026", "19/10/2026", "NUM", False, False),
            "19.10.2026": ("reject", "not_a_schedule_request", "19.10.2026", "19.10.2026", "NUM", False, False),
            "2026-292": ("reject", "not_a_schedule_request", "2026", "2026", "NUM", False, False),
            "2026-10-19T19:00:00Z": ("reject", "not_a_schedule_request", "2026", "2026", "NUM", False, False),
            "2026-W43": ("reject", "not_a_schedule_request", "W43", "W43", "PROPN", False, False),
            "2026-W43-1": ("reject", "not_a_schedule_request", "W43", "W43", "PROPN", False, False),
            "mother-in-law": ("reject", "not_a_schedule_request", "mother", "mother", "NOUN", False, False),
            "vis-à-vis": ("reject", "not_a_schedule_request", "vis", "vis", "X", False, True),
            "peut-être": ("reject", "not_a_schedule_request", "être", "être", "NOUN", False, True),
            "va-et-vient": ("reject", "not_a_schedule_request", "vient", "vient", "NOUN", False, False),
            "c'est-à-dire": ("reject", "not_a_schedule_request", "c'est", "c'est", "PROPN", False, True),
            "celui-ci": ("reject", "not_a_schedule_request", "ci", "ci", "NOUN", False, False),
            "n'est-ce": ("reject", "not_a_schedule_request", "ce", "ce", "NOUN", False, True),
            "qu'à": ("reject", "not_a_schedule_request", "qu'à", "qu'à", "ADV", False, True),
            "m'en": ("reject", "not_a_schedule_request", "m'en", "m'en", "NOUN", False, False),
            "d'abord": ("reject", "not_a_schedule_request", "d'abord", "d'abord", "PROPN", False, False),
            "aren't": ("reject", "not_a_schedule_request", "are", "be", "AUX", False, False),
            "wouldn't": ("reject", "not_a_schedule_request", "would", "would", "AUX", False, False),
            "must've": ("reject", "not_a_schedule_request", "'ve", "'ve", "AUX", False, False),
            "Gen.": ("reject", "not_a_schedule_request", "Gen.", "Gen.", "PROPN", False, False),
            "Mme.": ("reject", "not_a_schedule_request", "Mme", "Mme", "PROPN", False, False),
            "Ltd.": ("reject", "not_a_schedule_request", "Ltd.", "Ltd.", "PROPN", False, False),
            "U.K.": ("reject", "not_a_schedule_request", "U.K.", "U.K.", "PROPN", False, False),
            "a.k.a.": ("reject", "not_a_schedule_request", "a.k.a", "a.k.a", "ADJ", False, False),
            "৳": ("reject", "not_a_schedule_request", "৳", "৳", "PUNCT", False, False),
            "µ": ("reject", "not_a_schedule_request", "µ", "µ", "PROPN", False, False),
            "±": ("reject", "not_a_schedule_request", "±", "±", "NOUN", False, False),
            "×": ("reject", "not_a_schedule_request", "×", "×", "X", False, False),
            "∞": ("reject", "not_a_schedule_request", "∞", "∞", "X", False, False),
            "√": ("reject", "not_a_schedule_request", "√", "√", "PROPN", False, False),
            "π": ("reject", "not_a_schedule_request", "π", "π", "PROPN", False, False),
            "Ω": ("reject", "not_a_schedule_request", "Ω", "Ω", "PROPN", False, False),
            "²": ("reject", "not_a_schedule_request", "²", "²", "NUM", False, False),
            "₁": ("reject", "not_a_schedule_request", "₁", "₁", "X", False, False),
            "⅓": ("reject", "not_a_schedule_request", "⅓", "⅓", "X", False, False),
            "⅛": ("reject", "not_a_schedule_request", "⅛", "⅛", "NUM", False, False),
            "𝟐": ("reject", "not_a_schedule_request", "𝟐", "𝟐", "NUM", False, False),
            "𝟙": ("reject", "not_a_schedule_request", "𝟙", "𝟙", "X", False, False),
            "߁": ("reject", "not_a_schedule_request", "߁", "߁", "NUM", False, False),
            "༡": ("reject", "not_a_schedule_request", "༡", "༡", "NUM", False, False),
            "၁": ("reject", "not_a_schedule_request", "၁", "၁", "NUM", False, False),
            "១": ("reject", "not_a_schedule_request", "១", "១", "NUM", False, False),
            "๑": ("reject", "not_a_schedule_request", "๑", "๑", "NUM", False, False),
            "᱑": ("reject", "not_a_schedule_request", "᱑", "᱑", "NUM", False, False),
            "１９": ("reject", "not_a_schedule_request", "１９", "１９", "NUM", False, False),
            "ə": ("reject", "not_a_schedule_request", "ə", "ə", "X", False, False),
            "ɛ": ("reject", "not_a_schedule_request", "ɛ", "ɛ", "X", False, False),
            "œ": ("reject", "not_a_schedule_request", "œ", "œ", "NOUN", False, True),
            "æ": ("reject", "not_a_schedule_request", "æ", "æ", "X", False, True),
            "ÿ": ("reject", "not_a_schedule_request", "ÿ", "ÿ", "X", False, False),
            "č": ("reject", "not_a_schedule_request", "č", "č", "X", False, False),
            "ʔ": ("reject", "not_a_schedule_request", "ʔ", "ʔ", "X", False, False),
            "〖": ("reject", "not_a_schedule_request", "〖", "〖", "PROPN", False, False),
            "„": ("reject", "not_a_schedule_request", "„", "„", "PUNCT", False, False),
            "✓": ("reject", "not_a_schedule_request", "✓", "✓", "NOUN", False, False),
            "☕": ("reject", "not_a_schedule_request", "☕", "☕", "NOUN", False, False),
            "🇫🇷": ("reject", "not_a_schedule_request", "🇫", "🇫", "NOUN", False, False),
            "⌚️": ("reject", "not_a_schedule_request", "️", "️", "NOUN", False, False),
            "lundi matin": ("clarify", "temporal_signal_without_clear_request", "lundi", "lundi", "VERB", False, True),
            "mercredi matin": ("clarify", "temporal_signal_without_clear_request", "matin", "matin", "PROPN", False, True),
            "jeudi soir": ("clarify", "temporal_signal_without_clear_request", "soir", "soir", "NOUN", False, True),
            "vendredi midi": ("clarify", "temporal_signal_without_clear_request", "midi", "midi", "PROPN", False, True),
            "après-demain soir": ("clarify", "temporal_signal_without_clear_request", "soir", "soir", "NOUN", False, True),
            "cet après-midi": ("clarify", "temporal_signal_without_clear_request", "midi", "midi", "NOUN", False, True),
            "dans une heure": ("clarify", "temporal_signal_without_clear_request", "dans", "dans", "PROPN", False, False),
            "à 19h": ("clarify", "temporal_signal_without_clear_request", "19h", "19h", "NOUN", False, True),
            "vers 19h": ("clarify", "temporal_signal_without_clear_request", "vers", "ver", "NOUN", False, True),
            "dès 19h": ("clarify", "temporal_signal_without_clear_request", "19h", "19h", "NOUN", False, True),
            "19 heures": ("clarify", "temporal_signal_without_clear_request", "heures", "heure", "NOUN", False, False),
            "dix-neuf heures": ("reject", "not_a_schedule_request", "heures", "heure", "NOUN", False, False),
            "chaque midi": ("clarify", "temporal_signal_without_clear_request", "midi", "midi", "PROPN", False, True),
            "ce midi": ("clarify", "temporal_signal_without_clear_request", "midi", "midi", "PROPN", False, False),
            "fin de journée": ("reject", "not_a_schedule_request", "journée", "journée", "PROPN", False, True),
            "début de journée": ("reject", "not_a_schedule_request", "journée", "journée", "PROPN", False, True),
        }
        # Subject-less verb mistags, including the French polite formula.
        requests = {
            "s'il vous plaît": ("plaît", "plaît", "NOUN", False, True),
            "n'a": ("n'a", "n'a", "VERB", False, False),
            "j'y": ("j'y", "j'y", "VERB", False, False),
            "l'on": ("l'on", "l'on", "VERB", False, False),
            "haven't": ("have", "have", "VERB", False, False),
            "doesn't": ("does", "do", "VERB", False, False),
            "let's": ("let", "let", "VERB", False, False),
            "₱": ("₱", "₱", "VERB", False, False),
            "₵": ("₵", "₵", "VERB", False, False),
            "‰": ("‰", "‰", "VERB", False, False),
            "′": ("′", "′", "VERB", False, False),
            "†": ("†", "†", "VERB", False, False),
            "🔔": ("🔔", "🔔", "VERB", False, False),
        }
        for text, (decision, reason, root_text, lemma, pos, subject, french) in expected.items():
            with self.subTest(text=text):
                self._assert_october_decision(
                    text, decision, reason, root_text, lemma, pos, subject, french,
                )

        for text, (root_text, lemma, pos, subject, french) in requests.items():
            with self.subTest(text=text):
                self._assert_october_decision(
                    text,
                    "clarify",
                    "request_without_temporal_signal",
                    root_text,
                    lemma,
                    pos,
                    subject,
                    french,
                )

        self._assert_october_decision(
            "who'd've",
            "reject",
            "informational_question_not_schedule_request",
            "'ve",
            "'ve",
            "AUX",
            True,
            False,
        )
        # "cinq" is a subject-less verb and the French duration regex matches,
        # so the English heuristic allows the phrase.
        self._assert_october_decision(
            "dans cinq jours",
            "allow",
            "request_with_temporal_signal",
            "cinq",
            "cinq",
            "VERB",
            False,
            False,
        )

    def _assert_october_decision(self, text, decision, reason, root_text, lemma, pos, subject, french):
        self.assertEqual(french, text_looks_french(text))
        with patch("intent.duckling_parse", return_value=[]):
            result = classify(text)
        root = result["features"]["root"]
        self.assertEqual(text, result["text"])
        self.assertEqual((decision, reason), (result["decision"], result["reason"]))
        self.assertEqual(root_text, root["text"])
        self.assertEqual(lemma, root["lemma"])
        self.assertEqual(pos, root["pos"])
        self.assertEqual(subject, result["features"]["has_subject"])
        self.assertEqual(
            reason in {"temporal_signal_without_clear_request", "request_with_temporal_signal"},
            result["features"]["has_temporal_signal"],
        )
        self.assertEqual(
            reason in {"request_without_temporal_signal", "request_with_temporal_signal"},
            result["features"]["looks_like_request"],
        )


if __name__ == "__main__":
    unittest.main()
