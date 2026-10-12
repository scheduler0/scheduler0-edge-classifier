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

    def test_calendar_units_compounds_and_script_marks(self):
        # Forms the earlier numeral and abbreviation passes do not lock.
        # Chinese splits to characters unless a case below says otherwise.
        not_like_plus = {"fa", "grc", "la", "lb", "lt", "nl", "pl", "si", "ta", "te", "tl", "uk", "yo"}
        split_plus = {"bn", "sl", "zh"}
        iso_glue = {"da", "el", "es", "fi", "hu", "lt", "nb", "nl", "nn", "pt", "ro", "sv", "vi"}
        dmy_hyphen_glue = iso_glue | {"tr"}
        slash_split = {"de", "id", "ms"}
        unit_glue = {"grc", "ro", "vi"}
        unit_stop = {"sl", "sq", "yo"}
        month_apr = {"am", "ar", "da", "de", "en", "fa", "fo", "grc", "id", "la", "ms", "sv", "ti", "vi"}
        month_jun = {"am", "ar", "da", "de", "en", "fa", "grc", "id", "ms", "sv", "ti", "vi"}
        month_jul = {"am", "ar", "de", "en", "fa", "grc", "id", "ms", "ti", "vi"}
        weekday_dot = {"am", "ar", "fa", "grc", "ti", "vi"}
        a_stop = {"ca", "cs", "de", "dsb", "en", "es", "fr", "ga", "hr", "hsb", "it", "lij", "lt", "pl", "pt", "sk", "sl", "sq", "yo"}
        in_stop = {"af", "az", "en", "ga", "it", "la", "lij", "sl"}
        hyphen_glue = {"ca", "da", "de", "el", "es", "fi", "hu", "ky", "lb", "nb", "nl", "nn", "pt", "ro", "sr", "sv", "tt", "vi"}
        how_glue = {"am", "ar", "bn", "da", "de", "el", "es", "fa", "fi", "grc", "hu", "nb", "nl", "nn", "pl", "ro", "sl", "sr", "sv", "ti", "vi"}
        elision = {"ca", "fr", "it", "lij"}
        file_glue = {"hu", "lb", "nl", "ro", "vi"}
        currency = ["\u09f3", "\u09f2", "\u0bf9", "\u0af1", "\u20b6", "\u20b7"]
        bare_marks = [
            "\u00ac", "\u00a6", "\u00a8", "\u00af", "\u00b4", "\u00b8",
            "\u2219", "\u22c5", "\u2150", "\u2151", "\u2152",
            "\u2615", "\u231a", "\u231b", "\u23f3", "\U0001f514", "\U0001f4c6",
            "\u2669", "\u266c",
        ]
        punct_marks = ["\u2034", "\u2035", "\u2043", "\u2027", "\uff65"]
        script_digits = ["\u1c50", "\u07c0", "\u19d0"]

        for code, pipeline in self.pipelines.items():
            if pipeline is None:
                continue
            with self.subTest(code=code):
                for text in currency:
                    token = self.round_trip(pipeline, text)[0]
                    self.assertTrue(token.is_currency)
                    self.assertFalse(token.is_punct)
                    self.assertFalse(token.is_stop)
                for text in bare_marks:
                    token = self.round_trip(pipeline, text)[0]
                    self.assertEqual(text, token.text)
                    self.assertFalse(token.is_alpha or token.is_digit or token.is_punct)
                    self.assertFalse(token.is_space or token.is_stop or token.is_currency)
                    self.assertFalse(token.like_num)
                for text in punct_marks:
                    token = self.round_trip(pipeline, text)[0]
                    self.assertTrue(token.is_punct)
                    self.assertFalse(token.is_quote or token.is_bracket or token.is_stop)
                for text in script_digits:
                    token = self.round_trip(pipeline, text)[0]
                    self.assertTrue(token.is_digit)
                    self.assertEqual(code != "grc", token.like_num)
                    self.assertFalse(token.is_stop)
                self.assert_cased(pipeline, code, "\u0194", alpha=True, upper=True, title=True)
                for text in ("\u025b", "\u0254", "\u0257", "\u0253", "\u1d00"):
                    self.assert_cased(pipeline, code, text, alpha=True, lower=True)
                self.assert_cased(pipeline, code, "\u0294", alpha=True)
                self.assert_cased(pipeline, code, "\u2170", lower=True)
                self.assert_cased(pipeline, code, "\u2169", upper=True, title=True)
                self.assert_cased(pipeline, code, "\u217c", lower=True)

                self.assert_decimal(pipeline, code, "2.71828")
                self.assert_decimal(pipeline, code, "00.5")
                self.assert_signed_zero(pipeline, code, not_like_plus, split_plus)
                self.assert_apostrophe_number(pipeline, code, "1'234.56", dot=True)
                self.assert_apostrophe_number(pipeline, code, "1'234,56", dot=False)
                self.assert_power(pipeline, code, "10\u00b2")
                self.assert_power(pipeline, code, "2\u00b3")
                self.assert_whole_token(pipeline, code, "km\u00b2", is_lower=True, is_alpha=False, like_num=False)
                self.assert_whole_token(pipeline, code, "3\u00bd", like_num=False, is_punct=False)
                self.assert_whole_token(pipeline, code, "Inf", is_alpha=True, is_title=True, is_lower=False)
                self.assert_whole_token(pipeline, code, "P1DT2H", is_upper=True, is_title=False, is_alpha=False)
                self.assert_whole_token(pipeline, code, "T09:00:00", is_upper=True, is_title=True, is_alpha=False)
                self.assert_whole_token(pipeline, code, "9mins", is_lower=True, like_num=False, is_alpha=False)
                self.assert_acronym(pipeline, code, "XLII", like_num=code == "la")
                self.assert_letters(pipeline, code, "mm", like_num=code == "la", lower=True)
                for text in ("ASAP", "FYI", "TBD", "EOD", "COB", "YTD"):
                    self.assert_letters(pipeline, code, text, like_num=False, lower=False)
                self.assert_half_year(pipeline, code)
                self.assert_iso_duration(pipeline, code, "PT1H30M", "PT1H30", "M", unit_glue, unit_stop)
                self.assert_iso_duration(pipeline, code, "2h30m", "2h30", "m", unit_glue, unit_stop)
                self.assert_iso_date(pipeline, code, "2026-10-12", iso_glue)
                self.assert_iso_date(pipeline, code, "10-12-2026", dmy_hyphen_glue)
                self.assert_dotted_date(pipeline, code, "12.10.2026")
                self.assert_slashed_date(pipeline, code, "12/10/2026", slash_split)
                self.assert_timestamp(pipeline, code)
                self.assert_relation(pipeline, code, "\u22485")
                self.assert_relation(pipeline, code, "\u22655")
                self.assert_relation(pipeline, code, "\u22600")
                self.assert_relation(pipeline, code, "\u00ac5")
                leq = self.round_trip(pipeline, "\u226410")
                if code == "zh":
                    self.assertEqual(["\u2264", "1", "0"], [token.text for token in leq])
                    self.assertFalse(leq[0].is_punct or leq[0].is_stop)
                    self.assertTrue(all(token.like_num and token.is_stop for token in leq[1:]))
                else:
                    self.assertEqual(["\u226410"], [token.text for token in leq])
                    self.assertFalse(leq[0].like_num or leq[0].is_punct)

                self.assert_final_dot(pipeline, code, "Apr.", month_apr)
                self.assert_final_dot(pipeline, code, "Nov.", month_apr)
                self.assert_final_dot(pipeline, code, "Jun.", month_jun)
                self.assert_final_dot(pipeline, code, "Jul.", month_jul)
                for text in ("Sun.", "Thu.", "Fri.", "Sat.", "Rd.", "Blvd."):
                    self.assert_final_dot(pipeline, code, text, weekday_dot)
                hon = self.round_trip(pipeline, "Hon.")
                if code == "zh":
                    self.assertEqual(["H", "o", "n", "."], [token.text for token in hon])
                    self.assertTrue(hon[-1].is_stop)
                elif code in weekday_dot:
                    self.assertEqual(["Hon."], [token.text for token in hon])
                elif code == "sv":
                    self.assertEqual(["Hon", "."], [token.text for token in hon])
                    self.assertTrue(hon[0].is_stop)
                else:
                    self.assertEqual(["Hon", "."], [token.text for token in hon])
                    self.assertFalse(hon[0].is_stop)
                self.assert_initials(pipeline, code, "N.B.", stop_letter=None)
                self.assert_initials(pipeline, code, "P.O.", stop_letter="O")

                self.assert_ll_contraction(pipeline, code, "they'll", "they", elision)
                self.assert_ll_contraction(pipeline, code, "what'll", "what", elision)
                self.assert_s_contraction(pipeline, code, "how's", "how", how_glue)
                self.assert_s_contraction(pipeline, code, "when's", "when", how_glue)
                self.assert_nt_contraction(pipeline, code, "hadn't", "had", "hadn")
                self.assert_double_contraction(pipeline, code)
                for text in ("imma", "gimme", "kinda", "sorta"):
                    self.assert_whole_token(
                        pipeline, code, text, is_alpha=True, is_lower=True, like_num=False, is_stop=False,
                    )

                self.assert_hyphen_compound(
                    pipeline, code, "follow-up", hyphen_glue | {"id", "ms"},
                    stops={"en": {"up"}},
                )
                self.assert_hyphen_compound(
                    pipeline, code, "check-in", hyphen_glue | {"id"},
                    stops={lang: {"in"} for lang in in_stop},
                )
                self.assert_hyphen_compound(
                    pipeline, code, "re-schedule", hyphen_glue | {"fr"},
                    stops={"en": {"re"}, "tn": {"re"}},
                    like_num={"yo": {"schedule"}},
                )
                for text, stops in (
                    ("all-day", {"en": {"all"}, "it": {"all"}}),
                    ("part-time", {"en": {"part"}}),
                    ("well-being", {"en": {"well", "being"}}),
                    ("out-of-office", {"en": {"out", "of"}, "az": {"of"}}),
                    ("mother-in-law", {lang: {"in"} for lang in in_stop}),
                    ("in-person", {lang: {"in"} for lang in in_stop}),
                ):
                    self.assert_hyphen_compound(pipeline, code, text, hyphen_glue, stops=stops)

                self.assert_url(pipeline, code, "https://user:pass@example.com", yo_num=True)
                self.assert_url(pipeline, code, "http://localhost:3000", yo_num=False)
                self.assert_url(pipeline, code, "www.example.co.uk", yo_num=True)
                idn = self.round_trip(pipeline, "https://\u4f8b.jp")
                if code == "zh":
                    self.assertIn("\u4f8b", [token.text for token in idn])
                    self.assertFalse(any(token.like_url for token in idn))
                    self.assertTrue(idn[8].is_alpha)
                    self.assertFalse(idn[8].is_lower)
                else:
                    self.assertEqual(["https://\u4f8b.jp"], [token.text for token in idn])
                    self.assertTrue(idn[0].like_url)
                    self.assertFalse(idn[0].like_num)
                bare = self.round_trip(pipeline, "a@b")
                if code == "zh":
                    self.assertEqual(["a", "@", "b"], [token.text for token in bare])
                    self.assertTrue(bare[1].is_stop)
                else:
                    self.assertEqual(["a@b"], [token.text for token in bare])
                    self.assertFalse(bare[0].like_email or bare[0].like_url or bare[0].like_num)
                self.assert_file_url(pipeline, code, file_glue, a_stop)
                self.assert_dici(pipeline, code)
                self.assert_a_lheure(pipeline, code)

    def assert_cased(self, pipeline, code, text, alpha=False, lower=False, upper=False, title=False):
        token = self.round_trip(pipeline, text)[0]
        self.assertEqual(text, token.text)
        self.assertEqual(alpha, token.is_alpha)
        self.assertEqual(lower, token.is_lower)
        self.assertEqual(upper, token.is_upper)
        self.assertEqual(title, token.is_title)
        self.assertFalse(token.like_num or token.is_punct)

    def assert_whole_token(self, pipeline, code, text, **flags):
        doc = self.round_trip(pipeline, text)
        if code == "zh":
            self.assertEqual(list(text), [token.text for token in doc])
            return doc
        self.assertEqual([text], [token.text for token in doc])
        for name, expected in flags.items():
            self.assertEqual(expected, getattr(doc[0], name))
        return doc

    def assert_decimal(self, pipeline, code, text):
        doc = self.round_trip(pipeline, text)
        if code == "zh":
            self.assertEqual(list(text), [token.text for token in doc])
            self.assertTrue(all(token.is_stop for token in doc))
            self.assertTrue(all(token.like_num for token in doc if token.text.isdigit()))
        elif code == "pl":
            self.assertEqual(3, len(doc))
            self.assertEqual(".", doc[1].text)
            self.assertTrue(doc[0].like_num and doc[2].like_num)
        elif code in {"grc", "la"}:
            self.assertEqual([text], [token.text for token in doc])
            self.assertFalse(doc[0].like_num)
        else:
            self.assertEqual([text], [token.text for token in doc])
            self.assertTrue(doc[0].like_num)

    def assert_signed_zero(self, pipeline, code, not_like, split):
        doc = self.round_trip(pipeline, "+0")
        if code in split:
            self.assertEqual(["+", "0"], [token.text for token in doc])
            self.assertTrue(doc[1].like_num)
            self.assertEqual(code == "zh", doc[1].is_stop)
        elif code in not_like:
            self.assertEqual(["+0"], [token.text for token in doc])
            self.assertFalse(doc[0].like_num)
        else:
            self.assertEqual(["+0"], [token.text for token in doc])
            self.assertTrue(doc[0].like_num)

    def assert_apostrophe_number(self, pipeline, code, text, dot):
        doc = self.round_trip(pipeline, text)
        if code == "zh":
            self.assertEqual(list(text), [token.text for token in doc])
            self.assertTrue(all(token.is_stop for token in doc))
        elif dot and code == "pl":
            self.assertEqual(["1'234", ".", "56"], [token.text for token in doc])
            self.assertFalse(doc[0].like_num)
            self.assertTrue(doc[2].like_num)
        else:
            self.assertEqual([text], [token.text for token in doc])
            self.assertFalse(doc[0].like_num)

    def assert_power(self, pipeline, code, text):
        doc = self.round_trip(pipeline, text)
        if code == "zh":
            self.assertEqual(list(text), [token.text for token in doc])
            self.assertTrue(all(token.like_num for token in doc))
            self.assertTrue(all(token.is_stop for token in doc if token.text.isascii()))
            self.assertFalse(doc[-1].is_stop)
        elif code == "grc":
            self.assertEqual([text], [token.text for token in doc])
            self.assertTrue(doc[0].is_digit)
            self.assertFalse(doc[0].like_num)
        else:
            self.assertEqual([text], [token.text for token in doc])
            self.assertTrue(doc[0].is_digit and doc[0].like_num)

    def assert_acronym(self, pipeline, code, text, like_num):
        doc = self.round_trip(pipeline, text)
        if code == "zh":
            self.assertEqual(list(text), [token.text for token in doc])
            self.assertTrue(all(token.is_upper and token.is_title and not token.like_num for token in doc))
        else:
            self.assertEqual([text], [token.text for token in doc])
            self.assertTrue(doc[0].is_alpha and doc[0].is_upper)
            self.assertFalse(doc[0].is_title)
            self.assertEqual(like_num, doc[0].like_num)

    def assert_letters(self, pipeline, code, text, like_num, lower):
        doc = self.round_trip(pipeline, text)
        if code == "zh":
            self.assertEqual(list(text), [token.text for token in doc])
            self.assertTrue(all(token.is_alpha for token in doc))
            self.assertFalse(any(token.like_num for token in doc))
        else:
            self.assertEqual([text], [token.text for token in doc])
            self.assertTrue(doc[0].is_alpha)
            self.assertEqual(lower, doc[0].is_lower)
            self.assertEqual(not lower, doc[0].is_upper)
            self.assertFalse(doc[0].is_title)
            self.assertEqual(like_num, doc[0].like_num)

    def assert_half_year(self, pipeline, code):
        doc = self.round_trip(pipeline, "H1")
        if code == "zh":
            self.assertEqual(["H", "1"], [token.text for token in doc])
            self.assertTrue(doc[0].is_upper and doc[0].is_title)
            self.assertTrue(doc[1].like_num and doc[1].is_stop)
        else:
            self.assertEqual(["H1"], [token.text for token in doc])
            self.assertTrue(doc[0].is_upper and doc[0].is_title)
            self.assertFalse(doc[0].is_alpha or doc[0].like_num)

    def assert_iso_duration(self, pipeline, code, text, head, tail, glue, stop):
        doc = self.round_trip(pipeline, text)
        if code == "zh":
            self.assertEqual(list(text), [token.text for token in doc])
        elif code in glue:
            self.assertEqual([text], [token.text for token in doc])
            self.assertFalse(doc[0].like_num)
        else:
            self.assertEqual([head, tail], [token.text for token in doc])
            self.assertFalse(doc[0].like_num)
            self.assertEqual(code == "la", doc[1].like_num)
            self.assertEqual(code in stop, doc[1].is_stop)

    def assert_iso_date(self, pipeline, code, text, glue):
        doc = self.round_trip(pipeline, text)
        pieces = text.split("-")
        if code == "zh":
            self.assertEqual(list(text), [token.text for token in doc])
            self.assertTrue(all(token.is_stop for token in doc))
        elif code in glue:
            self.assertEqual([text], [token.text for token in doc])
            self.assertFalse(doc[0].like_num)
        elif code == "grc":
            self.assertEqual([pieces[0], "-", pieces[1], "-", pieces[2]], [token.text for token in doc])
            self.assertTrue(all(token.is_digit and not token.like_num for token in doc if token.text != "-"))
        else:
            self.assertEqual([pieces[0], "-", pieces[1], "-", pieces[2]], [token.text for token in doc])
            self.assertTrue(all(token.like_num for token in doc if token.text != "-"))

    def assert_dotted_date(self, pipeline, code, text):
        doc = self.round_trip(pipeline, text)
        if code == "zh":
            self.assertEqual(list(text), [token.text for token in doc])
        elif code == "pl":
            self.assertEqual(["12", ".", "10", ".", "2026"], [token.text for token in doc])
            self.assertTrue(all(token.like_num for token in doc if token.text != "."))
        elif code in {"grc", "la"}:
            self.assertEqual([text], [token.text for token in doc])
            self.assertFalse(doc[0].like_num)
        else:
            self.assertEqual([text], [token.text for token in doc])
            self.assertTrue(doc[0].like_num)

    def assert_slashed_date(self, pipeline, code, text, slash_split):
        doc = self.round_trip(pipeline, text)
        if code == "zh":
            self.assertEqual(list(text), [token.text for token in doc])
            self.assertTrue(all(token.is_stop for token in doc if token.text == "/"))
        elif code in slash_split:
            self.assertEqual(["12", "/", "10", "/", "2026"], [token.text for token in doc])
            self.assertTrue(all(token.like_num for token in (doc[0], doc[2], doc[4])))
        elif code == "fa":
            self.assertEqual([text], [token.text for token in doc])
            self.assertTrue(doc[0].like_num)
        else:
            self.assertEqual([text], [token.text for token in doc])
            self.assertFalse(doc[0].like_num)

    def assert_timestamp(self, pipeline, code):
        doc = self.round_trip(pipeline, "09:00:00.000Z")
        if code == "zh":
            self.assertEqual(list("09:00:00.000Z"), [token.text for token in doc])
        elif code == "pl":
            self.assertEqual(["09:00:00", ".", "000Z"], [token.text for token in doc])
            self.assertTrue(doc[2].is_upper and doc[2].is_title)
            self.assertFalse(doc[0].like_num)
        else:
            self.assertEqual(["09:00:00.000Z"], [token.text for token in doc])
            self.assertTrue(doc[0].is_upper and doc[0].is_title)
            self.assertFalse(doc[0].is_alpha or doc[0].like_num)

    def assert_relation(self, pipeline, code, text):
        doc = self.round_trip(pipeline, text)
        if code == "zh":
            self.assertEqual([text[0], text[1:]], [token.text for token in doc])
            self.assertFalse(doc[0].is_punct)
            self.assertEqual(text[0] == "\u2248", doc[0].is_stop)
            self.assertTrue(doc[1].like_num and doc[1].is_stop)
        else:
            self.assertEqual([text], [token.text for token in doc])
            self.assertFalse(doc[0].like_num or doc[0].is_punct or doc[0].is_stop)

    def assert_final_dot(self, pipeline, code, text, kept):
        doc = self.round_trip(pipeline, text)
        stem = text[:-1]
        if code == "zh":
            self.assertEqual(list(stem) + ["."], [token.text for token in doc])
            self.assertTrue(doc[0].is_upper and doc[0].is_title)
            self.assertTrue(doc[-1].is_punct and doc[-1].is_stop)
        elif code in kept:
            self.assertEqual([text], [token.text for token in doc])
            self.assertTrue(doc[0].is_title)
            self.assertFalse(doc[0].is_alpha)
        else:
            self.assertEqual([stem, "."], [token.text for token in doc])
            self.assertTrue(doc[0].is_alpha and doc[0].is_title)
            self.assertTrue(doc[1].is_punct)

    def assert_initials(self, pipeline, code, text, stop_letter):
        doc = self.round_trip(pipeline, text)
        first, second = text[0], text[2]
        if code == "zh":
            self.assertEqual([first, ".", second, "."], [token.text for token in doc])
            self.assertTrue(doc[1].is_stop and doc[3].is_stop)
        elif code in {"lt", "sr"}:
            self.assertEqual([text[:-1], "."], [token.text for token in doc])
            self.assertTrue(doc[0].is_upper and doc[0].is_title)
            self.assertFalse(doc[0].is_alpha)
        elif code == "sl":
            self.assertEqual([first + ".", second + "."], [token.text for token in doc])
            self.assertTrue(all(token.is_upper and token.is_title for token in doc))
        elif code == "pl":
            self.assertEqual([first, ".", second, "."], [token.text for token in doc])
            self.assertEqual(second == stop_letter, doc[2].is_stop)
            self.assertFalse(doc[0].is_stop)
        else:
            self.assertEqual([text], [token.text for token in doc])
            self.assertTrue(doc[0].is_upper and doc[0].is_title)
            self.assertFalse(doc[0].is_alpha or doc[0].like_num)

    def assert_ll_contraction(self, pipeline, code, text, stem, elision):
        doc = self.round_trip(pipeline, text)
        if code == "en":
            self.assertEqual([stem, "'ll"], [token.text for token in doc])
            self.assertTrue(doc[0].is_stop and doc[1].is_stop)
        elif code in elision:
            self.assertEqual([stem + "'", "ll"], [token.text for token in doc])
            self.assertFalse(doc[0].is_stop or doc[1].is_stop)
        elif code in {"ky", "tt"}:
            self.assertEqual([stem, "'", "ll"], [token.text for token in doc])
            self.assertFalse(any(token.is_stop for token in doc))
        elif code == "zh":
            self.assertEqual(list(text), [token.text for token in doc])
        else:
            self.assertEqual([text], [token.text for token in doc])
            self.assertFalse(doc[0].is_stop or doc[0].like_num)

    def assert_s_contraction(self, pipeline, code, text, stem, glue):
        doc = self.round_trip(pipeline, text)
        if code == "en":
            self.assertEqual([stem, "'s"], [token.text for token in doc])
            self.assertTrue(doc[0].is_stop and doc[1].is_stop)
        elif code in {"ca", "fr"}:
            self.assertEqual([stem + "'", "s"], [token.text for token in doc])
            self.assertFalse(doc[0].is_stop or doc[1].is_stop)
        elif code in glue:
            self.assertEqual([text], [token.text for token in doc])
            self.assertFalse(doc[0].is_stop)
        elif code == "zh":
            self.assertEqual(list(text), [token.text for token in doc])
        else:
            self.assertEqual([stem, "'s"], [token.text for token in doc])
            self.assertFalse(doc[0].is_stop or doc[1].is_stop)

    def assert_nt_contraction(self, pipeline, code, text, stem, short):
        doc = self.round_trip(pipeline, text)
        if code == "en":
            self.assertEqual([stem, "n't"], [token.text for token in doc])
            self.assertTrue(doc[0].is_stop and doc[1].is_stop)
        elif code == "ca":
            self.assertEqual([short, "'t"], [token.text for token in doc])
            self.assertFalse(doc[0].is_stop or doc[1].is_stop)
        elif code in {"fr", "it", "lij"}:
            self.assertEqual([short + "'", "t"], [token.text for token in doc])
            self.assertFalse(doc[0].is_stop or doc[1].is_stop)
        elif code in {"ky", "tt"}:
            self.assertEqual([short, "'", "t"], [token.text for token in doc])
            self.assertFalse(any(token.is_stop for token in doc))
        elif code == "zh":
            self.assertEqual(list(text), [token.text for token in doc])
        else:
            self.assertEqual([text], [token.text for token in doc])
            self.assertFalse(doc[0].is_stop or doc[0].like_num)

    def assert_double_contraction(self, pipeline, code):
        text = "shouldn't've"
        doc = self.round_trip(pipeline, text)
        if code == "en":
            self.assertEqual(["should", "n't", "'ve"], [token.text for token in doc])
            self.assertTrue(all(token.is_stop for token in doc))
        elif code == "ca":
            self.assertEqual(["shouldn", "'t", "'", "ve"], [token.text for token in doc])
            self.assertFalse(any(token.is_stop for token in doc))
        elif code in {"fr", "it"}:
            self.assertEqual(["shouldn'", "t'", "ve"], [token.text for token in doc])
            self.assertTrue(doc[1].is_stop)
            self.assertFalse(doc[0].is_stop or doc[2].is_stop)
        elif code == "lij":
            self.assertEqual(["shouldn'", "t'", "ve"], [token.text for token in doc])
            self.assertFalse(any(token.is_stop for token in doc))
        elif code in {"ky", "tt"}:
            self.assertEqual(["shouldn", "'", "t", "'", "ve"], [token.text for token in doc])
            self.assertFalse(any(token.is_stop for token in doc))
        elif code == "zh":
            self.assertEqual(list(text), [token.text for token in doc])
        else:
            self.assertEqual([text], [token.text for token in doc])
            self.assertFalse(doc[0].is_stop or doc[0].like_num)

    def assert_hyphen_compound(self, pipeline, code, text, glue, stops=None, like_num=None):
        doc = self.round_trip(pipeline, text)
        stops = stops or {}
        like_num = like_num or {}
        if code == "zh":
            self.assertEqual(list(text), [token.text for token in doc])
            self.assertTrue(all(token.is_stop for token in doc if token.text == "-"))
        elif code in glue:
            self.assertEqual([text], [token.text for token in doc])
            self.assertFalse(doc[0].is_stop or doc[0].like_num)
        else:
            pieces = text.split("-")
            expected = []
            for index, piece in enumerate(pieces):
                if index:
                    expected.append("-")
                expected.append(piece)
            self.assertEqual(expected, [token.text for token in doc])
            for token in doc:
                if token.text == "-":
                    self.assertTrue(token.is_punct)
                    self.assertFalse(token.is_stop)
                    continue
                self.assertEqual(token.text in stops.get(code, ()), token.is_stop)
                self.assertEqual(token.text in like_num.get(code, ()), token.like_num)

    def assert_url(self, pipeline, code, text, yo_num):
        doc = self.round_trip(pipeline, text)
        if code == "zh":
            self.assertGreater(len(doc), 1)
            self.assertFalse(any(token.like_url or token.like_email for token in doc))
        else:
            self.assertEqual([text], [token.text for token in doc])
            self.assertTrue(doc[0].like_url)
            self.assertFalse(doc[0].like_email)
            self.assertEqual(code == "yo" and yo_num, doc[0].like_num)

    def assert_file_url(self, pipeline, code, glue, a_stop):
        doc = self.round_trip(pipeline, "file:///tmp/a")
        if code == "zh":
            self.assertEqual(list("file:///tmp/a"), [token.text for token in doc])
            self.assertFalse(any(token.like_url for token in doc))
        elif code in glue:
            self.assertEqual(["file:///tmp/a"], [token.text for token in doc])
            self.assertFalse(doc[0].like_url or doc[0].like_num)
        else:
            self.assertEqual(["file:///tmp", "/", "a"], [token.text for token in doc])
            self.assertFalse(doc[0].like_url)
            self.assertEqual(code == "yo", doc[0].like_num)
            self.assertEqual(code in a_stop, doc[2].is_stop)

    def assert_dici(self, pipeline, code):
        doc = self.round_trip(pipeline, "d'ici 9h")
        if code == "zh":
            self.assertEqual(list("d'ici") + ["9", "h"], [token.text for token in doc])
        elif code in {"fr", "it", "lb", "lij"}:
            self.assertEqual(["d'", "ici", "9h"], [token.text for token in doc])
            self.assertTrue(doc[0].is_stop)
            self.assertFalse(doc[1].is_stop or doc[2].like_num)
        elif code == "ca":
            self.assertEqual(["d'", "ici", "9h"], [token.text for token in doc])
            self.assertFalse(doc[0].is_stop)
        elif code in {"ky", "tt"}:
            self.assertEqual(["d", "'", "ici", "9h"], [token.text for token in doc])
        else:
            self.assertEqual(["d'ici", "9h"], [token.text for token in doc])
            self.assertFalse(doc[1].like_num)
            self.assertEqual(" ", doc[0].whitespace_)

    def assert_a_lheure(self, pipeline, code):
        doc = self.round_trip(pipeline, "\u00e0 l'heure")
        if code == "zh":
            self.assertEqual(["\u00e0"] + list("l'heure"), [token.text for token in doc])
        elif code in {"fr", "lij"}:
            self.assertEqual(["\u00e0", "l'", "heure"], [token.text for token in doc])
            self.assertTrue(doc[0].is_stop and doc[1].is_stop)
        elif code == "it":
            self.assertEqual(["\u00e0", "l'", "heure"], [token.text for token in doc])
            self.assertTrue(doc[1].is_stop)
            self.assertFalse(doc[0].is_stop)
        elif code == "ca":
            self.assertEqual(["\u00e0", "l'", "heure"], [token.text for token in doc])
            self.assertFalse(doc[0].is_stop or doc[1].is_stop)
        elif code in {"ky", "tt"}:
            self.assertEqual(["\u00e0", "l", "'", "heure"], [token.text for token in doc])
        elif code in {"lb", "pt", "vi", "yo"}:
            self.assertEqual(["\u00e0", "l'heure"], [token.text for token in doc])
            self.assertTrue(doc[0].is_stop)
        else:
            self.assertEqual(["\u00e0", "l'heure"], [token.text for token in doc])
            self.assertFalse(doc[0].is_stop)

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

    def test_recurrence_clocks_and_new_symbols_keep_their_decisions(self):
        # French lexicon hits with a time or recurrence grain, and no request.
        # "lundi" is a subject-less verb, but the French path does not treat
        # that shape as a scheduling request.
        french_clarify = {
            "d'ici 9h": ("NOUN", "9h"),
            "d'ici demain": ("NOUN", "demain"),
            "9h pile": ("NOUN", "pile"),
            "chaque jour": ("PROPN", "jour"),
            "tous les jours": ("NOUN", "jour"),
            "lundi matin": ("VERB", "lundi"),
            "apr\u00e8s-demain \u00e0 9h": ("NOUN", "9h"),
            "rendez-vous \u00e0 9h": ("NOUN", "9h"),
        }
        french_reject = {
            "\u00e0 bient\u00f4t": ("NOUN", "bient\u00f4t"),
            "fin de journ\u00e9e": ("PROPN", "journ\u00e9e"),
            "\u00e0 l'heure": ("PROPN", "l'heure"),
        }
        # "midi" is temporal on the English path. "dans une heure" is too,
        # even though the French marker does not claim the phrase.
        english_clarify_temporal = {
            "dans une heure": ("PROPN", "dans"),
            "ce midi": ("PROPN", "midi"),
        }
        english_clarify_request = {
            "hadn't": ("VERB", "have"),
            "follow-up": ("NOUN", "up"),
            "check-in": ("VERB", "check"),
            "Inf": ("VERB", "inf"),
            "\U0001f514": ("VERB", "\U0001f514"),
            "\u00ac": ("VERB", "\u00ac"),
            "km\u00b2": ("VERB", "km\u00b2"),
        }
        english_reject = {
            "they'll": ("AUX", "will", True, "not_a_schedule_request"),
            "how's": ("AUX", "be", False, "informational_question_not_schedule_request"),
            "when's": ("AUX", "be", False, "informational_question_not_schedule_request"),
            "what'll": ("AUX", "will", True, "informational_question_not_schedule_request"),
            "shouldn't've": ("AUX", "'ve", False, "not_a_schedule_request"),
            "ce week-end": ("NOUN", "end", False, "not_a_schedule_request"),
            "tout de suite": ("NOUN", "suite", False, "not_a_schedule_request"),
            "imma": ("PROPN", "imma", False, "not_a_schedule_request"),
            "gimme": ("NOUN", "gimme", False, "not_a_schedule_request"),
            "kinda": ("ADV", "kinda", False, "not_a_schedule_request"),
            "sorta": ("ADV", "sorta", False, "not_a_schedule_request"),
            "out-of-office": ("ADP", "out", False, "not_a_schedule_request"),
            "mother-in-law": ("NOUN", "mother", False, "not_a_schedule_request"),
            "all-day": ("NOUN", "day", False, "not_a_schedule_request"),
            "part-time": ("NOUN", "time", False, "not_a_schedule_request"),
            "well-being": ("NOUN", "being", False, "not_a_schedule_request"),
            "Apr.": ("PROPN", "Apr.", False, "not_a_schedule_request"),
            "Sun.": ("PROPN", "Sun", False, "not_a_schedule_request"),
            "N.B.": ("PROPN", "N.B.", False, "not_a_schedule_request"),
            "ASAP": ("NOUN", "ASAP", False, "not_a_schedule_request"),
            "H1": ("NOUN", "H1", False, "not_a_schedule_request"),
            "2026-10-12": ("NUM", "2026", False, "not_a_schedule_request"),
            "PT1H30M": ("PROPN", "M", False, "not_a_schedule_request"),
            "P1DT2H": ("PUNCT", "P1DT2H", False, "not_a_schedule_request"),
            "9mins": ("NOUN", "9min", False, "not_a_schedule_request"),
            "2h30m": ("NUM", "2h30", False, "not_a_schedule_request"),
            "\u09f3": ("PUNCT", "\u09f3", False, "not_a_schedule_request"),
            "\u2150": ("NOUN", "\u2150", False, "not_a_schedule_request"),
            "\u2615": ("NOUN", "\u2615", False, "not_a_schedule_request"),
            "\u231a": ("PROPN", "\u231a", False, "not_a_schedule_request"),
            "XLII": ("PROPN", "XLII", False, "not_a_schedule_request"),
            "10\u00b2": ("NUM", "10\u00b2", False, "not_a_schedule_request"),
            "3\u00bd": ("NUM", "3\u00bd", False, "not_a_schedule_request"),
            "\u22655": ("PROPN", "\u22655", False, "not_a_schedule_request"),
            "https://user:pass@example.com": (
                "NOUN", "https://user:pass@example.com", False, "not_a_schedule_request",
            ),
            "http://localhost:3000": (
                "NUM", "http://localhost:3000", False, "not_a_schedule_request",
            ),
            "a@b": ("NOUN", "a@b", False, "not_a_schedule_request"),
            "file:///tmp/a": ("PROPN", "file:///tmp", False, "not_a_schedule_request"),
        }

        for text, (pos, lemma) in french_clarify.items():
            with self.subTest(text=text):
                result = self.classify_without_entity(text)
                self.assertTrue(text_looks_french(text))
                self.assertEqual("fr_FR", self.duckling_locale(text))
                self.assertEqual(
                    ("clarify", "temporal_signal_without_clear_request"),
                    (result["decision"], result["reason"]),
                )
                self.assertTrue(result["features"]["has_temporal_signal"])
                self.assertFalse(result["features"]["looks_like_request"])
                self.assertFalse(result["features"]["has_subject"])
                self.assertEqual(pos, result["features"]["root"]["pos"])
                self.assertEqual(lemma, result["features"]["root"]["lemma"])

        for text, (pos, lemma) in french_reject.items():
            with self.subTest(text=text):
                result = self.classify_without_entity(text)
                self.assertTrue(text_looks_french(text))
                self.assertEqual("fr_FR", self.duckling_locale(text))
                self.assertEqual(
                    ("reject", "not_a_schedule_request"),
                    (result["decision"], result["reason"]),
                )
                self.assertFalse(result["features"]["has_temporal_signal"])
                self.assertFalse(result["features"]["looks_like_request"])
                self.assertEqual(pos, result["features"]["root"]["pos"])
                self.assertEqual(lemma, result["features"]["root"]["lemma"])

        for text, (pos, lemma) in english_clarify_temporal.items():
            with self.subTest(text=text):
                result = self.classify_without_entity(text)
                self.assertFalse(text_looks_french(text))
                self.assertEqual("en_GB", self.duckling_locale(text))
                self.assertEqual(
                    ("clarify", "temporal_signal_without_clear_request"),
                    (result["decision"], result["reason"]),
                )
                self.assertTrue(result["features"]["has_temporal_signal"])
                self.assertFalse(result["features"]["looks_like_request"])
                self.assertEqual(pos, result["features"]["root"]["pos"])
                self.assertEqual(lemma, result["features"]["root"]["lemma"])

        with patch("intent.duckling_parse", return_value=[]):
            midi = classify("midi et demi")
        self.assertFalse(text_looks_french("midi et demi"))
        self.assertEqual("en_GB", self.duckling_locale("midi et demi"))
        self.assertEqual(
            ("allow", "request_with_temporal_signal"),
            (midi["decision"], midi["reason"]),
        )
        self.assertEqual("VERB", midi["features"]["root"]["pos"])
        self.assertEqual("midi", midi["features"]["root"]["lemma"])
        self.assertFalse(midi["features"]["has_subject"])
        self.assertTrue(midi["features"]["has_temporal_signal"])
        self.assertTrue(midi["features"]["looks_like_request"])

        for text, (pos, lemma) in english_clarify_request.items():
            with self.subTest(text=text):
                result = self.classify_without_entity(text)
                self.assertFalse(text_looks_french(text))
                self.assertEqual(
                    ("clarify", "request_without_temporal_signal"),
                    (result["decision"], result["reason"]),
                )
                self.assertFalse(result["features"]["has_temporal_signal"])
                self.assertTrue(result["features"]["looks_like_request"])
                self.assertFalse(result["features"]["has_subject"])
                self.assertEqual(pos, result["features"]["root"]["pos"])
                self.assertEqual(lemma, result["features"]["root"]["lemma"])

        for text, (pos, lemma, subject, reason) in english_reject.items():
            with self.subTest(text=text):
                result = self.classify_without_entity(text)
                self.assertFalse(text_looks_french(text))
                self.assertEqual(("reject", reason), (result["decision"], result["reason"]))
                self.assertFalse(result["features"]["has_temporal_signal"])
                self.assertFalse(result["features"]["looks_like_request"])
                self.assertEqual(subject, result["features"]["has_subject"])
                self.assertEqual(pos, result["features"]["root"]["pos"])
                self.assertEqual(lemma, result["features"]["root"]["lemma"])
                doc = english_nlp(text)
                self.assertEqual(text, "".join(token.text_with_ws for token in doc))

    def classify_without_entity(self, text):
        with patch("intent.duckling_parse", return_value=[]):
            return classify(text)

    def duckling_locale(self, text):
        with patch("intent.requests.post") as post:
            post.return_value.json.return_value = []
            post.return_value.raise_for_status.return_value = None
            classify(text)
        return post.call_args.kwargs["data"]["locale"]


if __name__ == "__main__":
    unittest.main()
