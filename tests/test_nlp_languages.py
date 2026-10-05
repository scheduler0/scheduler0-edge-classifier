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

    def test_fractions_compounds_clocks_and_contractions(self):
        # Eighths, sixths, and fifths stay one token and are never numerals.
        # Superscript digits are numerals except in Ancient Greek. Roman
        # numeral characters are cased symbols, not like_num, even in Latin
        # and Russian where ASCII "XII" is a numeral.
        vulgar = ("⅛", "⅜", "⅝", "⅞", "⅙", "⅚", "⅕", "⅖", "⅗", "⅘", "⅟", "↉")
        superscript_digits = ("⁴", "⁵", "₃")
        neutral_marks = ("⁺", "⁻", "₊", "©", "®", "™", "℠", "℗")
        currencies = ("؋", "﷼", "₠", "₢", "₰", "₯", "₧", "₻", "֏")
        math_symbols = ("∞", "√", "÷", "≠", "≤", "≥", "∆", "∈", "∂", "∑")
        punct_marks = ("‡", "⁂", "⸮")
        here_glued = {
            "am", "ar", "bn", "da", "de", "el", "es", "fa", "fi", "grc", "hu",
            "nb", "nl", "nn", "pl", "ro", "sl", "sr", "sv", "ti", "vi",
        }
        n_t_elision = {"fr", "it", "lij"}
        ll_elision = {"ca", "fr", "it", "lij"}
        apostrophe_parts = {"ky", "tt"}
        quelqu_like_num = {"ca", "fr"}
        quelqu_stop = {"it", "lij"}
        dabord_both_stop = {"fr"}
        dabord_elision_stop = {"it", "lb", "lij"}
        peut_etre_glued = {
            "ca", "da", "de", "es", "fi", "fr", "hu", "ky", "lb", "nb", "nl",
            "nn", "pt", "ro", "sr", "sv", "tt", "vi",
        }
        questce_glued = {
            "da", "de", "es", "fi", "hu", "lb", "nb", "nl", "nn", "ro", "sr",
            "sv", "vi",
        }
        questce_prefix = {"ca", "el", "pt"}
        hyphen_glued = {
            "ca", "da", "de", "el", "es", "fi", "fr", "hu", "ky", "lb", "nb",
            "nl", "nn", "pt", "ro", "sr", "sv", "tt", "vi",
        }
        ltd_keeps_dot = {"am", "ar", "en", "fa", "grc", "hu", "nl", "pt", "ti", "tr", "vi"}
        aka_keeps_dot = {"am", "ar", "fa", "grc", "ti", "vi"}
        nr_keeps_dot = {"am", "ar", "da", "de", "fa", "fo", "grc", "hu", "nl", "ro", "ti", "vi"}
        dh_keeps_dot = {"am", "ar", "de", "fa", "grc", "nl", "ti", "vi"}
        caret_glued = {
            "da", "de", "fi", "hu", "ky", "lb", "nb", "nl", "nn", "pl", "sv",
            "tt", "vi",
        }
        signed_not_num = {"fa", "grc", "la", "lb", "lt", "nl", "si", "ta", "te", "tl", "uk", "yo"}
        degree_f_glued = {"hu", "nb", "nn", "ro", "vi"}
        numero_glued = {"hu", "nb", "ro", "vi"}
        co_glued = {"da", "de", "hu", "lb", "nl", "ro", "vi"}
        co_o_stop = {"az", "ca", "cs", "es", "fr", "hr", "lij", "lt", "pl", "pt", "sk", "tl", "tn", "tr", "yo"}
        emoji_glued = {"hu", "vi"}

        for code, pipeline in self.pipelines.items():
            if pipeline is None:
                continue
            with self.subTest(code=code):
                for text in vulgar:
                    token = self.round_trip(pipeline, text)[0]
                    self.assertEqual(text, token.text)
                    self.assertFalse(token.like_num or token.is_punct or token.is_alpha or token.is_digit)

                for text in superscript_digits:
                    token = self.round_trip(pipeline, text)[0]
                    self.assertTrue(token.is_digit)
                    self.assertEqual(code != "grc", token.like_num)
                    self.assertFalse(token.is_alpha)

                for text in ("ⁿ", "ℓ"):
                    token = self.round_trip(pipeline, text)[0]
                    self.assertTrue(token.is_alpha and token.is_lower)
                    self.assertFalse(token.like_num or token.is_digit or token.is_punct)

                for text in neutral_marks:
                    token = self.round_trip(pipeline, text)[0]
                    self.assertEqual(text, token.text)
                    self.assertFalse(
                        token.is_punct or token.is_alpha or token.is_digit
                        or token.like_num or token.is_currency or token.is_space
                    )

                for text in currencies:
                    token = self.round_trip(pipeline, text)[0]
                    self.assertTrue(token.is_currency)
                    self.assertFalse(token.is_punct or token.is_alpha)

                for text in math_symbols:
                    token = self.round_trip(pipeline, text)[0]
                    self.assertFalse(token.is_punct or token.is_currency or token.is_alpha)

                for text in punct_marks + ("⟨", "⟩"):
                    token = self.round_trip(pipeline, text)[0]
                    self.assertTrue(token.is_punct)
                    self.assertFalse(token.is_bracket or token.is_quote or token.is_stop)

                for text in ("Ⅻ", "Ⅸ"):
                    token = self.round_trip(pipeline, text)[0]
                    self.assertTrue(token.is_upper and token.is_title)
                    self.assertFalse(token.is_alpha or token.like_num)
                roman_three = self.round_trip(pipeline, "ⅲ")[0]
                self.assertTrue(roman_three.is_lower)
                self.assertFalse(roman_three.is_alpha or roman_three.like_num)

                ohm = self.round_trip(pipeline, "Ω")[0]
                self.assertTrue(ohm.is_alpha and ohm.is_upper and ohm.is_title)
                self.assertEqual(code == "el", ohm.is_stop)
                angstrom = self.round_trip(pipeline, "Å")[0]
                self.assertTrue(angstrom.is_alpha and angstrom.is_upper)
                self.assertEqual(code == "nb", angstrom.is_stop)
                space_symbol = self.round_trip(pipeline, "␣")[0]
                self.assertFalse(space_symbol.is_space or space_symbol.is_punct)

                shouldnt = self.round_trip(pipeline, "shouldn't")
                youll = self.round_trip(pipeline, "you'll")
                heres = self.round_trip(pipeline, "here's")
                if code == "zh":
                    self.assertEqual(list("shouldn't"), [token.text for token in shouldnt])
                    self.assertTrue(shouldnt[7].is_quote and shouldnt[7].is_stop)
                    self.assertEqual(list("you'll"), [token.text for token in youll])
                    self.assertEqual(list("here's"), [token.text for token in heres])
                elif code == "en":
                    self.assertEqual(["should", "n't"], [token.text for token in shouldnt])
                    self.assertTrue(all(token.is_stop for token in shouldnt))
                    self.assertEqual(["you", "'ll"], [token.text for token in youll])
                    self.assertTrue(all(token.is_stop for token in youll))
                    self.assertEqual(["here", "'s"], [token.text for token in heres])
                    self.assertTrue(all(token.is_stop for token in heres))
                elif code in apostrophe_parts and code not in here_glued:
                    self.assertEqual(["shouldn", "'", "t"], [token.text for token in shouldnt])
                    self.assertTrue(shouldnt[1].is_quote)
                    self.assertEqual(["you", "'", "ll"], [token.text for token in youll])
                    self.assertEqual(["here", "'s"], [token.text for token in heres])
                    self.assertFalse(any(token.is_stop for token in heres))
                else:
                    if code == "ca":
                        self.assertEqual(["shouldn", "'t"], [token.text for token in shouldnt])
                    elif code in n_t_elision:
                        self.assertEqual(["shouldn'", "t"], [token.text for token in shouldnt])
                    else:
                        self.assertEqual(["shouldn't"], [token.text for token in shouldnt])
                        self.assertFalse(shouldnt[0].like_num)
                    if code in ll_elision:
                        self.assertEqual(["you'", "ll"], [token.text for token in youll])
                    elif code in apostrophe_parts:
                        self.assertEqual(["you", "'", "ll"], [token.text for token in youll])
                    else:
                        self.assertEqual(["you'll"], [token.text for token in youll])
                    if code in here_glued:
                        self.assertEqual(["here's"], [token.text for token in heres])
                    elif code in {"ca", "fr"}:
                        self.assertEqual(["here'", "s"], [token.text for token in heres])
                    elif code == "sq":
                        self.assertEqual(["here", "'s"], [token.text for token in heres])
                        self.assertTrue(heres[0].is_stop)
                        self.assertFalse(heres[1].is_stop)
                    else:
                        self.assertEqual(["here", "'s"], [token.text for token in heres])
                        self.assertFalse(any(token.is_stop for token in heres))

                quelqu = self.round_trip(pipeline, "quelqu'un")
                dabord = self.round_trip(pipeline, "d'abord")
                peut = self.round_trip(pipeline, "peut-être")
                quest = self.round_trip(pipeline, "qu'est-ce")
                if code == "zh":
                    self.assertEqual(list("quelqu'un"), [token.text for token in quelqu])
                    self.assertEqual(list("d'abord"), [token.text for token in dabord])
                    self.assertEqual(list("peut-être"), [token.text for token in peut])
                    self.assertTrue(peut[5].is_alpha)
                    self.assertEqual(list("qu'est-ce"), [token.text for token in quest])
                elif code in apostrophe_parts:
                    self.assertEqual(["quelqu", "'", "un"], [token.text for token in quelqu])
                    self.assertFalse(quelqu[2].is_stop or quelqu[2].like_num)
                    self.assertEqual(["d", "'", "abord"], [token.text for token in dabord])
                    self.assertEqual(["peut-être"], [token.text for token in peut])
                    self.assertEqual(["qu", "'", "est-ce"], [token.text for token in quest])
                    self.assertTrue(quest[1].is_quote)
                else:
                    if code in quelqu_like_num:
                        self.assertEqual(["quelqu'", "un"], [token.text for token in quelqu])
                        self.assertTrue(quelqu[1].is_stop and quelqu[1].like_num)
                    elif code in quelqu_stop:
                        self.assertEqual(["quelqu'", "un"], [token.text for token in quelqu])
                        self.assertTrue(quelqu[1].is_stop)
                        self.assertFalse(quelqu[1].like_num)
                    else:
                        self.assertEqual(["quelqu'un"], [token.text for token in quelqu])
                    if code in dabord_both_stop:
                        self.assertEqual(["d'", "abord"], [token.text for token in dabord])
                        self.assertTrue(all(token.is_stop for token in dabord))
                    elif code in dabord_elision_stop:
                        self.assertEqual(["d'", "abord"], [token.text for token in dabord])
                        self.assertTrue(dabord[0].is_stop)
                        self.assertFalse(dabord[1].is_stop)
                    elif code == "ca":
                        self.assertEqual(["d'", "abord"], [token.text for token in dabord])
                        self.assertFalse(any(token.is_stop for token in dabord))
                    else:
                        self.assertEqual(["d'abord"], [token.text for token in dabord])
                    if code in peut_etre_glued:
                        self.assertEqual(["peut-être"], [token.text for token in peut])
                    else:
                        self.assertEqual(["peut", "-", "être"], [token.text for token in peut])
                        self.assertTrue(peut[1].is_punct)
                    if code == "fr":
                        self.assertEqual(["qu'", "est", "-ce"], [token.text for token in quest])
                        self.assertTrue(quest[0].is_stop and quest[1].is_stop)
                        self.assertFalse(quest[2].is_stop)
                    elif code in {"it", "lij"}:
                        self.assertEqual(["qu'", "est", "-", "ce"], [token.text for token in quest])
                        self.assertFalse(any(token.is_stop for token in quest))
                    elif code in questce_prefix:
                        self.assertEqual(["qu'", "est-ce"], [token.text for token in quest])
                    elif code in questce_glued:
                        self.assertEqual(["qu'est-ce"], [token.text for token in quest])
                    else:
                        self.assertEqual(["qu'est", "-", "ce"], [token.text for token in quest])

                for text in ("8h15", "14h", "23h59", "0h", "24h", "9hr", "9min", "09:30:00"):
                    doc = self.round_trip(pipeline, text)
                    if code == "zh":
                        self.assertEqual(list(text), [token.text for token in doc])
                        self.assertTrue(all(token.like_num for token in doc if token.is_digit))
                    else:
                        self.assertEqual([text], [token.text for token in doc])
                        self.assertFalse(doc[0].like_num)

                demi = self.round_trip(pipeline, "demi-heure")
                if code == "zh":
                    self.assertEqual(list("demi-heure"), [token.text for token in demi])
                elif code in hyphen_glued:
                    self.assertEqual(["demi-heure"], [token.text for token in demi])
                else:
                    self.assertEqual(["demi", "-", "heure"], [token.text for token in demi])
                    self.assertEqual(code in {"id", "ms"}, demi[0].is_stop)

                self.assert_final_dot(pipeline, "Ltd.", code in ltd_keeps_dot, code)
                self.assert_final_dot(pipeline, "a.k.a.", code in aka_keeps_dot, code)
                self.assert_final_dot(pipeline, "d.h.", code in dh_keeps_dot, code)
                nr = self.round_trip(pipeline, "Nr.")
                if code == "zh":
                    self.assertEqual(list("Nr."), [token.text for token in nr])
                elif code in nr_keeps_dot:
                    self.assertEqual(["Nr."], [token.text for token in nr])
                else:
                    self.assertEqual(["Nr", "."], [token.text for token in nr])
                    self.assertEqual(code == "sv", nr[0].is_stop)

                co = self.round_trip(pipeline, "c/o")
                if code == "zh":
                    self.assertEqual(["c", "/", "o"], [token.text for token in co])
                    self.assertTrue(co[1].is_punct and co[1].is_stop)
                elif code in co_glued:
                    self.assertEqual(["c/o"], [token.text for token in co])
                elif code == "la":
                    self.assertEqual(["c", "/", "o"], [token.text for token in co])
                    self.assertTrue(co[0].like_num)
                    self.assertTrue(co[2].is_stop)
                elif code == "sl":
                    self.assertEqual(["c", "/", "o"], [token.text for token in co])
                    self.assertTrue(co[0].is_stop and co[2].is_stop)
                elif code == "sq":
                    self.assertEqual(["c", "/", "o"], [token.text for token in co])
                    self.assertTrue(co[0].is_stop)
                    self.assertFalse(co[2].is_stop)
                elif code in co_o_stop:
                    self.assertEqual(["c", "/", "o"], [token.text for token in co])
                    self.assertFalse(co[0].is_stop)
                    self.assertTrue(co[2].is_stop)
                else:
                    self.assertEqual(["c", "/", "o"], [token.text for token in co])
                    self.assertFalse(co[0].is_stop or co[2].is_stop)

                millions = self.round_trip(pipeline, "1,000,000")
                dotted_millions = self.round_trip(pipeline, "1.000.000")
                if code == "zh":
                    self.assertEqual(list("1,000,000"), [token.text for token in millions])
                    self.assertTrue(all(token.is_stop for token in millions))
                    self.assertEqual(list("1.000.000"), [token.text for token in dotted_millions])
                elif code == "pl":
                    self.assertEqual(["1,000,000"], [token.text for token in millions])
                    self.assertTrue(millions[0].like_num)
                    self.assertEqual(
                        ["1", ".", "000", ".", "000"],
                        [token.text for token in dotted_millions],
                    )
                    self.assertEqual(
                        [True, False, True, False, True],
                        [token.like_num for token in dotted_millions],
                    )
                else:
                    self.assertEqual(["1,000,000"], [token.text for token in millions])
                    self.assertEqual(code not in {"grc", "la", "ne"}, millions[0].like_num)
                    self.assertEqual(["1.000.000"], [token.text for token in dotted_millions])
                    self.assertEqual(code not in {"grc", "la"}, dotted_millions[0].like_num)

                bond = self.round_trip(pipeline, "007")
                if code == "zh":
                    self.assertEqual(list("007"), [token.text for token in bond])
                    self.assertTrue(all(token.like_num and token.is_stop for token in bond))
                else:
                    self.assertEqual(["007"], [token.text for token in bond])
                    self.assertTrue(bond[0].is_digit)
                    self.assertEqual(code != "grc", bond[0].like_num)

                scientific = self.round_trip(pipeline, "1e+10")
                avogadro = self.round_trip(pipeline, "6.022e23")
                if code == "zh":
                    self.assertEqual(list("1e+10"), [token.text for token in scientific])
                    self.assertTrue(scientific[2].is_stop)
                    self.assertEqual(list("6.022e23"), [token.text for token in avogadro])
                elif code == "pl":
                    self.assertEqual(["1e+10"], [token.text for token in scientific])
                    self.assertFalse(scientific[0].like_num)
                    self.assertEqual(["6", ".", "022e23"], [token.text for token in avogadro])
                    self.assertTrue(avogadro[0].like_num)
                    self.assertFalse(avogadro[2].like_num)
                else:
                    self.assertEqual(["1e+10"], [token.text for token in scientific])
                    self.assertFalse(scientific[0].like_num)
                    self.assertEqual(["6.022e23"], [token.text for token in avogadro])
                    self.assertFalse(avogadro[0].like_num)

                power = self.round_trip(pipeline, "2^10")
                if code == "zh":
                    self.assertEqual(["2", "^", "1", "0"], [token.text for token in power])
                    self.assertTrue(power[1].is_stop)
                elif code in caret_glued:
                    self.assertEqual(["2^10"], [token.text for token in power])
                    self.assertFalse(power[0].like_num)
                elif code == "grc":
                    self.assertEqual(["2", "^", "10"], [token.text for token in power])
                    self.assertFalse(any(token.like_num for token in power))
                else:
                    self.assertEqual(["2", "^", "10"], [token.text for token in power])
                    self.assertEqual([True, False, True], [token.like_num for token in power])
                    self.assertFalse(power[1].is_punct)

                absolute = self.round_trip(pipeline, "-273.15")
                if code == "zh":
                    self.assertEqual(list("-273.15"), [token.text for token in absolute])
                    self.assertTrue(absolute[0].is_punct and absolute[0].is_stop)
                elif code == "pl":
                    self.assertEqual(["-273", ".", "15"], [token.text for token in absolute])
                    self.assertFalse(absolute[0].like_num)
                    self.assertTrue(absolute[2].like_num)
                elif code in {"ca", "sl"}:
                    self.assertEqual(["-", "273.15"], [token.text for token in absolute])
                    self.assertTrue(absolute[0].is_punct)
                    self.assertTrue(absolute[1].like_num)
                else:
                    self.assertEqual(["-273.15"], [token.text for token in absolute])
                    self.assertEqual(code not in signed_not_num, absolute[0].like_num)

                fahrenheit = self.round_trip(pipeline, "5°F")
                if code in degree_f_glued:
                    self.assertEqual(["5°F"], [token.text for token in fahrenheit])
                    self.assertFalse(fahrenheit[0].like_num)
                elif code == "it":
                    self.assertEqual(["5°", "F"], [token.text for token in fahrenheit])
                    self.assertFalse(fahrenheit[0].like_num)
                elif code in {"sl", "yo"}:
                    self.assertEqual(["5", "°", "F"], [token.text for token in fahrenheit])
                    self.assertTrue(fahrenheit[0].like_num and fahrenheit[2].is_stop)
                elif code == "grc":
                    self.assertEqual(["5", "°", "F"], [token.text for token in fahrenheit])
                    self.assertFalse(fahrenheit[0].like_num or fahrenheit[2].is_stop)
                else:
                    self.assertEqual(["5", "°", "F"], [token.text for token in fahrenheit])
                    self.assertTrue(fahrenheit[0].like_num)
                    self.assertFalse(fahrenheit[1].is_punct or fahrenheit[2].is_stop)

                ordinal_o = self.round_trip(pipeline, "1º")
                if code == "zh":
                    self.assertEqual(["1", "º"], [token.text for token in ordinal_o])
                    self.assertTrue(ordinal_o[0].like_num)
                    self.assertTrue(ordinal_o[1].is_alpha and ordinal_o[1].is_lower)
                else:
                    self.assertEqual(["1º"], [token.text for token in ordinal_o])
                    self.assertEqual(code == "pt", ordinal_o[0].like_num)

                numero = self.round_trip(pipeline, "N°")
                if code in numero_glued:
                    self.assertEqual(["N°"], [token.text for token in numero])
                elif code in {"sl", "yo"}:
                    self.assertEqual(["N", "°"], [token.text for token in numero])
                    self.assertTrue(numero[0].is_stop)
                else:
                    self.assertEqual(["N", "°"], [token.text for token in numero])
                    self.assertFalse(numero[0].is_stop or numero[1].is_punct)

                angles = self.round_trip(pipeline, "⟨hello⟩")
                singles = self.round_trip(pipeline, "‹hello›")
                if code == "zh":
                    self.assertEqual(["⟨", "h", "e", "l", "l", "o", "⟩"], [token.text for token in angles])
                    self.assertTrue(angles[0].is_punct and not angles[0].is_quote)
                    self.assertEqual(
                        ["‹", "h", "e", "l", "l", "o", "›"],
                        [token.text for token in singles],
                    )
                    self.assertTrue(singles[0].is_left_punct and singles[0].is_quote)
                    self.assertTrue(singles[-1].is_right_punct)
                else:
                    self.assertEqual(["⟨hello⟩"], [token.text for token in angles])
                    self.assertEqual(["‹hello›"], [token.text for token in singles])

                heart = self.round_trip(pipeline, "❤️")
                rainbow = self.round_trip(pipeline, "🏳️‍🌈")
                bang = self.round_trip(pipeline, "‼️")
                if code in emoji_glued:
                    self.assertEqual(["❤️"], [token.text for token in heart])
                    self.assertEqual(["🏳️‍🌈"], [token.text for token in rainbow])
                    self.assertEqual(["‼️"], [token.text for token in bang])
                elif code == "zh":
                    self.assertEqual(["❤", "️"], [token.text for token in heart])
                    self.assertEqual(["🏳", "️", "\u200d", "🌈"], [token.text for token in rainbow])
                    self.assertEqual(["‼", "️"], [token.text for token in bang])
                    self.assertTrue(bang[0].is_punct)
                    self.assertFalse(bang[1].is_punct)
                else:
                    self.assertEqual(["❤", "️"], [token.text for token in heart])
                    self.assertEqual(["🏳", "️\u200d", "🌈"], [token.text for token in rainbow])
                    self.assertEqual(["‼️"], [token.text for token in bang])
                    self.assertFalse(bang[0].is_punct)

                noon = self.round_trip(pipeline, "noon")
                if code == "zh":
                    self.assertEqual(list("noon"), [token.text for token in noon])
                else:
                    self.assertEqual(["noon"], [token.text for token in noon])
                    self.assertEqual(code == "tl", noon[0].is_stop)
                    self.assertTrue(noon[0].is_alpha)

        # Nepali rejects a comma group and still accepts the dotted group.
        self.assertFalse(self.pipelines["ne"]("1,000,000")[0].like_num)
        self.assertTrue(self.pipelines["ne"]("1.000.000")[0].like_num)

    def assert_final_dot(self, pipeline, text, keeps_dot, code):
        doc = self.round_trip(pipeline, text)
        if code == "zh":
            self.assertEqual(list(text), [token.text for token in doc])
        elif keeps_dot:
            self.assertEqual([text], [token.text for token in doc])
        else:
            self.assertEqual([text[:-1], "."], [token.text for token in doc])
            self.assertTrue(doc[-1].is_punct)

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

    def test_clocks_compounds_and_symbol_decisions_are_stable(self):
        # (decision, reason, pos, lemma, has_subject, french locale)
        expected = {
            "⅟": ("clarify", "request_without_temporal_signal", "VERB", "⅟", False, False),
            "⁂": ("clarify", "request_without_temporal_signal", "VERB", "⁂", False, False),
            "⅛": ("reject", "not_a_schedule_request", "NUM", "⅛", False, False),
            "⅙": ("reject", "not_a_schedule_request", "ADV", "⅙", False, False),
            "℗": ("reject", "not_a_schedule_request", "PUNCT", "℗", False, False),
            "℠": ("reject", "not_a_schedule_request", "NOUN", "℠", False, False),
            "™": ("reject", "not_a_schedule_request", "X", "™", False, False),
            "Å": ("reject", "not_a_schedule_request", "NOUN", "å", False, False),
            "Ⅻ": ("reject", "not_a_schedule_request", "X", "ⅻ", False, False),
            "shouldn't": ("reject", "not_a_schedule_request", "AUX", "should", False, False),
            "you'll": ("reject", "not_a_schedule_request", "AUX", "will", True, False),
            "here's": ("reject", "not_a_schedule_request", "AUX", "be", False, False),
            "what's": ("reject", "informational_question_not_schedule_request", "AUX", "be", True, False),
            "quelqu'un": ("reject", "not_a_schedule_request", "PROPN", "quelqu'un", False, False),
            "d'abord": ("reject", "not_a_schedule_request", "PROPN", "d'abord", False, False),
            "peut-être": ("reject", "not_a_schedule_request", "NOUN", "être", False, True),
            "rendez-vous": ("reject", "not_a_schedule_request", "ADJ", "vous", False, True),
            "qu'est-ce": ("reject", "informational_question_not_schedule_request", "NOUN", "ce", False, True),
            "c'est-à-dire": ("reject", "not_a_schedule_request", "PROPN", "c'est", False, True),
            "8h15": ("clarify", "temporal_signal_without_clear_request", "ADV", "8h15", False, True),
            "14h": ("clarify", "temporal_signal_without_clear_request", "NOUN", "14h", False, True),
            "23h59": ("clarify", "temporal_signal_without_clear_request", "NUM", "23h59", False, True),
            "0h": ("clarify", "temporal_signal_without_clear_request", "PROPN", "0h", False, True),
            "24h": ("clarify", "temporal_signal_without_clear_request", "NOUN", "24h", False, True),
            "1,000,000": ("reject", "not_a_schedule_request", "NUM", "1,000,000", False, False),
            "1.000.000": ("reject", "not_a_schedule_request", "NUM", "1.000.000", False, False),
            "007": ("reject", "not_a_schedule_request", "NUM", "007", False, False),
            "1e+10": ("reject", "not_a_schedule_request", "NUM", "1e+10", False, False),
            "6.022e23": ("reject", "not_a_schedule_request", "X", "6.022e23", False, False),
            "2^10": ("reject", "not_a_schedule_request", "NUM", "2", False, False),
            "-273.15": ("reject", "not_a_schedule_request", "NOUN", "-273.15", False, False),
            "5°F": ("reject", "not_a_schedule_request", "NOUN", "f", False, False),
            "1º": ("reject", "not_a_schedule_request", "NUM", "1º", False, False),
            "noon": ("reject", "not_a_schedule_request", "NOUN", "noon", False, False),
            "09:30:00": ("reject", "not_a_schedule_request", "NUM", "09:30:00", False, False),
            "❤️": ("reject", "not_a_schedule_request", "PUNCT", "❤", False, False),
            "🏳️‍🌈": ("reject", "not_a_schedule_request", "PROPN", "🌈", False, False),
            "‼️": ("reject", "not_a_schedule_request", "PUNCT", "‼️", False, False),
        }
        for text, (decision, reason, pos, lemma, subject, french) in expected.items():
            with self.subTest(text=text):
                self.assertEqual(french, text_looks_french(text))
                with patch("intent.requests.post") as post:
                    post.return_value.json.return_value = []
                    post.return_value.raise_for_status.return_value = None
                    result = classify(text)
                self.assertEqual("fr_FR" if french else "en_GB", post.call_args.kwargs["data"]["locale"])
                self.assertEqual(text, post.call_args.kwargs["data"]["text"])
                self.assertEqual((decision, reason), (result["decision"], result["reason"]))
                self.assertEqual(pos, result["features"]["root"]["pos"])
                self.assertEqual(lemma, result["features"]["root"]["lemma"])
                self.assertEqual(subject, result["features"]["has_subject"])
                self.assertEqual(text, result["text"])


if __name__ == "__main__":
    unittest.main()
