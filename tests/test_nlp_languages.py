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

    def test_fractions_elisions_clocks_and_marks(self):
        # Slash fractions are like_num when the tokenizer keeps them whole.
        # de/id/ms split on "/", and Ancient Greek, Latin, and Yoruba keep the
        # token but do not treat it as a number. A fraction slash (U+2044) is
        # a different character and is never like_num.
        not_like_slash = {"grc", "la", "yo"}
        slash_split = {"de", "id", "ms"}
        not_like_signed = {"fa", "grc", "la", "lb", "lt", "nl", "pl", "si", "ta", "te", "tl", "uk", "yo"}
        trailing_dot_glued = {"am", "ar", "bn", "da", "de", "el", "fa", "hu", "nb", "nn", "sl", "sr", "ti", "tr", "vi"}
        apostrophe_three = {"ky", "tt"}
        apostrophe_elision = {"ca", "fr", "it", "lij"}
        possessive_glue = {"am", "ar", "bn", "da", "de", "el", "es", "fa", "fi", "grc", "hu", "nb", "nl", "nn", "pl", "ro", "sl", "sr", "sv", "ti", "vi"}
        curly_three = {"da", "de", "fi", "hu", "ky", "nb", "nl", "nn", "pl", "sv", "tt"}
        hyphen_glue = {"da", "el", "es", "fi", "hu", "lt", "nb", "nl", "nn", "pt", "ro", "sv", "vi"}
        pre_meeting_glue = {"ca", "da", "de", "el", "es", "fi", "hu", "ky", "lb", "nb", "nl", "nn", "pt", "ro", "sr", "sv", "tt", "vi"}
        a_stop = {"ca", "cs", "de", "dsb", "en", "es", "fr", "ga", "hr", "hsb", "hu", "it", "lb", "lij", "lt", "pl", "pt", "ro", "sk", "sl", "sq", "yo"}
        script_zeros = ("૦", "੦", "௦", "౦", "೦", "൦", "၀", "໐", "༠", "٠", "۰", "０")
        extra_currency = ("₲", "₵", "₭", "₮", "₳", "₣", "₤")
        extra_punct = ("§", "¶", "¡", "‼", "⁇", "⁉", "‥", "〜", "・")

        for code, pipeline in self.pipelines.items():
            if pipeline is None:
                continue
            with self.subTest(code=code):
                half = self.round_trip(pipeline, "1/2")
                week = self.round_trip(pipeline, "24/7")
                pi_approx = self.round_trip(pipeline, "22/7")
                if code == "zh":
                    self.assertEqual(["1", "/", "2"], [token.text for token in half])
                    self.assertTrue(all(token.like_num for token in (half[0], half[2])))
                    self.assertTrue(half[1].is_punct and half[0].is_stop)
                    self.assertEqual(["2", "4", "/", "7"], [token.text for token in week])
                    self.assertEqual(["2", "2", "/", "7"], [token.text for token in pi_approx])
                elif code in slash_split:
                    self.assertEqual(["1", "/", "2"], [token.text for token in half])
                    self.assertEqual(["24", "/", "7"], [token.text for token in week])
                    self.assertEqual(["22", "/", "7"], [token.text for token in pi_approx])
                    self.assertTrue(all(token.like_num for token in (half[0], half[2], week[0], week[2])))
                    self.assertTrue(half[1].is_punct)
                elif code in not_like_slash:
                    self.assertEqual(["1/2"], [token.text for token in half])
                    self.assertEqual(["24/7"], [token.text for token in week])
                    self.assertEqual(["22/7"], [token.text for token in pi_approx])
                    self.assertFalse(any(token.like_num for token in half))
                else:
                    self.assert_token_pattern(pipeline, "1/2", ["1/2"], like_num=True)
                    self.assert_token_pattern(pipeline, "24/7", ["24/7"], like_num=True)
                    self.assert_token_pattern(pipeline, "22/7", ["22/7"], like_num=True)

                stacked = self.round_trip(pipeline, "1/2/3")
                if code == "zh":
                    self.assertEqual(["1", "/", "2", "/", "3"], [token.text for token in stacked])
                elif code in slash_split:
                    self.assertEqual(["1", "/", "2", "/", "3"], [token.text for token in stacked])
                    self.assertFalse(stacked[0].is_stop)
                elif code == "fa":
                    self.assert_token_pattern(pipeline, "1/2/3", ["1/2/3"], like_num=True)
                else:
                    self.assert_token_pattern(pipeline, "1/2/3", ["1/2/3"], like_num=False)

                fraction_slash = self.round_trip(pipeline, "1\u20442")
                if code == "zh":
                    self.assertEqual(["1", "\u2044", "2"], [token.text for token in fraction_slash])
                    self.assertFalse(fraction_slash[1].is_punct)
                    self.assertFalse(fraction_slash[1].like_num)
                else:
                    self.assert_token_pattern(pipeline, "1\u20442", ["1\u20442"], like_num=False)

                plus_minus = self.round_trip(pipeline, "±5")
                tilde = self.round_trip(pipeline, "~5")
                if code == "zh":
                    self.assertEqual(["±", "5"], [token.text for token in plus_minus])
                    self.assertFalse(plus_minus[0].is_punct)
                    self.assertTrue(plus_minus[1].like_num and plus_minus[1].is_stop)
                    self.assertEqual(["~", "5"], [token.text for token in tilde])
                    self.assertTrue(tilde[0].is_stop)
                    self.assertFalse(tilde[0].is_punct)
                elif code == "sl":
                    self.assert_token_pattern(pipeline, "±5", ["±5"], like_num=True)
                    self.assertEqual(["~", "5"], [token.text for token in tilde])
                    self.assertFalse(tilde[0].is_punct)
                    self.assertTrue(tilde[1].like_num)
                elif code in not_like_signed:
                    self.assert_token_pattern(pipeline, "±5", ["±5"], like_num=False)
                    self.assert_token_pattern(pipeline, "~5", ["~5"], like_num=False)
                else:
                    self.assert_token_pattern(pipeline, "±5", ["±5"], like_num=True)
                    self.assert_token_pattern(pipeline, "~5", ["~5"], like_num=True)

                self.assertEqual("±", self.round_trip(pipeline, "±")[0].text)
                self.assertFalse(self.round_trip(pipeline, "±")[0].is_punct)
                self.assertFalse(self.round_trip(pipeline, "±")[0].is_currency)

                leading_dot = self.round_trip(pipeline, ".5")
                trailing_dot = self.round_trip(pipeline, "5.")
                if code == "zh":
                    self.assertEqual([".", "5"], [token.text for token in leading_dot])
                    self.assertTrue(leading_dot[0].is_punct and leading_dot[0].is_stop)
                    self.assertEqual(["5", "."], [token.text for token in trailing_dot])
                    self.assertTrue(trailing_dot[1].is_stop)
                elif code in {"grc", "la"}:
                    self.assert_token_pattern(pipeline, ".5", [".5"], like_num=False)
                    if code == "grc":
                        self.assert_token_pattern(pipeline, "5.", ["5."], like_num=False)
                    else:
                        self.assertEqual(["5", "."], [token.text for token in trailing_dot])
                        self.assertTrue(trailing_dot[0].like_num and trailing_dot[1].is_punct)
                elif code in trailing_dot_glued:
                    self.assert_token_pattern(pipeline, ".5", [".5"], like_num=True)
                    self.assert_token_pattern(pipeline, "5.", ["5."], like_num=True)
                else:
                    self.assert_token_pattern(pipeline, ".5", [".5"], like_num=True)
                    self.assertEqual(["5", "."], [token.text for token in trailing_dot])
                    self.assertTrue(trailing_dot[0].like_num)

                leading_comma = self.round_trip(pipeline, ",5")
                trailing_comma = self.round_trip(pipeline, "5,")
                if code == "vi":
                    self.assert_token_pattern(pipeline, ",5", [",5"], like_num=True)
                    self.assert_token_pattern(pipeline, "5,", ["5,"], like_num=True)
                elif code == "grc":
                    self.assertEqual([",", "5"], [token.text for token in leading_comma])
                    self.assertFalse(leading_comma[1].like_num)
                    self.assertTrue(leading_comma[1].is_digit)
                    self.assertEqual(["5", ","], [token.text for token in trailing_comma])
                    self.assertFalse(trailing_comma[0].like_num)
                elif code == "zh":
                    self.assertEqual([",", "5"], [token.text for token in leading_comma])
                    self.assertTrue(leading_comma[0].is_stop and leading_comma[1].is_stop)
                else:
                    self.assertEqual([",", "5"], [token.text for token in leading_comma])
                    self.assertTrue(leading_comma[1].like_num and leading_comma[0].is_punct)
                    self.assertEqual(["5", ","], [token.text for token in trailing_comma])
                    self.assertTrue(trailing_comma[0].like_num)

                euro_group = self.round_trip(pipeline, "1.234,56")
                us_group = self.round_trip(pipeline, "1,234.56")
                if code == "zh":
                    self.assertEqual(list("1.234,56"), [token.text for token in euro_group])
                    self.assertEqual(list("1,234.56"), [token.text for token in us_group])
                elif code == "pl":
                    self.assertEqual(["1", ".", "234,56"], [token.text for token in euro_group])
                    self.assertTrue(euro_group[0].like_num and euro_group[2].like_num)
                    self.assertEqual(["1,234", ".", "56"], [token.text for token in us_group])
                    self.assertTrue(us_group[0].like_num and us_group[2].like_num)
                elif code in {"grc", "la", "ne"}:
                    self.assert_token_pattern(pipeline, "1.234,56", ["1.234,56"], like_num=False)
                    self.assert_token_pattern(pipeline, "1,234.56", ["1,234.56"], like_num=False)
                else:
                    self.assert_token_pattern(pipeline, "1.234,56", ["1.234,56"], like_num=True)
                    self.assert_token_pattern(pipeline, "1,234.56", ["1,234.56"], like_num=True)

                # An ASCII space between digit groups is ordinary whitespace.
                # A narrow no-break space is its own space token.
                spaced = self.round_trip(pipeline, "1 000")
                narrow = self.round_trip(pipeline, "1\u202f000")
                if code == "zh":
                    self.assertEqual(["1", "0", "0", "0"], [token.text for token in spaced])
                    self.assertEqual(" ", spaced[0].whitespace_)
                    self.assertFalse(any(token.is_space for token in spaced))
                    self.assertEqual(["1", "\u202f", "0", "0", "0"], [token.text for token in narrow])
                    self.assertTrue(narrow[1].is_space)
                elif code == "grc":
                    self.assertEqual(["1", "000"], [token.text for token in spaced])
                    self.assertEqual(" ", spaced[0].whitespace_)
                    self.assertFalse(spaced[0].like_num or spaced[1].like_num)
                    self.assertEqual(["1", "\u202f", "000"], [token.text for token in narrow])
                    self.assertTrue(narrow[1].is_space)
                    self.assertFalse(narrow[0].like_num)
                else:
                    self.assertEqual(["1", "000"], [token.text for token in spaced])
                    self.assertEqual(" ", spaced[0].whitespace_)
                    self.assertTrue(spaced[0].like_num and spaced[1].like_num)
                    self.assertEqual(["1", "\u202f", "000"], [token.text for token in narrow])
                    self.assertTrue(narrow[1].is_space)
                    self.assertTrue(narrow[0].like_num and narrow[2].like_num)
                if code == "zh":
                    self.assertEqual(["1", "_", "0", "0", "0"], [token.text for token in self.round_trip(pipeline, "1_000")])
                else:
                    self.assert_token_pattern(pipeline, "1_000", ["1_000"], like_num=False)

                per_mille = self.round_trip(pipeline, "5\u2030")
                if code == "zh":
                    self.assertEqual(["5", "\u2030"], [token.text for token in per_mille])
                    self.assertTrue(per_mille[1].is_punct)
                else:
                    self.assert_token_pattern(pipeline, "5\u2030", ["5\u2030"], like_num=False)

                for digit in script_zeros:
                    token = self.round_trip(pipeline, digit)[0]
                    self.assertTrue(token.is_digit)
                    self.assertEqual(code != "grc", token.like_num)
                    self.assertFalse(token.is_alpha)
                    self.assertEqual(code == "zh" and digit == "０", token.is_stop)

                for symbol in extra_currency:
                    token = self.round_trip(pipeline, symbol)[0]
                    self.assertTrue(token.is_currency)
                    self.assertFalse(token.is_punct)
                for symbol in extra_punct:
                    token = self.round_trip(pipeline, symbol)[0]
                    self.assertTrue(token.is_punct)
                    self.assertFalse(token.is_stop)
                    self.assertFalse(token.is_quote)

                micro = self.round_trip(pipeline, "µ")[0]
                self.assertTrue(micro.is_alpha and micro.is_lower)
                self.assertFalse(micro.like_num)
                pi = self.round_trip(pipeline, "π")[0]
                self.assertTrue(pi.is_alpha)
                self.assertEqual(code == "grc", pi.like_num)
                omega = self.round_trip(pipeline, "Ω")[0]
                self.assertTrue(omega.is_alpha)
                self.assertEqual(code == "el", omega.is_stop)
                self.assertFalse(omega.like_num)
                iteration = self.round_trip(pipeline, "々")[0]
                self.assertTrue(iteration.is_alpha)
                self.assertFalse(iteration.is_punct)
                celsius = self.round_trip(pipeline, "℃")[0]
                self.assertFalse(celsius.is_punct)
                self.assertEqual(code == "zh", celsius.is_stop)
                self.assertFalse(self.round_trip(pipeline, "℉")[0].is_punct)

                self._assert_contraction(
                    pipeline, code, "isn't",
                    en=["is", "n't"],
                    catalan=["isn", "'t"],
                    elided=["isn'", "t"],
                )
                didnt = self.round_trip(pipeline, "didn't")
                if code == "yo":
                    self.assertEqual(["didn't"], [token.text for token in didnt])
                    self.assertTrue(didnt[0].like_num)
                else:
                    self._assert_contraction(
                        pipeline, code, "didn't",
                        en=["did", "n't"],
                        catalan=["didn", "'t"],
                        elided=["didn'", "t"],
                    )
                self._assert_contraction(
                    pipeline, code, "could've",
                    en=["could", "'ve"],
                    catalan=["could'", "ve"],
                    elided=["could'", "ve"],
                )
                self._assert_contraction(
                    pipeline, code, "I've",
                    en=["I", "'ve"],
                    catalan=["I'", "ve"],
                    elided=["I'", "ve"],
                )
                self._assert_contraction(
                    pipeline, code, "he'd",
                    en=["he", "'d"],
                    catalan=["he'", "d"],
                    elided=["he'", "d"],
                )
                self._assert_contraction(
                    pipeline, code, "I'd",
                    en=["I", "'d"],
                    catalan=["I'", "d"],
                    elided=["I'", "d"],
                )
                for text in ("that's", "there's"):
                    doc = self.round_trip(pipeline, text)
                    stem = text.split("'")[0]
                    if code == "zh":
                        self.assertGreater(len(doc), 2)
                    elif code == "en":
                        self.assertEqual([stem, "'s"], [token.text for token in doc])
                        self.assertTrue(all(token.is_stop for token in doc))
                    elif code in {"ca", "fr"}:
                        self.assertEqual([stem + "'", "s"], [token.text for token in doc])
                        self.assertFalse(any(token.is_stop for token in doc))
                    elif code in possessive_glue:
                        self.assertEqual([text], [token.text for token in doc])
                        self.assertFalse(doc[0].is_stop)
                    else:
                        self.assertEqual([stem, "'s"], [token.text for token in doc])
                        self.assertFalse(any(token.is_stop for token in doc))

                gotta = self.round_trip(pipeline, "gotta")
                if code == "zh":
                    self.assertEqual(list("gotta"), [token.text for token in gotta])
                elif code == "en":
                    self.assertEqual(["got", "ta"], [token.text for token in gotta])
                    self.assertFalse(any(token.is_stop for token in gotta))
                else:
                    self.assertEqual(["gotta"], [token.text for token in gotta])
                    self.assertTrue(gotta[0].is_alpha)

                self._assert_french_elision(pipeline, code, "s'il", {
                    "fr": (["s'", "il"], True, True),
                    "it": (["s'", "il"], True, True),
                    "lij": (["s'", "il"], True, False),
                    "ca": (["s'", "il"], False, False),
                })
                une = self.round_trip(pipeline, "d'une")
                if code == "zh":
                    self.assertEqual(list("d'une"), [token.text for token in une])
                elif code in apostrophe_three:
                    self.assertEqual(["d", "'", "une"], [token.text for token in une])
                elif code == "fr":
                    self.assertEqual(["d'", "une"], [token.text for token in une])
                    self.assertTrue(une[0].is_stop and une[1].is_stop and une[1].like_num)
                elif code in {"it", "lb", "lij"}:
                    self.assertEqual(["d'", "une"], [token.text for token in une])
                    self.assertTrue(une[0].is_stop)
                    self.assertFalse(une[1].is_stop or une[1].like_num)
                elif code == "ca":
                    self.assertEqual(["d'", "une"], [token.text for token in une])
                    self.assertFalse(une[0].is_stop or une[1].is_stop)
                else:
                    self.assertEqual(["d'une"], [token.text for token in une])
                elle = self.round_trip(pipeline, "qu'elle")
                if code == "yo":
                    self.assertEqual(["qu'elle"], [token.text for token in elle])
                    self.assertTrue(elle[0].like_num)
                else:
                    self._assert_french_elision(pipeline, code, "qu'elle", {
                        "fr": (["qu'", "elle"], True, True),
                        "ca": (["qu'", "elle"], False, False),
                        "it": (["qu'", "elle"], False, False),
                        "lij": (["qu'", "elle"], False, False),
                    })
                self._assert_french_elision(pipeline, code, "n'a", {
                    "fr": (["n'", "a"], True, True),
                    "lij": (["n'", "a"], True, True),
                    "ca": (["n'", "a"], False, True),
                    "it": (["n'", "a"], False, True),
                })
                self._assert_french_elision(pipeline, code, "l'équipe", {
                    "fr": (["l'", "équipe"], True, False),
                    "it": (["l'", "équipe"], True, False),
                    "lij": (["l'", "équipe"], True, False),
                    "ca": (["l'", "équipe"], False, False),
                })

                curly_sil = self.round_trip(pipeline, "s\u2019il")
                if code == "zh":
                    self.assertEqual(["s", "\u2019", "i", "l"], [token.text for token in curly_sil])
                elif code == "fr":
                    self.assertEqual(["s\u2019", "il"], [token.text for token in curly_sil])
                    self.assertTrue(curly_sil[0].is_stop and curly_sil[1].is_stop)
                elif code == "it":
                    self.assertEqual(["s\u2019", "il"], [token.text for token in curly_sil])
                    self.assertTrue(curly_sil[1].is_stop)
                    self.assertFalse(curly_sil[0].is_stop)
                elif code == "hu":
                    self.assertEqual(["s", "\u2019", "il"], [token.text for token in curly_sil])
                    self.assertTrue(curly_sil[0].is_stop and curly_sil[1].is_quote)
                elif code in (curly_three - {"hu", "de"}):
                    self.assertEqual(["s", "\u2019", "il"], [token.text for token in curly_sil])
                    self.assertTrue(curly_sil[1].is_quote)
                    self.assertFalse(any(token.is_stop for token in curly_sil))
                elif code in {"ca", "de", "lij"}:
                    self.assertEqual(["s\u2019", "il"], [token.text for token in curly_sil])
                    self.assertFalse(any(token.is_stop for token in curly_sil))
                else:
                    self.assertEqual(["s\u2019il"], [token.text for token in curly_sil])
                curly_equipe = self.round_trip(pipeline, "l\u2019équipe")
                if code == "zh":
                    self.assertGreater(len(curly_equipe), 3)
                elif code == "fr":
                    self.assertEqual(["l\u2019", "équipe"], [token.text for token in curly_equipe])
                    self.assertTrue(curly_equipe[0].is_stop)
                elif code in {"ca", "it", "lij"}:
                    self.assertEqual(["l\u2019", "équipe"], [token.text for token in curly_equipe])
                    self.assertFalse(curly_equipe[0].is_stop)
                elif code in curly_three:
                    self.assertEqual(["l", "\u2019", "équipe"], [token.text for token in curly_equipe])
                    self.assertTrue(curly_equipe[1].is_quote)
                else:
                    self.assertEqual(["l\u2019équipe"], [token.text for token in curly_equipe])
                curly_cest = self.round_trip(pipeline, "c\u2019est")
                if code == "zh":
                    self.assertEqual(["c", "\u2019", "e", "s", "t"], [token.text for token in curly_cest])
                elif code == "fr":
                    self.assertEqual(["c\u2019", "est"], [token.text for token in curly_cest])
                    self.assertTrue(all(token.is_stop for token in curly_cest))
                elif code in {"ca", "it", "lij"}:
                    self.assertEqual(["c\u2019", "est"], [token.text for token in curly_cest])
                    self.assertFalse(any(token.is_stop for token in curly_cest))
                elif code in curly_three:
                    self.assertEqual(["c", "\u2019", "est"], [token.text for token in curly_cest])
                else:
                    self.assertEqual(["c\u2019est"], [token.text for token in curly_cest])
                curly_today = self.round_trip(pipeline, "aujourd\u2019hui")
                if code == "zh":
                    self.assertGreater(len(curly_today), 3)
                elif code in {"ca", "it", "lij"}:
                    self.assertEqual(["aujourd\u2019", "hui"], [token.text for token in curly_today])
                elif code in curly_three:
                    self.assertEqual(["aujourd", "\u2019", "hui"], [token.text for token in curly_today])
                    self.assertTrue(curly_today[1].is_quote)
                else:
                    self.assertEqual(["aujourd\u2019hui"], [token.text for token in curly_today])

                self._assert_final_dot(pipeline, code, "Prof.", {"am", "ar", "da", "de", "en", "es", "fa", "grc", "hu", "id", "ms", "nl", "ro", "sv", "ti", "tr", "vi"})
                self._assert_final_dot(pipeline, code, "Dept.", {"am", "ar", "fa", "grc", "ti", "vi"})
                self._assert_final_dot(pipeline, code, "Fig.", {"am", "ar", "da", "fa", "grc", "nl", "ro", "sv", "ti", "vi"})
                ca_abbr = self.round_trip(pipeline, "ca.")
                if code == "zh":
                    self.assertEqual(list("ca."), [token.text for token in ca_abbr])
                elif code in {"am", "ar", "da", "de", "fa", "grc", "hu", "nb", "nl", "nn", "sl", "ti", "vi"}:
                    self.assertEqual(["ca."], [token.text for token in ca_abbr])
                    self.assertFalse(ca_abbr[0].is_stop)
                elif code in {"en", "ro", "sq"}:
                    self.assertEqual(["ca", "."], [token.text for token in ca_abbr])
                    self.assertTrue(ca_abbr[0].is_stop and ca_abbr[1].is_punct)
                else:
                    self.assertEqual(["ca", "."], [token.text for token in ca_abbr])
                    self.assertFalse(ca_abbr[0].is_stop)

                for text, special in (
                    ("U.K.", None),
                    ("U.N.", "fr"),
                    ("B.A.", "vi"),
                ):
                    doc = self.round_trip(pipeline, text)
                    if code == "zh":
                        self.assertEqual(list(text), [token.text for token in doc])
                    elif code == "pl":
                        self.assertEqual([text[0], ".", text[2], "."], [token.text for token in doc])
                        stopped = doc[0] if text != "B.A." else doc[2]
                        self.assertTrue(stopped.is_stop)
                    elif code == "sl":
                        self.assertEqual([text[:2], text[2:]], [token.text for token in doc])
                    elif code in {"lt", "sr"}:
                        self.assertEqual([text[:-1], "."], [token.text for token in doc])
                    elif special is not None and code == special:
                        self.assertEqual([text], [token.text for token in doc])
                        self.assertTrue(doc[0].like_num)
                    else:
                        self.assertEqual([text], [token.text for token in doc])
                        self.assertFalse(doc[0].like_num)

                ie = self.round_trip(pipeline, "i.e.,")
                eg = self.round_trip(pipeline, "e.g.,")
                if code == "zh":
                    self.assertEqual(list("i.e.,"), [token.text for token in ie])
                    self.assertEqual(list("e.g.,"), [token.text for token in eg])
                elif code == "vi":
                    self.assertEqual(["i.e.,"], [token.text for token in ie])
                    self.assertEqual(["e.g.,"], [token.text for token in eg])
                else:
                    if code in {"am", "ar", "da", "de", "en", "fa", "grc", "hu", "lb", "ms", "nl", "pt", "ti"}:
                        self.assertEqual(["i.e.", ","], [token.text for token in ie])
                    else:
                        self.assertEqual(["i.e", ".", ","], [token.text for token in ie])
                    if code in {"am", "ar", "de", "en", "fa", "grc", "nl", "pt", "ti"}:
                        self.assertEqual(["e.g.", ","], [token.text for token in eg])
                    else:
                        self.assertEqual(["e.g", ".", ","], [token.text for token in eg])
                    self.assertTrue(ie[-1].is_punct and eg[-1].is_punct)

                na = self.round_trip(pipeline, "n/a")
                upper_na = self.round_trip(pipeline, "N/A")
                without = self.round_trip(pipeline, "w/o")
                if code == "zh":
                    self.assertEqual(["n", "/", "a"], [token.text for token in na])
                    self.assertTrue(na[1].is_stop)
                    self.assertEqual(["N", "/", "A"], [token.text for token in upper_na])
                    self.assertEqual(["w", "/", "o"], [token.text for token in without])
                elif code in {"hu", "lb", "nl", "ro", "vi"}:
                    self.assertEqual(["n/a"], [token.text for token in na])
                    self.assertEqual(["N/A"], [token.text for token in upper_na])
                elif code == "ms":
                    self.assertEqual(["n", "/", "a"], [token.text for token in na])
                    self.assertFalse(na[0].is_stop or na[2].is_stop)
                    self.assertEqual(["N/A"], [token.text for token in upper_na])
                elif code in {"sl", "yo"}:
                    self.assertEqual(["n", "/", "a"], [token.text for token in na])
                    self.assertTrue(na[0].is_stop and na[2].is_stop)
                    self.assertEqual(["N", "/", "A"], [token.text for token in upper_na])
                    self.assertTrue(upper_na[0].is_stop and upper_na[2].is_stop)
                elif code in a_stop:
                    self.assertEqual(["n", "/", "a"], [token.text for token in na])
                    self.assertTrue(na[2].is_stop)
                    self.assertFalse(na[0].is_stop)
                    self.assertEqual(["N", "/", "A"], [token.text for token in upper_na])
                    self.assertTrue(upper_na[2].is_stop)
                    self.assertFalse(upper_na[0].is_stop)
                else:
                    self.assertEqual(["n", "/", "a"], [token.text for token in na])
                    self.assertFalse(na[0].is_stop or na[2].is_stop)
                    self.assertEqual(["N", "/", "A"], [token.text for token in upper_na])
                    self.assertFalse(upper_na[0].is_stop or upper_na[2].is_stop)
                if code == "zh":
                    pass
                elif code in {"en", "hu", "lb", "nl", "ro", "vi"}:
                    self.assertEqual(["w/o"], [token.text for token in without])
                elif code in {"pl", "yo"}:
                    self.assertEqual(["w", "/", "o"], [token.text for token in without])
                    self.assertTrue(without[0].is_stop and without[2].is_stop)
                elif code in {"az", "ca", "cs", "es", "fr", "hr", "la", "lij", "lt", "pt", "sk", "sl", "tl", "tn", "tr"}:
                    self.assertEqual(["w", "/", "o"], [token.text for token in without])
                    self.assertTrue(without[2].is_stop)
                    self.assertFalse(without[0].is_stop)
                else:
                    self.assertEqual(["w", "/", "o"], [token.text for token in without])
                    self.assertFalse(without[0].is_stop or without[2].is_stop)

                covid = self.round_trip(pipeline, "COVID-19")
                if code == "zh":
                    self.assertEqual(list("COVID-19"), [token.text for token in covid])
                elif code in {"id", "ms", "pl"}:
                    self.assertEqual(["COVID", "-", "19"], [token.text for token in covid])
                    self.assertTrue(covid[1].is_punct and covid[2].like_num)
                else:
                    self.assertEqual(["COVID-19"], [token.text for token in covid])
                    self.assertFalse(covid[0].like_num)
                meeting = self.round_trip(pipeline, "pre-meeting")
                if code == "zh":
                    self.assertEqual(list("pre-meeting"), [token.text for token in meeting])
                elif code in pre_meeting_glue:
                    self.assertEqual(["pre-meeting"], [token.text for token in meeting])
                elif code == "sk":
                    self.assertEqual(["pre", "-", "meeting"], [token.text for token in meeting])
                    self.assertTrue(meeting[0].is_stop and meeting[1].is_punct)
                else:
                    self.assertEqual(["pre", "-", "meeting"], [token.text for token in meeting])
                    self.assertFalse(meeting[0].is_stop)
                shift = self.round_trip(pipeline, "9-5")
                if code == "zh":
                    self.assertEqual(["9", "-", "5"], [token.text for token in shift])
                    self.assertTrue(shift[1].is_stop)
                elif code in hyphen_glue:
                    self.assertEqual(["9-5"], [token.text for token in shift])
                    self.assertFalse(shift[0].like_num)
                elif code == "grc":
                    self.assertEqual(["9", "-", "5"], [token.text for token in shift])
                    self.assertFalse(shift[0].like_num or shift[2].like_num)
                else:
                    self.assertEqual(["9", "-", "5"], [token.text for token in shift])
                    self.assertTrue(shift[0].like_num and shift[2].like_num and shift[1].is_punct)
                ampm = self.round_trip(pipeline, "9am-5pm")
                if code == "zh":
                    self.assertEqual(list("9am-5pm"), [token.text for token in ampm])
                elif code in {"id", "ms", "pl"}:
                    self.assertEqual(["9am", "-", "5pm"], [token.text for token in ampm])
                else:
                    self.assertEqual(["9am-5pm"], [token.text for token in ampm])

                clock = self.round_trip(pipeline, "9 h 30")
                if code == "zh":
                    self.assertEqual(["9", "h", "3", "0"], [token.text for token in clock])
                elif code == "grc":
                    self.assertEqual(["9", "h", "30"], [token.text for token in clock])
                    self.assertFalse(clock[0].like_num or clock[2].like_num)
                    self.assertEqual(" ", clock[0].whitespace_)
                elif code in {"ro", "sl"}:
                    self.assertEqual(["9", "h", "30"], [token.text for token in clock])
                    self.assertTrue(clock[1].is_stop)
                    self.assertTrue(clock[0].like_num and clock[2].like_num)
                else:
                    self.assertEqual(["9", "h", "30"], [token.text for token in clock])
                    self.assertFalse(clock[1].is_stop)
                    self.assertTrue(clock[0].like_num and clock[2].like_num)
                for text in ("18h00", "9H30", "0900h", "９：００"):
                    doc = self.round_trip(pipeline, text)
                    if code == "zh":
                        self.assertGreater(len(doc), 1)
                    else:
                        self.assertEqual([text], [token.text for token in doc])
                        self.assertFalse(doc[0].like_num)

                morning = self.round_trip(pipeline, "9:00 a.m.")
                if code == "zh":
                    self.assertGreater(len(morning), 4)
                elif code in {"am", "ar", "en", "fa", "grc", "nb", "nl", "nn", "ro", "ti", "vi"}:
                    self.assertEqual(["9:00", "a.m."], [token.text for token in morning])
                else:
                    self.assertEqual(["9:00", "a.m", "."], [token.text for token in morning])
                    self.assertTrue(morning[2].is_punct)

                stamp = self.round_trip(pipeline, "2026-10-04T09:00:00Z")
                if code == "zh":
                    self.assertIn("T", [token.text for token in stamp])
                    self.assertGreater(len(stamp), 5)
                elif code in hyphen_glue - {"tr"}:
                    self.assertEqual(["2026-10-04T09:00:00Z"], [token.text for token in stamp])
                elif code == "grc":
                    self.assertEqual(["2026", "-", "10", "-", "04T09:00:00Z"], [token.text for token in stamp])
                    self.assertFalse(stamp[0].like_num)
                else:
                    self.assertEqual(["2026", "-", "10", "-", "04T09:00:00Z"], [token.text for token in stamp])
                    self.assertTrue(stamp[0].like_num and stamp[2].like_num)
                    self.assertFalse(stamp[4].like_num)
                day_first = self.round_trip(pipeline, "30/09/2026")
                if code == "zh":
                    self.assertEqual(list("30/09/2026"), [token.text for token in day_first])
                elif code in slash_split:
                    self.assertEqual(["30", "/", "09", "/", "2026"], [token.text for token in day_first])
                    self.assertTrue(all(token.like_num for token in day_first if token.text != "/"))
                elif code == "fa":
                    self.assert_token_pattern(pipeline, "30/09/2026", ["30/09/2026"], like_num=True)
                else:
                    self.assert_token_pattern(pipeline, "30/09/2026", ["30/09/2026"], like_num=False)

                for text in ("\u201chello\u201d", "\u2018hello\u2019"):
                    doc = self.round_trip(pipeline, text)
                    if code == "vi":
                        self.assertEqual([text], [token.text for token in doc])
                        self.assertFalse(doc[0].is_quote)
                    elif code == "zh":
                        self.assertEqual([text[0], "h", "e", "l", "l", "o", text[-1]], [token.text for token in doc])
                        self.assertTrue(doc[0].is_left_punct and doc[0].is_quote and doc[0].is_stop)
                        self.assertTrue(doc[-1].is_right_punct and doc[-1].is_quote and doc[-1].is_stop)
                    else:
                        self.assertEqual([text[0], "hello", text[-1]], [token.text for token in doc])
                        self.assertTrue(doc[0].is_left_punct and doc[0].is_quote)
                        self.assertFalse(doc[0].is_right_punct or doc[0].is_stop)
                        self.assertTrue(doc[2].is_right_punct and doc[2].is_quote)
                        self.assertFalse(doc[2].is_left_punct)
                        self.assertTrue(doc[1].is_alpha)
                corner = self.round_trip(pipeline, "\u300chello\u300d")
                if code == "vi":
                    self.assertEqual(["\u300chello\u300d"], [token.text for token in corner])
                elif code == "zh":
                    self.assertEqual(["\u300c", "h", "e", "l", "l", "o", "\u300d"], [token.text for token in corner])
                    self.assertTrue(corner[0].is_punct and not corner[0].is_quote)
                    self.assertTrue(corner[-1].is_punct and corner[-1].is_stop)
                    self.assertFalse(corner[-1].is_quote)
                else:
                    self.assertEqual(["\u300c", "hello", "\u300d"], [token.text for token in corner])
                    self.assertTrue(corner[0].is_punct and corner[2].is_punct)
                    self.assertFalse(corner[0].is_quote or corner[2].is_quote)
                lenticular = self.round_trip(pipeline, "\u300ea\u300f")
                if code == "vi":
                    self.assertEqual(["\u300ea\u300f"], [token.text for token in lenticular])
                elif code == "zh":
                    self.assertEqual(["\u300e", "a", "\u300f"], [token.text for token in lenticular])
                    self.assertTrue(lenticular[0].is_stop and lenticular[2].is_stop)
                    self.assertFalse(lenticular[1].is_stop)
                else:
                    self.assertEqual(["\u300e", "a", "\u300f"], [token.text for token in lenticular])
                    self.assertEqual(code in a_stop, lenticular[1].is_stop)
                    self.assertFalse(lenticular[0].is_quote)

                tilde_letter = self.round_trip(pipeline, "a\u0303")
                if code == "zh":
                    self.assertEqual(["a", "\u0303"], [token.text for token in tilde_letter])
                    self.assertTrue(tilde_letter[0].is_alpha)
                    self.assertFalse(tilde_letter[1].is_alpha)
                else:
                    self.assertEqual(["a\u0303"], [token.text for token in tilde_letter])
                    self.assertFalse(tilde_letter[0].is_alpha)
                    self.assertTrue(tilde_letter[0].is_lower)

    def _assert_contraction(self, pipeline, code, text, en, catalan, elided):
        doc = self.round_trip(pipeline, text)
        if code == "zh":
            self.assertGreater(len(doc), 2)
            self.assertTrue(any(token.is_quote for token in doc))
        elif code == "en":
            self.assertEqual(en, [token.text for token in doc])
            self.assertTrue(all(token.is_stop for token in doc))
        elif code == "ca":
            self.assertEqual(catalan, [token.text for token in doc])
            self.assertFalse(any(token.is_stop for token in doc))
        elif code in {"fr", "it", "lij"}:
            self.assertEqual(elided, [token.text for token in doc])
            self.assertFalse(any(token.is_stop for token in doc))
        elif code in {"ky", "tt"}:
            self.assertEqual(3, len(doc))
            self.assertTrue(doc[1].is_quote and doc[1].is_punct)
        else:
            self.assertEqual([text], [token.text for token in doc])
            self.assertFalse(doc[0].like_num)

    def _assert_french_elision(self, pipeline, code, text, special):
        doc = self.round_trip(pipeline, text)
        if code == "zh":
            self.assertGreater(len(doc), 2)
        elif code in {"ky", "tt"}:
            self.assertEqual(3, len(doc))
            self.assertTrue(doc[1].is_quote)
        elif code in special:
            texts, left_stop, right_stop = special[code]
            self.assertEqual(texts, [token.text for token in doc])
            self.assertEqual(left_stop, doc[0].is_stop)
            self.assertEqual(right_stop, doc[1].is_stop)
        else:
            self.assertEqual([text], [token.text for token in doc])

    def _assert_final_dot(self, pipeline, code, text, kept):
        doc = self.round_trip(pipeline, text)
        stem = text[:-1]
        if code == "zh":
            self.assertEqual(list(text), [token.text for token in doc])
        elif code in kept:
            self.assertEqual([text], [token.text for token in doc])
        else:
            self.assertEqual([stem, "."], [token.text for token in doc])
            self.assertTrue(doc[0].is_alpha and doc[1].is_punct)
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

    def test_fraction_clock_and_elision_decisions(self):
        # The English model tags a few symbols and contractions as subject-less
        # verbs. French clocks and "aujourd'hui" are temporal with no entity.
        # Straight "d'une" / "qu'elle" are not French markers.
        clarify = {
            "¡": ("VERB", "¡", False),
            "‼": ("VERB", "‼", False),
            "々": ("VERB", "々", False),
            "didn't": ("VERB", "do", False),
            "gotta": ("VERB", "get", False),
            "n'a": ("VERB", "n'a", False),
            "９：００": ("VERB", "９：００", False),
        }
        for text, (pos, lemma, subject) in clarify.items():
            with self.subTest(text=text):
                result = self.classify_without_time(text)
                self.assertFalse(text_looks_french(text))
                self.assertEqual(
                    ("clarify", "request_without_temporal_signal"),
                    (result["decision"], result["reason"]),
                )
                self.assertEqual(pos, result["features"]["root"]["pos"])
                self.assertEqual(lemma, result["features"]["root"]["lemma"])
                self.assertEqual(subject, result["features"]["has_subject"])
                self.assertTrue(result["features"]["looks_like_request"])

        french_temporal = {
            "18h00": ("NUM", "18h00"),
            "9H30": ("NOUN", "9h30"),
            "9 h 30": ("NOUN", "h"),
            "aujourd\u2019hui": ("NOUN", "aujourd\u2019hui"),
        }
        for text, (pos, lemma) in french_temporal.items():
            with self.subTest(text=text):
                self.assertTrue(text_looks_french(text))
                result = self.classify_without_time(text)
                self.assertEqual(
                    ("clarify", "temporal_signal_without_clear_request"),
                    (result["decision"], result["reason"]),
                )
                self.assertTrue(result["features"]["has_temporal_signal"])
                self.assertFalse(result["features"]["looks_like_request"])
                self.assertEqual(pos, result["features"]["root"]["pos"])
                self.assertEqual(lemma, result["features"]["root"]["lemma"])

        french_reject = {
            "s'il": ("VERB", "s'il"),
            "l'équipe": ("NOUN", "l'équipe"),
            "s\u2019il": ("NOUN", "s\u2019il"),
            "l\u2019équipe": ("PROPN", "l\u2019équipe"),
            "c\u2019est": ("PROPN", "c\u2019est"),
        }
        for text, (pos, lemma) in french_reject.items():
            with self.subTest(text=text):
                self.assertTrue(text_looks_french(text))
                result = self.classify_without_time(text)
                self.assertEqual(
                    ("reject", "not_a_schedule_request"),
                    (result["decision"], result["reason"]),
                )
                self.assertEqual(pos, result["features"]["root"]["pos"])
                self.assertEqual(lemma, result["features"]["root"]["lemma"])

        rejected = {
            "1/2": ("NUM", "1/2"),
            "1\u20442": ("NUM", "1\u20442"),
            "±5": ("ADJ", "±5"),
            "~5": ("NOUN", "~5"),
            ".5": ("PUNCT", ".5"),
            "5.": ("NUM", "5"),
            "1.234,56": ("NUM", "1.234,56"),
            "1 000": ("NUM", "000"),
            "1_000": ("NUM", "1_000"),
            "5\u2030": ("NUM", "5\u2030"),
            "π": ("PROPN", "π"),
            "Ω": ("PROPN", "Ω"),
            "℃": ("PUNCT", "℃"),
            "℉": ("PROPN", "℉"),
            "⏰": ("X", "⏰"),
            "📅": ("NOUN", "📅"),
            "isn't": ("AUX", "be"),
            "could've": ("AUX", "'ve"),
            "I've": ("AUX", "have"),
            "that's": ("AUX", "be"),
            "there's": ("VERB", "be"),
            "he'd": ("VERB", "would"),
            "I'd": ("VERB", "would"),
            "d'une": ("X", "d'une"),
            "qu'elle": ("PROPN", "qu'elle"),
            "Prof.": ("PROPN", "Prof."),
            "U.N.": ("PROPN", "U.N."),
            "B.A.": ("PROPN", "B.A."),
            "i.e.,": ("X", "i.e."),
            "e.g.,": ("ADV", "e.g."),
            "n/a": ("PRON", "a"),
            "w/o": ("ADP", "w/o"),
            "COVID-19": ("PROPN", "COVID-19"),
            "pre-meeting": ("NOUN", "pre"),
            "9-5": ("NUM", "9"),
            "9am-5pm": ("NOUN", "9am-5pm"),
            "0900h": ("X", "0900h"),
            "9hrs": ("NUM", "9hrs"),
            "30/09/2026": ("NUM", "30/09/2026"),
            "2026-10-04T09:00:00Z": ("NOUN", "04t09:00:00z"),
            "Q3": ("PROPN", "Q3"),
            "FY2026": ("PROPN", "FY2026"),
            "9:00 a.m.": ("NUM", "9:00"),
            "\u201chello\u201d": ("INTJ", "hello"),
            "\u300chello\u300d": ("PROPN", "\u300c"),
            "a\u0303": ("PROPN", "a\u0303"),
            "０": ("X", "０"),
            "٠": ("NUM", "٠"),
        }
        for text, (pos, lemma) in rejected.items():
            with self.subTest(text=text):
                self.assertFalse(text_looks_french(text))
                result = self.classify_without_time(text)
                self.assertEqual(
                    ("reject", "not_a_schedule_request"),
                    (result["decision"], result["reason"]),
                )
                self.assertEqual(pos, result["features"]["root"]["pos"])
                self.assertEqual(lemma, result["features"]["root"]["lemma"])
                self.assertFalse(result["features"]["has_temporal_signal"])

        with_subject = ("I've", "that's", "there's", "he'd", "I'd")
        without_subject = ("isn't", "could've", "1/2", "gotta")
        for text in with_subject:
            with self.subTest(text=text, subject=True):
                self.assertTrue(self.classify_without_time(text)["features"]["has_subject"])
        for text in without_subject:
            with self.subTest(text=text, subject=False):
                self.assertFalse(self.classify_without_time(text)["features"]["has_subject"])

    def classify_without_time(self, text):
        with patch("intent.duckling_parse", return_value=[]):
            return classify(text)


if __name__ == "__main__":
    unittest.main()
