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

from intent import classify, nlp as english_nlp, should_use_french_locale, text_looks_french


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

# Alphabetic stop lists with no lowercase/uppercase pair. Faroese, Nynorsk,
# and the multilingual class are empty; the rest are caseless scripts.
NO_CASED_ALPHA_STOP = {
    "am", "ar", "bn", "fa", "fo", "gu", "he", "hi", "ja", "kn", "ko", "ml",
    "mr", "ne", "nn", "sa", "si", "ta", "te", "th", "ti", "ur", "xx",
}

# "a" is a stop word. "b" is too in a smaller set.
A_STOP_LANGUAGES = {
    "ca", "cs", "de", "dsb", "en", "es", "fr", "ga", "hr", "hsb", "hu", "it",
    "lb", "lij", "lt", "pl", "pt", "sk",
}
AB_STOP_LANGUAGES = {"ro", "sl", "sq", "yo"}

# Hyphen compounds stay one token in these classes.
HYPHEN_GLUE_LANGUAGES = {
    "ca", "da", "de", "el", "es", "fi", "hu", "ky", "lb", "nb", "nl", "nn",
    "pt", "ro", "sr", "sv", "tt", "vi",
}
EMAIL_GLUE_LANGUAGES = (HYPHEN_GLUE_LANGUAGES | {"fr", "it"}) - {"ro"}
EMAIL_STOP_E = {"hr", "lg", "lij", "lt", "sl", "sq", "tn", "yo"}

# Final dot stays on the abbreviation.
IE_GLUE_LANGUAGES = {
    "am", "ar", "da", "de", "en", "fa", "grc", "hu", "lb", "ms", "nl", "pt",
    "ti", "vi",
}
EG_GLUE_LANGUAGES = {"am", "ar", "de", "en", "fa", "grc", "nl", "pt", "ti", "vi"}
MR_GLUE_LANGUAGES = {
    "am", "ar", "da", "de", "en", "fa", "fr", "grc", "hu", "nb", "nl", "nn",
    "pt", "sl", "ti", "vi",
}

# Straight apostrophes. Curly apostrophes are covered separately.
ITS_GLUE_LANGUAGES = {
    "am", "ar", "bn", "da", "de", "el", "es", "fa", "fi", "grc", "hu", "nb",
    "nl", "nn", "pl", "ro", "sl", "sr", "sv", "ti", "vi",
}
APOSTROPHE_PARTS = {"ky", "tt"}
LAN_SPLIT_STOP = {"fr", "it"}
DACCORD_SPLIT_STOP = {"fr", "it", "lb", "lij"}

EM_DASH_GLUE_LANGUAGES = {
    "ca", "da", "de", "es", "hu", "lb", "nb", "nl", "nn", "ro", "sr", "sv", "vi",
}
COLON_GLUE_LANGUAGES = {"fi", "sv", "vi"}
SLASH_GLUE_LANGUAGES = {"hu", "lb", "nl", "ro", "vi"}
UNIT_GLUE_LANGUAGES = {"grc", "ro", "vi"}
EURO_GLUE_LANGUAGES = {"bn", "hu", "vi"}
CLOCK_SUFFIX_SPLIT = {"ca", "es"}
AMPM_DOT_GLUE = {"am", "ar", "fa", "grc", "ti", "vi"}
HASH_GLUE_LANGUAGES = {"id", "ms", "nb", "nn", "vi"}
REPEAT_PUNCT_GLUE = {"vi"}
DASH_SPLIT_LANGUAGES = {"ca", "id", "ms", "pl"}
PRECOMPOSED_E_STOP = {"ga", "lij", "pt", "yo"}

# Duckling locales that have no spaCy language class.
DUCKLING_WITHOUT_SPACY = {"ka", "km", "lo", "mn", "my", "sw"}

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

    def test_case_compounds_clocks_and_amounts(self):
        for code, pipeline in self.pipelines.items():
            if pipeline is None:
                continue
            with self.subTest(code=code):
                if code == "zh":
                    for text in ("ABC", "Abc", "abc", "9:00am", "9h30", "09h00", "5USD", "foo_bar", "0x10", "C++"):
                        self.assertEqual(list(text), self.token_texts(pipeline, text))
                    continue

                upper = self.round_trip(pipeline, "ABC")[0]
                title = self.round_trip(pipeline, "Abc")[0]
                lower = self.round_trip(pipeline, "abc")[0]
                self.assertTrue(upper.is_upper and upper.is_ascii and upper.is_alpha)
                self.assertTrue(title.is_title and title.is_ascii and not title.is_upper)
                self.assertTrue(lower.is_lower and lower.is_ascii and lower.is_alpha)

                for text in ("9:00am", "9h30", "09h00", "foo_bar", "0x10", "@user", "hello|world"):
                    token = self.round_trip(pipeline, text)[0]
                    self.assertEqual(text, token.text)
                    self.assertFalse(token.like_num)
                    self.assertFalse(token.like_url)
                self.assertTrue(self.round_trip(pipeline, "5USD")[0].is_upper)
                self.assertTrue(self.round_trip(pipeline, "C++")[0].is_title)

                clock = self.round_trip(pipeline, "9.30")
                if code == "pl":
                    self.assertEqual(["9", ".", "30"], self.texts(clock))
                    self.assertEqual([True, False, True], [token.like_num for token in clock])
                elif code in {"grc", "la"}:
                    self.assertEqual(["9.30"], self.texts(clock))
                    self.assertFalse(clock[0].like_num)
                else:
                    self.assertEqual(["9.30"], self.texts(clock))
                    self.assertTrue(clock[0].like_num)

                for text, split in (("9am", CLOCK_SUFFIX_SPLIT | {"en"}), ("12pm", CLOCK_SUFFIX_SPLIT | {"en"})):
                    doc = self.round_trip(pipeline, text)
                    if code in split:
                        self.assertEqual([text[:-2], text[-2:]], self.texts(doc))
                        self.assertTrue(doc[0].like_num)
                        self.assertEqual(text == "9am" and code == "en", doc[1].is_stop)
                    else:
                        self.assertEqual([text], self.texts(doc))
                        self.assertFalse(doc[0].like_num)

                dotted = self.round_trip(pipeline, "12a.m.")
                if code in CLOCK_SUFFIX_SPLIT | {"en"}:
                    self.assertEqual(["12", "a.m."], self.texts(dotted))
                    self.assertTrue(dotted[0].like_num)
                elif code in AMPM_DOT_GLUE:
                    self.assertEqual(["12a.m."], self.texts(dotted))
                else:
                    self.assertEqual(["12a.m", "."], self.texts(dotted))

                euro = self.round_trip(pipeline, "€5")
                if code in EURO_GLUE_LANGUAGES:
                    self.assertEqual(["€5"], self.texts(euro))
                    self.assertFalse(euro[0].like_num)
                else:
                    self.assertEqual(["€", "5"], self.texts(euro))
                    self.assertTrue(euro[0].is_currency)
                    self.assertEqual(code != "grc", euro[1].like_num)

                pounds = self.round_trip(pipeline, "£5.00")
                if code in EURO_GLUE_LANGUAGES:
                    self.assertEqual(["£5.00"], self.texts(pounds))
                elif code == "pl":
                    self.assertEqual(["£", "5", ".", "00"], self.texts(pounds))
                elif code in {"grc", "la"}:
                    self.assertEqual(["£", "5.00"], self.texts(pounds))
                    self.assertFalse(pounds[1].like_num)
                else:
                    self.assertEqual(["£", "5.00"], self.texts(pounds))
                    self.assertTrue(pounds[0].is_currency)
                    self.assertTrue(pounds[1].like_num)

                compound = self.round_trip(pipeline, "well-known")
                if code in HYPHEN_GLUE_LANGUAGES:
                    self.assertEqual(["well-known"], self.texts(compound))
                else:
                    self.assertEqual(["well", "-", "known"], self.texts(compound))
                    self.assertEqual(code == "en", compound[0].is_stop)

                email = self.round_trip(pipeline, "e-mail")
                if code in EMAIL_GLUE_LANGUAGES:
                    self.assertEqual(["e-mail"], self.texts(email))
                elif code == "ro":
                    self.assertEqual(["e-", "mail"], self.texts(email))
                else:
                    self.assertEqual(["e", "-", "mail"], self.texts(email))
                    self.assertEqual(code in EMAIL_STOP_E, email[0].is_stop)

    def test_ordinals_units_dates_and_abbreviations(self):
        for code, pipeline in self.pipelines.items():
            if pipeline is None:
                continue
            with self.subTest(code=code):
                if code == "zh":
                    for text in ("1st", "2nd", "4th", "21st", "10,000", "1.000", "1'000", "10km", "5kg", "2026/09/30", "30.09.2026", "i.e.", "e.g.", "U.S.", "Mr.", "C#", "#1", "n°5"):
                        self.assertEqual(list(text), self.token_texts(pipeline, text))
                    continue

                for text in ("1st", "2nd", "4th", "21st"):
                    token = self.round_trip(pipeline, text)[0]
                    self.assertEqual(text, token.text)
                    like_num = code == "en" or (code == "tn" and text == "4th")
                    self.assertEqual(like_num, token.like_num)

                thousands = self.round_trip(pipeline, "10,000")
                self.assertEqual(["10,000"], self.texts(thousands))
                self.assertEqual(code not in {"grc", "la", "ne"}, thousands[0].like_num)
                dotted = self.round_trip(pipeline, "1.000")
                if code == "pl":
                    self.assertEqual(["1", ".", "000"], self.texts(dotted))
                else:
                    self.assertEqual(["1.000"], self.texts(dotted))
                    self.assertEqual(code not in {"grc", "la"}, dotted[0].like_num)
                apostrophe = self.round_trip(pipeline, "1'000")
                self.assertEqual(["1'000"], self.texts(apostrophe))
                self.assertFalse(apostrophe[0].like_num)

                for text in ("10km", "5kg"):
                    doc = self.round_trip(pipeline, text)
                    if code in UNIT_GLUE_LANGUAGES:
                        self.assertEqual([text], self.texts(doc))
                        self.assertFalse(doc[0].like_num)
                    else:
                        self.assertEqual([text[:-2], text[-2:]], self.texts(doc))
                        self.assertTrue(doc[0].like_num)

                slash_date = self.round_trip(pipeline, "2026/09/30")
                if code in {"de", "id", "ms"}:
                    self.assertEqual(["2026", "/", "09", "/", "30"], self.texts(slash_date))
                    self.assertTrue(all(token.like_num for token in slash_date if token.text != "/"))
                else:
                    self.assertEqual(["2026/09/30"], self.texts(slash_date))
                    self.assertEqual(code == "fa", slash_date[0].like_num)
                dotted_date = self.round_trip(pipeline, "30.09.2026")
                if code == "pl":
                    self.assertEqual(["30", ".", "09", ".", "2026"], self.texts(dotted_date))
                else:
                    self.assertEqual(["30.09.2026"], self.texts(dotted_date))
                    self.assertEqual(code not in {"grc", "la"}, dotted_date[0].like_num)

                ie = self.round_trip(pipeline, "i.e.")
                eg = self.round_trip(pipeline, "e.g.")
                self.assertEqual(["i.e."] if code in IE_GLUE_LANGUAGES else ["i.e", "."], self.texts(ie))
                self.assertEqual(["e.g."] if code in EG_GLUE_LANGUAGES else ["e.g", "."], self.texts(eg))

                us = self.round_trip(pipeline, "U.S.")
                if code in {"lt", "sr"}:
                    self.assertEqual(["U.S", "."], self.texts(us))
                elif code == "pl":
                    self.assertEqual(["U", ".", "S", "."], self.texts(us))
                    self.assertTrue(us[0].is_stop)
                elif code == "sl":
                    self.assertEqual(["U.", "S."], self.texts(us))
                else:
                    self.assertEqual(["U.S."], self.texts(us))
                    self.assertTrue(us[0].is_title)

                mr = self.round_trip(pipeline, "Mr.")
                self.assertEqual(["Mr."] if code in MR_GLUE_LANGUAGES else ["Mr", "."], self.texts(mr))

                sharp = self.round_trip(pipeline, "C#")
                if code == "vi":
                    self.assertEqual(["C#"], self.texts(sharp))
                else:
                    self.assertEqual(["C", "#"], self.texts(sharp))
                    self.assertEqual(code in {"ro", "sl", "sq"}, sharp[0].is_stop)
                    self.assertEqual(code == "la", sharp[0].like_num)

                numbered = self.round_trip(pipeline, "#1")
                if code in HASH_GLUE_LANGUAGES:
                    self.assertEqual(["#1"], self.texts(numbered))
                    self.assertFalse(numbered[0].like_num)
                else:
                    self.assertEqual(["#", "1"], self.texts(numbered))
                    self.assertTrue(numbered[0].is_punct)
                    self.assertEqual(code != "grc", numbered[1].like_num)

                numero = self.round_trip(pipeline, "n°5")
                if code in {"hu", "nb", "nn", "ro", "vi"}:
                    self.assertEqual(["n°5"], self.texts(numero))
                elif code in {"es", "fr", "it"}:
                    self.assertEqual(["n°", "5"], self.texts(numero))
                    self.assertTrue(numero[1].like_num)
                else:
                    self.assertEqual(["n", "°", "5"], self.texts(numero))
                    self.assertEqual(code in {"sl", "yo"}, numero[0].is_stop)
                    self.assertEqual(code != "grc", numero[2].like_num)

    def test_apostrophes_quotes_and_repeated_punctuation(self):
        for code, pipeline in self.pipelines.items():
            if pipeline is None:
                continue
            with self.subTest(code=code):
                if code == "zh":
                    for text in ("it's", "I'm", "don't", "l'an", "d'accord", "qu'il", "\"hello\"", "'hello'", "«hello»", "(hello)", "[hello]", "!!!", "??", "...", "…", "“", "”"):
                        self.assertEqual(list(text), self.token_texts(pipeline, text))
                    continue

                its = self.round_trip(pipeline, "it's")
                if code == "en":
                    self.assertEqual(["it", "'s"], self.texts(its))
                    self.assertTrue(all(token.is_stop for token in its))
                elif code in {"ca", "fr"}:
                    self.assertEqual(["it'", "s"], self.texts(its))
                elif code in {"lt", "lv"}:
                    self.assertEqual(["it", "'s"], self.texts(its))
                    self.assertTrue(its[0].is_stop)
                    self.assertFalse(its[1].is_stop)
                elif code in ITS_GLUE_LANGUAGES:
                    self.assertEqual(["it's"], self.texts(its))
                    self.assertFalse(its[0].is_stop)
                else:
                    self.assertEqual(["it", "'s"], self.texts(its))
                    self.assertFalse(any(token.is_stop for token in its))

                im = self.round_trip(pipeline, "I'm")
                if code == "en":
                    self.assertEqual(["I", "'m"], self.texts(im))
                    self.assertTrue(all(token.is_stop for token in im))
                elif code == "ca":
                    self.assertEqual(["I", "'m"], self.texts(im))
                    self.assertTrue(im[0].is_stop)
                    self.assertFalse(im[1].is_stop)
                elif code in {"fr", "it", "lij"}:
                    self.assertEqual(["I'", "m"], self.texts(im))
                elif code in APOSTROPHE_PARTS:
                    self.assertEqual(["I", "'", "m"], self.texts(im))
                    self.assertTrue(im[1].is_quote)
                else:
                    self.assertEqual(["I'm"], self.texts(im))

                dont = self.round_trip(pipeline, "don't")
                if code == "en":
                    self.assertEqual(["do", "n't"], self.texts(dont))
                    self.assertTrue(all(token.is_stop for token in dont))
                elif code == "ca":
                    self.assertEqual(["don", "'t"], self.texts(dont))
                elif code in {"fr", "it", "lij"}:
                    self.assertEqual(["don'", "t"], self.texts(dont))
                elif code in APOSTROPHE_PARTS:
                    self.assertEqual(["don", "'", "t"], self.texts(dont))
                elif code == "yo":
                    self.assertEqual(["don't"], self.texts(dont))
                    self.assertTrue(dont[0].like_num)
                else:
                    self.assertEqual(["don't"], self.texts(dont))
                    self.assertFalse(dont[0].like_num)

                lan = self.round_trip(pipeline, "l'an")
                if code in LAN_SPLIT_STOP:
                    self.assertEqual(["l'", "an"], self.texts(lan))
                    self.assertTrue(lan[0].is_stop)
                elif code == "lij":
                    self.assertEqual(["l'", "an"], self.texts(lan))
                    self.assertTrue(lan[0].is_stop and lan[1].is_stop)
                elif code == "ca":
                    self.assertEqual(["l'", "an"], self.texts(lan))
                    self.assertFalse(lan[0].is_stop)
                elif code in APOSTROPHE_PARTS:
                    self.assertEqual(["l", "'", "an"], self.texts(lan))
                else:
                    self.assertEqual(["l'an"], self.texts(lan))

                accord = self.round_trip(pipeline, "d'accord")
                if code in DACCORD_SPLIT_STOP:
                    self.assertEqual(["d'", "accord"], self.texts(accord))
                    self.assertTrue(accord[0].is_stop)
                elif code == "ca":
                    self.assertEqual(["d'", "accord"], self.texts(accord))
                    self.assertFalse(accord[0].is_stop)
                elif code in APOSTROPHE_PARTS:
                    self.assertEqual(["d", "'", "accord"], self.texts(accord))
                else:
                    self.assertEqual(["d'accord"], self.texts(accord))

                quil = self.round_trip(pipeline, "qu'il")
                if code == "fr":
                    self.assertEqual(["qu'", "il"], self.texts(quil))
                    self.assertTrue(quil[0].is_stop and quil[1].is_stop)
                elif code == "it":
                    self.assertEqual(["qu'", "il"], self.texts(quil))
                    self.assertTrue(quil[1].is_stop)
                    self.assertFalse(quil[0].is_stop)
                elif code in {"ca", "lij"}:
                    self.assertEqual(["qu'", "il"], self.texts(quil))
                    self.assertFalse(quil[0].is_stop or quil[1].is_stop)
                elif code in APOSTROPHE_PARTS:
                    self.assertEqual(["qu", "'", "il"], self.texts(quil))
                else:
                    self.assertEqual(["qu'il"], self.texts(quil))

                quoted = self.round_trip(pipeline, '"hello"')
                if code == "vi":
                    self.assertEqual(['"hello"'], self.texts(quoted))
                else:
                    self.assertEqual(['"', "hello", '"'], self.texts(quoted))
                    self.assertTrue(quoted[0].is_quote and quoted[0].is_left_punct and quoted[0].is_right_punct)

                single = self.round_trip(pipeline, "'hello'")
                if code == "vi":
                    self.assertEqual(["'hello'"], self.texts(single))
                elif code == "el":
                    self.assertEqual(["'", "hell", "o'"], self.texts(single))
                elif code == "fi":
                    self.assertEqual(["'", "hello'"], self.texts(single))
                else:
                    self.assertEqual(["'", "hello", "'"], self.texts(single))
                    self.assertTrue(single[0].is_quote)

                guillemet = self.round_trip(pipeline, "«hello»")
                if code == "vi":
                    self.assertEqual(["«hello»"], self.texts(guillemet))
                else:
                    self.assertEqual(["«", "hello", "»"], self.texts(guillemet))
                    self.assertTrue(guillemet[0].is_left_punct and guillemet[2].is_right_punct)

                for text in ("(hello)", "[hello]"):
                    wrapped = self.round_trip(pipeline, text)
                    if code == "vi":
                        self.assertEqual([text], self.texts(wrapped))
                        self.assertFalse(wrapped[0].is_punct)
                    else:
                        self.assertEqual([text[0], "hello", text[-1]], self.texts(wrapped))
                        self.assertTrue(wrapped[0].is_bracket and wrapped[0].is_left_punct)
                        self.assertTrue(wrapped[-1].is_bracket and wrapped[-1].is_right_punct)

                for text in ("!!!", "??"):
                    marks = self.round_trip(pipeline, text)
                    if code in REPEAT_PUNCT_GLUE:
                        self.assertEqual([text], self.texts(marks))
                    else:
                        self.assertEqual(list(text), self.texts(marks))
                    self.assertTrue(all(token.is_punct for token in marks))
                ellipsis = self.round_trip(pipeline, "...")
                self.assertEqual(["..."], self.texts(ellipsis))
                self.assertTrue(ellipsis[0].is_punct)
                single_ellipsis = self.round_trip(pipeline, "…")[0]
                self.assertTrue(single_ellipsis.is_punct)
                self.assertFalse(single_ellipsis.is_stop)
                for text, left in (("“", True), ("”", False)):
                    quote = self.round_trip(pipeline, text)[0]
                    self.assertTrue(quote.is_quote and quote.is_punct)
                    self.assertEqual(left, quote.is_left_punct)
                    self.assertEqual(not left, quote.is_right_punct)
                    self.assertFalse(quote.is_stop)

    def test_spaces_emoji_modifiers_and_degree_clusters(self):
        for code, pipeline in self.pipelines.items():
            if pipeline is None:
                continue
            with self.subTest(code=code):
                pair = self.round_trip(pipeline, "a  b")
                self.assertEqual(["a", " ", "b"], self.texts(pair))
                self.assertEqual(" ", pair[0].whitespace_)
                self.assertTrue(pair[1].is_space)
                self.assertEqual(code in A_STOP_LANGUAGES or code in AB_STOP_LANGUAGES, pair[0].is_stop)
                self.assertEqual(code in AB_STOP_LANGUAGES, pair[2].is_stop)

                for text in ("a\tb", "a\u00a0b"):
                    spaced = self.round_trip(pipeline, text)
                    self.assertEqual(["a", text[1], "b"], self.texts(spaced))
                    self.assertEqual("", spaced[0].whitespace_)
                    self.assertTrue(spaced[1].is_space)

                if code == "zh":
                    self.assertEqual(list("hello\nworld"), self.token_texts(pipeline, "hello\nworld"))
                    self.assertEqual(list("hello/world"), self.token_texts(pipeline, "hello/world"))
                    self.assertEqual(list("hello:world"), self.token_texts(pipeline, "hello:world"))
                    self.assertEqual(list("hello\u2014world"), self.token_texts(pipeline, "hello\u2014world"))
                    self.assertEqual(list("e\u0301"), self.token_texts(pipeline, "e\u0301"))
                    self.assertEqual(["👋", "🏻"], self.token_texts(pipeline, "👋🏻"))
                    self.assertEqual(["5", "°"], self.token_texts(pipeline, "5°"))
                    self.assertEqual(["-", "-"], self.token_texts(pipeline, "--"))
                    self.assertEqual(["-", "-", "-"], self.token_texts(pipeline, "---"))
                    continue

                lines = self.round_trip(pipeline, "hello\nworld")
                self.assertEqual(["hello", "\n", "world"], self.texts(lines))
                self.assertTrue(lines[1].is_space)
                self.assertFalse(lines[0].is_stop or lines[2].is_stop)

                slash = self.round_trip(pipeline, "hello/world")
                colon = self.round_trip(pipeline, "hello:world")
                dash = self.round_trip(pipeline, "hello\u2014world")
                self.assertEqual(
                    ["hello/world"] if code in SLASH_GLUE_LANGUAGES else ["hello", "/", "world"],
                    self.texts(slash),
                )
                self.assertEqual(
                    ["hello:world"] if code in COLON_GLUE_LANGUAGES else ["hello", ":", "world"],
                    self.texts(colon),
                )
                self.assertEqual(
                    ["hello\u2014world"] if code in EM_DASH_GLUE_LANGUAGES else ["hello", "\u2014", "world"],
                    self.texts(dash),
                )

                combining = self.round_trip(pipeline, "e\u0301")[0]
                self.assertEqual("e\u0301", combining.text)
                self.assertTrue(combining.is_lower)
                self.assertFalse(combining.is_alpha)
                acute = self.round_trip(pipeline, "é")[0]
                self.assertTrue(acute.is_alpha and acute.is_lower)
                self.assertEqual(code in PRECOMPOSED_E_STOP, acute.is_stop)

                wave = self.round_trip(pipeline, "👋🏻")
                coder = self.round_trip(pipeline, "👨\u200d💻")
                if code in {"hu", "vi"}:
                    self.assertEqual(["👋🏻"], self.texts(wave))
                    self.assertEqual(["👨\u200d💻"], self.texts(coder))
                else:
                    self.assertEqual(["👋", "🏻"], self.texts(wave))
                    self.assertEqual(["👨", "\u200d", "💻"], self.texts(coder))

                degree = self.round_trip(pipeline, "5°")
                if code in {"hu", "it", "nb", "ro", "vi"}:
                    self.assertEqual(["5°"], self.texts(degree))
                    self.assertFalse(degree[0].like_num)
                else:
                    self.assertEqual(["5", "°"], self.texts(degree))
                    self.assertEqual(code != "grc", degree[0].like_num)
                    self.assertFalse(degree[1].is_punct)

                doubled = self.round_trip(pipeline, "--")
                if code in DASH_SPLIT_LANGUAGES | {"sl"}:
                    self.assertEqual(["-", "-"], self.texts(doubled))
                else:
                    self.assertEqual(["--"], self.texts(doubled))
                    self.assertTrue(doubled[0].is_punct)
                tripled = self.round_trip(pipeline, "---")
                if code in DASH_SPLIT_LANGUAGES:
                    self.assertEqual(["-", "-", "-"], self.texts(tripled))
                elif code == "sl":
                    self.assertEqual(["-", "--"], self.texts(tripled))
                else:
                    self.assertEqual(["---"], self.texts(tripled))

    def test_uppercased_stop_words_stay_stops(self):
        caseless = set()
        for code, pipeline in self.pipelines.items():
            words = spacy.util.get_lang_class(code).Defaults.stop_words
            cased = sorted(
                word
                for word in words
                if word.isalpha() and " " not in word and word != word.upper()
            )
            if not cased:
                caseless.add(code)
                continue
            self.assertIsNotNone(pipeline)
            word = cased[0]
            with self.subTest(code=code, word=word):
                original = self.round_trip(pipeline, word)
                upper = self.round_trip(pipeline, word.upper())
                if code == "zh":
                    self.assertGreater(len(upper), 1)
                    self.assertTrue(all(not token.is_stop for token in upper))
                    self.assertTrue(all(token.is_alpha for token in upper))
                    continue
                self.assertEqual([word], self.texts(original))
                self.assertTrue(original[0].is_stop)
                self.assertEqual([word.upper()], self.texts(upper))
                self.assertTrue(upper[0].is_stop)
                self.assertTrue(upper[0].is_upper)
        self.assertEqual(NO_CASED_ALPHA_STOP, caseless)

    def test_discovered_codes_match_duckling_gaps(self):
        self.assertEqual(
            DUCKLING_WITHOUT_SPACY,
            set(DUCKLING_LANGUAGE_SAMPLES) - set(self.codes),
        )
        for code in DUCKLING_WITHOUT_SPACY:
            with self.subTest(code=code):
                with self.assertRaises(ImportError):
                    spacy.util.get_lang_class(code)

    def texts(self, doc):
        return [token.text for token in doc]

    def token_texts(self, pipeline, text):
        return self.texts(self.round_trip(pipeline, text))

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

    def test_clock_abbreviation_and_symbol_edges_classify_stably(self):
        # Decisions from en_core_web_sm with no Duckling entity. French clock
        # forms are temporal; a few ASCII strings are subject-less verbs.
        expected = {
            "9h30": ("clarify", "temporal_signal_without_clear_request", "NUM", False),
            "09h00": ("clarify", "temporal_signal_without_clear_request", "NUM", False),
            "9am": ("reject", "not_a_schedule_request", "NUM", False),
            "12pm": ("reject", "not_a_schedule_request", "NOUN", False),
            "12a.m.": ("reject", "not_a_schedule_request", "NUM", False),
            "Mr.": ("reject", "not_a_schedule_request", "PROPN", False),
            "i.e.": ("reject", "not_a_schedule_request", "X", False),
            "e.g.": ("reject", "not_a_schedule_request", "ADV", False),
            "U.S.": ("reject", "not_a_schedule_request", "PROPN", False),
            "it's": ("reject", "not_a_schedule_request", "AUX", True),
            "I'm": ("reject", "not_a_schedule_request", "AUX", True),
            "don't": ("clarify", "request_without_temporal_signal", "VERB", False),
            "l'an": ("reject", "not_a_schedule_request", "ADJ", False),
            "d'accord": ("reject", "not_a_schedule_request", "PROPN", False),
            "qu'il": ("reject", "not_a_schedule_request", "PROPN", False),
            "10km": ("reject", "not_a_schedule_request", "NOUN", False),
            "5kg": ("reject", "not_a_schedule_request", "NOUN", False),
            "€5": ("reject", "not_a_schedule_request", "NUM", False),
            "£5.00": ("reject", "not_a_schedule_request", "NUM", False),
            "well-known": ("clarify", "request_without_temporal_signal", "VERB", False),
            "e-mail": ("reject", "not_a_schedule_request", "X", False),
            "1st": ("reject", "not_a_schedule_request", "NOUN", False),
            "2nd": ("reject", "not_a_schedule_request", "ADJ", False),
            "4th": ("reject", "not_a_schedule_request", "ADJ", False),
            "21st": ("reject", "not_a_schedule_request", "NOUN", False),
            "10,000": ("reject", "not_a_schedule_request", "NUM", False),
            "30.09.2026": ("reject", "not_a_schedule_request", "NUM", False),
            "2026/09/30": ("reject", "not_a_schedule_request", "NUM", False),
            "n°5": ("reject", "not_a_schedule_request", "CCONJ", False),
            "C++": ("reject", "not_a_schedule_request", "NOUN", False),
            "C#": ("reject", "not_a_schedule_request", "NOUN", False),
            "@user": ("reject", "not_a_schedule_request", "ADV", False),
            "#1": ("reject", "not_a_schedule_request", "NUM", False),
            "0x10": ("reject", "not_a_schedule_request", "X", False),
            "«hello»": ("reject", "not_a_schedule_request", "INTJ", False),
            "e\u0301": ("clarify", "request_without_temporal_signal", "VERB", False),
            "…": ("reject", "not_a_schedule_request", "PUNCT", False),
            "!!!": ("reject", "not_a_schedule_request", "PUNCT", False),
            "(hello)": ("reject", "not_a_schedule_request", "INTJ", False),
            "foo_bar": ("reject", "not_a_schedule_request", "PROPN", False),
            "5°": ("reject", "not_a_schedule_request", "NUM", False),
            "100°C": ("reject", "not_a_schedule_request", "NOUN", False),
            "👋🏻": ("reject", "not_a_schedule_request", "NOUN", True),
        }
        for text, (decision, reason, pos, subject) in expected.items():
            with self.subTest(text=text):
                self.assertEqual(text in {"9h30", "09h00"}, text_looks_french(text))
                self.assertEqual(text in {"9h30", "09h00"}, should_use_french_locale(text))
                with patch("intent.requests.post") as post:
                    post.return_value.json.return_value = []
                    post.return_value.raise_for_status.return_value = None
                    result = classify(text)
                locale = post.call_args.kwargs["data"]["locale"]
                self.assertEqual("fr_FR" if text in {"9h30", "09h00"} else "en_GB", locale)
                self.assertEqual(text, result["text"])
                self.assertEqual((decision, reason), (result["decision"], result["reason"]))
                self.assertEqual(pos, result["features"]["root"]["pos"])
                self.assertEqual(subject, result["features"]["has_subject"])
                self.assertEqual(
                    reason == "request_without_temporal_signal",
                    result["features"]["looks_like_request"],
                )

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


if __name__ == "__main__":
    unittest.main()
