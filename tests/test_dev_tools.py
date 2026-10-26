import unittest
from pathlib import Path

from eval_adversarial import loopback_base_url
from fix_classifier import _UnsafePatch, assert_patch_source_is_safe


ROOT = Path(__file__).resolve().parents[1]


class DevToolGuardTests(unittest.TestCase):
    def test_current_classifier_source_passes_the_patch_guard(self):
        source = (ROOT / "intent.py").read_text(encoding="utf-8")
        assert_patch_source_is_safe(source)

    def test_generated_process_and_import_calls_are_rejected(self):
        samples = (
            "import os\ndef classify(text):\n    os.system(text)\n",
            "import subprocess\ndef classify(text):\n    return text\n",
            "import requests\ndef classify(text):\n    requests.get('http://169.254.169.254/')\n",
            "def classify(text):\n    return eval(text)\n",
        )
        for source in samples:
            with self.subTest(source=source.splitlines()[0]):
                with self.assertRaises(_UnsafePatch):
                    assert_patch_source_is_safe(source)

    def test_classifier_url_stays_on_loopback(self):
        self.assertEqual(
            "http://127.0.0.1:8080",
            loopback_base_url("http://127.0.0.1:8080/"),
        )
        self.assertEqual(
            "http://localhost:5000",
            loopback_base_url("http://localhost:5000"),
        )
        for url in (
            "http://169.254.169.254/",
            "http://example.com/v1",
            "http://10.0.0.5:8080",
            "file:///etc/passwd",
            "http://user:pass@127.0.0.1:8080",
            "http://127.0.0.1:8080/?next=http://evil",
        ):
            with self.subTest(url=url):
                with self.assertRaises(RuntimeError):
                    loopback_base_url(url)
