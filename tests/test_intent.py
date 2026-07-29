import unittest
from unittest.mock import patch

from intent import classify


TEST_CASES = {
    "allow": [
        ("Remind me every Monday at 9am.", True),
        ("Email Acme a recap every Monday at 9am.", True),
        ("Can you send me a digest every Friday?", True),
        ("Please notify the team tomorrow morning.", True),
        ("Create a reminder for next Friday at 9am.", True),
    ],
    "clarify": [
        ("Next Friday at 9am.", True),
        ("Please email John.", False),
        ("Don't remind me every Monday at 9am.", True),
    ],
    "reject": [
        ("I love waking up every Monday at 9am.", True),
        ("My backup runs every day.", True),
        ("We meet every Friday at 10am.", True),
        ("What is Kubernetes?", False),
        ("When is Easter next year?", True),
    ],
}

# Confirmed failures found by eval_adversarial.py are inserted in this list.
# Tuple format: (prompt, expected decision, Duckling detected time/duration).
ADVERSARIAL_CASES = [
    ("Please don't forget to remind me about the 3pm meeting", 'allow', True),
    ('What about scheduling our weekly sync every Monday at 10am?', 'allow', True),
    # eval_adversarial.py inserts confirmed cases above this marker.
]


def duckling_entities(has_time):
    if not has_time:
        return []
    return [
        {
            "body": "fixture time",
            "dim": "time",
            "value": {
                "type": "value",
                "value": "2026-07-27T09:00:00.000-07:00",
                "grain": "hour",
            },
        }
    ]


class IntentClassifierTests(unittest.TestCase):
    def test_prompt_classifications(self):
        for expected, cases in TEST_CASES.items():
            for text, has_time in cases:
                self.assert_classification(text, expected, has_time)

        for text, expected, has_time in ADVERSARIAL_CASES:
            self.assert_classification(text, expected, has_time)

    def assert_classification(self, text, expected, has_time):
        with self.subTest(expected=expected, text=text):
            with patch(
                "intent.duckling_parse",
                return_value=duckling_entities(has_time),
            ):
                result = classify(text)

            self.assertEqual(
                expected,
                result["decision"],
                (
                    f"{text!r} should be {expected!r}, got "
                    f"{result['decision']!r} ({result['reason']})"
                ),
            )


if __name__ == "__main__":
    unittest.main()
