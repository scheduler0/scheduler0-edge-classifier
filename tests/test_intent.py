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

# (text, expected decision, expected reason, Duckling detected time/duration).
# Recurrence phrases set has_time False on purpose: the weekday/period regex
# must count as a temporal signal even when Duckling returns nothing.
EDGE_CASES = [
    ("", "reject", "not_a_schedule_request", False),
    ("   ", "reject", "not_a_schedule_request", False),
    ("???", "reject", "not_a_schedule_request", False),
    (".", "reject", "not_a_schedule_request", False),
    ("ok", "reject", "not_a_schedule_request", False),
    ("0", "reject", "not_a_schedule_request", False),
    ("<script>alert(1)</script>", "reject", "not_a_schedule_request", False),
    ("", "clarify", "temporal_signal_without_clear_request", True),
    ("???", "clarify", "temporal_signal_without_clear_request", True),
    ("tomorrow", "clarify", "temporal_signal_without_clear_request", True),
    ("9am", "clarify", "temporal_signal_without_clear_request", True),
    ("Remind me.", "clarify", "request_without_temporal_signal", False),
    ("Please remind me.", "clarify", "request_without_temporal_signal", False),
    ("Please email John.", "clarify", "request_without_temporal_signal", False),
    ("Can you send the file?", "clarify", "request_without_temporal_signal", False),
    ("Schedule", "clarify", "request_without_temporal_signal", False),
    ("set up", "clarify", "request_without_temporal_signal", False),
    ("Don't remind me.", "clarify", "request_without_temporal_signal", False),
    ("Remind me.", "allow", "request_with_temporal_signal", True),
    ("Please remind me.", "allow", "request_with_temporal_signal", True),
    ("Can you send the file?", "allow", "request_with_temporal_signal", True),
    ("  Can you   remind me tomorrow?  ", "allow", "request_with_temporal_signal", True),
    ("please\nremind me tomorrow", "allow", "request_with_temporal_signal", True),
    ("\tRemind me\ttomorrow\n", "allow", "request_with_temporal_signal", True),
    ("Remind me tomorrow 😊", "allow", "request_with_temporal_signal", True),
    ("...Remind me tomorrow", "allow", "request_with_temporal_signal", True),
    ("I want you to email the report next Tuesday", "allow", "request_with_temporal_signal", True),
    ("Help me schedule a call tomorrow", "allow", "request_with_temporal_signal", True),
    ("would you schedule the review for Thursday?", "allow", "request_with_temporal_signal", True),
    ("help me remember this Friday", "allow", "request_with_temporal_signal", True),
    ("schedule the interview tomorrow afternoon", "allow", "request_with_temporal_signal", True),
    ("What about Friday?", "allow", "request_with_temporal_signal", True),
    ("What about Friday?", "clarify", "request_without_temporal_signal", False),
    ("Please don't forget to remind me about the 3pm meeting", "allow", "request_with_temporal_signal", True),
    ("Please do not forget to remind me tomorrow", "allow", "request_with_temporal_signal", True),
    ("Don't forget the 3pm meeting", "allow", "request_with_temporal_signal", True),
    ("Don't forget the 3pm meeting", "clarify", "request_without_temporal_signal", False),
    ("Don't remind me.", "clarify", "negated_schedule_like_request_needs_intent_confirmation", True),
    ("Don't remind me every Monday at 9am.", "clarify", "negated_schedule_like_request_needs_intent_confirmation", False),
    ("Don\u2019t remind me every Monday at 9am.", "clarify", "negated_schedule_like_request_needs_intent_confirmation", False),
    ("do NOT remind me each weekend", "clarify", "negated_schedule_like_request_needs_intent_confirmation", False),
    ("Never email me every Monday", "clarify", "negated_schedule_like_request_needs_intent_confirmation", False),
    ("STOP reminding me every day", "clarify", "negated_schedule_like_request_needs_intent_confirmation", False),
    ("I will not attend every Friday", "clarify", "negated_schedule_like_request_needs_intent_confirmation", False),
    ("Cancel the Monday reminder", "clarify", "negated_schedule_like_request_needs_intent_confirmation", True),
    ("remove the daily digest", "clarify", "negated_schedule_like_request_needs_intent_confirmation", True),
    ("dont cancel the Friday sync", "clarify", "negated_schedule_like_request_needs_intent_confirmation", True),
    ("Never mind the Monday reminder", "clarify", "negated_schedule_like_request_needs_intent_confirmation", True),
    ("every Monday", "clarify", "temporal_signal_without_clear_request", False),
    ("Thanks. Remind me every Monday at 9am.", "clarify", "temporal_signal_without_clear_request", False),
    ("Next Friday at 9am.", "clarify", "temporal_signal_without_clear_request", True),
    ("The meeting is tomorrow.", "reject", "declarative_schedule_not_request", True),
    ("I'll send the proposal tomorrow.", "reject", "declarative_schedule_not_request", True),
    ("I love waking up every Monday at 9am.", "reject", "declarative_schedule_not_request", False),
    ("Why do we meet every Monday?", "reject", "informational_question_not_schedule_request", False),
    ("Who scheduled the Friday standup?", "reject", "informational_question_not_schedule_request", False),
    ("Which day is the review?", "reject", "informational_question_not_schedule_request", False),
    ("Where is the Monday meeting?", "reject", "informational_question_not_schedule_request", False),
    ("What time is the standup?", "reject", "informational_question_not_schedule_request", True),
    ("How about tomorrow at 3?", "reject", "informational_question_not_schedule_request", True),
    ("When should we meet?", "reject", "informational_question_not_schedule_request", True),
    ("What is Kubernetes?", "reject", "informational_question_not_schedule_request", True),
]

RECURRENCE_GRAINS = [
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
    "day",
    "week",
    "month",
    "year",
    "morning",
    "afternoon",
    "evening",
    "night",
    "weekday",
    "weekend",
    "mondays",
    "fridays",
]

# Confirmed failures found by eval_adversarial.py are inserted in this list.
# Tuple format: (prompt, expected decision, Duckling detected time/duration).
ADVERSARIAL_CASES = [
    ("Please don't forget to remind me about the 3pm meeting", 'allow', True),
    ('What about scheduling our weekly sync every Monday at 10am?', 'allow', True),
    # Requests with a greeting or acknowledgment before the ask.
    ("Hi, can you schedule a sync every Monday at 10am?", "allow", True),
    ("Hey can you remind me tomorrow at 9am", "allow", True),
    ("Thanks, but can you move it to Thursday at 2?", "allow", True),
    # Proposals and first-person scheduling intents.
    ("Let's schedule a meeting for Friday at 3.", "allow", True),
    ("Let us meet tomorrow at 3pm.", "allow", True),
    ("How about we meet tomorrow at 4?", "allow", True),
    ("What if we move the standup to 10am every day?", "allow", True),
    ("Could we schedule a review for next Wednesday at 2pm?", "allow", True),
    ("Shall we meet tomorrow at 3?", "allow", True),
    ("Could someone remind me tomorrow at 9?", "allow", True),
    ("Can the team meet every Friday at 10am?", "allow", True),
    ("We should meet every Monday at 9am.", "allow", True),
    ("I need this done by Friday at 5pm.", "allow", True),
    ("When should we meet next Tuesday at 3pm?", "allow", True),
    ("I'd like to schedule a call for Thursday at 4pm.", "allow", True),
    ("I would like you to remind me every Monday at 9am.", "allow", True),
    ("I want a reminder every Monday at 9am.", "allow", True),
    ("I need a meeting tomorrow at 3pm.", "allow", True),
    ("Would it be possible to meet tomorrow at 9?", "allow", True),
    ("Do you mind scheduling a sync tomorrow at 3?", "allow", True),
    ("I was hoping you could book a room tomorrow at 2.", "allow", True),
    # Imperatives spaCy mistags, and embedded subjects that are not the root.
    ("Book conference room B for tomorrow at 10.", "allow", True),
    ("Snooze the reminder until tomorrow morning.", "allow", True),
    ("Circle back on this tomorrow morning.", "allow", True),
    ("Set this up so I get a nudge tomorrow morning.", "allow", True),
    ("Remind me every Monday because I forget.", "allow", True),
    # Negation false positives: visit, double negation, discourse marker.
    ("Stop by my office tomorrow at 3.", "allow", True),
    ("Please stop by the office every Monday at 9am.", "allow", True),
    ("Please do not let me forget the 4pm demo.", "allow", True),
    ("Never mind, remind me tomorrow at 9 instead.", "allow", True),
    ("Please stop emailing me every Monday.", "clarify", True),
    # Informational asks that mention a time but do not create a schedule.
    ("Can you tell me when the meeting is tomorrow?", "reject", True),
    ("Can you explain why we meet every Friday?", "reject", True),
    ("Please tell me about the Monday meeting.", "reject", True),
    ("Tell me the agenda for tomorrow's meeting.", "reject", True),
    ("Please describe our schedule every Monday.", "reject", True),
    ("How do I schedule a meeting for Friday?", "reject", True),
    ("What if it rains every Monday?", "reject", True),
    ("Book club meets every Monday at 9am.", "reject", True),
    ("I love to schedule meetings every Monday.", "reject", True),
    # Bugbot review fixes: preference statements, let-verb, unrelated proposals
    ("I like to schedule meetings every Monday at 9am.", "reject", True),
    ("This app lets me schedule meetings every Monday at 9am.", "reject", True),
    ("What if the book club meets every Monday at 9am?", "reject", True),
    ("How about my book club on Tuesday?", "reject", True),
    # French failures found without the Claude CLI. Duckling was assumed to
    # see a time whenever the phrase contains a date, clock, or recurrence;
    # the English heuristics still misclassified these.
    ("Rappelle-moi tous les lundis à 9h.", "allow", True),
    ("Rappelle-moi tous les lundis a 9h.", "allow", True),
    ("Envoie un récapitulatif à Acme tous les lundis à 9h.", "allow", True),
    ("Peux-tu m'envoyer un digest tous les vendredis ?", "allow", True),
    ("Peux-tu me prevenir demain matin ?", "allow", True),
    ("Préviens l'équipe demain matin.", "allow", True),
    ("Crée un rappel pour vendredi prochain à 9h.", "allow", True),
    ("S'il vous plaît, planifiez la réunion demain à 15h.", "allow", True),
    ("Pourriez-vous me rappeler chaque matin à 8h ?", "allow", True),
    ("Aide-moi à programmer un point chaque lundi.", "allow", True),
    ("Je voudrais planifier notre sync hebdomadaire tous les lundis à 10h.", "allow", True),
    ("N'oublie pas de me rappeler la réunion de 15h.", "allow", True),
    ("Merci de notifier l'équipe demain matin.", "allow", True),
    ("Il faut que tu m'envoies le rapport chaque vendredi.", "allow", True),
    ("Est-ce que tu peux réserver la salle demain à 14h ?", "allow", True),
    ("On se cale un point tous les mardis à 11h ?", "allow", True),
    ("Bloque-moi une heure demain après-midi.", "allow", True),
    ("Ajoute un rappel chaque premier du mois.", "allow", True),
    ("Programmez l'envoi du bulletin tous les soirs à 18h.", "allow", True),
    ("Rappelle-moi dans deux heures.", "allow", True),
    ("Fixe un rendez-vous lundi prochain à 9h30.", "allow", True),
    ("Pense à me rappeler demain matin.", "allow", True),
    ("Fais-moi un rappel pour demain 8h.", "allow", True),
    ("Mets un rappel lundi prochain.", "allow", True),
    ("Ne me préviens pas.", "clarify", False),
    ("Je pense à la réunion de demain.", "reject", True),
    ("Note que le train part à 7h.", "reject", True),
    ("Notez que la réunion est demain.", "reject", True),
    ("Mets la table demain.", "clarify", True),
    ("Vendredi prochain à 9h.", "clarify", True),
    ("Demain matin.", "clarify", True),
    ("Tous les lundis à 9h.", "clarify", True),
    ("Envoie un courriel à Jean.", "clarify", False),
    ("Peux-tu envoyer un message à Marie ?", "clarify", False),
    ("Planifie quelque chose.", "clarify", False),
    ("Ne me rappelle pas tous les lundis à 9h.", "clarify", True),
    ("Arrête de m'envoyer un digest chaque vendredi.", "clarify", True),
    ("Annule le rappel de tous les lundis.", "clarify", True),
    ("Ne programme plus la réunion de 15h.", "clarify", True),
    ("J'adore me réveiller tous les lundis à 9h.", "reject", True),
    ("Ma sauvegarde s'exécute tous les jours.", "reject", True),
    ("Nous nous réunissons tous les vendredis à 10h.", "reject", True),
    ("Le backup tourne chaque nuit.", "reject", True),
    ("La facture est due vendredi.", "reject", True),
    ("Mon train part à 7h.", "reject", True),
    ("La standup est tous les jours à 9h30.", "reject", True),
    ("Qu'est-ce que Kubernetes ?", "reject", False),
    ("Quand est Pâques l'année prochaine ?", "reject", True),
    ("Pourquoi la réunion est-elle à 9h ?", "reject", True),
    ("Où a lieu la réunion de demain ?", "reject", True),
    ("Qui participe au point de lundi ?", "reject", True),
    ("Pourriez-vous m'expliquer pourquoi nous nous réunissons tous les vendredis ?", "reject", True),
    ("Est-ce que la réunion est demain ?", "reject", True),
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

    def test_edge_case_classifications(self):
        for text, expected, reason, has_time in EDGE_CASES:
            with self.subTest(text=text, has_time=has_time):
                result = self.classify_with_time(text, has_time)
                self.assertEqual(expected, result["decision"], result["reason"])
                self.assertEqual(reason, result["reason"])

    def test_recurrence_regex_without_duckling(self):
        for grain in RECURRENCE_GRAINS:
            for quantifier in ("every", "each"):
                text = f"Remind me {quantifier} {grain}."
                with self.subTest(text=text):
                    result = self.classify_with_time(text, False)
                    self.assertTrue(result["features"]["recurrence_regex_match"])
                    self.assertFalse(result["features"]["duckling_has_time"])
                    self.assertTrue(result["features"]["has_temporal_signal"])
                    self.assertEqual("allow", result["decision"])
                    self.assertEqual("request_with_temporal_signal", result["reason"])

    def test_duration_entity_counts_as_temporal_signal(self):
        entities = [
            {
                "body": "2 hours",
                "dim": "duration",
                "value": {"type": "value", "value": 2, "unit": "hour"},
            }
        ]
        result = self.classify_with_entities("Remind me in a bit.", entities)
        self.assertTrue(result["features"]["duckling_has_time"])
        self.assertEqual("allow", result["decision"])
        self.assertEqual(entities[0]["dim"], result["duckling_entities"][0]["dim"])

    def test_non_time_entity_is_ignored(self):
        entities = [
            {"body": "42", "dim": "number", "value": {"type": "value", "value": 42}}
        ]
        result = self.classify_with_entities("Remind me.", entities)
        self.assertFalse(result["features"]["duckling_has_time"])
        self.assertEqual("clarify", result["decision"])
        self.assertEqual("request_without_temporal_signal", result["reason"])

    def test_duckling_is_always_called_with_english_locale(self):
        samples = [
            "Remind me tomorrow",
            "",
            "每星期一提醒我",
            "Напомни мне завтра",
        ]
        for text in samples:
            with self.subTest(text=text):
                with patch("intent.requests.post") as post:
                    response = post.return_value
                    response.raise_for_status.return_value = None
                    response.json.return_value = []
                    result = classify(text)
                self.assertIn(result["decision"], {"allow", "clarify", "reject"})
                post.assert_called_once()
                data = post.call_args.kwargs["data"]
                self.assertEqual("en_GB", data["locale"])
                self.assertEqual(text, data["text"])
                self.assertEqual('["time","duration"]', data["dims"])

    def test_feature_payload_for_imperative_and_declarative(self):
        allow = self.classify_with_time("Remind me every Monday at 9am.", False)
        features = allow["features"]
        self.assertTrue(features["looks_like_request"])
        self.assertFalse(features["looks_like_declarative"])
        self.assertFalse(features["looks_like_info_question"])
        self.assertFalse(features["has_negation"])
        self.assertFalse(features["has_subject"])
        self.assertEqual("remind", features["root"]["lemma"])
        self.assertEqual("VERB", features["root"]["pos"])
        self.assertTrue(allow["tokens"])

        statement = self.classify_with_time("I love waking up every Monday at 9am.", False)
        features = statement["features"]
        self.assertTrue(features["looks_like_declarative"])
        self.assertTrue(features["has_non_request_subject"])
        self.assertFalse(features["has_second_person_subject"])
        self.assertEqual("love", features["root"]["lemma"])

        question = self.classify_with_time("Can you send me a digest every Friday?", False)
        self.assertTrue(question["features"]["has_second_person_subject"])
        self.assertTrue(question["features"]["looks_like_request"])
        self.assertFalse(question["features"]["looks_like_declarative"])

    def test_long_text_does_not_crash(self):
        text = "Remind me every Monday at 9am. " * 50
        result = self.classify_with_time(text, False)
        self.assertEqual("allow", result["decision"])
        self.assertGreater(len(result["tokens"]), 10)

    def classify_with_time(self, text, has_time):
        return self.classify_with_entities(text, duckling_entities(has_time))

    def classify_with_entities(self, text, entities):
        with patch("intent.duckling_parse", return_value=entities):
            from intent import classify

            return classify(text)

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

    def test_duckling_locale_follows_language(self):
        with patch("intent.requests.post") as post:
            post.return_value.json.return_value = []
            post.return_value.raise_for_status.return_value = None
            classify("Rappelle-moi demain matin.")
            classify("Remind me tomorrow morning.")

        locales = [call.kwargs["data"]["locale"] for call in post.call_args_list]
        self.assertEqual(["fr_FR", "en_GB"], locales)

    def test_french_recurrence_grains_without_duckling(self):
        tous_les = [
            "lundi",
            "mardi",
            "mercredi",
            "jeudi",
            "vendredi",
            "samedi",
            "dimanche",
            "jour",
            "semaine",
            "mois",
            "an",
            "année",
            "matin",
            "après-midi",
            "apres-midi",
            "soir",
            "nuit",
            "lundis",
            "jours",
            "semaines",
            "nuits",
            "soirs",
        ]
        chaque = [
            "lundi",
            "mardi",
            "mercredi",
            "jeudi",
            "vendredi",
            "samedi",
            "dimanche",
            "jour",
            "semaine",
            "mois",
            "matin",
            "après-midi",
            "apres-midi",
            "soir",
            "nuit",
            "premier",
        ]
        phrases = [f"Rappelle-moi tous les {grain}." for grain in tous_les]
        phrases += [
            "Rappelle-moi toutes les semaines.",
            "Rappelle-moi toutes les nuits.",
        ]
        phrases += [f"Rappelle-moi chaque {grain}." for grain in chaque]
        phrases.append("Rappelle-moi hebdomadaire.")
        for text in phrases:
            with self.subTest(text=text):
                result = self.classify_with_time(text, False)
                self.assertTrue(result["features"]["recurrence_regex_match"])
                self.assertTrue(result["features"]["has_temporal_signal"])
                self.assertFalse(result["features"]["duckling_has_time"])
                self.assertEqual("allow", result["decision"])
                self.assertEqual("request_with_temporal_signal", result["reason"])

        # "heures" is a duration word, not one of the recurrence grains.
        hours = self.classify_with_time("Rappelle-moi toutes les heures.", False)
        self.assertFalse(hours["features"]["recurrence_regex_match"])
        self.assertFalse(hours["features"]["has_temporal_signal"])
        self.assertEqual("clarify", hours["decision"])
        self.assertEqual("request_without_temporal_signal", hours["reason"])

    def test_french_clock_and_calendar_words_without_duckling(self):
        phrases = [
            "demain",
            "demain matin",
            "aujourd'hui",
            "aujourd’hui",
            "hier",
            "lundi",
            "lundis",
            "lundi prochain",
            "prochaine semaine",
            "midi",
            "minuit",
            "matin",
            "soir",
            "après-midi",
            "dans deux heures",
            "dans 3 jours",
            "l'année prochaine",
            "année prochaine",
            "une heure",
            "9h",
            "9h30",
            "15 h",
        ]
        for phrase in phrases:
            text = f"Rappelle-moi {phrase}."
            with self.subTest(text=text):
                result = self.classify_with_time(text, False)
                self.assertTrue(result["features"]["has_temporal_signal"])
                self.assertFalse(result["features"]["duckling_has_time"])
                self.assertEqual("allow", result["decision"])
                self.assertEqual("request_with_temporal_signal", result["reason"])

    def test_french_surface_forms(self):
        cases = [
            ("  Rappelle-moi demain.  ", "clarify", "temporal_signal_without_clear_request"),
            ("Rappelle-moi\ndemain", "allow", "request_with_temporal_signal"),
            ("«Rappelle-moi demain»", "allow", "request_with_temporal_signal"),
            ("Rappelle-moi demain 😊", "allow", "request_with_temporal_signal"),
            ("RAPPELLE-MOI DEMAIN", "allow", "request_with_temporal_signal"),
            ("rappelle-moi demain", "allow", "request_with_temporal_signal"),
            ("Ne me rappelle pas demain.", "clarify", "negated_schedule_like_request_needs_intent_confirmation"),
            ("N'oublie pas demain.", "allow", "request_with_temporal_signal"),
            ("Pourquoi demain ?", "reject", "informational_question_not_schedule_request"),
            ("Qu'est-ce que demain ?", "reject", "informational_question_not_schedule_request"),
            ("Je me réveille demain.", "reject", "declarative_schedule_not_request"),
            ("Note que le train part demain.", "reject", "declarative_schedule_not_request"),
        ]
        for text, expected, reason in cases:
            with self.subTest(text=text):
                result = self.classify_with_time(text, False)
                self.assertEqual(expected, result["decision"], result["reason"])
                self.assertEqual(reason, result["reason"])
                self.assertEqual(text, result["text"])

    def test_french_clock_marker_on_english_text(self):
        # "9h" is a French clock, so these sentences take the French path.
        # English imperatives the model already knows still count as requests.
        # "I love 9h" does not match the French statement lexicon, so it stays
        # a bare temporal signal instead of a declarative rejection.
        cases = [
            ("Remind me at 9h", "allow", "request_with_temporal_signal"),
            ("Please remind me at 9h.", "allow", "request_with_temporal_signal"),
            ("Can you send this at 9h?", "allow", "request_with_temporal_signal"),
            ("Email Acme at 9h", "allow", "request_with_temporal_signal"),
            ("Don't remind me at 9h.", "clarify", "negated_schedule_like_request_needs_intent_confirmation"),
            ("9h", "clarify", "temporal_signal_without_clear_request"),
            ("9h30", "clarify", "temporal_signal_without_clear_request"),
            ("at 9 h", "clarify", "temporal_signal_without_clear_request"),
            ("the 9h meeting", "clarify", "temporal_signal_without_clear_request"),
            ("I love 9h", "clarify", "temporal_signal_without_clear_request"),
            ("My backup runs at 9h.", "clarify", "temporal_signal_without_clear_request"),
            ("What is 9h?", "reject", "informational_question_not_schedule_request"),
        ]
        for text, expected, reason in cases:
            with self.subTest(text=text):
                result = self.classify_with_time(text, False)
                self.assertTrue(result["features"]["has_temporal_signal"])
                self.assertFalse(result["features"]["duckling_has_time"])
                self.assertEqual(expected, result["decision"], result["reason"])
                self.assertEqual(reason, result["reason"])


if __name__ == "__main__":
    unittest.main()
