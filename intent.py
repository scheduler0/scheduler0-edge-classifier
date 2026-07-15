import json
import os
import re
import requests
import spacy

# Duckling runs on the EC2 host, not in this container; start.sh exports
# DUCKLING_URL with the host private IP. Fall back to localhost for local dev.
DUCKLING_URL = os.environ.get("DUCKLING_URL", "http://127.0.0.1:8000/parse")
nlp = spacy.load("en_core_web_sm")


REQUEST_STARTERS = re.compile(
    r"^\s*(please|can you|could you|would you|will you|i need you to|i want you to|help me|set up|create|schedule)\b",
    re.I,
)

QUESTION_REQUEST = re.compile(
    r"^\s*(can|could|would|will)\s+you\b",
    re.I,
)

INFO_QUESTION = re.compile(
    r"^\s*(what|why|how|when|where|who|which)\b",
    re.I,
)

RECURRENCE_PATTERN = re.compile(
    r"\b(every|each)\s+"
    r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"day|week|month|year|morning|afternoon|evening|night|weekday|weekend)s?\b",
    re.I,
)

NEGATION_PATTERN = re.compile(
    r"\b(don't|do not|dont|never|stop|cancel|remove)\b",
    re.I,
)


def duckling_parse(text):
    r = requests.post(
        DUCKLING_URL,
        data={
            "locale": "en_GB",
            "text": text,
            "dims": '["time","duration"]',
        },
        timeout=3,
    )
    r.raise_for_status()
    return r.json()


def root_token(doc):
    return next((t for t in doc if t.dep_ == "ROOT"), None)


def has_subject(doc):
    return any(t.dep_ in {"nsubj", "nsubjpass", "expl"} for t in doc)


def has_second_person_subject(doc):
    return any(
        t.dep_ in {"nsubj", "nsubjpass"}
        and t.text.lower() == "you"
        for t in doc
    )


def has_non_request_subject(doc):
    return any(
        t.dep_ in {"nsubj", "nsubjpass", "expl"}
        and t.text.lower() not in {"you"}
        for t in doc
    )


def duckling_has_time(entities):
    return any(e.get("dim") in {"time", "duration"} for e in entities)


def has_temporal_signal(text, entities):
    return duckling_has_time(entities) or bool(RECURRENCE_PATTERN.search(text))


def looks_like_request(doc, text):
    root = root_token(doc)

    if root is None:
        return False

    # "Can you send me a digest every Friday?"
    if QUESTION_REQUEST.search(text) and has_second_person_subject(doc):
        return True

    # "Please notify me tomorrow"
    # "I need you to remind me tomorrow"
    if REQUEST_STARTERS.search(text):
        return True

    # Imperative command:
    # "Remind me tomorrow"
    # "Email Acme every Monday"
    # Usually root verb with no explicit subject.
    if root.pos_ == "VERB" and not has_subject(doc):
        return True

    return False


def looks_like_declarative_statement(doc, text):
    root = root_token(doc)

    if root is None:
        return False

    # Do not treat request-shaped questions as declarative just because
    # they contain subject "you".
    if QUESTION_REQUEST.search(text) or REQUEST_STARTERS.search(text):
        return False

    # "I love waking up every Monday at 9am"
    # "My backup runs every day"
    # "We meet every Friday"
    if has_non_request_subject(doc) and root.pos_ in {"VERB", "AUX"}:
        return True

    return False


def looks_like_info_question(text):
    return bool(INFO_QUESTION.search(text))


def has_negation(text, doc):
    if NEGATION_PATTERN.search(text):
        return True

    return any(t.dep_ == "neg" for t in doc)


def classify(text):
    doc = nlp(text)
    entities = duckling_parse(text)

    root = root_token(doc)

    temporal = has_temporal_signal(text, entities)
    request = looks_like_request(doc, text)
    declarative = looks_like_declarative_statement(doc, text)
    info_question = looks_like_info_question(text)
    negated = has_negation(text, doc)

    if info_question:
        decision = "reject"
        reason = "informational_question_not_schedule_request"

    elif negated and temporal:
        decision = "clarify"
        reason = "negated_schedule_like_request_needs_intent_confirmation"

    elif temporal and request and not declarative:
        decision = "allow"
        reason = "request_with_temporal_signal"

    elif temporal and not request:
        decision = "clarify"
        reason = "temporal_signal_without_clear_request"

    elif request and not temporal:
        decision = "clarify"
        reason = "request_without_temporal_signal"

    else:
        decision = "reject"
        reason = "not_a_schedule_request"

    return {
        "text": text,
        "decision": decision,
        "reason": reason,
        "features": {
            "has_temporal_signal": temporal,
            "duckling_has_time": duckling_has_time(entities),
            "recurrence_regex_match": bool(RECURRENCE_PATTERN.search(text)),
            "looks_like_request": request,
            "looks_like_declarative": declarative,
            "looks_like_info_question": info_question,
            "has_negation": negated,
            "root": {
                "text": root.text if root else None,
                "lemma": root.lemma_ if root else None,
                "pos": root.pos_ if root else None,
                "dep": root.dep_ if root else None,
            },
            "has_subject": has_subject(doc),
            "has_second_person_subject": has_second_person_subject(doc),
            "has_non_request_subject": has_non_request_subject(doc),
        },
        "duckling_entities": [
            {
                "body": e.get("body"),
                "dim": e.get("dim"),
                "value": e.get("value"),
            }
            for e in entities
        ],
        "tokens": [
            {
                "text": t.text,
                "lemma": t.lemma_,
                "pos": t.pos_,
                "dep": t.dep_,
                "head": t.head.text,
            }
            for t in doc
        ],
    }


if __name__ == "__main__":
    tests = [
        # allow
        "Remind me every Monday at 9am.",
        "Email Acme a recap every Monday at 9am.",
        "Can you send me a digest every Friday?",
        "Please notify the team tomorrow morning.",
        "Create a reminder for next Friday at 9am.",

        # clarify
        "Next Friday at 9am.",
        "Please email John.",
        "Don't remind me every Monday at 9am.",

        # reject
        "I love waking up every Monday at 9am.",
        "My backup runs every day.",
        "We meet every Friday at 10am.",
        "What is Kubernetes?",
        "When is Easter next year?",
    ]

    for test in tests:
        print(json.dumps(classify(test), indent=2))
        print("-" * 80)