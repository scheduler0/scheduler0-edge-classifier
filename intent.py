import json
import os
import re
import requests
import spacy

# Duckling runs on the EC2 host, not in this container; start.sh exports
# DUCKLING_URL with the host private IP. Fall back to localhost for local dev.
DUCKLING_URL = os.environ.get("DUCKLING_URL", "http://127.0.0.1:8000/parse")
nlp = spacy.load("en_core_web_sm")


# Single-word or start-of-message commands. Kept anchored so a bare
# "schedule" or "email" inside a statement is not treated as a request.
REQUEST_STARTERS = re.compile(
    r"^\s*(please|can you|could you|would you|will you|i need you to|i want you to|help me|set up|create|schedule|email|what about)\b",
    re.I,
)

# Polite and proposal phrases that often follow a greeting or acknowledgment:
# "Hi, can you...", "Thanks, but could you...", "Let's schedule...".
REQUEST_PHRASES = re.compile(
    r"\b("
    r"please|can you|could you|would you|will you|"
    r"i need you to|i want you to|"
    r"help me|"
    r"what about|let's|lets|let us|"
    r"can we|could we|shall we|should we|"
    r"do you mind"
    r")\b",
    re.I,
)

QUESTION_REQUEST = re.compile(
    r"\b(?:can|could|would|will)\s+you\b",
    re.I,
)

INDEFINITE_REQUEST = re.compile(
    r"\b(?:can|could|would|will)\s+(?:someone|somebody|anyone)\b",
    re.I,
)

# "we should meet" proposes a schedule; factual "we meet" does not.
GROUP_PROPOSAL = re.compile(
    r"\bwe\s+should\s+(?:meet|schedule|book|reschedule|move)\b",
    re.I,
)

# "Can the team meet every Friday?" — collective scheduling question.
COLLECTIVE_REQUEST = re.compile(
    r"\b(?:can|could|shall|should|will|would)\s+(?:the\s+)?(?:team|group|crew)\s+"
    r"(?:meet|schedule|book|reschedule)\b",
    re.I,
)

# "I need this done by Friday" is a deadline, not a statement of fact.
DEADLINE_REQUEST = re.compile(
    r"\bi\s+(?:need|want)\b.{0,60}\b(?:done|finished|completed|sent)\s+by\b",
    re.I,
)

# "I'd like to schedule", "I want a reminder", "I need a meeting".
FIRST_PERSON_SCHEDULE = re.compile(
    r"\bi(?:['’]?d|\s+would)?\s+(?:like|want|need)\b(?:\s+you)?\s+to\s+"
    r"(?:schedule|book|remind|set|create|send|email|notify|ping|meet|move|reschedule)\b"
    r"|"
    r"\bi(?:['’]?d|\s+would)?\s+(?:like|want|need)\s+(?:a|an|the)\s+"
    r"(?:reminder|meeting|call|sync|event|appointment)\b",
    re.I,
)

# "how about" / "what if" are proposals only when they name a scheduling act.
# "What if it rains every Monday?" stays informational.
PROPOSAL_PREFIX = re.compile(
    r"\b(?:how\s+about|what\s+if)\b",
    re.I,
)

SCHEDULE_CONTENT = re.compile(
    r"\b(?:meet|meeting|schedule|scheduling|book|reschedule|move|remind|reminder|"
    r"snooze|cancel|ping|email|notify|sync|standup|call|appointment)\b",
    re.I,
)

POLITE_INDIRECT = re.compile(
    r"\byou\s+(?:could|can|would)\s+(?:book|schedule|remind|send|email|move|reschedule|ping|notify)\b"
    r"|"
    r"\b(?:would\s+it\s+be\s+possible|is\s+it\s+possible)\s+to\s+(?:meet|schedule|book|remind)\b"
    r"|"
    r"\bcircle\s+back\b",
    re.I,
)

INFO_QUESTION = re.compile(
    r"^\s*(what(?!\s+about)|why|how|when|where|who|which)\b",
    re.I,
)

# "When should we meet Tuesday at 3?" proposes a meeting.
# "When is Easter?" does not.
SCHEDULING_PROPOSAL_QUESTION = re.compile(
    r"\b(?:should|shall|can|could)\s+we\s+(?:meet|schedule|book|reschedule|move)\b",
    re.I,
)

RECURRENCE_PATTERN = re.compile(
    r"\b(every|each)\s+"
    r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"day|week|month|year|morning|afternoon|evening|night|weekday|weekend)s?\b",
    re.I,
)

# "stop by" means visit, not cancel. "stop reminding" is still negation.
NEGATION_PATTERN = re.compile(
    r"\b(don't|do not|dont|never|stop(?!\s+by)|cancel|remove)\b",
    re.I,
)

# "don't forget to X" and "don't let me forget X" are double negations
# meaning "please remember to X".
FORGET_NEGATION = re.compile(
    r"\b(?:don'?t|do\s+not|dont)\s+(?:(?:let|make)\s+me\s+)?forget\b",
    re.I,
)

# "Never mind, remind me..." retracts the previous thought, then requests.
NEVER_MIND = re.compile(
    r"\bnever\s+mind\b",
    re.I,
)

# Sentence-initial imperatives spaCy often mistags as NOUN/PROPN ("Book", "Snooze").
LEADING_ACTIONS = {
    "remind",
    "email",
    "schedule",
    "book",
    "snooze",
    "ping",
    "notify",
    "create",
    "reschedule",
    "cancel",
    "add",
    "send",
    "follow",
    "move",
}

INFO_ROOT_LEMMAS = {"tell", "explain", "describe", "define", "summarize"}

SCHEDULE_ACTION_LEMMAS = {
    "remind",
    "schedule",
    "book",
    "create",
    "notify",
    "send",
    "email",
    "ping",
    "reschedule",
    "move",
    "snooze",
    "cancel",
    "meet",
}


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


def _root_subjects(doc):
    root = root_token(doc)
    if root is None:
        return []
    return [
        t
        for t in doc
        if t.head == root and t.dep_ in {"nsubj", "nsubjpass", "expl"}
    ]


def has_subject(doc):
    # Only the root's subject counts. A subject inside "because I forget"
    # must not turn "Remind me ..." into a statement.
    return bool(_root_subjects(doc))


def has_second_person_subject(doc):
    return any(
        t.dep_ in {"nsubj", "nsubjpass"}
        and t.text.lower() == "you"
        for t in doc
    )


def has_non_request_subject(doc):
    return any(t.text.lower() != "you" for t in _root_subjects(doc))


def has_scheduling_proposal(text):
    return bool(PROPOSAL_PREFIX.search(text) and SCHEDULE_CONTENT.search(text))


def has_request_phrase(text):
    return any(
        pattern.search(text)
        for pattern in (
            REQUEST_STARTERS,
            REQUEST_PHRASES,
            QUESTION_REQUEST,
            INDEFINITE_REQUEST,
            GROUP_PROPOSAL,
            COLLECTIVE_REQUEST,
            DEADLINE_REQUEST,
            FIRST_PERSON_SCHEDULE,
            POLITE_INDIRECT,
        )
    ) or has_scheduling_proposal(text)


def leading_imperative(doc):
    first = next((t for t in doc if t.is_alpha), None)
    if first is None:
        return False
    if first.text.lower() not in LEADING_ACTIONS and first.lemma_.lower() not in LEADING_ACTIONS:
        return False
    # "Book club meets every Monday" has a real subject on the root.
    if has_non_request_subject(doc):
        return False
    return True


def duckling_has_time(entities):
    return any(e.get("dim") in {"time", "duration"} for e in entities)


def has_temporal_signal(text, entities):
    return duckling_has_time(entities) or bool(RECURRENCE_PATTERN.search(text))


def looks_like_request(doc, text):
    root = root_token(doc)

    if root is None:
        return False

    # "Can you send me a digest every Friday?"
    # "Hi, can you schedule a sync every Monday?"
    if QUESTION_REQUEST.search(text) and has_second_person_subject(doc):
        return True

    # "Please notify me tomorrow"
    # "Let's schedule a meeting for Friday"
    # "I want a reminder every Monday"
    # "I'd like to schedule a call"
    if has_request_phrase(text):
        return True

    # Imperative command:
    # "Remind me tomorrow"
    # "Email Acme every Monday"
    # Usually root verb with no explicit subject.
    if root.pos_ == "VERB" and not has_subject(doc):
        return True

    # "Book conference room B", "Snooze the reminder" — spaCy drops the verb.
    if leading_imperative(doc):
        return True

    return False


def looks_like_declarative_statement(doc, text):
    root = root_token(doc)

    if root is None:
        return False

    # Do not treat request-shaped questions as declarative just because
    # they contain subject "you" or "we".
    if has_request_phrase(text):
        return False

    # "I love waking up every Monday at 9am"
    # "My backup runs every day"
    # "We meet every Friday"
    if has_non_request_subject(doc) and root.pos_ in {"VERB", "AUX"}:
        return True

    return False


def looks_like_info_question(text):
    if not INFO_QUESTION.search(text):
        return False
    # Scheduling proposals that merely open with a wh-word.
    if has_scheduling_proposal(text) or SCHEDULING_PROPOSAL_QUESTION.search(text):
        return False
    return True


def looks_like_info_request(doc):
    """Polite asks whose action is only to explain or describe, not to schedule.

    "Can you explain why we meet every Friday?" is informational.
    "Tell me to schedule the review" still names a scheduling action.
    """
    root = root_token(doc)
    if root is None or root.lemma_.lower() not in INFO_ROOT_LEMMAS:
        return False
    return not any(
        t.head == root
        and t.dep_ == "xcomp"
        and t.lemma_.lower() in SCHEDULE_ACTION_LEMMAS
        for t in doc
    )


def has_negation(text, doc):
    # Strip double negations and discourse "never mind" before looking
    # for a real cancellation of the scheduling intent.
    check_text = FORGET_NEGATION.sub(" ", text)
    check_text = NEVER_MIND.sub(" ", check_text)
    if NEGATION_PATTERN.search(check_text):
        return True

    forget_request = bool(FORGET_NEGATION.search(text))
    never_mind = bool(NEVER_MIND.search(text))
    for token in doc:
        if token.dep_ != "neg":
            continue
        head_lemma = token.head.lemma_.lower()
        if forget_request and head_lemma in {"forget", "let", "make"}:
            continue
        if never_mind and head_lemma == "mind":
            continue
        return True
    return False


def classify(text):
    doc = nlp(text)
    entities = duckling_parse(text)

    root = root_token(doc)

    temporal = has_temporal_signal(text, entities)
    request = looks_like_request(doc, text)
    declarative = looks_like_declarative_statement(doc, text)
    info_question = looks_like_info_question(text)
    info_request = looks_like_info_request(doc)
    negated = has_negation(text, doc)

    if info_question:
        decision = "reject"
        reason = "informational_question_not_schedule_request"

    elif info_request:
        decision = "reject"
        reason = "informational_request_not_schedule_request"

    elif negated and temporal:
        decision = "clarify"
        reason = "negated_schedule_like_request_needs_intent_confirmation"

    elif temporal and declarative:
        decision = "reject"
        reason = "declarative_schedule_not_request"

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
            "looks_like_info_request": info_request,
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
