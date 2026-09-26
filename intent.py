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
    r"what about|can we|could we|shall we|should we|"
    r"do you mind"
    r")\b"
    r"|\blet'?s(?!\s+(?:us|me|him|her|them)\b)(?=\s)"
    r"|\blet\s+us(?!\s+(?:me|him|her|them)\b)(?=\s)",
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
    r"\bi(?:['']d|\s+would)\s+like\b(?:\s+you)?\s+to\s+"
    r"(?:schedule|book|remind|set|create|send|email|notify|ping|meet|move|reschedule)\b"
    r"|"
    r"\bi(?:['']d|\s+would)\s+like\s+(?:a|an|the)\s+"
    r"(?:reminder|meeting|call|sync|event|appointment)\b"
    r"|"
    r"\bi\s+(?:want|need)\b(?:\s+you)?\s+to\s+"
    r"(?:schedule|book|remind|set|create|send|email|notify|ping|meet|move|reschedule)\b"
    r"|"
    r"\bi\s+(?:want|need)\s+(?:a|an|the)\s+"
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

# The loaded spaCy model is English, so French part-of-speech tags are not
# usable (they turn "Pourquoi" and "Ne" into imperative verbs). French
# decisions use these lexicons and the same allow/clarify/reject order.
FRENCH_MARKER = re.compile(
    r"[àâäçéèêëîïôùûüœæ]"
    r"|\b\d{1,2}\s*h\d{0,2}\b"
    r"|\btous\s+les\b|\btoutes\s+les\b"
    r"|\bs['’]il\b|\bqu['’]est\b|\bc['’]est\b"
    r"|\b(?:peux|pouvez|pourrais|pourriez|voudrais|aimerais|veuillez|"
    r"rappelle|rappelez|rappeler|rappel|planifie|planifiez|planifier|"
    r"previens|prevenez|previent|envoi\w*|programmez|annule|annuler|annulez|"
    r"arrete|arretez|supprime|supprimez|supprimer|demain|aujourd|"
    r"lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche|rendez-vous|"
    r"courriel|pourquoi|combien|quand|quel|quelle|quels|quelles|est-ce|"
    r"chaque|svp|stp|oublie|oubliez|bonjour|hebdomadaire|matin|soir|soirs|"
    r"nuit|nuits|semaine|semaines|fonctionne|calendrier|sauvegarde|ajoute|"
    r"ajoutez|bloque|bloquez|fixez|merci)\b",
    re.I,
)

FRENCH_INFO_QUESTION = re.compile(
    r"^\s*(?:[àa]\s+)?"
    r"(?:qu['’]est-ce|c['’]est\s+quoi|pourquoi|comment|quand|o[uù]|qui|"
    r"quel(?:le)?s?|combien)\b",
    re.I,
)

FRENCH_YESNO = re.compile(r"^\s*est-ce que\b", re.I)

FRENCH_YESNO_REQUEST = re.compile(
    r"^\s*est-ce que\s+(?:tu|vous|on)\s+"
    r"(?:peux|pouvez|peut|pourrais|pourriez|pourrait)\b",
    re.I,
)

FRENCH_EXPLAIN = re.compile(
    r"\b(?:expliquer|explique(?:z|-moi)?|d[eé]cri(?:s|re|vez)|"
    r"dis-moi\s+(?:pourquoi|comment|quand|o[uù]|qui))\b",
    re.I,
)

FRENCH_POLITE = re.compile(
    r"\b(?:s['’]il\s+(?:te|vous)\s+pla[iî]t|svp|stp|merci\s+d['’e]|veuillez|"
    r"peux[-\s]tu|pouvez[-\s]vous|pourrais[-\s]tu|pourriez[-\s]vous|"
    r"tu\s+peux|vous\s+pouvez|tu\s+pourrais|vous\s+pourriez|veux[-\s]tu|"
    r"j['’]ai\s+besoin|je\s+voudrais|j['’]aimerais|je\s+veux(?!\s+dire\b)|"
    r"je\s+souhaite|il\s+faut|nous\s+devrions|on\s+devrait|on\s+pourrait|"
    r"nous\s+pourrions|aide[-\s]moi|aidez[-\s]moi)\b",
    re.I,
)

FRENCH_IMPERATIVE = re.compile(
    r"(?:^|[,:;]\s*|«\s*|\"\s*)"
    r"(?:rappelle(?:z)?(?:-(?:moi|nous|le|la|les))?|"
    r"pr[eé]vien[st]|pr[eé]venez|envoie(?:z)?|cr[eé]e(?:z)?|"
    r"planifie(?:z)?|"
    r"programme(?:z)?(?=\s+(?:la|le|les|un|une|l['’]|moi|nous|ça|ca|ce|cet|cette)\b)|"
    r"bloque(?:z)?(?:-moi)?|ajoute(?:z)?|"
    r"fixe(?:z)?|r[eé]serve(?:z)?|notifie(?:z)?|d[eé]place(?:z)?|"
    r"cale(?:z)?|aide(?:z)?-moi)\b",
    re.I,
)

FRENCH_PROPOSAL = re.compile(
    r"\b(?:on se cale|on cale|on se bloque)\b",
    re.I,
)

FRENCH_CANCEL_REQUEST = re.compile(
    r"(?:^|[,:;]\s*)(?:annule(?:r|z)?|arr[eê]te(?:z)?|supprime(?:r|z)?)\b",
    re.I,
)

# "Pense à me rappeler" is "remember to remind me", same intent as "don't forget".
FRENCH_REMEMBER = re.compile(
    r"(?:^|[,:;]\s*)pense(?:z)?\s+[àa]\b",
    re.I,
)

# "Ne me préviens pas" is still a request; the "ne" keeps the verb off the
# start-of-sentence imperative pattern.
FRENCH_NEGATED_IMPERATIVE = re.compile(
    r"\bne\s+(?:me\s+|nous\s+|lui\s+|leur\s+)?"
    r"(?:rappelle(?:z)?|pr[eé]vien[st]|pr[eé]venez|envoie(?:z)?|notifie(?:z)?|"
    r"programme(?:z)?|planifie(?:z)?|r[eé]serve(?:z)?)\s+"
    r"(?:pas|plus|jamais)\b",
    re.I,
)

# "fais-moi" / "mets" / "note" are requests only next to a scheduling noun.
# Bare "Mets la table demain" is not one.
FRENCH_LIGHT_IMPERATIVE = re.compile(
    r"(?:^|[,:;]\s*)(?:fais-moi|faites-moi|mets|mettez|note(?:z)?(?!\s+que\b))\b",
    re.I,
)

FRENCH_SCHEDULE_NOUN = re.compile(
    r"\b(?:rappels?|rendez-vous|r[eé]unions?|calendrier|notifications?|"
    r"digest|compte-rendu|cr[eé]neaux|cr[eé]neau|alarmes?)\b",
    re.I,
)

FRENCH_SCHEDULE_ACTION = re.compile(
    r"\b(?:rappeler|rappelle(?:z)?|envoyer|envoie(?:z)?|pr[eé]venir|"
    r"pr[eé]vien[st]|pr[eé]venez|notifier|notifie(?:z)?|planifier|"
    r"planifie(?:z)?|programmer|programmez|cr[eé]er|cr[eé]e(?:z)?|"
    r"r[eé]server|r[eé]serve(?:z)?|bloquer|bloque(?:z)?|ajouter|ajoute(?:z)?|"
    r"fixer|fixe(?:z)?|d[eé]placer|d[eé]place(?:z)?|caler|cale(?:z)?)\b",
    re.I,
)

FRENCH_FORGET = re.compile(
    r"\bn['’]oublie(?:z)?\s+pas\b|\bne\s+pas\s+oublier\b|\boublie(?:z)?\s+pas\b",
    re.I,
)

FRENCH_NEGATION = re.compile(
    r"\b(?:jamais|arr[eê]te(?:z)?|annule(?:r|z)?|supprime(?:r|z)?)\b"
    r"|\bne\b.+\b(?:pas|plus|jamais)\b"
    r"|\bn['’]\w+\s+(?:pas|plus|jamais)\b",
    re.I,
)

FRENCH_RECURRENCE = re.compile(
    r"\b(?:tous|toutes)\s+les\s+"
    r"(?:lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche|"
    r"jour|semaine|mois|an|ann[eé]e|matin|apr[eè]s-midi|apres-midi|soir|nuit)s?\b"
    r"|\bchaque\s+"
    r"(?:lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche|"
    r"jour|semaine|mois|matin|apr[eè]s-midi|apres-midi|soir|nuit|premier)\b"
    r"|\bhebdomadaire\b",
    re.I,
)

FRENCH_TIME = re.compile(
    r"\b\d{1,2}\s*h\d{0,2}\b"
    r"|\bdemain(?:\s+(?:matin|soir|apr[eè]s-midi|apres-midi))?\b"
    r"|\baujourd['’]?hui\b"
    r"|\bhier\b"
    r"|\b(?:lundis?|mardis?|mercredis?|jeudis?|vendredis?|samedis?|dimanches?)"
    r"(?:\s+prochain(?:e)?)?\b"
    r"|\bprochaine?\s+(?:semaine|mois|lundi|mardi|mercredi|jeudi|vendredi|"
    r"samedi|dimanche)\b"
    r"|\bdans\s+(?:\d+|une|deux|trois|quatre|cinq|six|sept|huit|neuf|dix)\s+"
    r"(?:minutes?|heures?|jours?|semaines?|mois)\b"
    r"|\b(?:une|deux|trois|\d+)\s+heures?\b"
    r"|\bl['’]ann[eé]e\s+prochaine\b"
    r"|\bann[eé]e\s+prochaine\b"
    r"|\b(?:midi|minuit|matin|soir|apr[eè]s-midi|apres-midi)\b",
    re.I,
)

FRENCH_STATEMENT = re.compile(
    r"^\s*(?:j['’]|je\b|tu\b|il\b|elle\b|on\b|nous\b|ils\b|elles\b|"
    r"mon\b|ma\b|mes\b|notre\b|nos\b|son\b|sa\b|ses\b|leur\b|leurs\b)"
    r"|^\s*(?:le|la|les|un|une|ce|cet|cette)\s+\S+\s+"
    r"(?:est|sont|tourne|part|d[eé]marre|demarre|s['’]ex[eé]cute|s['’]execute)\b"
    r"|^\s*note(?:z)?\s+que\b",
    re.I,
)

# Imperatives the English model actually knows. Used only as a fallback so a
# French clock time ("9h") cannot swallow an otherwise English command.
ENGLISH_IMPERATIVE_LEMMAS = {
    "remind",
    "email",
    "notify",
    "schedule",
    "create",
    "send",
    "book",
    "set",
    "ping",
    "message",
    "call",
    "text",
    "invite",
}


def text_looks_french(text):
    return bool(FRENCH_MARKER.search(text))


def duckling_parse(text):
    # French clock times and weekdays are missed when Duckling is pinned to
    # en_GB, so switch locale only for text that looks French.
    locale = "fr_FR" if text_looks_french(text) else "en_GB"
    r = requests.post(
        DUCKLING_URL,
        data={
            "locale": locale,
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
    # "how about" / "what if" are proposals only when followed by scheduling action.
    # "What if the book club meets..." has "book" but it's not a scheduling proposal.
    match = PROPOSAL_PREFIX.search(text)
    if not match:
        return False
    # Look for explicit scheduling proposal patterns after "how about"/"what if"
    text_after = text[match.end():]
    # Match patterns like "we schedule", "I schedule", "scheduling a", "moving the", etc.
    # This avoids matching "the book club" or "my book" where "book" is a noun
    proposal_pattern = re.compile(
        r"\b(?:we|i)\s+(?:schedule|book|meet|reschedule|move|cancel)\b"
        r"|\b(?:scheduling|booking|meeting|rescheduling|moving|canceling)\s+(?:a|an|the)\b"
        r"|\b(?:schedule|book|meet|reschedule|move|cancel)\s+(?:a|an|the|our|my)\b",
        re.I
    )
    return bool(proposal_pattern.search(text_after))


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
    if has_scheduling_proposal(text):
        return False
    # "should we meet" patterns are proposals only when they specify a time/date
    # "When should we meet?" is informational; "When should we meet Tuesday at 3?" is a proposal
    match = SCHEDULING_PROPOSAL_QUESTION.search(text)
    if match and len(text[match.end():].strip(' ?')) > 0:
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


def looks_like_french_info_question(text):
    if FRENCH_INFO_QUESTION.search(text) or INFO_QUESTION.search(text):
        return True
    if FRENCH_YESNO.search(text) and not FRENCH_YESNO_REQUEST.search(text):
        return True
    # "Peux-tu m'expliquer pourquoi..." asks for information, not a schedule.
    if (
        FRENCH_EXPLAIN.search(text)
        and not FRENCH_IMPERATIVE.search(text)
        and not FRENCH_SCHEDULE_ACTION.search(text)
    ):
        return True
    return False


def looks_like_french_request(text, doc):
    if looks_like_french_info_question(text):
        return False
    if FRENCH_FORGET.search(text) or FORGET_NEGATION.search(text):
        return True
    if (
        FRENCH_POLITE.search(text)
        or FRENCH_IMPERATIVE.search(text)
        or FRENCH_PROPOSAL.search(text)
        or FRENCH_CANCEL_REQUEST.search(text)
        or FRENCH_REMEMBER.search(text)
        or FRENCH_NEGATED_IMPERATIVE.search(text)
        or (
            FRENCH_LIGHT_IMPERATIVE.search(text)
            and FRENCH_SCHEDULE_NOUN.search(text)
        )
        or FRENCH_YESNO_REQUEST.search(text)
        or REQUEST_STARTERS.search(text)
        or QUESTION_REQUEST.search(text)
    ):
        return True

    root = root_token(doc)
    if (
        root is not None
        and root.pos_ == "VERB"
        and not has_subject(doc)
        and root.lemma_.lower() in ENGLISH_IMPERATIVE_LEMMAS
    ):
        return True
    return False


def looks_like_french_declarative(text, doc):
    if looks_like_french_request(text, doc):
        return False
    return bool(FRENCH_STATEMENT.search(text))


def has_french_negation(text):
    # "n'oublie pas de me rappeler" means "don't forget to remind me".
    check = FRENCH_FORGET.sub(" ", text)
    check = FORGET_NEGATION.sub(" ", check)
    return bool(FRENCH_NEGATION.search(check) or NEGATION_PATTERN.search(check))


def has_french_temporal_signal(text, entities):
    return (
        duckling_has_time(entities)
        or bool(FRENCH_RECURRENCE.search(text))
        or bool(FRENCH_TIME.search(text))
        or bool(RECURRENCE_PATTERN.search(text))
    )


def has_negation(text, doc):
    # Strip double negations and discourse "never mind" before looking
    # for a real cancellation of the scheduling intent.
    check_text = FORGET_NEGATION.sub(" ", text)
    check_text_after_never_mind = NEVER_MIND.sub(" ", check_text)
    if NEGATION_PATTERN.search(check_text_after_never_mind):
        return True

    # "Never mind X" without a subsequent request is a cancellation
    never_mind_match = NEVER_MIND.search(text)
    if never_mind_match:
        # Check if there's a clear request after "never mind"
        after_text = text[never_mind_match.end():].strip()
        # If it continues with a comma and more substantial content (request), it's a retraction + new request
        # If it's just "the X reminder" or similar, it's a cancellation
        if after_text and not after_text.startswith(','):
            # "Never mind the Monday reminder" - cancellation
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
    french = text_looks_french(text)

    if french:
        temporal = has_french_temporal_signal(text, entities)
        request = looks_like_french_request(text, doc)
        declarative = looks_like_french_declarative(text, doc)
        info_question = looks_like_french_info_question(text)
        info_request = False  # French doesn't have info_request check yet
        negated = has_french_negation(text)
        recurrence = bool(
            FRENCH_RECURRENCE.search(text) or RECURRENCE_PATTERN.search(text)
        )
    else:
        temporal = has_temporal_signal(text, entities)
        request = looks_like_request(doc, text)
        declarative = looks_like_declarative_statement(doc, text)
        info_question = looks_like_info_question(text)
        info_request = looks_like_info_request(doc)
        negated = has_negation(text, doc)
        recurrence = bool(RECURRENCE_PATTERN.search(text))

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
            "recurrence_regex_match": recurrence,
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
