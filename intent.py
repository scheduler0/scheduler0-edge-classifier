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

# "Email is down" / "Schedule is full" / "Set up is hard" name a fact.
# The leading command word is the subject of a copula, not an imperative.
COPULAR_COMMAND = re.compile(
    r"^\s*(?:email|schedule|create|set\s+up)\s+"
    r"(?:is|are|was|were|isn't|aren't|wasn't|weren't)\b",
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

# "Couldn't you" / "Won't you" / "Can't you" / "Might you" are polite
# challenges, not refusals. "can't" and "won't" are irregular contractions.
_MODAL_YOU_RE = (
    r"\b(?:can't|won't|(?:can|could|would|will|should|might|may)"
    r"(?:n't|\s+not)?)\s+you\b"
)
QUESTION_REQUEST = re.compile(_MODAL_YOU_RE, re.I)
POLITE_CHALLENGE = re.compile(
    r"\b(?:can't|won't|(?:can|could|would|will|should|might|may)"
    r"(?:n't|\s+not))\s+you\b",
    re.I,
)

INDEFINITE_REQUEST = re.compile(
    r"\b(?:can|could|would|will)\s+(?:someone|somebody|anyone)\b",
    re.I,
)

# "we should meet" proposes a schedule; factual "we meet" does not.
# "we can meet the deadline" is not a meeting, so "meet" cannot take an object.
GROUP_PROPOSAL = re.compile(
    r"\bwe\s+(?:should|could|can|shall|must|ought\s+to|have\s+to|gotta)\s+"
    r"(?:schedule|book|reschedule|move)\b"
    r"|"
    r"\bwe\s+(?:should|could|can|shall|must|ought\s+to|have\s+to|gotta)\s+meet\b"
    r"(?!\s+(?:the|a|an|our|my|your|his|her|their)\b)",
    re.I,
)

# "We need to meet", "I need to be reminded", "the team needs a meeting".
# Kept on scheduling verbs so "I need to leave every Monday" stays a statement.
NEED_TO_SCHEDULE = re.compile(
    r"\b(?:i|we|the\s+team)\s+needs?\s+to\s+(?:meet|schedule|book|reschedule)\b"
    r"|"
    r"\b(?:i|we|the\s+team)\s+needs?\s+(?:a|an|the)\s+"
    r"(?:meeting|reminder|call|sync|appointment)\b"
    r"|"
    r"\bi\s+needs?\s+to\s+be\s+reminded\b"
    r"|"
    r"\bi\s+needs?\s+reminding\b"
    r"|"
    r"\b(?:someone|somebody)\s+should\s+"
    r"(?:remind|email|schedule|book|ping|notify|send)\b"
    r"|"
    r"^\s*(?:someone|somebody)\s+"
    r"(?:remind|email|schedule|book|ping|notify|send)\b",
    re.I,
)

# Polite hedges and colloquial proposals that never use "can you" / "let's".
HEDGED_PROPOSAL = re.compile(
    r"\b(?:wondering|hoping)\b.{0,50}\b(?:we|you|i)\s+(?:could|can|would)\s+"
    r"(?:meet|schedule|book|remind|reschedule)\b"
    r"|"
    r"\bany\s+chance\b.{0,40}\b(?:we|you|i)\s+(?:could|can|would)\s+"
    r"(?:meet|schedule|book|remind|reschedule)\b"
    r"|"
    r"\bmind\s+if\s+(?:we|i|you)\s+(?:meet|schedule|book|reschedule)\b"
    r"|"
    r"\b(?:are|is)\s+(?:you|someone|somebody|anyone)\s+able\s+to\s+"
    r"(?:meet|schedule|book|remind|reschedule)\b"
    r"|"
    r"\bit\s+would\s+be\s+(?:great|nice|good)\s+if\s+(?:we|you)\s+"
    r"(?:could|can|would)\s+(?:meet|schedule|book|remind|reschedule)\b"
    r"|"
    r"\bit\s+would\s+be\s+(?:great|nice|good)\s+if\s+you\s+reminded\b"
    r"|"
    r"\b(?:i\s+)?wish(?:ed)?\b.{0,40}\bwe\s+could\s+"
    r"(?:meet|schedule|book|reschedule)\b"
    r"|"
    r"^\s*(?:mind\s+meeting|mind\s+(?:a|an)\s+(?:call|meeting|sync|chat))\b"
    r"|"
    r"^\s*wanna\s+(?:meet|schedule|book|reschedule)\b"
    r"|"
    r"^\s*fancy\s+(?:a|an)\s+(?:call|meeting|sync|chat)\b"
    r"|"
    r"^\s*up\s+for\s+(?:a|an)\s+(?:call|meeting|sync|chat)\b"
    r"|"
    r"\bi(?:'m|\s+am)\s+down\s+to\s+(?:meet|schedule|book)\b"
    r"(?!\s+(?:the|a|an)\b)"
    r"|"
    r"\bsuppose\s+we\s+(?:meet|schedule|book)\b"
    r"(?!\s+(?:the|a|an)\b)",
    re.I,
)

# "Why don't we meet" / "couldn't we meet" propose a time. The negation is
# rhetorical, unlike "don't remind me" or "why do we meet".
RHETORICAL_PROPOSAL = re.compile(
    r"^\s*why\s+(?:don't|do\s+not|dont|not)\s+(?:we\s+)?"
    r"(?:meet|schedule|book|reschedule|move)\b"
    r"|"
    r"^\s*why\s+(?:don't|do\s+not|dont|not)\s+we\s+do\s+"
    r"(?:tomorrow|today|tonight|monday|tuesday|wednesday|thursday|friday|"
    r"saturday|sunday|next|\d)\b"
    r"|"
    r"^\s*what\s+(?:do\s+you\s+)?say\s+we\s+(?:meet|schedule|book|reschedule|move)\b"
    r"|"
    r"\b(?:could|would|should)(?:n't|\s+not)\s+we\s+"
    r"(?:meet|schedule|book|reschedule|move)\b"
    r"|"
    r"\bdon't\s+you\s+think\s+we\s+should\s+"
    r"(?:meet|schedule|book|reschedule|move)\b"
    r"|"
    r"\bwould(?:n't|\s+not)\s+it\s+be\s+(?:better|possible|good|okay|ok)\s+to\s+"
    r"(?:meet|schedule|book|reschedule)\b",
    re.I,
)

# Weekday or day-part used in "remind me Friday, not Monday".
CONTRASTIVE_TIME_WORDS = {
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
    "tomorrow",
    "today",
    "tonight",
    "morning",
    "afternoon",
    "evening",
    "night",
    "week",
    "weekend",
    "weekday",
}

# "Can the team meet every Friday?" — collective scheduling question.
COLLECTIVE_REQUEST = re.compile(
    r"\b(?:can|could|shall|should|will|would)\s+(?:the\s+)?(?:team|group|crew)\s+"
    r"(?:meet|schedule|book|reschedule)\b",
    re.I,
)

# "I need this done by Friday" is a deadline, not a statement of fact.
DEADLINE_REQUEST = re.compile(
    r"\bi\s+(?:need|want)\b.{0,60}\b(?:done|finished|completed|sent)\s+by\b"
    r"|"
    r"\bi\s+(?:need|want)\b.{0,40}\b(?:scheduled|booked)\b"
    r"|"
    r"\bi\s+(?:need|want)\s+(?:this|it|that)\s+by\b",
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
    r"(?:reminder|meeting|call|sync|event|appointment)\b"
    r"|"
    r"\bi(?:['’]d|\s+would)\s+appreciate\s+(?:a|an|the)\s+"
    r"(?:reminder|meeting|call|sync|appointment)\b"
    r"|"
    r"\bi\s+should\s+be\s+reminded\b"
    r"|"
    r"\bi(?:['’]d|\s+would)\s+love\s+to\s+"
    r"(?:schedule|book|remind|set|meet|move|reschedule)\b"
    r"|"
    r"\bi(?:['’]d|\s+would)\s+love\s+(?:a|an|the)\s+"
    r"(?:reminder|meeting|call|sync|appointment)\b"
    r"|"
    r"\bi\s+wanted\s+to\s+"
    r"(?:schedule|book|remind|meet|move|reschedule)\b",
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
    r"\bwould\s+it\s+be\s+(?:okay|ok|alright|fine)\s+to\s+(?:meet|schedule|book|remind)\b"
    r"|"
    r"\b(?:shall|should|can|could|may|might)\s+i\s+"
    r"(?:book|schedule|reschedule|meet|remind|email|send|ping|notify|call)\b"
    r"|"
    r"\b(?:can|could|may|might)\s+i\s+(?:get|have)\s+(?:a|an|the)\s+"
    r"(?:reminder|meeting|call|sync|appointment)\b"
    r"|"
    r"\b(?:be|make)\s+sure\s+to\s+"
    r"(?:remind|email|schedule|book|ping|notify|send|call|meet)\b"
    r"|"
    r"\bblock(?:\s+off)?\s+(?:\d+\s+)?(?:minutes?|hours?|mins?)\b"
    r"|"
    r"\bclear\s+(?:my|the|our)\s+calendar\b"
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

# "When can you remind me tomorrow?" names the ask. "When can you remind me?" does not.
WHEN_YOU_SCHEDULE = re.compile(
    r"^\s*when\s+(?:can|could|would|will)\s+you\s+"
    r"(?:remind|schedule|book|email|ping|notify|send|call|meet)\b",
    re.I,
)

_LEFTOVER_TIME_WORD = re.compile(
    r"\b(?:tomorrow|today|tonight|monday|tuesday|wednesday|thursday|friday|"
    r"saturday|sunday|morning|afternoon|evening|night|daily|weekly|hourly|"
    r"nightly|every|each|next|\d)\b",
    re.I,
)

# "every other Monday" / "every half hour" still count when Duckling is empty.
# Bare "daily" / "hourly" / "weekly" are the same recurrence signal.
RECURRENCE_PATTERN = re.compile(
    r"\b(every|each)\s+"
    r"(?:(?:other|single|half|\d+|two|three|four|five)\s+)?"
    r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"day|week|month|year|morning|afternoon|evening|night|weekday|weekend|"
    r"hour|minute|quarter)s?\b"
    r"|"
    r"\b(?:hourly|daily|weekly|nightly|monthly|yearly|annually|biweekly|fortnightly)\b",
    re.I,
)

# "stop by" means visit, not cancel. "stop reminding" is still negation.
NEGATION_PATTERN = re.compile(
    r"\b(don't|do not|dont|never|stop(?!\s+by)|cancel|remove|"
    r"quit(?=\s+(?:remind|email|schedul|send|notify|ping)))\b",
    re.I,
)

# "don't forget to X" and "don't let me forget X" are double negations
# meaning "please remember to X".
FORGET_NEGATION = re.compile(
    r"\b(?:don'?t|do\s+not|dont)\s+(?:(?:let|make)\s+me\s+)?forget\b",
    re.I,
)

# "Don't let me miss the 3pm demo" is the same reminder, not a cancellation.
MISS_NEGATION = re.compile(
    r"\b(?:don'?t|do\s+not|dont)\s+let\s+me\s+miss\b",
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

# "You need to remind me" / "someone should remind me" is a directive.
# "You need to leave every Monday" is not: leave/attend stay off this list.
OBLIGATION_REQUEST = re.compile(
    r"\byou\s+(?:need|have|ought)\s+to\s+"
    r"(?:remind|email|schedule|book|ping|notify|send|call|meet)\b"
    r"|"
    r"\byou\s+gotta\s+(?:remind|email|schedule|book|ping|notify|send|call|meet)\b"
    r"|"
    r"\byou(?:['’]ve|\s+have)\s+got\s+to\s+"
    r"(?:remind|email|schedule|book|ping|notify|send|call)\b"
    r"|"
    r"\byou\s+should\s+(?:remind|email|schedule|book|ping|notify|send|call|meet)\b"
    r"|"
    r"\b(?:someone|somebody)\s+should\s+remind\b"
    r"|"
    r"\bi\s+need\s+to\s+be\s+reminded\b"
    r"|"
    r"\bi\s+need\s+reminding\b",
    re.I,
)

# Headlines that only borrow a command word: "Schedule conflicts every Monday",
# "Email traffic spikes", "Book sales peak", "Create buttons appear".
# The root is an event predicate, not the thing being scheduled ("Email the recap").
EVENT_HEADLINE_LEMMAS = {
    "conflict",
    "spike",
    "peak",
    "surge",
    "rise",
    "climb",
    "jump",
    "appear",
    "happen",
    "occur",
    "increase",
    "change",
    "double",
    "triple",
    "drop",
    "fall",
    "grow",
    "shrink",
    "decline",
    "decrease",
    "plunge",
    "dip",
    "crash",
    "soar",
    "tumble",
    "vanish",
    "disappear",
}

# Looking up a calendar, or a rhetorical "can you believe", is not a request
# to create one. "Show me how to book" asks for instructions.
INFO_LOOKUP = re.compile(
    r"\b(?:show|list|display|pull\s+up|walk)\b.{0,50}"
    r"\b(?:schedules?|calendars?|agendas?|meetings)\b"
    r"|"
    r"\bcheck\b.{0,40}\b(?:schedules?|calendars?|agendas?)\b"
    r"|"
    r"\b(?:can|could|would|will)(?:n't|\s+not)?\s+you\s+(?:believe|imagine)\b"
    r"|"
    r"\bhow\s+to\s+(?:schedule|book|reschedule|set\s+up|cancel)\b"
    r"|"
    r"\bhelp\s+me\s+(?:understand|know|learn|figure|see\s+why)\b"
    r"|"
    r"\blet\s+me\s+know\b"
    r"|"
    r"\bnote\s+that\b",
    re.I,
)

# The loaded spaCy model is English, so French part-of-speech tags are not
# usable (they turn "Pourquoi" and "Ne" into imperative verbs). French
# decisions use these lexicons and the same allow/clarify/reject order.
_FRENCH_LEXICON_RE = (
    r"\btous\s+les\b|\btoutes\s+les\b"
    r"|\bs['’]il\b|\bqu['’]est\b|\bc['’]est\b"
    r"|\b(?:peux|pouvez|pourrais|pourriez|voudrais|aimerais|veuillez|"
    r"rappelle|rappelez|rappeler|rappel|planifie|planifiez|planifier|"
    r"previens|prevenez|previent|envoi\w*|programmez|annule|annuler|annulez|"
    r"arrete|arretez|supprime|supprimez|supprimer|demain|aujourd|"
    r"lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche|rendez-vous|"
    r"courriel|pourquoi|combien|quand|quel|quelle|quels|quelles|est-ce|"
    r"chaque|svp|stp|oublie|oubliez|bonjour|hebdomadaire|matin|soir|soirs|"
    r"nuit|nuits|semaine|semaines|fonctionne|calendrier|sauvegarde|ajoute|"
    r"ajoutez|bloque|bloquez|fixez|merci)\b"
)

# Diacritics ("café") and "9h" clocks also show up in English. The lexicon
# portion is what actually decides a French sentence once those are removed.
FRENCH_MARKER = re.compile(
    r"[àâäçéèêëîïôùûüœæ]"
    r"|\b\d{1,2}\s*h\d{0,2}\b"
    r"|" + _FRENCH_LEXICON_RE,
    re.I,
)
_FRENCH_LEXICON = re.compile(_FRENCH_LEXICON_RE, re.I)
_CLOCK_TIME = re.compile(r"\b\d{1,2}\s*h\d{0,2}\b", re.I)
_DIACRITIC = re.compile(r"[àâäçéèêëîïôùûüœæ]", re.I)
_ENGLISH_SYNTAX = re.compile(
    r"\b(?:the|my|your|our|their|is|are|was|were|every|each|"
    r"remind|please|let'?s|can|could|would|should|how|we|"
    r"tomorrow|today|tonight|meeting|schedule|email|i)\b",
    re.I,
)
_ENGLISH_TEMPORAL = re.compile(
    r"\b(?:tomorrow|today|tonight|yesterday|every|each|daily|weekly|monthly)\b",
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
    r"^\s*est-ce qu?[''e](?:\s+)?(?:tu|vous|on)\s+"
    r"(?:peux|pouvez|peut|pourrais|pourriez|pourrait)\b",
    re.I,
)

FRENCH_EXPLAIN = re.compile(
    r"\b(?:expliquer|explique(?:z|-moi)?|d[eé]cri(?:s|re|vez)|"
    r"dis-moi\s+(?:pourquoi|comment|quand|o[uù]|qui)|"
    r"me\s+dire\s+(?:pourquoi|comment|quand|o[uù]|qui|l['’]heure|si))\b",
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
    r"appel(?:le(?:s|z)?|ez)(?:-(?:moi|nous))?|"
    r"pr[eé]vien[st]|pr[eé]venez|envoie(?:z)?|cr[eé]e(?:z)?|"
    r"planifie(?:z)?|"
    r"programme(?:z)?(?=\s+(?:la|le|les|un|une|l['’]|moi|nous|ça|ca|ce|cet|cette)\b)|"
    r"bloque(?:z)?(?:-moi)?|ajoute(?:z)?|"
    r"fixe(?:z)?|r[eé]serve(?:z)?|notifie(?:z)?|d[eé]place(?:z)?|"
    r"cale(?:z)?|aide(?:z)?-moi)\b",
    re.I,
)

FRENCH_PROPOSAL = re.compile(
    r"\b(?:on se cale|on cale|on se bloque)\b"
    # "On se parle demain ?" proposes a call. Without "?" it is a statement.
    r"|\bon\s+se\s+parle\b[^?\n]{0,80}\?"
    r"|\bon\s+s['’]appelle\b[^?\n]{0,80}\?"
    r"|\bon\s+doit\s+se\s+(?:voir|rencontrer|caler|parler|retrouver)\b",
    re.I,
)

# Soft French proposals the imperative/polite lists miss.
FRENCH_SOFT_PROPOSAL = re.compile(
    r"\bserait-il\s+possible\s+de\s+(?:se\s+voir|se\s+retrouver|"
    r"se\s+r[eé]unir|planifier|programmer|r[eé]server|bloquer|caler)\b"
    r"|"
    r"\bet\s+si\s+(?:on|nous)\s+(?:se\s+)?"
    r"(?:voyait|voyions|retrouvait|retrouvions|r[eé]unissait|calait|planifiait)\b"
    r"|"
    r"\bpourquoi\s+ne\s+pas\s+(?:se\s+voir|se\s+retrouver|se\s+r[eé]unir|"
    r"planifier|se\s+caler|caler)\b"
    r"|"
    r"\bje\s+(?:te|vous)\s+propose\b"
    r"|"
    r"\bça\s+(?:te|vous)\s+dit\b"
    r"|"
    r"\bon\s+se\s+(?:voit|retrouve|r[eé]unit)\b[^.?!]*\?"
    r"|"
    r"\bdis-moi\s+de\s+(?:bloquer|r[eé]server|planifier|rappeler|"
    r"programmer|caler|d[eé]placer)\b"
    r"|"
    r"\bfaudrait\b.{0,80}\b(?:se\s+voir|se\s+voie|se\s+retrouver|"
    r"se\s+r[eé]unir|planifier|r[eé]server|caler|rappeler)\b"
    r"|"
    r"\b(?:ça|ca|ce)\s+serait\s+possible\s+de\s+(?:se\s+voir|se\s+retrouver|"
    r"se\s+r[eé]unir|planifier|r[eé]server|caler)\b",
    re.I,
)

# "Tu dois me rappeler" / "Faut me rappeler" direct the listener to schedule.
# "Tu dois partir demain" has no scheduling verb and stays a statement.
FRENCH_OBLIGATION = re.compile(
    r"\b(?:tu|vous)\s+(?:dois|devez)\s+(?:me\s+|nous\s+)?"
    r"(?:rappeler|appeler|r[eé]server|planifier|pr[eé]venir)\b"
    r"|"
    r"\bfaut\b.{0,60}\brappel",
    re.I,
)

FRENCH_CANCEL_REQUEST = re.compile(
    r"(?:^|[,:;]\s*)(?:annule(?:r|z)?|arr[eê]te(?:z)?|supprime(?:r|z)?)\b",
    re.I,
)

# "Pense à me rappeler" is "remember to remind me", same intent as "don't forget".
FRENCH_REMEMBER = re.compile(
    r"(?:^|[,:;]\s*)pense(?:z)?\s+[àa]\b"
    r"|(?:^|[,:;]\s*)souviens-toi\b"
    r"|^\s*souvenez-vous\b",
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
    r"|^\s*(?:le|la|les|un|une|ce|cet|cette)\s+(?:\S+\s+){1,6}"
    r"(?:est|sont|tourne|part|d[eé]marre|demarre|s['’]ex[eé]cute|s['’]execute|"
    r"ouvre|ouvrent|ferme|ferment)\b"
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


def should_use_french_locale(text):
    """Determine if Duckling should use fr_FR locale.
    
    This is separate from text_looks_french() because some English texts with
    "9h" should use fr_FR for Duckling (to parse the clock time) but still use
    English classification patterns.
    """
    if not FRENCH_MARKER.search(text):
        return False
    # English temporal words mean English locale can handle it
    stripped = _DIACRITIC.sub("", _CLOCK_TIME.sub(" ", text))
    if _ENGLISH_TEMPORAL.search(stripped):
        return False
    # French lexicon means French locale
    if _FRENCH_LEXICON.search(stripped):
        return True
    # "9h" clock time should use fr_FR even in English sentences
    if _CLOCK_TIME.search(text):
        return True
    # Diacritics in non-English text should try fr_FR
    # Only return False if it's clearly English (has many English syntax words)
    if _DIACRITIC.search(text):
        # Count English words - if there are many, it's English with a loanword
        english_count = len(_ENGLISH_SYNTAX.findall(stripped))
        if english_count >= 3:
            return False
        # Few/no English words with diacritics → try French locale
        return True
    return False


def text_looks_french(text):
    if not FRENCH_MARKER.search(text):
        return False
    # English temporal words (tomorrow, today, etc.) override "9h" or loanwords.
    # Strip diacritics and clock times to check what remains.
    stripped = _DIACRITIC.sub("", _CLOCK_TIME.sub(" ", text))
    if _ENGLISH_TEMPORAL.search(stripped):
        return False
    # French lexicon words are strong evidence of French.
    if _FRENCH_LEXICON.search(stripped):
        return True
    # At this point: FRENCH_MARKER present (9h or diacritic), no English temporal,
    # no French lexicon. Check if sentence is predominantly English.
    # "Remind me at 9h" is English with French clock → use English classification.
    # "9h" alone or "Rappelle-moi à 9h" is French → use French classification.
    # Non-English text with diacritics → treat as French for classification.
    if _DIACRITIC.search(text):
        # Count English words - if there are many, it's English with a loanword
        english_count = len(_ENGLISH_SYNTAX.findall(stripped))
        if english_count >= 3:
            return False
        # Few/no English words with diacritics → treat as French classification
        return True
    if _ENGLISH_SYNTAX.search(stripped):
        # English syntax words present → English classification
        return False
    return True


def duckling_parse(text):
    # French clock times and weekdays are missed when Duckling is pinned to
    # en_GB, so switch locale when we need French temporal parsing.
    locale = "fr_FR" if should_use_french_locale(text) else "en_GB"
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
    # "How about meeting tomorrow" and "How about a call tomorrow" propose a
    # slot. "How about tomorrow" and "How about meeting notes" do not.
    if re.search(
        r"^\s*how\s+about\s+meeting(?=\s*(?:[?.!]|$)|"
        r"\s+(?:tomorrow|today|tonight|again|at|on|every|each|next|this|"
        r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
        r"morning|afternoon|evening|night|\d)\b)",
        text,
        re.I,
    ):
        return True
    if re.search(
        r"^\s*how\s+about\s+(?:a|an)\s+(?:call|meeting|sync|chat|appointment)\b",
        text,
        re.I,
    ):
        return True
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
        r"|\b(?:we|i)\s+do\s+(?:tomorrow|today|tonight|monday|tuesday|wednesday|"
        r"thursday|friday|saturday|sunday|next|\d)\b"
        r"|\b(?:scheduling|booking|meeting|rescheduling|moving|canceling)\s+(?:a|an|the)\b"
        r"|\b(?:schedule|book|meet|reschedule|move|cancel)\s+(?:a|an|the|our|my)\b",
        re.I
    )
    return bool(proposal_pattern.search(text_after))


def has_request_phrase(text):
    patterns = (
        REQUEST_STARTERS,
        REQUEST_PHRASES,
        QUESTION_REQUEST,
        INDEFINITE_REQUEST,
        GROUP_PROPOSAL,
        NEED_TO_SCHEDULE,
        HEDGED_PROPOSAL,
        RHETORICAL_PROPOSAL,
        COLLECTIVE_REQUEST,
        DEADLINE_REQUEST,
        FIRST_PERSON_SCHEDULE,
        POLITE_INDIRECT,
        OBLIGATION_REQUEST,
        FRENCH_POLITE,
        FRENCH_IMPERATIVE,
        FRENCH_PROPOSAL,
        FRENCH_SOFT_PROPOSAL,
        FRENCH_CANCEL_REQUEST,
    )
    # A copular subject ("Schedule is full") must not count as "schedule ...".
    if COPULAR_COMMAND.search(text):
        patterns = tuple(pattern for pattern in patterns if pattern is not REQUEST_STARTERS)
    return any(pattern.search(text) for pattern in patterns) or has_scheduling_proposal(text)


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
    # Check Duckling entities, recurrence patterns, and French time markers (like "9h")
    return (
        duckling_has_time(entities) 
        or bool(RECURRENCE_PATTERN.search(text))
        or bool(FRENCH_TIME.search(text))
    )


def false_leading_command(doc):
    """A command word used as a noun modifier of an event headline.

    "Schedule conflicts every Monday" and "Email traffic spikes" are facts.
    "Email Acme a recap" stays a command: its root is the recipient or object,
    not an event predicate like conflict/spike/peak/appear.
    """
    first = next((t for t in doc if t.is_alpha), None)
    if first is None:
        return False
    if first.lemma_.lower() not in LEADING_ACTIONS and first.text.lower() not in LEADING_ACTIONS:
        return False
    root = root_token(doc)
    if root is None or root == first:
        return False
    return root.lemma_.lower() in EVENT_HEADLINE_LEMMAS


def _separated_by_break(doc, earlier, later):
    if earlier.i >= later.i:
        return False
    return any(t.text in {",", ":"} for t in doc[earlier.i + 1 : later.i])


def _lead_in_is_quoted_speech(doc, verb):
    """A finite verb before the comma is reported speech, not a vocative."""
    comma_at = max(
        (i for i in range(verb.i) if doc[i].text in {",", ":"}),
        default=None,
    )
    if comma_at is None:
        return False
    for token in doc[:comma_at]:
        if token.pos_ != "VERB":
            continue
        if token.dep_ == "ROOT":
            return True
        if any(child.dep_ in {"nsubj", "nsubjpass"} for child in token.children):
            return True
    return False


def preamble_imperative(doc):
    """Scheduling imperative after a short lead-in.

    "Okay, book the room" and "Reminder: send the report" are requests.
    "Thanks. Remind me" is a new sentence and is left to the root check.
    "She said, remind me" quotes someone else and is not a request.
    """
    for token in doc:
        if token.text in {".", "?", "!"}:
            break
        lemma = token.lemma_.lower()
        surface = token.text.lower()
        if lemma not in LEADING_ACTIONS and surface not in LEADING_ACTIONS:
            continue
        if token.i == 0:
            continue
        if token.dep_ in {"compound", "amod", "nsubj", "nsubjpass", "dobj", "pobj", "attr"}:
            continue
        if not any(part.text in {",", ":"} for part in doc[: token.i]):
            continue
        if _lead_in_is_quoted_speech(doc, token):
            continue
        subjects = [
            child
            for child in token.children
            if child.dep_ in {"nsubj", "nsubjpass", "expl"}
        ]
        if any(not _separated_by_break(doc, subject, token) for subject in subjects):
            continue
        return True
    return False


def looks_like_request(doc, text):
    root = root_token(doc)

    if root is None or false_leading_command(doc):
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
    # "Does Thursday work for a call?" is a question whose root is "do",
    # not an imperative. Emphatic "Do remind me" has root "remind".
    if root.pos_ == "VERB" and not has_subject(doc):
        if not (root.lemma_.lower() == "do" and "?" in text):
            return True

    # "Book conference room B", "Snooze the reminder" — spaCy drops the verb.
    if leading_imperative(doc):
        return True

    # "Okay, book the room" or "Reminder: send the report"
    if preamble_imperative(doc):
        return True

    return False


def looks_like_declarative_statement(doc, text):
    root = root_token(doc)

    if root is None:
        return False

    # Preamble imperatives are requests, not declaratives
    if preamble_imperative(doc):
        return False

    if false_leading_command(doc):
        return True

    # Do not treat request-shaped questions as declarative just because
    # they contain subject "you" or "we".
    if has_request_phrase(text):
        return False

    # "Set up is hard" — spaCy tags the command word as csubj, not nsubj.
    if COPULAR_COMMAND.search(text):
        return True

    # "I love waking up every Monday at 9am"
    # "My backup runs every day"
    # "We meet every Friday"
    if has_non_request_subject(doc) and root.pos_ in {"VERB", "AUX"}:
        return True

    return False


def looks_like_info_question(text):
    # "Why don't we meet" and "what say we meet" are proposals, not questions
    # about an existing schedule. "Why do we meet" still falls through.
    if RHETORICAL_PROPOSAL.search(text):
        return False
    when_request = WHEN_YOU_SCHEDULE.search(text)
    if when_request and _LEFTOVER_TIME_WORD.search(text[when_request.end() :]):
        return False
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


def _negation_is_reminder_content(token):
    """'Remind me not to miss the meeting' negates the payload, not the request."""
    head = token.head
    return head.dep_ in {"xcomp", "ccomp", "advcl"} and head.head.lemma_.lower() in {
        "remind",
        "remember",
        "tell",
    }


def _dep_negation_cancels_schedule(token):
    """A spaCy neg edge cancels the request only when it scopes over a verb.

    'Not sure, but can you remind me' negates an adjective. 'It's not urgent,
    but please remind me' negates the copula. Neither cancels the reminder.
    """
    head = token.head
    if head.pos_ not in {"VERB", "AUX"}:
        return False
    if head.lemma_.lower() == "be":
        return False
    return True


def _bare_not_is_contrastive(doc, token):
    """True when "not" picks an alternate time instead of canceling the ask.

    "Remind me tomorrow, not Monday" and "Not tomorrow — remind me Friday"
    still request a reminder. "I will not attend" does not.
    """
    if token.lemma_.lower() != "not":
        return False
    lowered = token.text.lower()
    if lowered in {"n't", "n't"} or lowered.endswith("n't"):
        return False
    if token.head.pos_ not in {"VERB", "AUX"}:
        return True
    if token.i + 1 >= len(doc):
        return False
    nxt = doc[token.i + 1]
    if nxt.pos_ in {"VERB", "AUX"}:
        return False
    if nxt.lemma_.lower() not in CONTRASTIVE_TIME_WORDS and nxt.pos_ not in {
        "PROPN",
        "NOUN",
        "NUM",
    }:
        return False
    if token.i == 0:
        return True
    return any(part.text in {",", ";", "—", "–", "?", "!"} for part in doc[: token.i])


def looks_like_info_request(doc):
    """Polite asks whose action is only to explain or describe, not to schedule.

    "Can you explain why we meet every Friday?" is informational.
    "Can you show me the schedule" and "can you believe we meet" are too.
    "Tell me to schedule the review" still names a scheduling action.
    """
    if INFO_LOOKUP.search(doc.text):
        return True

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
    # "Pourquoi ne pas se voir demain ?" proposes a meeting.
    if FRENCH_PROPOSAL.search(text) or FRENCH_SOFT_PROPOSAL.search(text):
        return False
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
        or FRENCH_SOFT_PROPOSAL.search(text)
        or FRENCH_CANCEL_REQUEST.search(text)
        or FRENCH_REMEMBER.search(text)
        or FRENCH_NEGATED_IMPERATIVE.search(text)
        or (
            FRENCH_LIGHT_IMPERATIVE.search(text)
            and FRENCH_SCHEDULE_NOUN.search(text)
        )
        or FRENCH_YESNO_REQUEST.search(text)
        or FRENCH_OBLIGATION.search(text)
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
    # "Pourquoi ne pas se voir" is a proposal, not a cancellation.
    check = FRENCH_FORGET.sub(" ", text)
    check = FORGET_NEGATION.sub(" ", check)
    check = FRENCH_SOFT_PROPOSAL.sub(" ", check)
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
    check_text = MISS_NEGATION.sub(" ", check_text)
    check_text_after_never_mind = NEVER_MIND.sub(" ", check_text)
    rhetorical = RHETORICAL_PROPOSAL.search(text)
    if rhetorical:
        check_text_after_never_mind = RHETORICAL_PROPOSAL.sub(
            " ", check_text_after_never_mind
        )
    challenge = POLITE_CHALLENGE.search(text)
    if challenge:
        check_text_after_never_mind = POLITE_CHALLENGE.sub(
            " ", check_text_after_never_mind
        )
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
    miss_request = bool(MISS_NEGATION.search(text))
    never_mind = bool(NEVER_MIND.search(text))
    for token in doc:
        if token.dep_ != "neg":
            continue
        head_lemma = token.head.lemma_.lower()
        if forget_request and head_lemma in {"forget", "let", "make"}:
            continue
        if miss_request and head_lemma in {"miss", "let"}:
            continue
        if never_mind and head_lemma == "mind":
            continue
        if _negation_is_reminder_content(token):
            continue
        if _bare_not_is_contrastive(doc, token):
            continue
        if (
            rhetorical
            and rhetorical.start() <= token.idx < rhetorical.end()
        ):
            continue
        if challenge and challenge.start() <= token.idx < challenge.end():
            continue
        if not _dep_negation_cancels_schedule(token):
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

    elif temporal and declarative and request:
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
