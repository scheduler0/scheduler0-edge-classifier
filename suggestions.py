"""Deterministic conversation-analysis engine for Scheduler0 Suggestions.

This module analyzes an ordered sequence of conversation messages and produces
structured, explainable suggestions (commitments, requests, deadlines, follow-ups,
etc.) plus the obligations they map to. It reuses the same spaCy model and Duckling
deployment as ``intent.py`` and follows the same conventions: a module-level ``nlp``
singleton, small ``looks_like_*``/``has_*`` helpers, stable string constants, and
plain-dict returns.

The engine is stateless across requests: obligation state (update / complete /
cancel / reschedule) is reconciled only within the message array submitted to a
single ``analyze()`` call. There is no persistence here.

English only: the ``locale`` option accepts empty (default ``en``) or any ``en*``
value; any other value raises :class:`UnsupportedLocaleError`, which the FastAPI
layer surfaces as a ``400 UNSUPPORTED_LOCALE`` error.
"""

import os
import re
import uuid
from datetime import datetime, timedelta, timezone as dt_timezone

import requests

try:  # Python 3.9+
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - fallback for older interpreters
    ZoneInfo = None

# Reuse the singleton spaCy pipeline loaded by intent.py so we do not load the
# model twice inside the same process.
from intent import nlp

# Duckling runs on the EC2 host; start.sh exports DUCKLING_URL with the host
# private IP. Fall back to localhost for local development.
DUCKLING_URL = os.environ.get("DUCKLING_URL", "http://127.0.0.1:8000/parse")

# Engine identity — stored on every response so past decisions can be reproduced.
ENGINE_VERSION = "1.0.0"
RULE_SET_VERSION = "2026-07-01"
SPACY_MODEL = "en_core_web_sm"

# ---------------------------------------------------------------------------
# Suggestion / obligation vocabulary
# ---------------------------------------------------------------------------

TYPE_COMMITMENT = "COMMITMENT"
TYPE_REQUEST = "REQUEST"
TYPE_CHECK_BACK = "CHECK_BACK"
TYPE_DEADLINE = "DEADLINE"
TYPE_FOLLOW_UP = "FOLLOW_UP"
TYPE_REMINDER = "REMINDER"
TYPE_DEPENDENCY = "DEPENDENCY"

STATUS_OPEN = "OPEN"
STATUS_COMPLETED = "COMPLETED"
STATUS_CANCELLED = "CANCELLED"
STATUS_SUPERSEDED = "SUPERSEDED"

ACTOR_UNRESOLVED = "UNRESOLVED"
ACTOR_GROUP = "GROUP"

# Defaults (overridable via request options). Mirrors spec section 36.
DEFAULT_DUE_TIME = "09:00"
DEFAULT_DEADLINE_TIME = "17:00"
DEFAULT_MINIMUM_CONFIDENCE = 0.65
DEFAULT_TIMEZONE = "America/Toronto"
OBLIGATION_MATCH_THRESHOLD = 0.75

# ---------------------------------------------------------------------------
# Locale handling
# ---------------------------------------------------------------------------


class UnsupportedLocaleError(Exception):
    """Raised when a non-English locale is requested.

    The FastAPI layer maps this to a 400 response with code UNSUPPORTED_LOCALE.
    """

    def __init__(self, locale):
        self.locale = locale
        super().__init__(
            f"The locale {locale} is not currently supported. "
            "Suggestions analysis currently only supports English (en*)."
        )


def is_english_locale(locale):
    """Report whether a locale string is English (or empty, which defaults to en)."""
    if locale is None:
        return True
    locale = str(locale).strip().lower()
    return locale == "" or locale.startswith("en")


def resolve_duckling_locale(locale):
    """Normalize a requested locale to a Duckling-supported English locale.

    Raises UnsupportedLocaleError for any non-English locale so the caller can
    reject it before spending spaCy/Duckling work.
    """
    if not is_english_locale(locale):
        raise UnsupportedLocaleError(locale)

    normalized = (locale or "en").strip().replace("-", "_")
    # Duckling ships a handful of English locales; map to the closest supported
    # one and fall back to en_GB (the value the intent classifier already uses).
    supported = {"en_gb", "en_us", "en_au", "en_ca", "en_nz", "en_bz", "en_in"}
    lowered = normalized.lower()
    if lowered in supported:
        return normalized
    return "en_GB"


# ---------------------------------------------------------------------------
# Duckling temporal extraction (per-message reference time + timezone)
# ---------------------------------------------------------------------------


def duckling_parse(text, reftime_ms=None, tz=None, locale="en_GB"):
    """Call Duckling with an explicit reference time and timezone.

    ``reftime_ms`` is epoch milliseconds derived from the message timestamp, and
    ``tz`` is an IANA timezone name. Passing both ensures relative expressions
    ("tomorrow", "in two hours") resolve against the message context rather than
    the server's local clock.
    """
    data = {
        "locale": locale,
        "text": text,
        "dims": '["time","duration"]',
    }
    if reftime_ms is not None:
        data["reftime"] = str(int(reftime_ms))
    if tz:
        data["tz"] = tz

    r = requests.post(DUCKLING_URL, data=data, timeout=3)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Timestamp + timezone helpers
# ---------------------------------------------------------------------------


def parse_timestamp(value):
    """Parse an ISO 8601 timestamp, tolerating a trailing 'Z'."""
    if value is None:
        raise ValueError("timestamp is required")
    if isinstance(value, datetime):
        return value
    s = str(value).strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


def get_zone(tz_name):
    """Return a tzinfo for an IANA name, defaulting to UTC when unavailable."""
    if tz_name and ZoneInfo is not None:
        try:
            return ZoneInfo(tz_name)
        except Exception:
            return dt_timezone.utc
    return dt_timezone.utc


def to_epoch_ms(dt):
    """Convert a datetime to epoch milliseconds (assumes UTC when naive)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=dt_timezone.utc)
    return int(dt.timestamp() * 1000)


def parse_hhmm(value, fallback_hour, fallback_minute):
    try:
        hh, mm = str(value).split(":")
        return int(hh), int(mm)
    except Exception:
        return fallback_hour, fallback_minute


def next_business_day(dt):
    """Return the next weekday (Mon-Fri) after dt, preserving tzinfo."""
    candidate = dt + timedelta(days=1)
    while candidate.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        candidate += timedelta(days=1)
    return candidate


def iso(dt):
    return dt.isoformat()


# ---------------------------------------------------------------------------
# Normalization (spec section 12)
# ---------------------------------------------------------------------------

# Unicode punctuation that we fold to ASCII before rule matching.
UNICODE_PUNCT = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u2026": "...",
    "\u00a0": " ",
}

CONTRACTIONS = {
    "i'll": "i will",
    "we'll": "we will",
    "you'll": "you will",
    "he'll": "he will",
    "she'll": "she will",
    "they'll": "they will",
    "it'll": "it will",
    "i'm": "i am",
    "you're": "you are",
    "we're": "we are",
    "they're": "they are",
    "i've": "i have",
    "we've": "we have",
    "you've": "you have",
    "they've": "they have",
    "can't": "cannot",
    "don't": "do not",
    "won't": "will not",
    "didn't": "did not",
    "doesn't": "does not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "couldn't": "could not",
    "let's": "let us",
    "i'd": "i would",
}


def normalize_unicode(text):
    for bad, good in UNICODE_PUNCT.items():
        text = text.replace(bad, good)
    return text


def expand_contractions(lowered):
    def repl(match):
        return CONTRACTIONS[match.group(0)]

    pattern = re.compile(
        "|".join(re.escape(k) for k in sorted(CONTRACTIONS, key=len, reverse=True))
    )
    return pattern.sub(repl, lowered)


def build_matching_form(original):
    """Lowercased, contraction-expanded, unicode-folded form used for matching."""
    folded = normalize_unicode(original)
    lowered = folded.lower()
    return expand_contractions(lowered)


# ---------------------------------------------------------------------------
# Input parsing (spec section 8) — supports full and minimal message shapes
# ---------------------------------------------------------------------------


def _speaker_fields(raw_speaker):
    """Return (speaker_id, display_name, timezone) from a speaker value.

    Supports both the object form ({"id", "display_name", "timezone"}) and the
    minimal string form (just the display name).
    """
    if isinstance(raw_speaker, dict):
        return (
            raw_speaker.get("id"),
            raw_speaker.get("display_name") or raw_speaker.get("name"),
            raw_speaker.get("timezone"),
        )
    return (None, raw_speaker, None)


def parse_participants(request):
    participants = []
    for p in request.get("participants") or []:
        if isinstance(p, dict):
            participants.append(
                {
                    "id": p.get("id"),
                    "display_name": p.get("display_name") or p.get("name"),
                    "timezone": p.get("timezone"),
                }
            )
        else:
            participants.append({"id": None, "display_name": p, "timezone": None})
    return participants


def parse_messages(request, default_timezone):
    """Normalize messages into an internal representation.

    Each normalized message has: id, speaker_id, speaker_name, timezone, ts
    (datetime), original, matching (lowercased/expanded), and index.
    """
    raw_messages = request.get("messages") or []
    parsed = []
    for idx, raw in enumerate(raw_messages):
        speaker_id, speaker_name, speaker_tz = _speaker_fields(raw.get("speaker"))
        ts = parse_timestamp(raw.get("timestamp"))
        original = raw.get("message") or ""
        parsed.append(
            {
                "id": raw.get("id") or f"msg_{idx + 1:03d}",
                "speaker_id": speaker_id,
                "speaker_name": speaker_name,
                "timezone": speaker_tz or default_timezone,
                "ts": ts,
                "original": original,
                "matching": build_matching_form(original),
                "index": idx,
            }
        )

    # Stable sort by timestamp, preserving original order on ties (spec 12).
    parsed.sort(key=lambda m: (m["ts"], m["index"]))
    return parsed


def infer_participants(messages, explicit):
    """Fill in the participant list from message speakers when not provided."""
    if explicit:
        return explicit
    seen = {}
    for m in messages:
        key = m["speaker_id"] or m["speaker_name"]
        if key and key not in seen:
            seen[key] = {
                "id": m["speaker_id"],
                "display_name": m["speaker_name"],
                "timezone": m["timezone"],
            }
    return list(seen.values())


# ---------------------------------------------------------------------------
# Marker regexes (spec section 15)
# ---------------------------------------------------------------------------

COMMITMENT_MARKERS = re.compile(
    r"\b(i will|i am going to|i can|let me|i plan to|i should be able to|"
    r"i'll|i am about to|i intend to)\b",
    re.I,
)
UNCERTAIN_MARKERS = re.compile(
    r"\b(i might|maybe|i could possibly|if i have time|i may|possibly|perhaps)\b",
    re.I,
)
REQUEST_MARKERS = re.compile(
    r"\b(can you|could you|would you|will you|please|do you mind|can someone|"
    r"could someone)\b",
    re.I,
)
CHECK_BACK_MARKERS = re.compile(
    r"\b(let us revisit|let us talk|revisit|circle back|talk later|discuss "
    r"tomorrow|come back to this|pick this up|let us discuss|let us circle back|"
    r"let us come back|talk about it)\b",
    re.I,
)
DEADLINE_MARKERS = re.compile(
    r"\b(is due|due by|due on|due|need this|needs to be|must be|has to be|by the "
    r"end of|closes|deadline)\b",
    re.I,
)
FOLLOW_UP_MARKERS = re.compile(
    r"\b(still waiting|waiting for|waiting on|have you heard|haven't received|"
    r"have not received|any update|following up|checking in)\b",
    re.I,
)
REMINDER_MARKERS = re.compile(
    r"\b(remind me|remind us|don't let me forget|do not let me forget|remind "
    r"them)\b",
    re.I,
)
DEPENDENCY_MARKERS = re.compile(
    r"\b(once|after|as soon as|when .* (approve|finish|complete|review)|waiting "
    r"for the .* to)\b",
    re.I,
)
COMPLETION_MARKERS = re.compile(
    r"\b(i sent|i have sent|i've sent|sent it|has been sent|i submitted|has been "
    r"submitted|i finished|i completed|i've completed|done|handled it|took care "
    r"of it|i sent the|i've done)\b",
    re.I,
)
CANCELLATION_MARKERS = re.compile(
    r"\b(never mind|nevermind|don't worry about it|do not worry about it|we no "
    r"longer need|no longer need it|cancel that|skip it|forget it|drop it|"
    r"don't worry about it anymore|do not worry about it anymore)\b",
    re.I,
)
RESCHEDULE_MARKERS = re.compile(
    r"\b(instead|actually|move it to|not tomorrow|make that|i'll do it|let's "
    r"make it|change it to|push it to|reschedule)\b",
    re.I,
)
HYPOTHETICAL_MARKERS = re.compile(r"\b(if|unless|assuming|in case|should they|were i to)\b", re.I)
NEGATION_MARKERS = re.compile(r"\b(not|never|no longer|cannot|won't|will not|don't|do not)\b", re.I)
EXAMPLE_MARKERS = re.compile(r"\b(for example|for instance|e\.g\.|such as|might say)\b", re.I)
QUOTED_LINE = re.compile(r'^\s*[">]')


# ---------------------------------------------------------------------------
# spaCy feature extraction (spec section 13)
# ---------------------------------------------------------------------------

FIRST_PERSON = {"i", "we"}
SECOND_PERSON = {"you"}


def _grammatical_person(token_text):
    t = token_text.lower()
    if t in {"i", "we", "me", "us", "my", "our"}:
        return "FIRST"
    if t in {"you", "your"}:
        return "SECOND"
    return "THIRD"


def extract_features(doc):
    """Extract linguistic features from a spaCy doc for the whole message.

    Returns a dict with subject, subject_person, verb, verb_lemma, direct_object,
    indirect_object, negated, modal, imperative, interrogative.
    """
    root = next((t for t in doc if t.dep_ == "ROOT"), None)

    subject = None
    subject_person = None
    for t in doc:
        if t.dep_ in {"nsubj", "nsubjpass", "expl"}:
            subject = t.text
            subject_person = _grammatical_person(t.text)
            break

    verb = None
    verb_lemma = None
    if root is not None and root.pos_ in {"VERB", "AUX"}:
        verb = root.text
        verb_lemma = root.lemma_
    else:
        v = next((t for t in doc if t.pos_ == "VERB"), None)
        if v is not None:
            verb = v.text
            verb_lemma = v.lemma_

    direct_object = _extract_object(doc, {"dobj", "obj"})
    indirect_object = _extract_object(doc, {"dative", "iobj"})

    negated = any(t.dep_ == "neg" for t in doc)
    modal = next((t.text for t in doc if t.tag_ == "MD"), None)
    interrogative = doc.text.strip().endswith("?")
    imperative = (
        root is not None
        and root.pos_ == "VERB"
        and subject is None
        and not interrogative
    )

    return {
        "subject": subject,
        "subject_person": subject_person,
        "verb": verb,
        "verb_lemma": verb_lemma,
        "direct_object": direct_object,
        "indirect_object": indirect_object,
        "negated": negated,
        "modal": modal,
        "imperative": imperative,
        "interrogative": interrogative,
    }


DETERMINERS = {"the", "a", "an", "this", "that", "these", "those", "my", "your", "our", "their"}


def _clean_noun_phrase(text):
    words = text.strip().split()
    while words and words[0].lower() in DETERMINERS:
        words = words[1:]
    return " ".join(words).strip().rstrip(".?!,")


def _extract_object(doc, deps):
    """Find an object token matching deps and return its cleaned noun chunk."""
    for token in doc:
        if token.dep_ in deps:
            # Prefer the enclosing noun chunk for a fuller phrase.
            for chunk in doc.noun_chunks:
                if chunk.start <= token.i < chunk.end:
                    return _clean_noun_phrase(chunk.text)
            return _clean_noun_phrase(token.text)
    return None


# Object fallbacks pulled straight from the matching text when the parse misses.
OBJECT_AFTER = re.compile(
    r"\b(?:waiting for|waiting on|revisit|revisiting|discuss|review|send|sent)\s+"
    r"(?:the\s+|a\s+|an\s+)?([a-z][a-z \-]+?)(?:\s+(?:tomorrow|today|by|next|"
    r"this|on|at|instead|again)\b|[.?!,]|$)",
    re.I,
)


def extract_object(features, matching):
    obj = features.get("direct_object")
    if obj:
        return obj
    m = OBJECT_AFTER.search(matching)
    if m:
        return _clean_noun_phrase(m.group(1))
    return None


# ---------------------------------------------------------------------------
# Actor resolution (spec section 16)
# ---------------------------------------------------------------------------


def _other_participant(participants, speaker_name, speaker_id):
    others = [
        p
        for p in participants
        if (p.get("id") or p.get("display_name")) != (speaker_id or speaker_name)
    ]
    if len(others) == 1:
        return others[0]
    return None


def _named_participant(matching, participants):
    for p in participants:
        name = (p.get("display_name") or "").strip()
        if not name:
            continue
        if re.search(r"@?\b" + re.escape(name.lower()) + r"\b", matching):
            return p
    return None


def resolve_actor(features, matching, message, participants, suggestion_type):
    """Resolve the actor for a suggestion using deterministic precedence.

    Returns a tuple (actor_dict_or_none, requested_by_dict_or_none, resolution_note).
    """
    speaker = {
        "id": message["speaker_id"],
        "display_name": message["speaker_name"],
    }

    if suggestion_type == TYPE_COMMITMENT:
        return speaker, None, "The first-person subject refers to the speaker."

    if suggestion_type == TYPE_REQUEST:
        named = _named_participant(matching, participants)
        if named:
            return (
                {"id": named.get("id"), "display_name": named.get("display_name")},
                speaker,
                "The named participant is the requested actor.",
            )
        # Second person in a direct (2-party) conversation.
        if len(participants) == 2:
            other = _other_participant(participants, message["speaker_name"], message["speaker_id"])
            if other:
                return (
                    {"id": other.get("id"), "display_name": other.get("display_name")},
                    speaker,
                    "In a direct conversation, 'you' refers to the other participant.",
                )
        return (
            {"id": None, "display_name": ACTOR_UNRESOLVED},
            speaker,
            "The requested actor could not be resolved in a group conversation.",
        )

    if suggestion_type in (TYPE_FOLLOW_UP, TYPE_REMINDER):
        return speaker, None, "The speaker is following up / asked to be reminded."

    if suggestion_type == TYPE_CHECK_BACK:
        # A deferred discussion has no single actor (spec 5.3).
        return None, None, "Deferred discussion has no single actor."

    return speaker, None, "Defaulted actor to the speaker."


# ---------------------------------------------------------------------------
# Rule engine (spec section 15)
# ---------------------------------------------------------------------------


def _rule(rule_id, suggestion_type, score, features):
    return {
        "rule_id": rule_id,
        "suggestion_type": suggestion_type,
        "score": score,
        "features": features,
    }


def match_rules(features, matching, message):
    """Return a list of matched rules for a message (may be empty)."""
    matches = []
    first_person = features["subject_person"] == "FIRST"
    has_verb = features["verb_lemma"] is not None

    # Commitment: first-person + future/intentional marker.
    if COMMITMENT_MARKERS.search(matching) and not features["negated"]:
        matches.append(
            _rule(
                "commitment.first_person_future",
                TYPE_COMMITMENT,
                0.45,
                {
                    "first_person_subject": first_person,
                    "action_verb": has_verb,
                    "future_marker": True,
                },
            )
        )

    # Request: imperative / interrogative + request marker.
    if REQUEST_MARKERS.search(matching):
        matches.append(
            _rule(
                "request.second_person",
                TYPE_REQUEST,
                0.45,
                {
                    "request_marker": True,
                    "interrogative": features["interrogative"],
                    "imperative": features["imperative"],
                },
            )
        )

    # Deferred discussion / check-back.
    if CHECK_BACK_MARKERS.search(matching):
        matches.append(
            _rule("check_back.deferred_discussion", TYPE_CHECK_BACK, 0.4, {"defer_marker": True})
        )

    # Reminder request.
    if REMINDER_MARKERS.search(matching):
        matches.append(
            _rule("reminder.explicit", TYPE_REMINDER, 0.5, {"reminder_marker": True})
        )

    # Follow-up / waiting state.
    if FOLLOW_UP_MARKERS.search(matching):
        matches.append(
            _rule("follow_up.waiting_state", TYPE_FOLLOW_UP, 0.4, {"waiting_marker": True})
        )

    # Dependency (conditional / blocked).
    if DEPENDENCY_MARKERS.search(matching):
        matches.append(
            _rule("dependency.blocked_action", TYPE_DEPENDENCY, 0.35, {"dependency_marker": True})
        )

    # Deadline (only when it is not already a request/commitment).
    if DEADLINE_MARKERS.search(matching) and not matches:
        matches.append(
            _rule("deadline.explicit", TYPE_DEADLINE, 0.4, {"deadline_marker": True})
        )

    return matches


def detect_state_change(features, matching):
    """Detect completion / cancellation / reschedule signals for the state engine."""
    if CANCELLATION_MARKERS.search(matching):
        return "cancel"
    if COMPLETION_MARKERS.search(matching) and not features["negated"]:
        return "complete"
    if RESCHEDULE_MARKERS.search(matching):
        return "reschedule"
    return None


# ---------------------------------------------------------------------------
# Temporal resolution (spec section 18)
# ---------------------------------------------------------------------------

GRAIN_TO_PRECISION = {
    "second": "SECOND",
    "minute": "MINUTE",
    "hour": "HOUR",
    "day": "DAY",
    "week": "WEEK",
    "month": "MONTH",
    "year": "YEAR",
}


def _first_time_entity(entities):
    for e in entities:
        if e.get("dim") == "time":
            return e
    return None


def resolve_temporal(entities, matching, message, options):
    """Resolve a temporal expression into a due time.

    Returns a temporal dict or None. When no explicit expression exists, applies
    the deterministic fallback policy and marks is_inferred=True.
    """
    tz_name = message["timezone"]
    zone = get_zone(tz_name)
    ref = message["ts"].astimezone(zone)

    due_hour, due_min = parse_hhmm(options.get("default_due_time", DEFAULT_DUE_TIME), 9, 0)
    dl_hour, dl_min = parse_hhmm(options.get("default_deadline_time", DEFAULT_DEADLINE_TIME), 17, 0)
    is_deadline = bool(re.search(r"\bby\b", matching)) or bool(DEADLINE_MARKERS.search(matching))

    entity = _first_time_entity(entities)
    if entity is not None:
        value = entity.get("value") or {}
        if value.get("type") == "interval":
            # A "by X" deadline closes at the "to" boundary; an open range
            # ("next week", "between 9 and 11") starts at the "from" boundary.
            boundary = (value.get("to") if is_deadline else value.get("from")) or value.get("to") or value.get("from") or {}
            due_raw = boundary.get("value")
            grain = boundary.get("grain", "day")
        else:
            due_raw = value.get("value")
            grain = value.get("grain", "day")

        if due_raw:
            try:
                due_dt = parse_timestamp(due_raw).astimezone(zone)
            except Exception:
                due_dt = None

            if due_dt is not None:
                coarse = grain in ("day", "week", "month", "year")
                at_midnight = due_dt.hour == 0 and due_dt.minute == 0 and due_dt.second == 0

                if is_deadline:
                    # Deadlines normalize to the deadline time-of-day on the target
                    # date unless the message carried an explicit time.
                    if coarse or at_midnight:
                        due_dt = due_dt.replace(hour=dl_hour, minute=dl_min, second=0, microsecond=0)
                    ttype, precision = "DATE", "DAY"
                elif coarse:
                    due_dt = due_dt.replace(hour=due_hour, minute=due_min, second=0, microsecond=0)
                    ttype, precision = "DATE", GRAIN_TO_PRECISION.get(grain, "DAY")
                else:
                    ttype, precision = "TIME", GRAIN_TO_PRECISION.get(grain, "MINUTE")

                return {
                    "source_text": entity.get("body"),
                    "type": ttype,
                    "due_at": iso(due_dt),
                    "timezone": tz_name,
                    "precision": precision,
                    "is_inferred": False,
                }

    # Ambiguous "later" -> low-confidence, still inferred to later today.
    if re.search(r"\blater\b", matching):
        due_dt = ref.replace(second=0, microsecond=0) + timedelta(hours=3)
        return {
            "source_text": "later",
            "type": "TIME",
            "due_at": iso(due_dt),
            "timezone": tz_name,
            "precision": "HOUR",
            "is_inferred": True,
            "inference_policy": "later_today",
            "ambiguous": True,
        }

    # No explicit time -> deterministic fallback (next business day at default time).
    due_dt = next_business_day(ref).replace(hour=due_hour, minute=due_min, second=0, microsecond=0)
    return {
        "source_text": None,
        "type": "DATE",
        "due_at": iso(due_dt),
        "timezone": tz_name,
        "precision": "DAY",
        "is_inferred": True,
        "inference_policy": "next_business_day",
    }


# ---------------------------------------------------------------------------
# Confidence scoring (spec section 21)
# ---------------------------------------------------------------------------


def score_confidence(rule, features, actor, temporal, obj, matching):
    score = 0.35  # base matching rule
    factors = []

    if actor and actor.get("display_name") not in (None, ACTOR_UNRESOLVED, ACTOR_GROUP):
        score += 0.15
        factors.append("explicit_actor")
    if features.get("verb_lemma"):
        score += 0.10
        factors.append("explicit_action")
    if obj:
        score += 0.10
        factors.append("explicit_object")
    if temporal and not temporal.get("is_inferred"):
        score += 0.15
        factors.append("explicit_temporal")
    if rule["features"].get("request_marker") or rule["features"].get("future_marker"):
        score += 0.10
        factors.append("direct_marker")

    if obj and obj.lower() in {"it", "this", "that"}:
        score -= 0.10
        factors.append("ambiguous_pronoun")
    if UNCERTAIN_MARKERS.search(matching):
        score -= 0.15
        factors.append("uncertain_modal")
    if HYPOTHETICAL_MARKERS.search(matching):
        score -= 0.20
        factors.append("hypothetical_condition")
    if actor and actor.get("display_name") in (ACTOR_UNRESOLVED, ACTOR_GROUP):
        score -= 0.15
        factors.append("unresolved_group_actor")

    return max(0.0, min(1.0, round(score, 2))), factors


# ---------------------------------------------------------------------------
# Obligation matching similarity (spec section 20)
# ---------------------------------------------------------------------------


def _object_similarity(a, b):
    if not a or not b:
        return 0.0
    a = a.lower()
    b = b.lower()
    if a == b:
        return 1.0
    a_words = set(a.split())
    b_words = set(b.split())
    if not a_words or not b_words:
        return 0.0
    overlap = len(a_words & b_words)
    return overlap / max(len(a_words), len(b_words))


def obligation_similarity(obligation, actor, action_lemma, obj):
    score = 0.0
    ob_actor = obligation.get("actor") or {}
    if actor and ob_actor.get("display_name") == actor.get("display_name"):
        score += 0.30
    if action_lemma and obligation.get("action") == action_lemma:
        score += 0.25
    score += 0.25 * _object_similarity(obligation.get("object"), obj)
    # Participant + temporal proximity are constant within a single conversation
    # analysis, so award them when actor matches.
    if actor and ob_actor.get("display_name") == actor.get("display_name"):
        score += 0.10 + 0.10
    return round(score, 3)


# ---------------------------------------------------------------------------
# Suppression (spec section 23)
# ---------------------------------------------------------------------------


def suppression_reason(features, matching, temporal, confidence, minimum_confidence):
    if EXAMPLE_MARKERS.search(matching):
        return "example_text"
    if QUOTED_LINE.match(matching):
        return "quoted_text"
    if HYPOTHETICAL_MARKERS.search(matching):
        return "hypothetical"
    if features.get("negated"):
        return "negated_action"
    if confidence < minimum_confidence:
        return "below_confidence_threshold"
    return None


# ---------------------------------------------------------------------------
# Fingerprint for deduplication / idempotency (spec section 28)
# ---------------------------------------------------------------------------


def suggestion_fingerprint(conversation_id, actor, suggestion_type, action_lemma, obj, due_at):
    actor_key = (actor or {}).get("display_name") or ""
    return "|".join(
        [
            str(conversation_id or ""),
            actor_key,
            suggestion_type,
            action_lemma or "",
            (obj or "").lower(),
            due_at or "",
        ]
    )


# ---------------------------------------------------------------------------
# Main analysis entry point
# ---------------------------------------------------------------------------


def analyze(request):
    """Analyze a conversation and return suggestions + obligations + warnings.

    ``request`` is the parsed JSON body (see spec section 8). Raises
    UnsupportedLocaleError for a non-English locale.
    """
    options = request.get("options") or {}
    locale = options.get("locale")
    duckling_locale = resolve_duckling_locale(locale)  # raises for non-English

    default_timezone = options.get("default_timezone") or DEFAULT_TIMEZONE
    minimum_confidence = options.get("minimum_confidence", DEFAULT_MINIMUM_CONFIDENCE)
    include_low_confidence = bool(options.get("include_low_confidence", False))
    conversation_id = request.get("conversation_id")

    messages = parse_messages(request, default_timezone)
    participants = infer_participants(messages, parse_participants(request))
    participant_views = [
        {"id": p.get("id"), "display_name": p.get("display_name")} for p in participants
    ]

    warnings = []
    obligations = []  # active obligation records (internal + surfaced)
    suggestions = []

    for message in messages:
        text = message["original"]
        if not text.strip():
            continue

        matching = message["matching"]

        # spaCy features (fail-soft per message).
        try:
            doc = nlp(text)
            features = extract_features(doc)
        except Exception:
            warnings.append({"message_id": message["id"], "code": "SPACY_PARSE_FAILED"})
            continue

        # Duckling temporal extraction (fail-soft per message).
        entities = []
        try:
            reftime_ms = to_epoch_ms(message["ts"])
            entities = duckling_parse(
                text, reftime_ms=reftime_ms, tz=message["timezone"], locale=duckling_locale
            )
        except Exception:
            warnings.append({"message_id": message["id"], "code": "TEMPORAL_PARSE_FAILED"})
            entities = []

        # --- State reconciliation first (completion / cancellation / reschedule) ---
        state_change = detect_state_change(features, matching)
        if state_change in ("complete", "cancel") and obligations:
            obj_for_match = extract_object(features, matching)
            action_lemma = features.get("verb_lemma")
            target = _best_open_obligation(obligations, message, action_lemma, obj_for_match)
            if target is not None:
                target["status"] = STATUS_COMPLETED if state_change == "complete" else STATUS_CANCELLED
                target["source_message_ids"].append(message["id"])
                _mark_suggestion_status(suggestions, target["suggestion_id"], target["status"])
                continue

        # --- Rule matching ---
        rule_matches = match_rules(features, matching, message)
        if not rule_matches:
            continue

        # Reschedule: if a reschedule marker is present and we have an open
        # obligation, update it instead of creating a duplicate.
        if state_change == "reschedule" and obligations:
            obj_for_match = extract_object(features, matching)
            action_lemma = features.get("verb_lemma")
            target = _best_open_obligation(obligations, message, action_lemma, obj_for_match)
            if target is not None:
                new_temporal = resolve_temporal(entities, matching, message, options)
                target["due_at"] = new_temporal["due_at"] if new_temporal else target["due_at"]
                target["source_message_ids"].append(message["id"])
                _update_suggestion_temporal(suggestions, target["suggestion_id"], new_temporal, message)
                continue

        # Pick the highest-scoring rule for this message.
        rule = max(rule_matches, key=lambda r: r["score"])
        suggestion_type = rule["suggestion_type"]

        actor, requested_by, actor_note = resolve_actor(
            features, matching, message, participants, suggestion_type
        )
        obj = extract_object(features, matching)
        obj_resolved = obj is not None

        # Pronoun object -> search prior obligations/messages for antecedent.
        if obj and obj.lower() in {"it", "this", "that"}:
            antecedent = _resolve_pronoun(obligations)
            if antecedent:
                obj = antecedent
                obj_resolved = True
            else:
                obj_resolved = False

        temporal = resolve_temporal(entities, matching, message, options)
        action_lemma = features.get("verb_lemma") or _action_from_type(suggestion_type)

        confidence, factors = score_confidence(rule, features, actor, temporal, obj, matching)

        suppress = suppression_reason(
            features, matching, temporal, confidence, minimum_confidence
        )
        if suppress and not (include_low_confidence and suppress == "below_confidence_threshold"):
            warnings.append(
                {"message_id": message["id"], "code": "SUGGESTION_SUPPRESSED", "reason": suppress}
            )
            continue

        # Deduplicate against existing suggestions via fingerprint.
        fp = suggestion_fingerprint(
            conversation_id, actor, suggestion_type, action_lemma, obj, temporal["due_at"]
        )
        if any(s.get("_fingerprint") == fp for s in suggestions):
            warnings.append(
                {"message_id": message["id"], "code": "DUPLICATE_SUPPRESSED"}
            )
            continue

        obligation_id = f"obl_{len(obligations) + 1:03d}"
        suggestion_id = f"sug_{len(suggestions) + 1:03d}"
        now_iso = iso(message["ts"])

        obligation = {
            "id": obligation_id,
            "conversation_id": conversation_id,
            "type": suggestion_type,
            "actor": actor,
            "actor_id": (actor or {}).get("id"),
            "requester_id": (requested_by or {}).get("id"),
            "action": action_lemma,
            "object": obj,
            "status": STATUS_OPEN,
            "due_at": temporal["due_at"],
            "source_message_ids": [message["id"]],
            "suggestion_id": suggestion_id,
            "created_at": now_iso,
            "updated_at": now_iso,
        }
        obligations.append(obligation)

        suggestion = {
            "id": suggestion_id,
            "type": suggestion_type,
            "status": STATUS_OPEN,
            "actor": actor,
            "requested_by": requested_by,
            "participants": participant_views,
            "action": {"lemma": action_lemma, "surface": features.get("verb") or action_lemma},
            "object": {"text": obj, "normalized": (obj or "").lower() or None, "resolved": obj_resolved},
            "temporal": temporal,
            "recommended_action": _recommended_action(suggestion_type, temporal),
            "confidence": confidence,
            "reason": _build_reason(suggestion_type, actor, action_lemma, obj, temporal),
            "evidence": [
                {
                    "message_id": message["id"],
                    "text": text,
                    "start": 0,
                    "end": len(text),
                }
            ],
            "rule_ids": [rule["rule_id"]]
            + (["temporal.explicit_date"] if temporal and not temporal.get("is_inferred") else ["temporal.inferred"]),
            "explanation": {
                "actor_resolution": actor_note,
                "action_resolution": f"The main verb is '{action_lemma}'.",
                "object_resolution": (
                    f"The object is '{obj}'." if obj else "No object could be resolved."
                ),
                "temporal_resolution": (
                    "Resolved relative to the message timestamp."
                    if temporal and not temporal.get("is_inferred")
                    else f"Inferred using the {temporal.get('inference_policy')} policy."
                ),
                "time_inferred": bool(temporal and temporal.get("is_inferred")),
                "confidence_factors": factors,
            },
            "created_at": now_iso,
            "_fingerprint": fp,
        }
        suggestions.append(suggestion)

    include_resolved = bool(options.get("include_resolved_obligations", False))

    surfaced_suggestions = [
        {k: v for k, v in s.items() if not k.startswith("_")}
        for s in suggestions
        if include_resolved or s["status"] == STATUS_OPEN
    ]
    surfaced_obligations = [
        {
            "id": o["id"],
            "status": o["status"],
            "suggestion_id": o["suggestion_id"],
            "type": o["type"],
            "due_at": o["due_at"],
        }
        for o in obligations
        if include_resolved or o["status"] == STATUS_OPEN
    ]

    return {
        "request_id": f"req_{uuid.uuid4().hex[:12]}",
        "conversation_id": conversation_id,
        "analyzed_at": iso(datetime.now(dt_timezone.utc)),
        "suggestions": surfaced_suggestions,
        "obligations": surfaced_obligations,
        "warnings": warnings,
        "engine": {
            "engine_version": ENGINE_VERSION,
            "rule_set_version": RULE_SET_VERSION,
            "spacy_model": SPACY_MODEL,
        },
    }


# ---------------------------------------------------------------------------
# State-engine helpers
# ---------------------------------------------------------------------------


def _best_open_obligation(obligations, message, action_lemma, obj):
    """Return the best-matching OPEN obligation for a new event, or None."""
    best = None
    best_score = 0.0
    speaker_actor = {"id": message["speaker_id"], "display_name": message["speaker_name"]}
    for o in obligations:
        if o["status"] != STATUS_OPEN:
            continue
        score = obligation_similarity(o, speaker_actor, action_lemma, obj)
        # A cancellation from any participant should still match on object/recency
        # even when the actor differs; give a small baseline for open obligations.
        if score < OBLIGATION_MATCH_THRESHOLD and _object_similarity(o.get("object"), obj) >= 0.5:
            score = max(score, OBLIGATION_MATCH_THRESHOLD)
        if score > best_score:
            best_score = score
            best = o
    if best is not None and best_score >= OBLIGATION_MATCH_THRESHOLD:
        return best
    # Fallback: a single open obligation is the unambiguous target.
    open_obs = [o for o in obligations if o["status"] == STATUS_OPEN]
    if len(open_obs) == 1:
        return open_obs[0]
    return None


def _mark_suggestion_status(suggestions, suggestion_id, status):
    for s in suggestions:
        if s["id"] == suggestion_id:
            s["status"] = status


def _update_suggestion_temporal(suggestions, suggestion_id, temporal, message):
    for s in suggestions:
        if s["id"] == suggestion_id and temporal:
            s["temporal"] = temporal
            s["recommended_action"] = _recommended_action(s["type"], temporal)
            s.setdefault("evidence", []).append(
                {
                    "message_id": message["id"],
                    "text": message["original"],
                    "start": 0,
                    "end": len(message["original"]),
                }
            )


def _resolve_pronoun(obligations):
    for o in reversed(obligations):
        if o.get("object"):
            return o["object"]
    return None


def _action_from_type(suggestion_type):
    return {
        TYPE_FOLLOW_UP: "follow_up",
        TYPE_CHECK_BACK: "revisit",
        TYPE_REMINDER: "remind",
    }.get(suggestion_type, "follow_up")


def _recommended_action(suggestion_type, temporal):
    if not temporal:
        return None
    return {"type": "CREATE_REMINDER", "remind_at": temporal["due_at"]}


def _build_reason(suggestion_type, actor, action_lemma, obj, temporal):
    who = (actor or {}).get("display_name") or "Someone"
    obj_part = f" the {obj}" if obj else ""
    when = ""
    if temporal and temporal.get("source_text"):
        when = f" {temporal['source_text']}"
    if suggestion_type == TYPE_COMMITMENT:
        return f"{who} committed to {action_lemma}ing{obj_part}{when}.".replace("ing e", "ing")
    if suggestion_type == TYPE_REQUEST:
        return f"{who} was asked to {action_lemma}{obj_part}{when}."
    if suggestion_type == TYPE_CHECK_BACK:
        return f"The conversation should revisit{obj_part}{when}."
    if suggestion_type == TYPE_FOLLOW_UP:
        return f"{who} is waiting for{obj_part} and should follow up."
    if suggestion_type == TYPE_REMINDER:
        return f"{who} asked to be reminded to {action_lemma}{obj_part}{when}."
    if suggestion_type == TYPE_DEADLINE:
        return f"There is a deadline for{obj_part}{when}."
    if suggestion_type == TYPE_DEPENDENCY:
        return f"An action is blocked until a dependency is resolved{when}."
    return f"{who} has a follow-up{obj_part}{when}."


# ---------------------------------------------------------------------------
# Smoke tests (spec section 5 use cases)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    two_party = [
        {"id": "user_123", "display_name": "Victor", "timezone": "America/Toronto"},
        {"id": "user_456", "display_name": "John", "timezone": "America/Toronto"},
    ]

    def conv(messages, participants=two_party):
        return {
            "conversation_id": "conv_smoke",
            "messages": messages,
            "participants": participants,
            "options": {"default_timezone": "America/Toronto", "locale": "en_CA"},
        }

    scenarios = {
        "5.1 commitment": conv(
            [
                {"speaker": "Victor", "timestamp": "2026-07-17T10:00:00-04:00", "message": "I'll send you the proposal tomorrow."},
                {"speaker": "John", "timestamp": "2026-07-17T10:01:00-04:00", "message": "Sounds good."},
            ]
        ),
        "5.2 request": conv(
            [
                {"speaker": "Victor", "timestamp": "2026-07-17T10:00:00-04:00", "message": "John, can you review the pull request by Friday?"},
            ]
        ),
        "5.3 deferred": conv(
            [
                {"speaker": "Victor", "timestamp": "2026-07-17T10:00:00-04:00", "message": "Let's revisit pricing next week."},
            ]
        ),
        "5.4 waiting": conv(
            [
                {"speaker": "John", "timestamp": "2026-07-17T10:00:00-04:00", "message": "I'm still waiting for the signed agreement."},
            ]
        ),
        "5.5 update": conv(
            [
                {"speaker": "Victor", "timestamp": "2026-07-17T10:00:00-04:00", "message": "I'll send the proposal tomorrow."},
                {"speaker": "Victor", "timestamp": "2026-07-17T10:05:00-04:00", "message": "Actually, I'll send it Friday instead."},
            ]
        ),
        "5.6 completion": conv(
            [
                {"speaker": "Victor", "timestamp": "2026-07-17T10:00:00-04:00", "message": "I'll send the proposal tomorrow."},
                {"speaker": "Victor", "timestamp": "2026-07-17T12:00:00-04:00", "message": "I sent the proposal."},
            ]
        ),
        "5.7 cancellation": conv(
            [
                {"speaker": "Victor", "timestamp": "2026-07-17T10:00:00-04:00", "message": "I'll send the proposal tomorrow."},
                {"speaker": "John", "timestamp": "2026-07-17T10:10:00-04:00", "message": "Don't worry about it anymore."},
            ]
        ),
    }

    for name, req in scenarios.items():
        print("=" * 80)
        print(name)
        try:
            result = analyze(req)
            print(json.dumps({"suggestions": result["suggestions"], "obligations": result["obligations"], "warnings": result["warnings"]}, indent=2))
        except UnsupportedLocaleError as e:
            print("UNSUPPORTED_LOCALE:", e)
