import hashlib
import hmac
import logging
import os
from typing import Annotated, Any, Dict, List, Optional, Union

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from intent import classify, nlp, post_duckling
from suggestions import UnsupportedLocaleError, analyze

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("intent_classifier")

# Scheduling text is short. Caps keep spaCy, Duckling, and the regex engine
# from being used as a CPU or memory sink.
MAX_CLASSIFY_CHARS = 4_000
MAX_MESSAGE_CHARS = 8_000
MAX_MESSAGES = 50
MAX_PARTICIPANTS = 50
MAX_ID_CHARS = 128
MAX_NAME_CHARS = 200
MAX_LOCALE_CHARS = 64
MAX_TIME_CHARS = 64
MAX_HHMM_CHARS = 8
MAX_BODY_BYTES = 2 * 1024 * 1024

_DEPLOYED_ENVS = {"production", "staging"}


def docs_enabled():
    """Interactive docs stay on for local development and off on ECS."""
    return os.environ.get("ENV", "").strip().lower() not in _DEPLOYED_ENVS


_DOCS = docs_enabled()
app = FastAPI(
    title="scheduler0-intent-classifier",
    version="1.0.0",
    docs_url="/docs" if _DOCS else None,
    redoc_url="/redoc" if _DOCS else None,
    openapi_url="/openapi.json" if _DOCS else None,
)


def _header_value(scope, name):
    target = name.lower()
    for key, value in scope.get("headers", []):
        if key.lower() == target:
            return value
    return None


async def _json_response(send, status, detail):
    body = ('{"detail":"%s"}' % detail).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body, "more_body": False})


class LimitRequestBodyMiddleware:
    """Reject bodies larger than MAX_BODY_BYTES before the route runs.

    A declared Content-Length is refused immediately. Chunked bodies are
    counted as they arrive and are not forwarded once the cap is crossed.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        raw_length = _header_value(scope, b"content-length")
        if raw_length is not None:
            # Refuse a huge digit string before converting it. Ten digits is
            # enough to express every length we would otherwise reject as 413.
            if len(raw_length) > 10 or not raw_length.isdigit():
                await _json_response(send, 400, "Invalid Content-Length")
                return
            if int(raw_length) > MAX_BODY_BYTES:
                await _json_response(send, 413, "Request body is too large")
                return

        chunks = []
        received = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            if message["type"] != "http.request":
                continue
            chunk = message.get("body", b"")
            received += len(chunk)
            if received > MAX_BODY_BYTES:
                await _json_response(send, 413, "Request body is too large")
                return
            chunks.append(chunk)
            if not message.get("more_body", False):
                break

        body = b"".join(chunks)
        sent = False

        async def replay():
            nonlocal sent
            if sent:
                return {"type": "http.disconnect"}
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}

        await self.app(scope, replay, send)


class SecurityHeadersMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                present = {name.lower() for name, _ in headers}
                extra = (
                    (b"x-content-type-options", b"nosniff"),
                    (b"x-frame-options", b"DENY"),
                    (b"referrer-policy", b"no-referrer"),
                    (b"cache-control", b"no-store"),
                    (b"content-security-policy", b"default-src 'none'; frame-ancestors 'none'"),
                )
                for name, value in extra:
                    if name not in present:
                        headers.append((name, value))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_headers)


# Added last so it is the outermost middleware and covers 413 responses too.
app.add_middleware(LimitRequestBodyMiddleware)
app.add_middleware(SecurityHeadersMiddleware)


def _secret_equal(provided: str, expected: str) -> bool:
    # Hash both sides so the comparison does not leak the secret's length.
    return hmac.compare_digest(
        hashlib.sha256(provided.encode("utf-8")).digest(),
        hashlib.sha256(expected.encode("utf-8")).digest(),
    )


def enforce_api_key(authorization: Optional[str]) -> None:
    """Require a bearer token when CLASSIFIER_API_KEY is configured.

    Unset means the process keeps the previous behavior (network policy is
    the only control). Health checks do not call this.
    """
    expected = os.environ.get("CLASSIFIER_API_KEY", "").strip()
    if not expected:
        return
    prefix = "Bearer "
    provided = authorization[len(prefix) :] if authorization and authorization.startswith(prefix) else ""
    if not provided or not _secret_equal(provided, expected):
        raise HTTPException(
            status_code=401,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Bearer"},
        )


class ClassifyRequest(BaseModel):
    text: str = Field(max_length=MAX_CLASSIFY_CHARS)


class SuggestionParticipant(BaseModel):
    id: Optional[str] = Field(default=None, max_length=MAX_ID_CHARS)
    display_name: Optional[str] = Field(default=None, max_length=MAX_NAME_CHARS)
    timezone: Optional[str] = Field(default=None, max_length=MAX_LOCALE_CHARS)


class SuggestionMessage(BaseModel):
    id: Optional[str] = Field(default=None, max_length=MAX_ID_CHARS)
    # speaker may be the full object or a bare display-name string (minimal shape).
    speaker: Union[SuggestionParticipant, Annotated[str, Field(max_length=MAX_NAME_CHARS)]]
    timestamp: str = Field(min_length=1, max_length=MAX_TIME_CHARS)
    message: str = Field(max_length=MAX_MESSAGE_CHARS)


class SuggestionOptions(BaseModel):
    reference_time: Optional[str] = Field(default=None, max_length=MAX_TIME_CHARS)
    locale: Optional[str] = Field(default=None, max_length=MAX_LOCALE_CHARS)
    default_timezone: Optional[str] = Field(default=None, max_length=MAX_LOCALE_CHARS)
    minimum_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    include_low_confidence: Optional[bool] = None
    include_resolved_obligations: Optional[bool] = None
    default_due_time: Optional[str] = Field(default=None, max_length=MAX_HHMM_CHARS)
    default_deadline_time: Optional[str] = Field(default=None, max_length=MAX_HHMM_CHARS)


class AnalyzeRequest(BaseModel):
    conversation_id: Optional[str] = Field(default=None, max_length=MAX_ID_CHARS)
    messages: List[SuggestionMessage] = Field(max_length=MAX_MESSAGES)
    participants: Optional[List[SuggestionParticipant]] = Field(default=None, max_length=MAX_PARTICIPANTS)
    options: Optional[SuggestionOptions] = None


@app.get("/healthz")
def healthz():
    try:
        nlp("test")
    except Exception as e:
        logger.warning("healthz: spaCy unavailable: %s", e)
        raise HTTPException(status_code=503, detail="spaCy unavailable")

    try:
        r = post_duckling(
            {"locale": "en_GB", "text": "tomorrow", "dims": '["time"]'},
            timeout=2,
        )
        r.raise_for_status()
    except Exception as e:
        logger.error("healthz: Duckling unreachable: %s", e)
        raise HTTPException(status_code=503, detail="Duckling unreachable")

    return {"status": "ok"}


@app.post("/v1/intents/classify")
def classify_intent(
    req: ClassifyRequest,
    authorization: Optional[str] = Header(default=None),
):
    enforce_api_key(authorization)
    try:
        return classify(req.text)
    except Exception:
        logger.exception("intent classify failed (text_len=%d)", len(req.text))
        raise HTTPException(status_code=500, detail="Intent classification failed")


@app.post("/v1/suggestions/analyze")
def analyze_suggestions(
    req: AnalyzeRequest,
    authorization: Optional[str] = Header(default=None),
):
    enforce_api_key(authorization)
    # model_dump keeps the flexible speaker union intact for the engine, which
    # accepts both the object and bare-string speaker shapes.
    payload: Dict[str, Any] = req.model_dump(exclude_none=True)
    try:
        return analyze(payload)
    except UnsupportedLocaleError as e:
        # English-only for the first release; surface as a structured 400 so the
        # Go proxy and clients can map it to UNSUPPORTED_LOCALE.
        raise HTTPException(
            status_code=400,
            detail={"code": "UNSUPPORTED_LOCALE", "message": str(e)},
        )
    except Exception:
        logger.exception("suggestions analyze failed (messages=%d)", len(req.messages))
        raise HTTPException(status_code=500, detail="Suggestions analysis failed")
