import logging
import os
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
import requests

from intent import classify, nlp, safe_duckling_url
from suggestions import UnsupportedLocaleError, analyze

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("intent_classifier")

# Scheduling text is short. These caps keep one request from pinning a worker
# inside spaCy or Duckling. The body cap is enforced again before parsing.
MAX_TEXT_CHARS = 8_000
MAX_MESSAGES = 100
MAX_PARTICIPANTS = 50
MAX_ID_CHARS = 200
MAX_NAME_CHARS = 200
MAX_LOCALE_CHARS = 64
MAX_TIMEZONE_CHARS = 64
MAX_TIMESTAMP_CHARS = 80
MAX_CLOCK_CHARS = 8
MAX_BODY_BYTES = 512 * 1024

SPACY_UNAVAILABLE = "spaCy unavailable"
DUCKLING_UNAVAILABLE = "Duckling unreachable"
CLASSIFY_FAILED = "Intent classification failed"
ANALYZE_FAILED = "Suggestions analysis failed"


def api_docs_urls(env_name):
    """Hide interactive docs outside local development.

    Production and staging are internal services. The schema is still in the
    repository; it does not need to be served from the deployed process.
    """
    if (env_name or "").strip().lower() in {"production", "staging"}:
        return None, None, None
    return "/docs", "/redoc", "/openapi.json"


def body_limit_status(content_length, body_length):
    """Return an HTTP status when a request body must be rejected."""
    if content_length is not None:
        try:
            declared = int(str(content_length).strip())
        except (TypeError, ValueError):
            return 400
        if declared < 0:
            return 400
        if declared > MAX_BODY_BYTES:
            return 413
    if body_length > MAX_BODY_BYTES:
        return 413
    return None


def _security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


_docs_url, _redoc_url, _openapi_url = api_docs_urls(os.environ.get("ENV", ""))
app = FastAPI(
    title="scheduler0-intent-classifier",
    version="1.0.0",
    docs_url=_docs_url,
    redoc_url=_redoc_url,
    openapi_url=_openapi_url,
)


class ClassifyRequest(BaseModel):
    text: str = Field(max_length=MAX_TEXT_CHARS)


class SuggestionParticipant(BaseModel):
    id: Optional[str] = Field(default=None, max_length=MAX_ID_CHARS)
    display_name: Optional[str] = Field(default=None, max_length=MAX_NAME_CHARS)
    timezone: Optional[str] = Field(default=None, max_length=MAX_TIMEZONE_CHARS)


class SuggestionMessage(BaseModel):
    id: Optional[str] = Field(default=None, max_length=MAX_ID_CHARS)
    # speaker may be the full object or a bare display-name string (minimal shape).
    speaker: Union[SuggestionParticipant, str]
    timestamp: str = Field(max_length=MAX_TIMESTAMP_CHARS)
    message: str = Field(max_length=MAX_TEXT_CHARS)

    @field_validator("speaker")
    @classmethod
    def speaker_is_bounded(cls, value):
        if isinstance(value, str) and len(value) > MAX_NAME_CHARS:
            raise ValueError("speaker name is too long")
        return value


class SuggestionOptions(BaseModel):
    reference_time: Optional[str] = Field(default=None, max_length=MAX_TIMESTAMP_CHARS)
    locale: Optional[str] = Field(default=None, max_length=MAX_LOCALE_CHARS)
    default_timezone: Optional[str] = Field(default=None, max_length=MAX_TIMEZONE_CHARS)
    minimum_confidence: Optional[float] = Field(default=None, ge=0, le=1)
    include_low_confidence: Optional[bool] = None
    include_resolved_obligations: Optional[bool] = None
    default_due_time: Optional[str] = Field(default=None, max_length=MAX_CLOCK_CHARS)
    default_deadline_time: Optional[str] = Field(default=None, max_length=MAX_CLOCK_CHARS)


class AnalyzeRequest(BaseModel):
    conversation_id: Optional[str] = Field(default=None, max_length=MAX_ID_CHARS)
    messages: List[SuggestionMessage] = Field(max_length=MAX_MESSAGES)
    participants: Optional[List[SuggestionParticipant]] = Field(
        default=None, max_length=MAX_PARTICIPANTS
    )
    options: Optional[SuggestionOptions] = None


@app.middleware("http")
async def limit_request_body(request: Request, call_next):
    declared = request.headers.get("content-length")
    status = body_limit_status(declared, 0)
    if status is not None:
        detail = (
            "Invalid Content-Length" if status == 400 else "Request body too large"
        )
        return _security_headers(
            JSONResponse(status_code=status, content={"detail": detail})
        )

    # Read with a hard cap. A missing or understated Content-Length must not
    # be allowed to buffer an unbounded body before the route runs.
    chunks = []
    received = 0
    async for chunk in request.stream():
        received += len(chunk)
        if received > MAX_BODY_BYTES:
            return _security_headers(
                JSONResponse(
                    status_code=413, content={"detail": "Request body too large"}
                )
            )
        chunks.append(chunk)
    # Cache the buffered body so the route can read it after the stream
    # has already been consumed.
    request._body = b"".join(chunks)
    response = await call_next(request)
    return _security_headers(response)


@app.get("/healthz")
def healthz():
    try:
        nlp("test")
    except Exception as e:
        logger.warning("healthz: spaCy unavailable: %s", e)
        raise HTTPException(status_code=503, detail=SPACY_UNAVAILABLE)

    try:
        r = requests.post(
            safe_duckling_url(),
            data={"locale": "en_GB", "text": "tomorrow", "dims": '["time"]'},
            timeout=2,
            allow_redirects=False,
        )
        r.raise_for_status()
    except Exception as e:
        logger.error("healthz: Duckling unreachable: %s", e)
        raise HTTPException(status_code=503, detail=DUCKLING_UNAVAILABLE)

    return {"status": "ok"}


@app.post("/v1/intents/classify")
def classify_intent(req: ClassifyRequest):
    try:
        return classify(req.text)
    except Exception as e:
        logger.exception("intent classify failed (text_len=%d)", len(req.text))
        raise HTTPException(status_code=500, detail=CLASSIFY_FAILED)


@app.post("/v1/suggestions/analyze")
def analyze_suggestions(req: AnalyzeRequest):
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
    except Exception as e:
        logger.exception("suggestions analyze failed (messages=%d)", len(req.messages))
        raise HTTPException(status_code=500, detail=ANALYZE_FAILED)
