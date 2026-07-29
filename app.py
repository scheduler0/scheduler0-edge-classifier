import logging
import os
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests

from intent import classify, nlp
from suggestions import UnsupportedLocaleError, analyze

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("intent_classifier")

app = FastAPI(title="scheduler0-intent-classifier", version="1.0.0")

# Duckling runs on the EC2 host; start.sh exports DUCKLING_URL with the
# host private IP so the container can reach it.  Falls back to localhost for
# local development.
DUCKLING_URL = os.environ.get("DUCKLING_URL", "http://127.0.0.1:8000/parse")


class ClassifyRequest(BaseModel):
    text: str


class SuggestionParticipant(BaseModel):
    id: Optional[str] = None
    display_name: Optional[str] = None
    timezone: Optional[str] = None


class SuggestionMessage(BaseModel):
    id: Optional[str] = None
    # speaker may be the full object or a bare display-name string (minimal shape).
    speaker: Union[SuggestionParticipant, str]
    timestamp: str
    message: str


class SuggestionOptions(BaseModel):
    reference_time: Optional[str] = None
    locale: Optional[str] = None
    default_timezone: Optional[str] = None
    minimum_confidence: Optional[float] = None
    include_low_confidence: Optional[bool] = None
    include_resolved_obligations: Optional[bool] = None
    default_due_time: Optional[str] = None
    default_deadline_time: Optional[str] = None


class AnalyzeRequest(BaseModel):
    conversation_id: Optional[str] = None
    messages: List[SuggestionMessage]
    participants: Optional[List[SuggestionParticipant]] = None
    options: Optional[SuggestionOptions] = None


@app.get("/healthz")
def healthz():
    try:
        nlp("test")
    except Exception as e:
        logger.warning("healthz: spaCy unavailable: %s", e)
        raise HTTPException(status_code=503, detail=f"spaCy unavailable: {e}")

    try:
        r = requests.post(
            DUCKLING_URL,
            data={"locale": "en_GB", "text": "tomorrow", "dims": '["time"]'},
            timeout=2,
        )
        r.raise_for_status()
    except Exception as e:
        logger.error("healthz: Duckling unreachable: %s", e)
        raise HTTPException(status_code=503, detail=f"Duckling unreachable: {e}")

    return {"status": "ok"}


@app.post("/v1/intents/classify")
def classify_intent(req: ClassifyRequest):
    try:
        return classify(req.text)
    except Exception as e:
        logger.exception("intent classify failed (text_len=%d)", len(req.text))
        raise HTTPException(status_code=500, detail=str(e))


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
        raise HTTPException(status_code=500, detail=str(e))
