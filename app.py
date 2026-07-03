from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests

from intent import classify, nlp

app = FastAPI(title="scheduler0-intent-classifier", version="1.0.0")

DUCKLING_URL = "http://127.0.0.1:8000/parse"


class ClassifyRequest(BaseModel):
    text: str


@app.get("/healthz")
def healthz():
    try:
        nlp("test")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"spaCy unavailable: {e}")

    try:
        r = requests.post(
            DUCKLING_URL,
            data={"locale": "en_GB", "text": "tomorrow", "dims": '["time"]'},
            timeout=2,
        )
        r.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Duckling unreachable: {e}")

    return {"status": "ok"}


@app.post("/v1/intents/classify")
def classify_intent(req: ClassifyRequest):
    try:
        return classify(req.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
