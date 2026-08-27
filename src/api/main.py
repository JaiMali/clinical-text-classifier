"""
Serves the fine-tuned DistilBERT classifier over a REST API.

Run locally:
    uvicorn src.api.main:app --reload

Docs auto-generated at /docs once running.
"""

import json
import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = os.environ.get("MODEL_DIR", "./model_output/final")

_tokenizer = None
_model = None
_label_names: list[str] = []


def load_model():
    global _tokenizer, _model, _label_names
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    _model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    _model.eval()

    label_path = os.path.join(MODEL_DIR, "label_names.json")
    if os.path.exists(label_path):
        with open(label_path) as f:
            _label_names = json.load(f)
    else:
        _label_names = [str(i) for i in range(_model.config.num_labels)]


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="Clinical Text Classifier",
    description="Classifies clinical text sentences using a fine-tuned DistilBERT model.",
    version="0.1.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label: str
    confidence: float
    all_scores: dict[str, float]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    inputs = _tokenizer(
        req.text, return_tensors="pt", truncation=True, padding=True, max_length=128
    )
    with torch.no_grad():
        logits = _model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze().tolist()

    scores = {_label_names[i]: round(p, 4) for i, p in enumerate(probs)}
    top_idx = max(range(len(probs)), key=lambda i: probs[i])

    return PredictResponse(
        label=_label_names[top_idx],
        confidence=round(probs[top_idx], 4),
        all_scores=scores,
    )
