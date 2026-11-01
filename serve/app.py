#!/usr/bin/env python3
"""Ru-Jev serving API, drop-in compatible with the Jev typed-decision format.

Exposes POST /v1/systemone accepting {model, state, questions} and returning
{model, answers, usage, providerMetadata}. A client built for Jev switches to
Ru-Jev by changing the base URL. Decisions are computed locally on CPU by the
Ru-Jev scoring model, so the text never leaves the host.

Run: uvicorn serve.app:app --host 0.0.0.0 --port 8080
Model dir via RU_JEV_MODEL env (default out/ru-jev-v1).
"""
import os, json
from typing import Any, Dict, List
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel

MODEL_DIR = os.environ.get("RU_JEV_MODEL", "out/ru-jev-v1")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Scorer(nn.Module):
    def __init__(self, path):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(path)
        self.head = nn.Linear(self.backbone.config.hidden_size, 1)
        self.head.load_state_dict(torch.load(os.path.join(path, "head.pt"), map_location="cpu"))

    def forward(self, ids, am):
        out = self.backbone(input_ids=ids, attention_mask=am)
        return self.head(out.last_hidden_state[:, 0]).squeeze(-1)


app = FastAPI(title="Ru-Jev", version="1.0")
_tok = None
_model = None
_T = 1.0


def _load():
    global _tok, _model, _T
    if _model is not None:
        return
    calib = json.load(open(os.path.join(MODEL_DIR, "calibration.json")))
    _T = float(calib.get("temperature", 1.0))
    _tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    _model = Scorer(MODEL_DIR).to(DEVICE).eval()


def _state_to_text(state: Any) -> str:
    if isinstance(state, str):
        return state
    if isinstance(state, list):
        parts = []
        for p in state:
            if isinstance(p, dict):
                parts.append(p.get("text") or json.dumps(p.get("value", p), ensure_ascii=False))
            else:
                parts.append(str(p))
        return "\n".join(parts)
    return json.dumps(state, ensure_ascii=False)


@torch.no_grad()
def _distribution(context: str, options: List[str]) -> List[float]:
    enc = _tok([context] * len(options), [str(o) for o in options],
               truncation=True, max_length=256, padding=True, return_tensors="pt").to(DEVICE)
    scores = _model(enc["input_ids"], enc["attention_mask"])
    probs = torch.softmax(scores / max(_T, 1e-2), dim=0)
    return probs.detach().cpu().tolist()


class DecideRequest(BaseModel):
    model: str | None = "ru-jev-latest"
    state: Any
    questions: Dict[str, Dict[str, Any]]


@app.get("/health")
def health():
    _load()
    return {"status": "ok", "model": MODEL_DIR, "device": DEVICE}


@app.post("/v1/systemone")
def systemone(req: DecideRequest):
    _load()
    context = _state_to_text(req.state)
    answers: Dict[str, Any] = {}
    confidence: Dict[str, float] = {}

    for qid, q in req.questions.items():
        qtype = q.get("type")
        instr = q.get("instructions", "")
        ctx = f"{instr}\n{context}" if instr else context
        crit = q.get("criteria")

        if qtype in ("boolean", "noul"):
            if isinstance(crit, dict) and (crit.get("true") or crit.get("false")):
                opts = [crit.get("true", "да"), crit.get("false", "нет")]
            else:
                opts = ["да", "нет"]
            p = _distribution(ctx, opts)
            p_true = float(p[0])
            answers[qid] = {"type": "boolean", "probability": p_true, "noul": p_true}
            confidence[qid] = max(p_true, 1 - p_true)

        elif qtype == "choice":
            if not isinstance(crit, dict) or not crit:
                raise HTTPException(400, f"choice question {qid} needs a nonempty criteria map")
            keys = list(crit.keys())
            descs = [crit[k] for k in keys]
            p = _distribution(ctx, descs)
            probs = {k: float(pi) for k, pi in zip(keys, p)}
            best = max(probs, key=probs.get)
            answers[qid] = {"type": "choice", "choice": best, "probabilities": probs}
            confidence[qid] = probs[best]

        elif qtype == "score":
            if not isinstance(crit, list) or not crit:
                raise HTTPException(400, f"score question {qid} needs a nonempty levels list")
            p = _distribution(ctx, crit)
            mean = sum(i * pi for i, pi in enumerate(p))
            answers[qid] = {"type": "score", "score": float(mean),
                            "probabilities": {str(i): float(pi) for i, pi in enumerate(p)}}
            confidence[qid] = float(max(p))
        else:
            raise HTTPException(400, f"unsupported question type: {qtype}")

    toks = len(_tok(context)["input_ids"])
    return {
        "model": req.model or "ru-jev-latest",
        "answers": answers,
        "usage": {"input_tokens": toks, "output_tokens": 0},
        "providerMetadata": {"ru_jev": {"confidence": confidence}},
    }
