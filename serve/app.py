#!/usr/bin/env python3
"""Ru-Jev serving API in the typed-decision request format.

Exposes POST /v1/systemone accepting {model, state, questions} and returning
{model, answers, usage, providerMetadata}. A client written against a closed
decision API moves over by changing the base URL. Decisions are computed locally
on CPU by the Ru-Jev scoring model, so the text never leaves the host.

Run: uvicorn serve.app:app --host 0.0.0.0 --port 8080
Weights via RU_JEV_MODEL, which takes a local directory or a Hub repo id. With
it unset the service prefers a fresh training output, then released weights
sitting next to this repository, and otherwise downloads besktech/ru-jev from
the Hub, so a clone of this repository alone is enough to start.
"""
import os, json
from typing import Any, Dict, List
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel

HERE = os.path.dirname(os.path.abspath(__file__))
HUB_REPO = "besktech/ru-jev"
# Prefer a fresh training output, then released weights sitting next to this
# repository. Those paths exist on the author's machine and not in a clone, so
# when neither is there the weights come from the Hub instead of failing with a
# missing-file error. Override everything with RU_JEV_MODEL, which accepts a
# local directory or a Hub repo id.
_CANDIDATES = (
    os.path.join(HERE, "..", "out", "ru-jev-v1.3"),
    os.path.join(HERE, "..", "..", "model", "ru-jev-v1.3", "ship"),
)
MODEL_SOURCE = os.environ.get("RU_JEV_MODEL") or next(
    (os.path.normpath(p) for p in _CANDIDATES if os.path.isdir(p)), HUB_REPO)
# Resolved to a real directory on first load. Downloading at import time would
# make importing this module hit the network, which breaks tests and tooling.
MODEL_DIR = MODEL_SOURCE if os.path.isdir(MODEL_SOURCE) else None
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
_MAX_LEN = 256


def _load():
    global _tok, _model, _T, _MAX_LEN, MODEL_DIR
    if _model is not None:
        return
    if MODEL_DIR is None:
        from huggingface_hub import snapshot_download
        MODEL_DIR = snapshot_download(MODEL_SOURCE)
    calib = json.load(open(os.path.join(MODEL_DIR, "calibration.json")))
    _T = float(calib.get("temperature", 1.0))
    _MAX_LEN = int(calib.get("max_len", 256))
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
               truncation=True, max_length=_MAX_LEN, padding=True,
               return_tensors="pt").to(DEVICE)
    scores = _model(enc["input_ids"], enc["attention_mask"])
    probs = torch.softmax(scores / max(_T, 1e-2), dim=0)
    return probs.detach().cpu().tolist()


class DecideRequest(BaseModel):
    model: str | None = "ru-jev-latest"
    state: Any
    questions: Dict[str, Dict[str, Any]]
    # Below this confidence the answer is withheld and the question is marked for
    # a human. Set per question with "threshold", or here for the whole request.
    # Omit it and nothing is withheld, which is the previous behaviour.
    threshold: float | None = None


@app.get("/health")
def health():
    _load()
    return {"status": "ok", "model": MODEL_SOURCE, "path": MODEL_DIR, "device": DEVICE,
            "max_len": _MAX_LEN, "temperature": _T}


@app.post("/v1/systemone")
def systemone(req: DecideRequest):
    _load()
    context = _state_to_text(req.state)
    answers: Dict[str, Any] = {}
    confidence: Dict[str, float] = {}

    review: List[str] = []

    for qid, q in req.questions.items():
        qtype = q.get("type")
        thr = q.get("threshold", req.threshold)
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

        if thr is not None and confidence[qid] < float(thr):
            answers[qid]["abstain"] = True
            review.append(qid)
        elif thr is not None:
            answers[qid]["abstain"] = False

    toks = len(_tok(context)["input_ids"])
    return {
        "model": req.model or "ru-jev-latest",
        "answers": answers,
        "usage": {"input_tokens": toks, "output_tokens": 0},
        "providerMetadata": {"ru_jev": {"confidence": confidence, "review": review}},
    }
