#!/usr/bin/env python3
"""Score one checkpoint over a file of decisions. Writes raw logits.

The temperature is NOT applied here. Raw logits let a temperature be fitted
afterwards on whatever split is wanted, and the argmax is invariant to it, so
accuracy does not depend on this choice at all.

Two row formats are accepted:
  wording bench:    text,  options, answer  (+ vocab, gold_clean, seen_from)
  task validation:  input, options, answer  (+ task, decision_type)

Usage:
  python scripts/score_rows.py --model model/ru-jev-v1.3/ship \\
      --rows data/val.jsonl --out val_v13.jsonl --tag ru-jev-v1.3
"""
import argparse
import json
import os

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, help="local directory or hub repo id")
ap.add_argument("--rows", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--tag", required=True, help="model name recorded in every row")
a = ap.parse_args()

# Fields copied through from the input row, so downstream scripts can slice the
# output without reading the input file again. row_id lets the per-row
# predictions of two checkpoints be lined up.
PASS = ("row_id", "vocab", "gold_clean", "gold_label", "seen_from", "task",
        "decision_type", "set", "gold", "licence")


def _load_head(path):
    """The scoring head ships as safetensors; .pt is accepted for older dumps."""
    st = os.path.join(path, "head.safetensors")
    if os.path.exists(st):
        from safetensors.torch import load_file
        return load_file(st)
    return torch.load(os.path.join(path, "head.pt"), map_location="cpu")


class Scorer(nn.Module):
    def __init__(self, path):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(path)
        self.head = nn.Linear(self.backbone.config.hidden_size, 1)
        self.head.load_state_dict(_load_head(path))

    def forward(self, ids, am):
        cls = self.backbone(input_ids=ids, attention_mask=am).last_hidden_state[:, 0]
        return self.head(cls).squeeze(-1)


dev = "cuda" if torch.cuda.is_available() else "cpu"
max_len = 256
cal_path = os.path.join(a.model, "calibration.json")
if os.path.exists(cal_path):
    max_len = int(json.load(open(cal_path)).get("max_len", 256))

tok = AutoTokenizer.from_pretrained(a.model)
model = Scorer(a.model).eval().to(dev)
print(f"{a.tag}: device {dev}, max_len {max_len}", flush=True)

rows = [json.loads(l) for l in open(a.rows, encoding="utf-8") if l.strip()]
ok = 0
with torch.no_grad(), open(a.out, "w", encoding="utf-8") as fh:
    for i, r in enumerate(rows):
        text = r.get("text", r.get("input"))
        opts = [str(o) for o in r["options"]]
        enc = tok([text] * len(opts), opts, truncation=True,
                  max_length=max_len, padding=True, return_tensors="pt").to(dev)
        logits = model(enc["input_ids"], enc["attention_mask"]).float().cpu()
        pick = int(torch.argmax(logits))
        rec = {"model": a.tag, "logits": [round(float(x), 6) for x in logits],
               "pick": pick, "answer": int(r["answer"])}
        for k in PASS:
            if k in r:
                rec[k] = r[k]
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        ok += int(pick == int(r["answer"]))
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{len(rows)} accuracy so far {ok/(i+1):.4f}", flush=True)

print(f"{a.tag}: done, {ok} of {len(rows)}, accuracy {ok/len(rows):.4f}", flush=True)
