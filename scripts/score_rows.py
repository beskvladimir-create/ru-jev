#!/usr/bin/env python3
"""Оценка одного чекпойнта Ру-Жев по строкам. Пишет сырые логиты.

Температуру здесь НЕ применяем. Логиты позволяют подобрать её потом точно,
а аргмакс к температуре нечувствителен, поэтому точность от этого не зависит.

Понимает два формата строк:
  стенд формулировок: text, options, answer (+ vocab, gold_clean, seen_from)
  валидация задач:    input, options, answer (+ task, decision_type)
"""
import argparse, json, os
import torch, torch.nn as nn
from transformers import AutoModel, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--rows", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--tag", required=True)
a = ap.parse_args()

PASS = ("row_id", "vocab", "gold_clean", "gold_label", "seen_from", "task", "decision_type",
        "set", "gold", "licence")


def _load_head(path):
    """Голова лежит в safetensors; .pt принимается для старых выгрузок."""
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
cal_path = os.path.join(a.model, "calibration.json")
max_len = 256
if os.path.exists(cal_path):
    max_len = int(json.load(open(cal_path)).get("max_len", 256))

tok = AutoTokenizer.from_pretrained(a.model)
model = Scorer(a.model).eval().to(dev)
print(f"{a.tag}: устройство {dev}, max_len {max_len}", flush=True)

rows = [json.loads(l) for l in open(a.rows, encoding="utf-8")]
ok = 0
with torch.no_grad(), open(a.out, "w", encoding="utf-8") as fh:
    for i, r in enumerate(rows):
        text = r.get("text", r.get("input"))
        opts = r["options"]
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
            print(f"  {i+1}/{len(rows)} точность пока {ok/(i+1):.4f}", flush=True)

print(f"{a.tag}: готово, {ok} из {len(rows)}, точность {ok/len(rows):.4f}", flush=True)
