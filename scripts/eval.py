#!/usr/bin/env python3
"""Evaluate ru-jev: per-task accuracy and calibration error (ECE).

Usage: python scripts/eval.py --model out/ru-jev-v1
"""
import json, os, argparse, math
from collections import defaultdict
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

HERE = os.path.dirname(__file__)


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


class Scorer(nn.Module):
    def __init__(self, path):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(path)
        h = self.backbone.config.hidden_size
        self.head = nn.Linear(h, 1)
        self.head.load_state_dict(torch.load(os.path.join(path, "head.pt"), map_location="cpu"))

    def forward(self, ids, am):
        out = self.backbone(input_ids=ids, attention_mask=am)
        return self.head(out.last_hidden_state[:, 0]).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(HERE, "..", "out", "ru-jev-v1"))
    ap.add_argument("--data", default=os.path.join(HERE, "..", "data", "val.jsonl"))
    args = ap.parse_args()

    device = pick_device()
    calib = json.load(open(os.path.join(args.model, "calibration.json")))
    T = calib.get("temperature", 1.0)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = Scorer(args.model).to(device).eval()
    rows = load_jsonl(args.data)

    per_task = defaultdict(lambda: [0, 0])     # task -> [correct, total]
    conf_bins = [[0.0, 0.0, 0] for _ in range(10)]  # sum_conf, sum_acc, n

    with torch.no_grad():
        for r in rows:
            opts = [str(o) for o in r["options"]]
            enc = tok([r["input"]] * len(opts), opts, truncation=True,
                      max_length=calib["max_len"], padding=True, return_tensors="pt").to(device)
            scores = model(enc["input_ids"], enc["attention_mask"])
            probs = torch.softmax(scores / max(T, 1e-2), dim=0)
            pred = int(torch.argmax(probs).item())
            correct = int(pred == r["answer"])
            per_task[r["task"]][0] += correct
            per_task[r["task"]][1] += 1
            conf = float(probs.max().item())
            b = min(9, int(conf * 10))
            conf_bins[b][0] += conf; conf_bins[b][1] += correct; conf_bins[b][2] += 1

    print("\n== Accuracy per task ==")
    tot_c = tot_n = 0
    for t in sorted(per_task):
        c, n = per_task[t]; tot_c += c; tot_n += n
        print(f"  {t:12s} {c/n:.3f}  (n={n})")
    print(f"  {'OVERALL':12s} {tot_c/max(tot_n,1):.3f}  (n={tot_n})")

    ece = 0.0
    N = sum(b[2] for b in conf_bins)
    for sc, sa, n in conf_bins:
        if n == 0:
            continue
        ece += (n / N) * abs(sa / n - sc / n)
    print(f"\n== Calibration ==\n  ECE = {ece:.4f}  (temperature T = {T:.3f})")


if __name__ == "__main__":
    main()
