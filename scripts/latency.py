#!/usr/bin/env python3
"""Latency of one decision at k candidates, on one and the same machine.

Why Ru-Jev gets slower as k grows. It scores each (context, option) pair in its
own pass of the encoder, so the cost of a decision is linear in k. Laya reads
all options in one pass, so its cost is almost flat in k. That is the price of
Ru-Jev's independence from the order of candidates, not a shortcoming.

What is measured is the whole decision: tokenisation, the forward passes and the
softmax. The median over runs is reported with the spread. Warm-up runs are
discarded first, otherwise the early runs carry one-off initialisation.

The two engines live in different environments, so the script is run twice, once
per engine with its own interpreter.
"""
import argparse, json, os, statistics, time

ap = argparse.ArgumentParser()
ap.add_argument("--engine", required=True, choices=["rujev", "laya"])
ap.add_argument("--model", default="model/ru-jev-v1.3/ship")
ap.add_argument("--checkpoint", default="convaiinnovations/laya-multilingual")
ap.add_argument("--rows", required=True, help="bench rows; the first runs+warmup are used")
ap.add_argument("--ks", default="2,5,18")
ap.add_argument("--runs", type=int, default=25)
ap.add_argument("--warmup", type=int, default=5)
ap.add_argument("--out", required=True)
a = ap.parse_args()

rows = [json.loads(l) for l in open(a.rows, encoding="utf-8") if l.strip()]
rows = [r for r in rows if len(r["options"]) >= 18][:a.runs + a.warmup]
assert len(rows) >= a.runs + a.warmup, f"too few rows: {len(rows)}"
ks = [int(x) for x in a.ks.split(",")]

if a.engine == "rujev":
    import torch, torch.nn as nn
    from transformers import AutoModel, AutoTokenizer

    class Scorer(nn.Module):
        def __init__(self, path):
            super().__init__()
            self.backbone = AutoModel.from_pretrained(path)
            self.head = nn.Linear(self.backbone.config.hidden_size, 1)
            st = os.path.join(path, "head.safetensors")
            if os.path.exists(st):
                from safetensors.torch import load_file
                self.head.load_state_dict(load_file(st))
            else:
                self.head.load_state_dict(
                    torch.load(os.path.join(path, "head.pt"), map_location="cpu"))

        def forward(self, ids, am):
            cls = self.backbone(input_ids=ids, attention_mask=am).last_hidden_state[:, 0]
            return self.head(cls).squeeze(-1)

    torch.set_num_threads(os.cpu_count())
    cal = json.load(open(os.path.join(a.model, "calibration.json")))
    T, max_len = cal["temperature"], int(cal.get("max_len", 256))
    tok = AutoTokenizer.from_pretrained(a.model)
    model = Scorer(a.model).eval()

    def decide(text, opts):
        with torch.no_grad():
            enc = tok([text] * len(opts), opts, truncation=True,
                      max_length=max_len, padding=True, return_tensors="pt")
            lg = model(enc["input_ids"], enc["attention_mask"])
            return torch.softmax(lg / T, dim=0).tolist()
    label = "ru-jev-v1.3"
else:
    import laya
    agent = laya.load(a.checkpoint)

    def decide(text, opts):
        crit = {f"o{j}": o for j, o in enumerate(opts)}
        res = agent.predict({"text": text},
                            {"q": {"type": "choice",
                                   "instructions": "К какой категории относится запрос пользователя?",
                                   "criteria": crit}})
        return list(res["answers"]["q"]["probabilities"].values())
    label = "laya-multilingual"

out = {"engine": label, "runs": a.runs, "warmup": a.warmup, "by_k": {}}
for k in ks:
    for r in rows[:a.warmup]:
        decide(r["text"], r["options"][:k])
    ts = []
    for r in rows[a.warmup:a.warmup + a.runs]:
        t0 = time.perf_counter()
        decide(r["text"], r["options"][:k])
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts.sort()
    out["by_k"][str(k)] = {"median_ms": statistics.median(ts),
                           "min_ms": ts[0], "max_ms": ts[-1],
                           "p25_ms": ts[len(ts) // 4], "p75_ms": ts[3 * len(ts) // 4]}
    print(f"{label} k={k:2d}: median {statistics.median(ts):7.1f} ms "
          f"(from {ts[0]:.1f} to {ts[-1]:.1f})", flush=True)
json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("written:", a.out)
