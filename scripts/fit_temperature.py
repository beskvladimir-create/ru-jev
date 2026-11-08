#!/usr/bin/env python3
"""Fit a model's temperature on your own data.

Why. The released temperature was fitted on our validation split. On your
decisions the distribution of confidence is different and the probabilities will
be biased. Accuracy does NOT depend on the temperature at all: the softmax over
candidates is monotone in it and the argmax does not move. What changes is only
how honest the number next to the answer is.

How many rows. A few hundred already give a stable estimate. The script fits the
temperature on half-splits and prints the spread, so it is visible whether there
were enough rows: if the edge of the spread is far from the middle, there were
not.

Input format, one JSON object per decision:
  {"input": "user request text", "options": ["candidate", ...], "answer": 0}
A "text" field is accepted instead of "input".

Per-option biases. The temperature cannot change which option is picked, so it
cannot repair a model that leans towards one of your labels whatever the input
says. One additive bias per option can: with --biases the script fits
softmax(s / T + b[option]) by maximum likelihood, temperature and biases
together, and writes the biases into the same calibration.json. They are a
correction of your label set, fitted on your rows, and they do not transfer to
a different set of labels. Without the flag nothing changes.

Example:
  python scripts/fit_temperature.py --model besktech/ru-jev \
      --rows my_decisions.jsonl --out my_calibration.json
  python scripts/fit_temperature.py --model besktech/ru-jev \
      --rows my_decisions.jsonl --out my_calibration.json --biases
"""
import argparse, json, math, os, random, statistics, sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import calib

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

SPLITS = 25
BINS = 15


class Scorer(nn.Module):
    def __init__(self, path):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(path)
        self.head = nn.Linear(self.backbone.config.hidden_size, 1)
        self.head.load_state_dict(_load_head(path))

    def forward(self, ids, mask):
        cls = self.backbone(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
        return self.head(cls).squeeze(-1)


def _load_head(path):
    """The head ships as safetensors; .pt is accepted for older dumps."""
    from huggingface_hub import hf_hub_download
    for name, reader in (("head.safetensors", None), ("head.pt", torch.load)):
        local = os.path.join(path, name)
        if os.path.exists(local):
            src = local
        else:
            try:
                src = hf_hub_download(path, name)
            except Exception:
                continue
        if reader is None:
            from safetensors.torch import load_file
            return load_file(src)
        return reader(src, map_location="cpu")
    raise SystemExit("no scoring head found: neither head.safetensors nor head.pt")


def nll(scored, T):
    s = 0.0
    for lg, ans in scored:
        p = [x / T for x in lg]
        m = max(p)
        s -= (p[ans] - m - math.log(sum(math.exp(x - m) for x in p)))
    return s / max(len(scored), 1)


def fit(scored, lo=0.2, hi=8.0, steps=300, rounds=4):
    """Grid search on the negative log-likelihood, then narrowing around it."""
    best = 1.0
    for _ in range(rounds):
        cand = [lo + (hi - lo) * i / steps for i in range(steps + 1)]
        best = min(cand, key=lambda t: nll(scored, t))
        step = (hi - lo) / steps
        lo, hi = max(0.05, best - step), best + step
    return best


def probs(lg, T):
    p = [x / T for x in lg]
    m = max(p)
    e = [math.exp(x - m) for x in p]
    z = sum(e)
    return [x / z for x in e]


def ece(scored, T):
    b = [[] for _ in range(BINS)]
    for lg, ans in scored:
        p = probs(lg, T)
        j = max(range(len(p)), key=lambda k: p[k])
        b[min(int(p[j] * BINS), BINS - 1)].append((p[j], int(j == ans)))
    n = sum(len(x) for x in b)
    return sum(len(x) / n * abs(sum(c for c, _ in x) / len(x)
                                - sum(h for _, h in x) / len(x))
               for x in b if x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local directory or hub repo id")
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", help="where to write a calibration.json with the new temperature")
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20261105, help="split seed, recorded in the report")
    ap.add_argument("--biases", action="store_true",
                    help="also fit one additive bias per option, which does move the answer")
    ap.add_argument("--no-temperature", action="store_true",
                    help="with --biases, hold the temperature at 1 and fit the biases alone")
    a = ap.parse_args()
    if a.no_temperature and not a.biases:
        raise SystemExit("--no-temperature only means something together with --biases")

    max_len = a.max_len
    if max_len is None:
        try:
            cal = os.path.join(a.model, "calibration.json")
            if os.path.exists(cal):
                max_len = int(json.load(open(cal)).get("max_len", 256))
            else:
                from huggingface_hub import hf_hub_download
                max_len = int(json.load(open(hf_hub_download(a.model, "calibration.json")))
                              .get("max_len", 256))
        except Exception:
            max_len = 256

    rows = [json.loads(l) for l in open(a.rows, encoding="utf-8") if l.strip()]
    if len(rows) < 100:
        print(f"warning: only {len(rows)} rows, the estimate will be noisy")
    tok = AutoTokenizer.from_pretrained(a.model)
    model = Scorer(a.model).eval()

    scored = []
    with torch.no_grad():
        for i, r in enumerate(rows):
            text = r.get("input", r.get("text"))
            opts = [str(o) for o in r["options"]]
            enc = tok([text] * len(opts), opts, truncation=True,
                      max_length=max_len, padding=True, return_tensors="pt")
            lg = model(enc["input_ids"], enc["attention_mask"]).float().tolist()
            scored.append((lg, opts, int(r["answer"])))
            if (i + 1) % 200 == 0:
                print(f"  scored {i+1}/{len(rows)}", flush=True)

    pairs = [(lg, ans) for lg, _opts, ans in scored]
    acc = sum(lg.index(max(lg)) == ans for lg, ans in pairs) / len(pairs)
    T_all = fit(pairs)

    idx = list(range(len(pairs)))
    ts, es = [], []
    for s in range(SPLITS):
        order = list(idx)
        random.Random(a.seed + s).shuffle(order)
        h = len(order) // 2
        T = fit([pairs[i] for i in order[:h]])
        ts.append(T)
        es.append(ece([pairs[i] for i in order[h:]], T))

    print(f"\nrows {len(scored)}, accuracy {acc:.4f} (independent of the temperature)")
    print(f"temperature on all rows: {T_all:.4f}")
    print(f"over {SPLITS} half-splits, seed {a.seed}: "
          f"mean {statistics.mean(ts):.4f}, from {min(ts):.3f} to {max(ts):.3f}")
    print(f"ECE at temperature 1: {ece(pairs, 1.0):.4f}")
    print(f"ECE after the fitted one: {statistics.mean(es):.4f}")
    spread = max(ts) - min(ts)
    if spread > 0.5:
        print(f"temperature spread {spread:.2f} is wide: add more rows")

    out = {"temperature": T_all, "max_len": max_len, "calibrated": True,
           "fitted_on": os.path.basename(a.rows), "n_rows": len(scored),
           "split_seed": a.seed, "accuracy": round(acc, 4)}

    if a.biases:
        fitted = calib.fit_calibration(scored, fit_temperature=not a.no_temperature)
        held = fitted["held_at_zero"]
        order_b = sorted(fitted["biases"].items(), key=lambda kv: -kv[1])

        # Out of fold, because biases can be fitted to noise in a way a single
        # temperature cannot. Each half is used once to fit and once to score,
        # so every row is answered by a fit that never saw it.
        accs = []
        for s in range(SPLITS):
            order = list(idx)
            random.Random(a.seed + s).shuffle(order)
            h = len(order) // 2
            halves = (order[:h], order[h:])
            for fit_ix, test_ix in (halves, halves[::-1]):
                f = calib.fit_calibration([scored[i] for i in fit_ix],
                                          fit_temperature=not a.no_temperature)
                hit = 0
                for i in test_ix:
                    lg, opts, ans = scored[i]
                    pr = calib.apply_calibration(lg, opts, f["temperature"],
                                                 f["biases"], warn=False)
                    hit += int(max(range(len(pr)), key=lambda k: pr[k]) == ans)
                accs.append(hit / max(len(test_ix), 1))
        oof = statistics.mean(accs)

        in_sample = 0
        for lg, opts, ans in scored:
            pr = calib.apply_calibration(lg, opts, fitted["temperature"],
                                         fitted["biases"], warn=False)
            in_sample += int(max(range(len(pr)), key=lambda k: pr[k]) == ans)
        in_sample /= len(scored)

        print(f"\nwith one bias per option, {len(fitted['options'])} options, "
              f"{'temperature fitted with them' if not a.no_temperature else 'temperature held at 1'}")
        print(f"temperature: {fitted['temperature']:.4f}")
        print(f"accuracy on the rows the biases were fitted on: {in_sample:.4f}")
        print(f"accuracy out of fold over {SPLITS} half-splits: {oof:.4f} "
              f"(from {min(accs):.3f} to {max(accs):.3f})")
        print(f"without biases it is {acc:.4f}")
        print(f"\nbiases, {held!r} held at zero:")
        for name, b in order_b:
            print(f"  {b:+.3f}  {name}")
        if oof <= acc:
            print("\nout of fold the biases do not help on these rows. "
                  "The lean they correct may not be there, or there are too few rows.")

        out["temperature"] = fitted["temperature"]
        out["biases"] = {k: round(v, 6) for k, v in fitted["biases"].items()}
        out["options"] = fitted["options"]
        out["bias_held_at_zero"] = held
        out["bias_temperature_fitted"] = not a.no_temperature
        out["accuracy_with_biases"] = round(oof, 4)

    if a.out:
        json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"\nwritten: {a.out}")


if __name__ == "__main__":
    main()
