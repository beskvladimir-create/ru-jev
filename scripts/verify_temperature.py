#!/usr/bin/env python3
"""Reproduce the released temperature of a checkpoint from its own validation split.

The trainer fits the temperature by minimising the negative log-likelihood over
the validation split with a coarse grid followed by narrowing, masking the
padded candidate slots after the division by T. This script repeats that and
compares the result with the number recorded in calibration.json, so the
released figure can be checked rather than taken on trust.

Accuracy does not depend on the temperature: the softmax over candidates is
monotone in it and the argmax does not move. Only the probability next to the
answer changes.

Two ways to run it. With a precomputed score dump, which is fast:

  python scripts/verify_temperature.py --model model/ru-jev-v1.3/ship \\
      --scores results/validation/val_v13.jsonl

Or scoring the rows here, which takes about half an hour on CPU for the full
validation split:

  python scripts/verify_temperature.py --model model/ru-jev-v1.3/ship \\
      --rows data/val.jsonl
"""
import argparse
import json
import math
import os


def nll(rows, T):
    s = 0.0
    for lg, ans in rows:
        p = [x / T for x in lg]
        m = max(p)
        s -= (p[ans] - m - math.log(sum(math.exp(x - m) for x in p)))
    return s / len(rows)


def fit_grid(rows, lo=0.2, hi=8.0, steps=300):
    """The 301-point grid the release was prepared with.

    The released 1.8640 is a point of exactly this grid: 0.2 + 7.8 * 64 / 300.
    Using a different grid, such as the narrowing search in scripts/train.py,
    lands a few thousandths away and would make an exact check impossible.
    """
    return min(((nll(rows, lo + (hi - lo) * i / steps), lo + (hi - lo) * i / steps)
                for i in range(steps + 1)))[1]


def fit_narrow(rows, lo=0.25, hi=6.0, points=48, rounds=6):
    """Coarse grid then narrowing, the criterion scripts/train.py uses."""
    best = 1.0
    for _ in range(rounds):
        step = (hi - lo) / points
        cand = [lo + step * i for i in range(points + 1)]
        best = min(cand, key=lambda t: nll(rows, t))
        lo, hi = max(0.05, best - step), best + step
    return best


def from_dump(path):
    out = []
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        # score_rows.py writes "logits"; the dump the release was prepared from
        # writes "scores". Both are raw scores before any temperature.
        sc = r.get("logits", r.get("scores"))
        if sc is None:
            raise SystemExit("no raw scores in the dump: expected logits or scores")
        out.append((sc, int(r["answer"])))
    return out


def from_model(model, rows_path):
    import torch
    import torch.nn as nn
    from transformers import AutoModel, AutoTokenizer

    def load_head(path):
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
            self.head.load_state_dict(load_head(path))

        def forward(self, ids, am):
            cls = self.backbone(input_ids=ids, attention_mask=am).last_hidden_state[:, 0]
            return self.head(cls).squeeze(-1)

    cal = json.load(open(os.path.join(model, "calibration.json")))
    max_len = int(cal.get("max_len", 256))
    tok = AutoTokenizer.from_pretrained(model)
    net = Scorer(model).eval()
    out = []
    rows = [json.loads(l) for l in open(rows_path, encoding="utf-8") if l.strip()]
    with torch.no_grad():
        for i, r in enumerate(rows):
            text = r.get("input", r.get("text"))
            opts = [str(o) for o in r["options"]]
            enc = tok([text] * len(opts), opts, truncation=True,
                      max_length=max_len, padding=True, return_tensors="pt")
            lg = net(enc["input_ids"], enc["attention_mask"]).float().tolist()
            out.append((lg, int(r["answer"])))
            if (i + 1) % 2000 == 0:
                print(f"  scored {i+1}/{len(rows)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="checkpoint directory")
    ap.add_argument("--scores", help="precomputed per-row scores from score_rows.py")
    ap.add_argument("--rows", help="validation rows, scored here if --scores is absent")
    ap.add_argument("--tolerance", type=float, default=0.01,
                    help="how far the refit may fall from the recorded value")
    ap.add_argument("--grid", choices=("release", "trainer"), default="release",
                    help="which temperature search to use: the 301-point grid the "
                         "release was prepared with, or the narrowing search in "
                         "train.py")
    ap.add_argument("--seed", type=int, default=17,
                    help="half-split seed. The released temperature was fitted on "
                         "one half of the validation split with seed 17, so the "
                         "default reproduces it; 0 fits on every row instead")
    a = ap.parse_args()

    if not a.scores and not a.rows:
        raise SystemExit("give either --scores or --rows")
    rows = from_dump(a.scores) if a.scores else from_model(a.model, a.rows)

    recorded = json.load(open(os.path.join(a.model, "calibration.json")))
    expected = float(recorded["temperature"])

    # The released temperature was fitted on one half of the validation split,
    # not on all of it, so the check has to use the same half. Fitting on every
    # row gives a different number and would fail a comparison that is correct.
    if a.seed:
        import random
        order = list(range(len(rows)))
        random.Random(a.seed).shuffle(order)
        fit_rows = [rows[i] for i in order[:len(order) // 2]]
    else:
        fit_rows = rows
    T = (fit_grid if a.grid == "release" else fit_narrow)(fit_rows)
    acc = sum(lg.index(max(lg)) == ans for lg, ans in rows) / len(rows)

    print(f"\nrows {len(rows)}, temperature fitted on {len(fit_rows)}")
    print(f"accuracy {acc:.4f} (independent of the temperature)")
    print(f"recorded temperature {expected:.4f}  (calibration.json, epoch "
          f"{recorded.get('epoch', 'not stated')})")
    print(f"refitted temperature {T:.4f}")
    print(f"difference {abs(T - expected):.4f}, grid {a.grid}, "
          f"NLL at the refit {nll(fit_rows, T):.4f}")
    if abs(T - expected) <= a.tolerance:
        print(f"\nOK: within {a.tolerance}")
        return 0
    print(f"\nMISMATCH: outside {a.tolerance}. The checkpoint, the validation "
          f"split or the fitting criterion is not the one the figure came from.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
