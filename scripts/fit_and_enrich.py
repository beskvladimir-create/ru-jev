#!/usr/bin/env python3
"""Fit a checkpoint's temperature the way train.py does, then add p_pick at
that temperature to the bench rows.

Why. A run stopped right after an epoch has no fitted temperature of its own:
calibration.json carries a placeholder. The published confidence and gap figures
are computed at the model's own temperature, so to make the comparison the same
rather than merely similar, the temperature is fitted on that checkpoint's own
validation split by grid search on the negative log-likelihood, which is the
criterion scripts/train.py uses.

Output: the bench file with p_pick and probs added, ready for aggregate.py.
"""
import argparse, json, math

ap = argparse.ArgumentParser()
ap.add_argument("--val", required=True, help="scores on the checkpoint's validation split")
ap.add_argument("--bench", required=True, help="scores on the wording bench")
ap.add_argument("--out", required=True)
a = ap.parse_args()


def rows(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def nll(data, T):
    s = 0.0
    for r in data:
        lg = r["logits"]
        m = max(x / T for x in lg)
        z = sum(math.exp(x / T - m) for x in lg)
        s -= (lg[r["answer"]] / T - m - math.log(z))
    return s / len(data)


val = rows(a.val)
# Coarse grid as in train.py, then narrowing around the minimum found.
lo, hi = 0.25, 6.0
for _ in range(6):
    step = (hi - lo) / 48
    cand = [lo + step * i for i in range(49)]
    best = min(cand, key=lambda t: nll(val, t))
    lo, hi = max(0.05, best - step), best + step
T = best

vacc = sum(r["pick"] == r["answer"] for r in val) / len(val)
per = {}
for r in val:
    t = r.get("task", "?")
    c, n = per.get(t, (0, 0))
    per[t] = (c + int(r["pick"] == r["answer"]), n + 1)

bench = rows(a.bench)
with open(a.out, "w", encoding="utf-8") as fh:
    for r in bench:
        lg = r["logits"]
        m = max(x / T for x in lg)
        e = [math.exp(x / T - m) for x in lg]
        z = sum(e)
        p = [x / z for x in e]
        r["probs"] = [round(x, 6) for x in p]
        r["p_pick"] = round(p[r["pick"]], 6)
        r["T_fitted"] = round(T, 4)
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")

bacc = sum(r["pick"] == r["answer"] for r in bench) / len(bench)
print(f"{a.val.split('/')[-1]}")
print(f"  temperature from validation: {T:.4f}  (NLL {nll(val, T):.4f})")
print(f"  whole validation: {vacc:.4f} on {len(val)} rows")
print(f"  whole bench:      {bacc:.4f} on {len(bench)} rows")
print("  by task: " + ", ".join(f"{t} {c/n:.3f} ({n})" for t, (c, n) in sorted(per.items())))
