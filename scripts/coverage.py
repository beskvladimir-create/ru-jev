#!/usr/bin/env python3
"""Coverage against accuracy: how much of a stream can be answered, and how well.

The temperature is fitted on one half of the rows and the curve is built on the
other, so the threshold is not chosen on the same data it is scored on.

The half-split seed is part of the published result: with 17, the seed the
release was prepared with, this reproduces the temperature 1.8640, the accuracy
0.8234 and the coverage table of the paper exactly.

Accuracy does not depend on the temperature, because the softmax over candidates
is monotone in it and the argmax does not move. The temperature decides where a
given coverage level falls, not how many answers are right.

Input: one JSON object per decision, as written by score_rujev.py, with raw
logits and the index of the correct candidate:

  {"logits": [...], "answer": 0, "task": "russe"}

Usage:
  python scripts/coverage.py results/validation/val_v13.jsonl
"""
import argparse
import json
import math
import random
from collections import defaultdict


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def softmax(xs, T):
    m = max(xs)
    e = [math.exp((x - m) / T) for x in xs]
    s = sum(e)
    return [v / s for v in e]


def scores_of(r):
    """Raw scores for a row. score_rujev.py writes "logits"; "scores" is accepted
    for older dumps, and "probs" for a model that returns probabilities."""
    for key in ("logits", "scores"):
        if key in r:
            return r[key]
    return [math.log(max(x, 1e-12)) for x in r["probs"]]


def nll(rows, T):
    return sum(-math.log(max(softmax(scores_of(r), T)[r["answer"]], 1e-12))
               for r in rows) / len(rows)


def fit_temperature(rows, lo=0.2, hi=8.0, steps=300):
    return min(((nll(rows, lo + (hi - lo) * i / steps), lo + (hi - lo) * i / steps)
                for i in range(steps + 1)))[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows", help="per-row scores from score_rujev.py")
    ap.add_argument("--seed", type=int, default=17,
                    help="seed of the half-split, recorded in the output. 17 is "
                         "the value the released temperature and the published "
                         "coverage table were produced with; changing it moves "
                         "the curve by a few thousandths")
    ap.add_argument("--target", type=float, default=0.95,
                    help="accuracy floor for the per-shape table")
    a = ap.parse_args()

    rows = load(a.rows)
    random.Random(a.seed).shuffle(rows)
    half = len(rows) // 2
    T = fit_temperature(rows[:half])
    test = rows[half:]

    scored = []
    for r in test:
        p = softmax(scores_of(r), T)
        i = max(range(len(p)), key=lambda k: p[k])
        scored.append((p[i], int(i == r["answer"]), r.get("task", "?"), len(p)))
    scored.sort(key=lambda t: -t[0])
    n = len(scored)

    print(f"{a.rows.split('/')[-1]}  T={T:.4f}  split seed {a.seed}  "
          f"fitted on {half}  scored on {n}\n")
    print(f"Accuracy on all scored rows: {sum(x[1] for x in scored) / n:.4f}\n")
    print("| Share answered | Threshold | Accuracy on answered | Errors per 1000 |")
    print("|---|---|---|---|")
    for cov in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        k = max(1, int(n * cov))
        sub = scored[:k]
        acc = sum(x[1] for x in sub) / k
        print(f"| {cov * 100:.0f}% | {sub[-1][0]:.2f} | {acc:.4f} | {(1 - acc) * 1000:.0f} |")

    print(f"\nBy decision shape, at accuracy on answered no lower than {a.target}:")
    groups = defaultdict(list)
    for conf, ok, task, k in scored:
        shape = "binary" if k == 2 else ("choice of 3 to 5" if k <= 5 else "choice of many")
        groups[shape].append((conf, ok))
    for shape in sorted(groups):
        s = sorted(groups[shape], key=lambda t: -t[0])
        m = len(s)
        best, thr = 0, 1.0
        for k in range(1, m + 1):
            if sum(x[1] for x in s[:k]) / k >= a.target:
                best, thr = k, s[k - 1][0]
        print(f"  {shape:18s} n={m:6d}  coverage {best / m * 100:5.1f}%  threshold {thr:.3f}")


if __name__ == "__main__":
    main()
