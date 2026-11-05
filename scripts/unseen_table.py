#!/usr/bin/env python3
"""Summary of the datasets no model trained on.

The method is the one in aggregate.py, so the figures stay comparable with the
wording bench: the temperature is fitted by grid search on the negative
log-likelihood of one half, the ECE is measured on the other, averaged over 25
half-splits.

Mean confidence uses each model's OWN temperature: for Ru-Jev the released one
from calibration.json, for Laya its own output, which is already calibrated.
"""
import json, math, random, statistics, sys

SPLITS, BINS = 25, 15

# Calibration lives in calib.py. An earlier copy of it here shuffled its index
# list in place without resetting it between splits, so split s was applied to
# the permutation left by s-1 and the result depended on the order of rows in
# the file. calib.py resets the indices, derives each split seed from a
# recorded SPLIT_SEED and puts rows in a canonical order.
from calib import calibration as _calibration  # noqa: E402


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def src_logprobs(r, T):
    lg = r.get("logits")
    if lg is not None:
        return [x / T for x in lg]
    return [math.log(max(x, 1e-12)) / T for x in r["probs"]]


def nll(rows, T):
    s = 0.0
    for r in rows:
        p = src_logprobs(r, T)
        m = max(p)
        z = sum(math.exp(x - m) for x in p)
        s -= (p[r["answer"]] - m - math.log(z))
    return s / max(len(rows), 1)


def fitT(rows):
    rows = [r for r in rows
            if r.get("logits") is not None or r["probs"][r["answer"]] > 0]
    best, bt = None, 1.0
    for i in range(301):
        T = 0.2 + 7.8 * i / 300
        v = nll(rows, T)
        if best is None or v < best:
            best, bt = v, T
    return bt


def probs_at(r, T):
    p = src_logprobs(r, T)
    m = max(p)
    e = [math.exp(x - m) for x in p]
    z = sum(e)
    return [x / z for x in e]


def ece(rows, T):
    b = [[] for _ in range(BINS)]
    for r in rows:
        p = probs_at(r, T)
        j = max(range(len(p)), key=lambda k: p[k])
        b[min(int(p[j] * BINS), BINS - 1)].append((p[j], int(j == r["answer"])))
    n = sum(len(x) for x in b)
    return sum(len(x) / n * abs(sum(c for c, _ in x) / len(x) - sum(h for _, h in x) / len(x))
               for x in b if x)



def _wrap(rows):
    """Returns (T, ECE after its own T, raw ECE), as the old signature did."""
    c = _calibration(rows)
    return c["T_mean"], c["ECE_own_T"], c["ECE_raw"]

calibration = calib = _wrap


# Each model's own temperature: the released one for Ru-Jev, 1.0 for Laya,
# which returns calibrated probabilities itself.
OWN_T = {"v1.3": 1.8640, "v1.1": 2.2754}


def main(args):
    models = [(a.split("=")[0], a.split("=", 1)[1]) for a in args]
    data = {t: load(p) for t, p in models}
    sets = []
    for rows in data.values():
        for r in rows:
            if r.get("set") and r["set"] not in sets:
                sets.append(r["set"])

    for s in sets:
        print(f"\n## {s}\n")
        print("| Model | Accuracy | Mean confidence | Gap | ECE as is | "
              "Fitted T | ECE after its own T |")
        print("|---|---|---|---|---|---|---|")
        for tag, rows in data.items():
            rs = [r for r in rows if r.get("set") == s]
            if not rs:
                continue
            acc = sum(r["pick"] == r["answer"] for r in rs) / len(rs)
            T0 = OWN_T.get(tag, 1.0)
            conf = statistics.mean(probs_at(r, T0)[r["pick"]] for r in rs)
            Tf, e_out, e_raw = calib(rs)
            print(f"| {tag} | {acc:.3f} | {conf:.3f} | {conf-acc:+.3f} | "
                  f"{e_raw:.4f} | {Tf:.3f} | {e_out:.4f} |")
        first = [r for r in next(iter(data.values())) if r.get("set") == s]
        n = len(first)
        # The candidate count comes from the length of the scores: the options
        # themselves are not written to the score dump, they live in the rows file.
        k = len(first[0].get("logits") or first[0]["probs"])
        print(f"\nRows {n}, candidates {k}, chance {1/k:.3f}.")


if __name__ == "__main__":
    main(sys.argv[1:])
