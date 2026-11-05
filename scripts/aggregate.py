#!/usr/bin/env python3
"""Summary of the wording bench for every model on identical rows.

The temperature is fitted by grid search on the negative log-likelihood of one
half and the ECE is measured on the other, averaged over 25 random half-splits.
This is the method the model card uses; any other would make the figures
incomparable between models.

The vocabulary names are the keys used in the score dumps and stay in Russian;
the printed table gives the English name of each alongside.
"""
import json, math, random, statistics, sys, collections

# Data keys, as written in the score dumps, with the English name used in the paper.
VOCAB_ORDER = ["русское слово", "английский ключ", "описание запроса",
               "русская фраза", "названия служб", "жаргон"]
VOCAB_EN = {"русское слово": "Russian word", "английский ключ": "English key",
            "описание запроса": "Descriptive sentence", "русская фраза": "Russian phrase",
            "названия служб": "Department name", "жаргон": "Anglicism jargon"}
SPLITS = 25
# NOTE. An earlier implementation of the splits here shuffled its index list IN
# PLACE and never reset it between splits, so split s was applied to the
# permutation left by s-1 and the result depended on the order of rows in the
# file. Calibration now lives in calib.py, which resets the indices, derives
# each split seed from a recorded SPLIT_SEED and puts rows in a canonical order.
from calib import calibration as _calibration  # noqa: E402

BINS = 15


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def nll(rows, T):
    s = 0.0
    for r in rows:
        lg = r.get("logits")
        if lg is not None:
            m = max(x / T for x in lg)
            z = sum(math.exp(x / T - m) for x in lg)
            s -= (lg[r["answer"]] / T - m - math.log(z))
        else:  # probabilities without logits: take log p and divide that by T
            p = [math.log(max(x, 1e-12)) / T for x in r["probs"]]
            m = max(p); z = sum(math.exp(x - m) for x in p)
            s -= (p[r["answer"]] - m - math.log(z))
    return s / max(len(rows), 1)


def fitT(rows):
    # Rows where the probability of the correct candidate arrived as zero,
    # because the model's output was rounded to four decimals, are left out of
    # the fit: log(0) would drag the temperature upwards. Their count is
    # reported separately.
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
    lg = r.get("logits")
    src = [x / T for x in lg] if lg is not None else [math.log(max(x, 1e-12)) / T for x in r["probs"]]
    m = max(src); e = [math.exp(x - m) for x in src]; z = sum(e)
    return [x / z for x in e]


def ece(rows, T):
    buckets = [[] for _ in range(BINS)]
    for r in rows:
        p = probs_at(r, T)
        pick = max(range(len(p)), key=lambda j: p[j])
        conf = p[pick]
        b = min(int(conf * BINS), BINS - 1)
        buckets[b].append((conf, int(pick == r["answer"])))
    n = sum(len(b) for b in buckets)
    return sum(len(b) / n * abs(sum(c for c, _ in b) / len(b) - sum(h for _, h in b) / len(b))
               for b in buckets if b)



def _wrap(rows):
    """Returns (T, ECE after its own T, raw ECE), as the old signature did."""
    c = _calibration(rows)
    return c["T_mean"], c["ECE_own_T"], c["ECE_raw"]

calibration = calib = _wrap


def main(files):
    models = {}
    for tag, path in files:
        models[tag] = load(path)

    print("## Wording bench: the same 1,000 questions, 18 categories, "
          "five candidates per question\n")
    head = "| Vocabulary | " + " | ".join(models) + " |"
    print(head)
    print("|" + "---|" * (len(models) + 1))
    for v in VOCAB_ORDER:
        cells = []
        for tag, rows in models.items():
            rs = [r for r in rows if r["vocab"] == v]
            acc = sum(r["pick"] == r["answer"] for r in rs) / len(rs)
            cells.append(f"{acc:.3f}")
        print(f"| {VOCAB_EN.get(v, v)} | " + " | ".join(cells) + " |")

    print("\n### Anglicism jargon in detail\n")
    print("| Model | 18 labels | 15 clean | Mean confidence | Confidence minus accuracy | Confident errors per 1000 |")
    print("|---|---|---|---|---|---|")
    for tag, rows in models.items():
        rs = [r for r in rows if r["vocab"] == "жаргон"]
        cl = [r for r in rs if r["gold_clean"]]
        acc = sum(r["pick"] == r["answer"] for r in rs) / len(rs)
        acc_cl = sum(r["pick"] == r["answer"] for r in cl) / len(cl)
        conf = sum(r["p_pick"] for r in rs) / len(rs)
        over = sum(1 for r in rs if r["pick"] != r["answer"] and r["p_pick"] > 0.8)
        print(f"| {tag} | {acc:.3f} | {acc_cl:.3f} | {conf:.3f} | {conf-acc:+.3f} | {over} |")

    print("\n### Calibration on all 6,000 bench rows\n")
    print("| Model | ECE as is | Fitted T | ECE after its own T |")
    print("|---|---|---|---|")
    for tag, rows in models.items():
        T, e_out, e_raw = calibration(rows)
        dropped = sum(1 for r in rows
                      if r.get("logits") is None and r["probs"][r["answer"]] <= 0)
        note = f" (rows left out of the fit: {dropped})" if dropped else ""
        print(f"| {tag} | {e_raw:.4f} | {T:.3f} | {e_out:.4f} |{note}")


if __name__ == "__main__":
    main([(a.split("=")[0], a.split("=", 1)[1]) for a in sys.argv[1:]])
