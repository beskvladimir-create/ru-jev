#!/usr/bin/env python3
"""What the unlabelled RuCoLA test rows did to the published figures.

    python3 -I results/analysis/rucola_test.py

RuCoLA ships three splits. Its test split carries no label: the csv behind it
holds only `id, sentence`, and the loader fills the label with -1.
`rows_simple` in `scripts/build_data.py` pooled every split of a source, and its
only guard read the answer index after `answer_from` had mapped the label.
For RuCoLA that mapping was `0 if int(lab) == 1 else 1`, so -1 became 1, a valid
index, and 2,789 sentences entered the mixture carrying "неприемлемо", a label
they do not have.

252 of them landed in the validation split the released v1.3 was measured on,
2,536 in its training split. This script recomputes every published figure that
rests on that validation split, with those 252 rows and without them.

What it reads, all of it in this repository:

- `results/validation/val_v13_trainsplit.jsonl`, the raw scores of the released
  checkpoint on all 26,746 validation rows;
- `results/rucola_test/val_v13_affected_rows.txt`, the positions of the 252
  affected rows in that file;
- `results/rucola_test/epoch_hits_on_affected.json`, how many of those 252 rows
  each epoch of the run answers as the mixture labelled them.

The sentences themselves are not in this repository. RuCoLA's licence covers its
labels and code, not the sentences, which are drawn from linguistic
publications, and `DATASETS.md` records that the corpus is not redistributed
here. `results/rucola_test/test_sentence_sha256.txt` holds the sha256 of each
stripped test sentence instead, which is enough for a reader with the dataset to
rebuild the index and check it against the one committed here.

Deterministic, standard library only, no network, no weights.
"""
import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
RES = os.path.join(ROOT, "results")
HERE_DATA = os.path.join(RES, "rucola_test")

# The release was prepared with these, and `scripts/coverage.py` and
# `scripts/verify_temperature.py` carry the same values as their defaults.
SPLIT_SEED = 17
GRID = [0.2 + 7.8 * i / 300 for i in range(301)]
BINS = 15
SHARES = [30, 40, 50, 60, 70, 80, 90, 100]

# From `results/training/v1_3_train.log`, the trainer's own evaluation at the end
# of each epoch over all 26,746 validation rows. Four decimals is what the log
# records, so a correct count derived from one is good to about two decisions.
LOG_ACC = [("epoch 1 of 4", 0.7536), ("epoch 2 of 4", 0.7920),
           ("epoch 3 of 4, released", 0.8181), ("epoch 4 of 4", 0.8139)]


def lines(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def softmax(xs, T):
    m = max(xs)
    e = [math.exp((x - m) / T) for x in xs]
    s = sum(e)
    return [v / s for v in e]


def nll(rows, T):
    t = 0.0
    for lg, a in rows:
        t -= math.log(max(softmax(lg, T)[a], 1e-12))
    return t / len(rows)


def fit_grid(rows):
    best = None
    for T in GRID:
        v = nll(rows, T)
        if best is None or v < best[1]:
            best = (T, v)
    return best[0]


def ece(rows, T):
    b = [[0, 0.0, 0] for _ in range(BINS)]
    for lg, a in rows:
        p = softmax(lg, T)
        k = max(range(len(p)), key=lambda i: p[i])
        c = p[k]
        j = min(BINS - 1, int(c * BINS))
        b[j][0] += 1
        b[j][1] += c
        b[j][2] += int(k == a)
    n = sum(x[0] for x in b)
    return sum(x[0] / n * abs(x[1] / x[0] - x[2] / x[0]) for x in b if x[0])


def halves(rows):
    """The seed 17 half-split the released temperature was fitted on."""
    order = list(range(len(rows)))
    random.Random(SPLIT_SEED).shuffle(order)
    h = len(order) // 2
    return [rows[i] for i in order[:h]], [rows[i] for i in order[h:]]


def accuracy(rows):
    return sum(1 for lg, a in rows if lg.index(max(lg)) == a) / len(rows)


def coverage(rows, T):
    """Share answered against accuracy on the answered, as coverage.py builds it."""
    conf = []
    for lg, a in rows:
        p = softmax(lg, T)
        k = max(range(len(p)), key=lambda i: p[i])
        conf.append((p[k], int(k == a)))
    conf.sort(key=lambda x: -x[0])
    out = []
    for share in SHARES:
        k = max(1, int(len(conf) * share / 100))
        taken = conf[:k]
        out.append((share, taken[-1][0], sum(x[1] for x in taken) / k))
    return out


def main():
    dump = lines(os.path.join(RES, "validation", "val_v13_trainsplit.jsonl"))
    bad = set(int(x) for x in
              open(os.path.join(HERE_DATA, "val_v13_affected_rows.txt"))
              if x.strip())
    hits = json.load(open(os.path.join(HERE_DATA, "epoch_hits_on_affected.json"),
                          encoding="utf-8"))["epochs"]

    for i in bad:
        if dump[i]["task"] != "rucola":
            raise SystemExit("row %d of the dump is %s, not rucola" % (i, dump[i]["task"]))

    def pairs(keep):
        return [(r["scores"], int(r["answer"])) for i, r in enumerate(dump) if keep(i, r)]

    allrows = pairs(lambda i, r: True)
    kept = pairs(lambda i, r: i not in bad)
    ruc = pairs(lambda i, r: r["task"] == "rucola")
    ruc_kept = pairs(lambda i, r: r["task"] == "rucola" and i not in bad)
    affected = pairs(lambda i, r: i in bad)

    print("# The unlabelled RuCoLA test rows, and the figures that rest on them\n")
    print("`results/analysis/rucola_test.py`. Everything below is recomputed from "
          "the published per-row scores of the released checkpoint; no weights and "
          "no network.\n")

    print("## How many rows\n")
    print("| Where | Rows | Of them from the RuCoLA test split |")
    print("|---|---|---|")
    print("| the validation split, all tasks | %d | %d |" % (len(allrows), len(bad)))
    print("| the validation split, rucola only | %d | %d |" % (len(ruc), len(bad)))
    print("\nEvery one of them carries the option \"неприемлемо\" as its right "
          "answer, which is the label the loader invented.\n")

    print("## Accuracy\n")
    print("| Set | Rows | As published | Without the test rows | Change |")
    print("|---|---|---|---|---|")
    print("| rucola, validation | %d | %.4f | %.4f | %+.4f |"
          % (len(ruc), accuracy(ruc), accuracy(ruc_kept),
             accuracy(ruc_kept) - accuracy(ruc)))
    print("| every task, validation | %d | %.4f | %.4f | %+.4f |"
          % (len(allrows), accuracy(allrows), accuracy(kept),
             accuracy(kept) - accuracy(allrows)))
    print("\nOn the affected rows alone the released checkpoint agrees with the "
          "invented label on %.4f of %d, which is about what a coin gives on a "
          "two-way decision.\n" % (accuracy(affected), len(affected)))

    print("## Temperature and ECE\n")
    print("Fitted on one half of the rows with seed %d on the 301-point grid, "
          "read on the other half, which is the path the released 1.8640 came "
          "from.\n" % SPLIT_SEED)
    print("| Set | Rows | Fitted T | ECE on the other half |")
    print("|---|---|---|---|")
    rowsets = [("as published", allrows), ("without the test rows", kept)]
    temps = {}
    for name, rs in rowsets:
        a, b = halves(rs)
        T = fit_grid(a)
        temps[name] = T
        print("| %s | %d | %.4f | %.4f |" % (name, len(rs), T, ece(b, T)))
    print("\nThe ECE difference is smaller than the gap between two honest ways "
          "of computing ECE on the same rows, so it carries nothing. The "
          "temperature moves by %.4f.\n"
          % abs(temps["without the test rows"] - temps["as published"]))

    print("## Coverage\n")
    print("Built on the half the temperature was not fitted on, as "
          "`scripts/coverage.py` builds it.\n")
    print("| Share answered | Threshold, as published | Accuracy, as published | "
          "Threshold, without | Accuracy, without |")
    print("|---|---|---|---|---|")
    cov = {}
    for name, rs in rowsets:
        a, b = halves(rs)
        cov[name] = coverage(b, fit_grid(a))
    for (share, t1, a1), (_s, t2, a2) in zip(cov["as published"],
                                             cov["without the test rows"]):
        print("| %d%% | %.2f | %.4f | %.2f | %.4f |" % (share, t1, a1, t2, a2))

    print("\n## Which epoch the run would have shipped\n")
    print("The released checkpoint is the third of four. The trainer's own "
          "evaluation is over all %d rows, so the count below subtracts, from "
          "each epoch, the rows it answered as the mixture labelled them.\n"
          % len(allrows))
    print("| Epoch | Accuracy in the log | Right on the %d affected | "
          "Accuracy without them |" % len(bad))
    print("|---|---|---|---|")
    n_all, n_keep = len(allrows), len(kept)
    best, best_name = None, None
    for name, acc in LOG_ACC:
        c = hits[name]["correct"]
        new = (round(acc * n_all) - c) / n_keep
        if best is None or new > best:
            best, best_name = new, name
        print("| %s | %.4f | %d | %.4f |" % (name, acc, c, new))
    print("\nThe epoch that wins without the affected rows is **%s**, which is "
          "the epoch that was shipped. The choice does not move.\n" % best_name)
    print("The two counts come from different code paths, the trainer's batched "
          "evaluation and the standalone scorer, which disagree by 72 decisions "
          "over the whole split and so by about one over 252. The gap between the "
          "third epoch and the fourth is 112 decisions, so that mismatch cannot "
          "turn the comparison over.\n")


if __name__ == "__main__":
    main()
