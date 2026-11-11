#!/usr/bin/env python3
"""Do the two rankings of the ladder agree across unseen datasets?

    python3 -I results/analysis/ladder_rank.py

`results/unseen/ladder_bias.md` ranks the ten distinct checkpoints of the ladder
on the news rubrics and on the CEDR emotions, once each checkpoint is allowed
its own additive bias per option. The two rankings disagree. With two datasets
there is no way to tell whether either of them travels, so further datasets are
added and the rule for reading them was written down first, in
`results/unseen3/README.md`, before any row was downloaded.

What is printed, for every dataset that is present:

- the accuracy of each checkpoint as published and out of fold with biases,
  using the functions of `unseen_bias.py`, 25 half-splits seeded 0 to 24;
- the Spearman rank correlation of the ten checkpoints between every pair of
  datasets, separately with biases and without;
- within each dataset, the correlation between the gain a checkpoint takes from
  the biases and where it ends up, and between the two rankings themselves;
- six fixed pairs of checkpoints with bootstrap intervals over rows and over
  labels, seed 20261105, 5,000 draws;
- the spread between the two published runs of the v1.1 recipe and the cells
  where the difference between the releases is no larger than it.

On that spread: the third run of the recipe, `seed-base-s3`, has no surviving
weights, only its bench logits, so on these datasets the spread is over two
runs, not three. It is labelled as such wherever it appears.

At ten checkpoints a rank correlation is a description, not a test. No p-value
is attached to any of these numbers.

Deterministic, standard library only, no network, no weights.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)
import unseen_bias as ub                                          # noqa: E402

R = os.path.normpath(os.path.join(HERE, "..", ".."))
RES = os.path.join(R, "results")

# name, rows file, the value of the "set" field, the directory of the dumps
DATASETS = [
    ("news rubrics", "unseen/unseen_rows.jsonl", "рубрики новостей", "unseen/ladder"),
    ("emotions", "unseen/unseen_rows.jsonl", "эмоции", "unseen/ladder_emotions"),
    ("scientific rubrics", "unseen3/scibench/rows.jsonl", "научные рубрики", "unseen3/scibench"),
    ("film reviews", "unseen3/kinopoisk/rows.jsonl", "отзывы о фильмах", "unseen3/kinopoisk"),
]

# the ten distinct checkpoints; v1.3 is the same weights as v1.3-ep3of4 and is
# scored only as a control, so it is not a row of any ranking
TEN = ["v1.1", "seed-base-s2", "v1.2", "seed-aug-s2", "seed-aug-s2-ep3", "seed-aug-s3",
       "v1.3-ep1of4", "v1.3-ep2of4", "v1.3-ep3of4", "v1.3-ep4of4"]
CONTROL = "v1.3"

PAIRS = [
    ("seed-base-s2", "v1.1", "a second seed of the v1.1 recipe against the release"),
    ("v1.3-ep3of4", "v1.1", "the two released checkpoints"),
    ("v1.3-ep3of4", "seed-base-s2", "the later release against that seed"),
    ("v1.3-ep2of4", "v1.3-ep3of4", "the epoch before the released one against it"),
    ("v1.3-ep2of4", "v1.1", "that earlier epoch against the earlier release"),
    ("seed-aug-s2-ep3", "seed-aug-s2", "early against late epoch in another run"),
]


def ranks(xs):
    """Average ranks, ties shared."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            out[order[k]] = r
        i = j + 1
    return out


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return num / (da * db) if da and db else float("nan")


def load(name, rows_path, which, dumps):
    path = os.path.join(RES, rows_path)
    if not os.path.exists(path):
        return None
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r["set"] == which]
    if not rows:
        return None
    got = {}
    for tag in TEN + [CONTROL]:
        p = os.path.join(RES, dumps, tag + ".jsonl")
        if not os.path.exists(p):
            continue
        s = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        if len(s) != len(rows):
            raise SystemExit("%s: %d rows against %d" % (p, len(s), len(rows)))
        for r, x in zip(rows, s):
            if r["answer"] != x["answer"] or r["gold"] != x["gold"]:
                raise SystemExit("%s is not aligned with its rows file" % p)
        got[tag] = s
    missing = [t for t in TEN if t not in got]
    if missing:
        return {"name": name, "rows": rows, "missing": missing}
    return {"name": name, "rows": rows, "scored": got, "missing": []}


def main():
    sets = []
    for name, rows_path, which, dumps in DATASETS:
        d = load(name, rows_path, which, dumps)
        if d is None:
            print("## %s\n\nno rows or no dumps, skipped\n" % name)
            continue
        if d["missing"]:
            print("## %s\n\nmissing checkpoints %s, skipped\n" % (name, ", ".join(d["missing"])))
            continue
        sets.append(d)

    print("Do the rankings of the ladder agree across unseen datasets?")
    print("%d half-splits, seeds 0 to %d, both halves scored; bootstrap %d draws, seed %d.\n"
          % (ub.SPLITS, ub.SPLITS - 1, ub.BOOT, ub.BOOT_SEED))

    for d in sets:
        rows = d["rows"]
        got = d["scored"]
        samples = {t: [(x["logits"], [str(o) for o in r["options"]], int(r["answer"]), r["gold"])
                       for r, x in zip(rows, got[t])] for t in got}
        d["raw"] = {t: ub.raw_hits(samples[t]) for t in TEN}
        d["fix"] = {t: ub.out_of_fold(samples[t], True) for t in TEN}
        d["labels"] = [r["gold"] for r in rows]
        if CONTROL in got:
            same = all(a["pick"] == b["pick"] for a, b in zip(got[CONTROL], got["v1.3-ep3of4"]))
            d["control"] = "the same answer on every row" if same else "DIFFERENT ANSWERS"
        else:
            d["control"] = "not scored"

        print("\n## %s, %d rows, %d candidates\n"
              % (d["name"], len(rows), len(rows[0]["options"])))
        print("| Checkpoint | as published | with biases | change |")
        print("|---|---|---|---|")
        for t in TEN:
            r, f = ub.mean(d["raw"][t]), ub.mean(d["fix"][t])
            print("| %s | %.3f | %.3f | %+.3f |" % (t, r, f, f - r))
        best_r = max(TEN, key=lambda t: ub.mean(d["raw"][t]))
        best_f = max(TEN, key=lambda t: ub.mean(d["fix"][t]))
        print("\nBest as published `%s` at %.3f, best with biases `%s` at %.3f. "
              "Control, v1.3 against v1.3-ep3of4: %s."
              % (best_r, ub.mean(d["raw"][best_r]), best_f, ub.mean(d["fix"][best_f]),
                 d["control"]))

        gain = [ub.mean(d["fix"][t]) - ub.mean(d["raw"][t]) for t in TEN]
        final = [ub.mean(d["fix"][t]) for t in TEN]
        pub = [ub.mean(d["raw"][t]) for t in TEN]
        print("\nWithin this dataset: the gain from the biases against where a checkpoint "
              "ends up, Spearman %+.2f; the ranking as published against the ranking with "
              "biases, %+.2f." % (spearman(gain, final), spearman(pub, final)))

    if len(sets) > 1:
        print("\n## Spearman between datasets, over the ten checkpoints\n")
        for how, key in (("with biases", "fix"), ("as published", "raw")):
            print("\n%s:\n" % how)
            print("| | " + " | ".join(d["name"] for d in sets) + " |")
            print("|" + "---|" * (len(sets) + 1))
            for a in sets:
                cells = []
                for b in sets:
                    if a is b:
                        cells.append("1.00")
                    else:
                        cells.append("%+.2f" % spearman([ub.mean(a[key][t]) for t in TEN],
                                                        [ub.mean(b[key][t]) for t in TEN]))
                print("| %s | %s |" % (a["name"], " | ".join(cells)))

    print("\n## The six fixed pairs, after the biases, out of fold\n")
    for d in sets:
        print("\n### %s\n" % d["name"])
        print("| Pair | difference | 95% over rows | 95% over labels | What it asks |")
        print("|---|---|---|---|---|")
        for a, b, note in PAIRS:
            diff = [x - y for y, x in zip(d["fix"][b], d["fix"][a])]
            lo_r, hi_r = ub.ci_rows(diff, ub.BOOT_SEED)
            lo_l, hi_l = ub.ci_labels(diff, d["labels"], ub.BOOT_SEED)
            print("| %s minus %s | %+.3f | [%+.3f, %+.3f] | [%+.3f, %+.3f] | %s |"
                  % (a, b, ub.mean(diff), lo_r, hi_r, lo_l, hi_l, note))

    print("\n## The v1.1 recipe across two runs\n")
    print("The third run of this recipe has no surviving weights, so this is two runs, "
          "not three.\n")
    print("| Dataset | v1.1 | seed-base-s2 | spread | v1.3 | v1.3 minus v1.1 | within the spread |")
    print("|---|---|---|---|---|---|---|")
    for how, key in (("as published", "raw"), ("with biases", "fix")):
        print("| **%s** | | | | | | |" % how)
        for d in sets:
            a = ub.mean(d[key]["v1.1"])
            b = ub.mean(d[key]["seed-base-s2"])
            v13 = ub.mean(d[key]["v1.3-ep3of4"])
            spread = abs(a - b)
            print("| %s | %.3f | %.3f | %.3f | %.3f | %+.3f | %s |"
                  % (d["name"], a, b, spread, v13, v13 - a,
                     "yes" if abs(v13 - a) <= spread else "no"))

    print("\n## The reading rule of results/unseen3/README.md\n")
    base = {d["name"]: d for d in sets}
    for new in ("scientific rubrics", "film reviews"):
        if new not in base:
            print("- **%s**: not available, no verdict." % new)
            continue
        for old in ("news rubrics", "emotions"):
            if old not in base:
                continue
            rho = spearman([ub.mean(base[old]["fix"][t]) for t in TEN],
                           [ub.mean(base[new]["fix"][t]) for t in TEN])
            verdict = "survives" if rho >= 0.5 else "does not survive"
            print("- The **%s** ranking %s **%s**: Spearman %+.2f, the rule asks for at least 0.50."
                  % (old, verdict, new, rho))
    print("\nAt ten checkpoints this is a description and not a statistical test.")


if __name__ == "__main__":
    main()
