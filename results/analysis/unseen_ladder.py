#!/usr/bin/env python3
"""Where between v1.1 and v1.3 the economy and politics boundary moves.

    python3 -I results/analysis/unseen_ladder.py results/unseen

Why this exists. After per-option biases are fitted, v1.3 still trails v1.1 on
the news rubrics, and the per-label recall in `bias.md` says the gap sits almost
entirely on two labels: `политика` and `экономика`. The question this script
answers is at which step of the training line that boundary gets worse, so the
answer is a step rather than a release.

What is measured. For every checkpoint, on the same 1,000 headlines:

- accuracy, and the count of `политика` answers, and recall for all six rubrics;
- the difference `logit(экономика) - logit(политика)` on each headline, which is
  the boundary itself, read before any softmax and so independent of the
  temperature;
- the AUC of that difference over the 340 headlines whose true label is one of
  those two rubrics, which is the probability that a random economic headline
  gets a higher difference than a random political one;
- a paired bootstrap of the AUC difference against v1.1, resampling the same 340
  headlines, 2,000 draws from seed 20261105;
- the median difference on economic and on political headlines separately, the
  median of the paired change against v1.1 on the same two groups, and the
  number of economic headlines where `политика` outscores `экономика`.

The checkpoints are read from `results/unseen/ladder/*.jsonl`, which hold the
raw logits of each one on these rows. The weights themselves are not published:
the intermediate checkpoints are working artefacts, and the two released ones
are on Hugging Face.

Deterministic, standard library only, no network.
"""
import json
import os
import random
import sys

BOOT = 2000
BOOT_SEED = 20261105
SET = "рубрики новостей"
POS, NEG = "экономика", "политика"

# The ladder, in training order. v1.3-ep3of4 is the released checkpoint; it was
# scored twice, once from the epoch directory and once from the ship directory,
# as a control on the scoring path.
ORDER = [
    ("v1.1", "the published v1.1"),
    ("seed-base-s2", "a second seed of the v1.1 recipe"),
    ("v1.2", "v1.2, phrase wordings added"),
    ("seed-aug-s2", "v1.3 schedule on a v1.2-level mixture, seed 2"),
    ("seed-aug-s2-ep3", "the same run at epoch 3"),
    ("seed-aug-s3", "v1.3 schedule on a v1.2-level mixture, seed 3"),
    ("v1.3-ep1of4", "v1.3 training, after epoch 1"),
    ("v1.3-ep2of4", "v1.3 training, after epoch 2"),
    ("v1.3-ep3of4", "v1.3 training, after epoch 3, the released weights"),
    ("v1.3", "the published v1.3, scored from the ship directory"),
    ("v1.3-ep4of4", "v1.3 training, after epoch 4, overtrained"),
]


def load_rows(directory):
    rows = [json.loads(l) for l in open(os.path.join(directory, "unseen_rows.jsonl"),
                                        encoding="utf-8") if l.strip()]
    return [r for r in rows if r["set"] == SET]


def load_scores(directory, tag, rows):
    path = os.path.join(directory, "ladder", tag + ".jsonl")
    if not os.path.exists(path):
        return None
    s = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    if len(s) != len(rows):
        raise SystemExit("%s has %d rows against %d" % (path, len(s), len(rows)))
    for r, x in zip(rows, s):
        if r["answer"] != x["answer"] or r["gold"] != x["gold"]:
            raise SystemExit("%s is not aligned with unseen_rows.jsonl" % path)
    return s


def margin(row, scored):
    """logit(экономика) - logit(политика) for one headline."""
    i = row["options"].index(POS)
    j = row["options"].index(NEG)
    return scored["logits"][i] - scored["logits"][j]


def auc(pos, neg):
    """Probability that a random positive outranks a random negative, ties half."""
    xs = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1][0] == xs[i][0]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = r
        i = j + 1
    sum_pos = sum(r for r, (_, lab) in zip(ranks, xs) if lab == 1)
    n_p, n_n = len(pos), len(neg)
    return (sum_pos - n_p * (n_p + 1) / 2.0) / (n_p * n_n)


def median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def main():
    directory = sys.argv[1] if len(sys.argv) > 1 else "results/unseen"
    rows = load_rows(directory)
    n = len(rows)
    labels = sorted({r["gold"] for r in rows})
    two = [i for i, r in enumerate(rows) if r["gold"] in (POS, NEG)]
    is_pos = [rows[i]["gold"] == POS for i in two]

    present = []
    stats = {}
    for tag, note in ORDER:
        s = load_scores(directory, tag, rows)
        if s is None:
            continue
        present.append((tag, note))
        m = [margin(r, x) for r, x in zip(rows, s)]
        picks = [r["options"][max(range(len(x["logits"])), key=lambda k: x["logits"][k])]
                 for r, x in zip(rows, s)]
        hit = [p == r["gold"] for p, r in zip(picks, rows)]
        rec = {}
        for lab in labels:
            idx = [i for i, r in enumerate(rows) if r["gold"] == lab]
            rec[lab] = sum(hit[i] for i in idx) / len(idx)
        stats[tag] = {
            "margin": m,
            "acc": sum(hit) / n,
            "picked_pol": sum(p == NEG for p in picks),
            "recall": rec,
            "auc": auc([m[i] for i, p in zip(two, is_pos) if p],
                       [m[i] for i, p in zip(two, is_pos) if not p]),
            "med_pos": median([m[i] for i, p in zip(two, is_pos) if p]),
            "med_neg": median([m[i] for i, p in zip(two, is_pos) if not p]),
            "econ_lost": sum(1 for i, p in zip(two, is_pos) if p and m[i] < 0),
        }

    base = "v1.1"
    if base not in stats:
        raise SystemExit("v1.1 is missing; the ladder is read against it")

    # Paired bootstrap of the AUC difference against v1.1, over the 340 headlines.
    for tag, _note in present:
        if tag == base:
            stats[tag]["ci"] = (0.0, 0.0)
            continue
        rng = random.Random(BOOT_SEED)
        draws = []
        k = len(two)
        for _ in range(BOOT):
            pick = [rng.randrange(k) for _ in range(k)]
            p_i = [two[i] for i in pick if is_pos[i]]
            n_i = [two[i] for i in pick if not is_pos[i]]
            if not p_i or not n_i:
                continue
            a = auc([stats[tag]["margin"][i] for i in p_i],
                    [stats[tag]["margin"][i] for i in n_i])
            b = auc([stats[base]["margin"][i] for i in p_i],
                    [stats[base]["margin"][i] for i in n_i])
            draws.append(a - b)
        draws.sort()
        stats[tag]["ci"] = (draws[int(0.025 * len(draws))],
                            draws[int(0.975 * len(draws)) - 1])

    print("Where the economy and politics boundary moves")
    print("%d headlines, %d of them %s or %s; bootstrap %d draws, seed %d\n"
          % (n, len(two), POS, NEG, BOOT, BOOT_SEED))

    print("| Checkpoint | Accuracy | AUC экономика vs политика | AUC minus v1.1 | 95% over headlines |")
    print("|---|---|---|---|---|")
    for tag, _note in present:
        s = stats[tag]
        d = s["auc"] - stats[base]["auc"]
        ci = "" if tag == base else "[%+.3f, %+.3f]" % s["ci"]
        print("| %s | %.3f | %.3f | %s | %s |"
              % (tag, s["acc"], s["auc"], "" if tag == base else "%+.3f" % d, ci))

    print("\n| Checkpoint | median margin, economic | median margin, political | "
          "median paired change vs v1.1, economic | the same, political | "
          "economic headlines lost to политика | answers политика |")
    print("|---|---|---|---|---|---|---|")
    base_m = stats[base]["margin"]
    for tag, _note in present:
        s = stats[tag]
        dp = median([s["margin"][i] - base_m[i] for i, p in zip(two, is_pos) if p])
        dn = median([s["margin"][i] - base_m[i] for i, p in zip(two, is_pos) if not p])
        print("| %s | %+.2f | %+.2f | %+.2f | %+.2f | %d | %d |"
              % (tag, s["med_pos"], s["med_neg"], dp, dn, s["econ_lost"], s["picked_pol"]))

    print("\n| Checkpoint | " + " | ".join(labels) + " |")
    print("|" + "---|" * (len(labels) + 1))
    for tag, _note in present:
        print("| %s | " % tag + " | ".join("%.3f" % stats[tag]["recall"][l] for l in labels) + " |")

    print("\nWhat each checkpoint is:\n")
    for tag, note in present:
        print("- `%s`: %s" % (tag, note))


if __name__ == "__main__":
    main()
