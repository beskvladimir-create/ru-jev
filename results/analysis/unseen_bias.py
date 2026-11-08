#!/usr/bin/env python3
"""Per-option biases on the two unseen datasets.

    python3 -I results/analysis/unseen_bias.py results/unseen

Why this exists. On the news rubrics v1.3 scores below v1.1, and the per-label
counts say most of that is not misreading: v1.3 answers "политика" 318 times
where only 166 headlines are about politics, against 232 for v1.1, and its
recall on "экономика" falls from 0.730 to 0.448. A lean towards one label is
not something the temperature can touch, because dividing every score by the
same number does not move the argmax. One additive bias per option does.

What is measured. For each dataset and each release: accuracy as published,
then accuracy after fitting biases, once with the temperature fitted alongside
them and once with it held at 1. Nothing is fitted and scored on the same rows.
Each of 25 half-splits is used twice, once in each direction, so every row is
answered 25 times by a fit that never saw it, and the accuracy printed is the
mean of those answers over every row.

The paired difference v1.3 minus v1.1 comes with two intervals. The one over
rows treats rows as the unit and says whether the gap is real given this
sample of headlines. The one over labels resamples whole labels and says
whether it survives a different choice of six rubrics, which is the harder
question and the one that matters for a claim about the release.

Deterministic: the splits are seeded 0 to 24, the bootstrap is seeded 20261105,
and nothing here touches the network or the weights. Standard library only.
"""
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.normpath(os.path.join(HERE, "..", "..", "scripts")))
import calib                                                    # noqa: E402

SPLITS = 25
BOOT = 5000
BOOT_SEED = 20261105
MODELS = (("ru-jev-v1.1", "unseen_v11.jsonl"), ("ru-jev-v1.3", "unseen_v13.jsonl"))


def load(directory):
    """Join the row file with each score dump, by position, checking the join."""
    rows = [json.loads(l) for l in open(os.path.join(directory, "unseen_rows.jsonl"),
                                        encoding="utf-8") if l.strip()]
    out = {}
    for name, fname in MODELS:
        scored = [json.loads(l) for l in open(os.path.join(directory, fname),
                                              encoding="utf-8") if l.strip()]
        if len(scored) != len(rows):
            raise SystemExit("%s has %d rows against %d in unseen_rows.jsonl"
                             % (fname, len(scored), len(rows)))
        for r, s in zip(rows, scored):
            if (r["answer"] != s["answer"] or r["set"] != s["set"]
                    or r["gold"] != s["gold"] or len(s["logits"]) != len(r["options"])):
                raise SystemExit("%s is not aligned with unseen_rows.jsonl" % fname)
        out[name] = scored
    return rows, out


def samples_for(rows, scored, which):
    """(scores, options, correct index, true label) for one dataset."""
    return [(s["logits"], [str(o) for o in r["options"]], int(r["answer"]), r["gold"])
            for r, s in zip(rows, scored) if r["set"] == which]


def raw_hits(samples):
    """1 where the released model, with no correction at all, is right."""
    return [int(max(range(len(lg)), key=lambda k: lg[k]) == ans)
            for lg, _opts, ans, _g in samples]


def out_of_fold(samples, fit_temperature):
    """How often each row is answered correctly by a fit that never saw it."""
    n = len(samples)
    hits = [0] * n
    seen = [0] * n
    for s in range(SPLITS):
        order = list(range(n))
        random.Random(s).shuffle(order)
        h = n // 2
        halves = (order[:h], order[h:])
        for fit_ix, test_ix in (halves, halves[::-1]):
            f = calib.fit_calibration([samples[i][:3] for i in fit_ix],
                                      fit_temperature=fit_temperature)
            T, b = f["temperature"], f["biases"]
            for i in test_ix:
                lg, opts, ans, _g = samples[i]
                p = calib.apply_calibration(lg, opts, T, b, warn=False)
                hits[i] += int(max(range(len(p)), key=lambda k: p[k]) == ans)
                seen[i] += 1
    return [h / s for h, s in zip(hits, seen)]


def mean(xs):
    return sum(xs) / len(xs)


def ci_rows(diff, seed):
    rng = random.Random(seed)
    n = len(diff)
    draws = []
    for _ in range(BOOT):
        draws.append(mean([diff[rng.randrange(n)] for _ in range(n)]))
    draws.sort()
    return draws[int(0.025 * BOOT)], draws[int(0.975 * BOOT) - 1]


def ci_labels(diff, labels, seed):
    """Resample whole labels, so the interval answers 'another six rubrics?'."""
    rng = random.Random(seed)
    by = {}
    for i, g in enumerate(labels):
        by.setdefault(g, []).append(i)
    names = sorted(by)
    draws = []
    for _ in range(BOOT):
        pool = []
        for _ in range(len(names)):
            pool.extend(by[names[rng.randrange(len(names))]])
        draws.append(mean([diff[i] for i in pool]))
    draws.sort()
    return draws[int(0.025 * BOOT)], draws[int(0.975 * BOOT) - 1]


def main():
    directory = sys.argv[1] if len(sys.argv) > 1 else "results/unseen"
    rows, scored = load(directory)
    sets = []
    for r in rows:
        if r["set"] not in sets:
            sets.append(r["set"])

    print("Per-option biases on the unseen datasets")
    print("%d half-splits, seeds 0 to %d, both halves scored; bootstrap %d draws, seed %d"
          % (SPLITS, SPLITS - 1, BOOT, BOOT_SEED))

    for which in sets:
        per_model = {}
        labels = None
        n_opts = None
        for name, _f in MODELS:
            samples = samples_for(rows, scored[name], which)
            labels = [g for _lg, _o, _a, g in samples]
            n_opts = len(samples[0][1])
            per_model[name] = {
                "n": len(samples),
                "raw": raw_hits(samples),
                "bias_T": out_of_fold(samples, True),
                "bias_noT": out_of_fold(samples, False),
            }

        print("\n## %s, %d rows, %d options\n" % (which, per_model[MODELS[0][0]]["n"], n_opts))
        print("| Model | as published | biases and T | biases, T held at 1 |")
        print("|---|---|---|---|")
        for name, _f in MODELS:
            m = per_model[name]
            print("| %s | %.3f | %.3f | %.3f |"
                  % (name, mean(m["raw"]), mean(m["bias_T"]), mean(m["bias_noT"])))

        a, b = MODELS[0][0], MODELS[1][0]
        print("\nPaired difference, %s minus %s:\n" % (b, a))
        print("| Setting | difference | 95% over rows | 95% over labels |")
        print("|---|---|---|---|")
        for label, key in (("as published", "raw"), ("after biases and T", "bias_T"),
                           ("after biases, T held at 1", "bias_noT")):
            diff = [y - x for x, y in zip(per_model[a][key], per_model[b][key])]
            lo_r, hi_r = ci_rows(diff, BOOT_SEED)
            lo_l, hi_l = ci_labels(diff, labels, BOOT_SEED)
            print("| %s | %+.3f | [%+.3f, %+.3f] | [%+.3f, %+.3f] |"
                  % (label, mean(diff), lo_r, hi_r, lo_l, hi_l))

        print("\nBiases fitted on all %d rows, for reading only, not used for any "
              "accuracy above:\n" % per_model[a]["n"])
        print("| Option | %s | %s | difference |" % (a, b))
        print("|---|---|---|---|")
        fits = {}
        for name, _f in MODELS:
            fits[name] = calib.fit_calibration(
                [s[:3] for s in samples_for(rows, scored[name], which)])
        opts = fits[a]["options"]
        # Both fits hold the same option at zero, so the difference column is
        # the only one that can be read on its own: it is how much further the
        # later release has to be pushed off that option.
        for o in sorted(opts, key=lambda o: fits[b]["biases"][o] - fits[a]["biases"][o]):
            print("| %s | %+.3f | %+.3f | %+.3f |"
                  % (o, fits[a]["biases"][o], fits[b]["biases"][o],
                     fits[b]["biases"][o] - fits[a]["biases"][o]))
        print("\nTemperature fitted with them: %s %.3f, %s %.3f. "
              "The option held at zero is %r."
              % (a, fits[a]["temperature"], b, fits[b]["temperature"], fits[a]["held_at_zero"]))


if __name__ == "__main__":
    main()
