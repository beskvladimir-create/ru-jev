#!/usr/bin/env python3
"""Per-option biases fitted on every checkpoint of the ladder.

    python3 -I results/analysis/ladder_bias.py results/unseen

Why this exists. `unseen_bias.py` fits one additive bias per option on the two
released checkpoints and shows that most of what separates them on the news
rubrics is a lean rather than a misreading. The same fit can be run on every
checkpoint of the ladder, and then the question is which checkpoint reads these
datasets best once each one is allowed its own correction. That is a different
ranking from accuracy as published, and it does not have to agree with the
validation accuracy the released epoch was chosen on.

Method, identical to `unseen_bias.py` and taken from its functions: 25 random
half-splits seeded 0 to 24, each used twice so that every row is answered 25
times by a fit that never saw it, and bootstrap intervals from seed 20261105
with 5,000 draws, over rows and over labels. Nothing is fitted and scored on the
same rows.

The pairs that are tested are fixed in the script. They are the comparisons that
answer a question rather than the full matrix: two seeds of the same recipe, the
two released checkpoints against each other and against that seed, and twice the
same question of an early against a late epoch inside one training run.

Deterministic, standard library only, no network. It reads the per-checkpoint
dumps in `ladder/` and `ladder_emotions/` and needs no weights.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)
import unseen_bias as ub                                          # noqa: E402

# dataset -> the directory holding one dump per checkpoint
SETS = [
    ("рубрики новостей", "ladder"),
    ("эмоции", "ladder_emotions"),
]

# the ladder in training order, by the tag each dump carries
ORDER = ["v1.1", "seed-base-s2", "v1.2", "seed-aug-s2", "seed-aug-s2-ep3",
         "seed-aug-s3", "v1.3-ep1of4", "v1.3-ep2of4", "v1.3-ep3of4", "v1.3",
         "v1.3-ep4of4"]

PAIRS = [
    ("seed-base-s2", "v1.1", "a second seed of the v1.1 recipe against the release"),
    ("v1.3", "v1.1", "the two released checkpoints"),
    ("v1.3", "seed-base-s2", "the later release against that seed rather than against v1.1"),
    ("v1.3-ep2of4", "v1.3-ep3of4", "the epoch before the released one against it"),
    ("v1.3-ep2of4", "v1.1", "that earlier epoch against the earlier release"),
    ("seed-aug-s2-ep3", "seed-aug-s2", "early against late epoch, the same question in another run"),
]


def load_rows(directory, which):
    rows = [json.loads(l) for l in open(os.path.join(directory, "unseen_rows.jsonl"),
                                        encoding="utf-8") if l.strip()]
    return [r for r in rows if r["set"] == which]


def samples(directory, sub, tag, rows):
    path = os.path.join(directory, sub, tag + ".jsonl")
    if not os.path.exists(path):
        return None
    scored = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    if len(scored) != len(rows):
        raise SystemExit("%s has %d rows against %d" % (path, len(scored), len(rows)))
    for r, s in zip(rows, scored):
        if r["answer"] != s["answer"] or r["gold"] != s["gold"]:
            raise SystemExit("%s is not aligned with unseen_rows.jsonl" % path)
    return [(s["logits"], [str(o) for o in r["options"]], int(r["answer"]), r["gold"])
            for r, s in zip(rows, scored)]


def main():
    directory = sys.argv[1] if len(sys.argv) > 1 else "results/unseen"

    print("Per-option biases on every checkpoint of the ladder")
    print("%d half-splits, seeds 0 to %d, both halves scored; bootstrap %d draws, seed %d"
          % (ub.SPLITS, ub.SPLITS - 1, ub.BOOT, ub.BOOT_SEED))

    for which, sub in SETS:
        rows = load_rows(directory, which)
        if not os.path.isdir(os.path.join(directory, sub)):
            print("\n## %s\n\nno dumps in %s/, skipped" % (which, sub))
            continue
        labels = [r["gold"] for r in rows]
        n_opts = len(rows[0]["options"])
        got = {}
        for tag in ORDER:
            s = samples(directory, sub, tag, rows)
            if s is None:
                continue
            got[tag] = {"raw": ub.raw_hits(s), "fix": ub.out_of_fold(s, True)}

        print("\n## %s, %d rows, %d options\n" % (which, len(rows), n_opts))
        print("| Checkpoint | as published | with biases and T | change |")
        print("|---|---|---|---|")
        for tag in ORDER:
            if tag not in got:
                continue
            raw, fix = ub.mean(got[tag]["raw"]), ub.mean(got[tag]["fix"])
            print("| %s | %.3f | %.3f | %+.3f |" % (tag, raw, fix, fix - raw))

        best = max(got, key=lambda t: ub.mean(got[t]["fix"]))
        print("\nBest with biases: `%s` at %.3f. Best as published: `%s` at %.3f."
              % (best, ub.mean(got[best]["fix"]),
                 max(got, key=lambda t: ub.mean(got[t]["raw"])),
                 max(ub.mean(got[t]["raw"]) for t in got)))

        print("\nPaired differences after the biases, out of fold:\n")
        print("| Pair | difference | 95% over rows | 95% over labels | What it asks |")
        print("|---|---|---|---|---|")
        for a, b, note in PAIRS:
            if a not in got or b not in got:
                print("| %s minus %s | | | | not available |" % (a, b))
                continue
            diff = [x - y for y, x in zip(got[b]["fix"], got[a]["fix"])]
            lo_r, hi_r = ub.ci_rows(diff, ub.BOOT_SEED)
            lo_l, hi_l = ub.ci_labels(diff, labels, ub.BOOT_SEED)
            print("| %s minus %s | %+.3f | [%+.3f, %+.3f] | [%+.3f, %+.3f] | %s |"
                  % (a, b, ub.mean(diff), lo_r, hi_r, lo_l, hi_l, note))


if __name__ == "__main__":
    main()
