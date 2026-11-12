#!/usr/bin/env python3
"""Everything the full score matrix allows, added after the reading rule.

    python3 -I results/analysis/matrix_sensitivity.py

`results/unseen3/matrix/` holds every one of the ten checkpoints scored on all
2,772 test texts of the GRNTI rubrics against all 28 rubric labels at once.
From that matrix any set of candidate labels can be assembled without touching
the weights again, because the model scores each (text, label) pair on its own.

This script is separate from `ladder_rank.py` on purpose. `ladder_rank.py`
computes what the rule in `results/unseen3/README.md` fixed before any scoring.
Everything here was decided after that rule was written and after its verdict
was known, and it is labelled as such wherever it is printed.

Three things:

- a check that the alphabetical ten assembled from the matrix is the same thing
  as the dump that was scored directly, logit for logit;
- a sensitivity check over twenty random tens of the 28 rubrics, seeds 0 to 19,
  everything else as the rule has it;
- the ten checkpoints with all 28 rubrics offered at once.

Deterministic, standard library only, no network, no weights.
"""
import json
import os
import random
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)
import unseen_bias as ub                                          # noqa: E402
import ladder_rank as lr                                          # noqa: E402

RES = os.path.normpath(os.path.join(HERE, "..", "..", "results"))
MATRIX = os.path.join(RES, "unseen3", "matrix")
SAMPLE_SEED = 20261105
TENS = 20
TEN_SIZE = 10
CANDIDATES = 5


def load_matrix():
    rows = [json.loads(l) for l in open(os.path.join(MATRIX, "rows.jsonl"), encoding="utf-8")
            if l.strip()]
    labels = rows[0]["options"]
    got = {}
    for tag in lr.TEN:
        p = os.path.join(MATRIX, tag + ".jsonl")
        if not os.path.exists(p):
            return None, None, None, tag
        s = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        if len(s) != len(rows):
            return None, None, None, "%s has %d rows of %d" % (tag, len(s), len(rows))
        for r, x in zip(rows, s):
            if r["answer"] != x["answer"] or r["gold"] != x["gold"]:
                raise SystemExit("%s is not aligned with the matrix rows" % tag)
        got[tag] = [x["logits"] for x in s]
    return rows, labels, got, None


def build_ten(rows, labels, ten):
    """The rows of one ten, by the same construction as build_unseen3.py.

    `build_unseen3.py` puts the rows of the source split in a canonical order,
    by label, then the source id, then the text, before it draws anything. The
    rows file of the matrix is that same order over all 28 classes, and the
    source ids it sorted on are not in it, so the order is taken from the file
    as it stands rather than rebuilt from the fields that survive. Restricting
    a sorted sequence to ten of its labels leaves it sorted, so the rows of one
    ten come out in the order the builder would have given them. The check
    above holds this to the dump that was scored directly.

    Every class of this split holds 99 test rows, so any ten of them is 990
    rows, under the 1,000 of the rule, and the stratified step of the builder
    does not run. The generator is therefore at its initial state when the
    candidates are drawn, exactly as it was for the ten that was published.
    """
    classes = sorted(ten)
    keep = [(i, r) for i, r in enumerate(rows) if r["gold"] in classes]
    rng = random.Random(SAMPLE_SEED)
    out = []
    for i, r in keep:
        gold = r["gold"]
        others = [c for c in classes if c != gold]
        opts = [gold] + rng.sample(others, CANDIDATES - 1)
        order = list(range(CANDIDATES))
        rng.shuffle(order)
        opts = [opts[k] for k in order]
        out.append((i, opts, opts.index(gold), gold))
    return out


def samples_for(built, labels, series):
    at = {lab: k for k, lab in enumerate(labels)}
    return [([series[i][at[o]] for o in opts], opts, ans, gold)
            for i, opts, ans, gold in built]


def main():
    rows, labels, got, missing = load_matrix()
    if rows is None:
        print("The matrix is not complete yet: %s" % missing)
        return

    print("# Added after the reading rule\n")
    print("Everything in this file was decided after the rule in "
          "`results/unseen3/README.md` was written and after its verdict was known. "
          "The verdict itself is unchanged and rests on the alphabetical ten.\n")
    print("The matrix holds %d texts against %d labels for each of %d checkpoints.\n"
          % (len(rows), len(labels), len(lr.TEN)))

    # ---------------------------------------------------------- the check
    alpha = sorted(labels)[:TEN_SIZE]
    built = build_ten(rows, labels, alpha)
    direct_rows = [json.loads(l) for l in
                   open(os.path.join(RES, "unseen3", "scibench", "rows.jsonl"), encoding="utf-8")
                   if l.strip()]
    print("## The alphabetical ten, assembled from the matrix against the direct scoring\n")
    same_rows = (len(built) == len(direct_rows)
                 and all(b[1] == d["options"] and b[2] == d["answer"]
                         and rows[b[0]]["text"] == d["text"] for b, d in zip(built, direct_rows)))
    print("| Check | Result |")
    print("|---|---|")
    print("| the rows are the same, text, options and answer | %s |"
          % ("yes" if same_rows else "NO"))
    worst = 0.0
    picks = 0
    total = 0
    for tag in lr.TEN:
        direct = [json.loads(l) for l in
                  open(os.path.join(RES, "unseen3", "scibench", tag + ".jsonl"), encoding="utf-8")
                  if l.strip()]
        s = samples_for(built, labels, got[tag])
        for (lg, _o, _a, _g), d in zip(s, direct):
            worst = max(worst, max(abs(p - q) for p, q in zip(lg, d["logits"])))
            picks += int(max(range(len(lg)), key=lambda k: lg[k]) == d["pick"])
            total += 1
    print("| largest disagreement between a matrix logit and a directly scored one | %.3g |" % worst)
    print("| rows where the answer is the same | %d of %d |" % (picks, total))

    print("\n| Checkpoint | from the matrix | published in ladder_rank.md |")
    print("|---|---|---|")
    alpha_fix = {}
    for tag in lr.TEN:
        s = samples_for(built, labels, got[tag])
        raw = ub.mean(ub.raw_hits(s))
        fix = ub.mean(ub.out_of_fold(s, True))
        alpha_fix[tag] = fix
        direct = [json.loads(l) for l in
                  open(os.path.join(RES, "unseen3", "scibench", tag + ".jsonl"), encoding="utf-8")
                  if l.strip()]
        draw = sum(1 for d in direct if d["pick"] == d["answer"]) / len(direct)
        print("| %s | %.3f as published, %.3f with biases | %.3f as published |"
              % (tag, raw, fix, draw))

    # ------------------------------------------------- the sensitivity check
    print("\n## Sensitivity check, added after the reading rule\n")
    print("Twenty random tens of the 28 rubrics, seeds 0 to 19. Everything else is as the "
          "rule has it: every row of the chosen ten, five candidates per question drawn with "
          "seed %d, biases fitted out of fold over %d half-splits.\n" % (SAMPLE_SEED, ub.SPLITS))

    base = {}
    for name, rows_path, which, dumps in lr.DATASETS[:2]:
        d = lr.load(name, rows_path, which, dumps)
        if d is None or d["missing"]:
            continue
        s = {t: [(x["logits"], [str(o) for o in r["options"]], int(r["answer"]), r["gold"])
                 for r, x in zip(d["rows"], d["scored"][t])] for t in lr.TEN}
        base[name] = [ub.mean(ub.out_of_fold(s[t], True)) for t in lr.TEN]

    print("| Seed | n | Spearman with news | Spearman with emotions | "
          "v1.3 minus v1.1 | v1.3 minus seed-base-s2 |")
    print("|---|---|---|---|---|---|")
    rho_news, rho_emo, d_v11, d_seed = [], [], [], []
    for seed in range(TENS):
        rng = random.Random(seed)
        ten = rng.sample(sorted(labels), TEN_SIZE)
        built = build_ten(rows, labels, ten)
        fix = {t: ub.mean(ub.out_of_fold(samples_for(built, labels, got[t]), True))
               for t in lr.TEN}
        vec = [fix[t] for t in lr.TEN]
        a = lr.spearman(base["news rubrics"], vec) if "news rubrics" in base else float("nan")
        b = lr.spearman(base["emotions"], vec) if "emotions" in base else float("nan")
        x = fix["v1.3-ep3of4"] - fix["v1.1"]
        y = fix["v1.3-ep3of4"] - fix["seed-base-s2"]
        rho_news.append(a); rho_emo.append(b); d_v11.append(x); d_seed.append(y)
        print("| %d | %d | %+.2f | %+.2f | %+.3f | %+.3f |" % (seed, len(built), a, b, x, y))

    print("\n| Quantity | median | lowest | highest | tens at or above 0.50 |")
    print("|---|---|---|---|---|")
    for name, xs, thresh in (("Spearman with news rubrics", rho_news, True),
                             ("Spearman with emotions", rho_emo, True),
                             ("v1.3 minus v1.1", d_v11, False),
                             ("v1.3 minus seed-base-s2", d_seed, False)):
        n_at = sum(1 for v in xs if v >= 0.5) if thresh else None
        print("| %s | %+.3f | %+.3f | %+.3f | %s |"
              % (name, statistics.median(xs), min(xs), max(xs),
                 ("%d of %d" % (n_at, len(xs))) if thresh else ""))

    # ------------------------------------------------------- all 28 offered
    print("\n## All 28 rubrics offered at once, added after the reading rule\n")
    print("Every one of the %d test texts, with all %d rubrics as candidates. Chance is %.3f.\n"
          % (len(rows), len(labels), 1 / len(labels)))
    full = [( [str(o) for o in r["options"]], int(r["answer"]), r["gold"]) for r in rows]
    print("| Checkpoint | as published | with biases | change |")
    print("|---|---|---|---|")
    for tag in lr.TEN:
        s = [(got[tag][i], full[i][0], full[i][1], full[i][2]) for i in range(len(rows))]
        raw = ub.mean(ub.raw_hits(s))
        fix = ub.mean(ub.out_of_fold(s, True))
        print("| %s | %.3f | %.3f | %+.3f |" % (tag, raw, fix, fix - raw))


if __name__ == "__main__":
    main()
