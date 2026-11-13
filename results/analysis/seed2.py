#!/usr/bin/env python3
"""Is the gap between the v1.1 and v1.3 recipes larger than the spread between
seeds of one recipe?

    python3 -I results/analysis/seed2.py

The rule that decides this was written down in `results/v13_seed2/README.md` and
committed before the GPU was rented. Everything printed here is what that rule
asks for, in the order it asks for it, and the epoch it reads was fixed there as
the third of four before training.

Four quantities on every dataset, each a per-row paired difference of accuracies
out of fold once every checkpoint is allowed its own additive bias per option,
by the functions of `unseen_bias.py`: 25 half-splits seeded 0 to 24, each used
twice, so no row is answered by a fit that saw it. Intervals are bootstrap over
rows and over labels, seed 20261105, 5,000 draws.

| Quantity | Definition |
|---|---|
| spread of the v1.3 recipe | the new run minus the released v1.3 |
| spread of the v1.1 recipe | `seed-base-s2` minus v1.1 |
| gap between the recipes | the mean of the two v1.3 runs minus the mean of the two v1.1 runs |
| what the forms are worth at one seed | the new run minus `seed-aug-s2-ep3` |

The rule: on a dataset the gap counts as established if it is larger in absolute
value than both within-recipe spreads. On the scientific rubrics it is read on
the median over the twenty random tens, with the count of tens in which it
holds beside it.

With two runs per recipe this is a description and not a statistical test. Two
runs give a range, not a distribution, and no p-value is attached to anything
here.

The five-candidate views are not scored again. The model scores each
(text, option) pair on its own, so five of the eighteen logits of a question are
the same numbers that scoring those five would give, which is what
`vocab_bias.py` already relies on. The control section holds that claim against
the published five-candidate dump.

Deterministic, standard library only, no network, no weights.
"""
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)
import unseen_bias as ub                                           # noqa: E402
import matrix_sensitivity as ms                                    # noqa: E402
import vocab_bias as vb                                            # noqa: E402

R = os.path.normpath(os.path.join(HERE, "..", ".."))
RES = os.path.join(R, "results")
SEED2 = os.path.join(RES, "v13_seed2")

NEW = "v1.3-seed2-ep3of4"
NEW_EPOCHS = ["v1.3-seed2-ep1of4", "v1.3-seed2-ep2of4",
              "v1.3-seed2-ep3of4", "v1.3-seed2-ep4of4"]
V13, V11, SEEDBASE, AUG = "v1.3", "v1.1", "seed-base-s2", "seed-aug-s2-ep3"
CORE = [NEW, V13, V11, SEEDBASE, AUG]
EXTRA = [t for t in NEW_EPOCHS if t != NEW] + ["v1.3-seed2-final"]
ALL_TAGS = CORE + EXTRA

QNAMES = ["spread of the v1.3 recipe", "spread of the v1.1 recipe",
          "gap between the recipes", "what the forms are worth at one seed"]

BENCH_SETS = ["английский ключ", "русское слово", "русская фраза",
              "жаргон", "названия служб", "описание запроса"]
PROBE_SETS = ["стиль_A", "стиль_B"]


def jload(path):
    return json.load(open(path, encoding="utf-8"))


def lines(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


# ------------------------------------------------------------------ loading

def dumps(directory, rows, tags, alias=None):
    """tag -> the scored rows of that checkpoint, aligned with `rows`.

    `alias` renames a tag for one directory only, for the case where the same
    weights were dumped there under a different name. The keys of the result
    are the tags as asked for, not the names on disk.
    """
    out, missing = {}, []
    alias = alias or {}
    for tag in tags:
        p = os.path.join(directory, alias.get(tag, tag) + ".jsonl")
        if not os.path.exists(p):
            missing.append(tag)
            continue
        s = lines(p)
        if len(s) != len(rows):
            raise SystemExit("%s holds %d rows of %d" % (p, len(s), len(rows)))
        for r, x in zip(rows, s):
            # `scripts/score_rujev.py` does not copy the text into its dump, it
            # carries the row's own extra fields forward, so the alignment is
            # held on the position of the right answer and on the gold label
            # wherever the dump carries one.
            if int(r["answer"]) != int(x["answer"]):
                raise SystemExit("%s is not aligned with its rows file" % p)
            for k in ("gold", "gold_label", "set"):
                if k in x and k in r and r[k] != x[k]:
                    raise SystemExit("%s is not aligned with its rows file, %s differs"
                                     % (p, k))
        out[tag] = s
    return out, missing


def samples(rows, scored):
    return [(x["logits"], [str(o) for o in r["options"]], int(r["answer"]),
             str(r.get("gold", r.get("gold_label"))))
            for r, x in zip(rows, scored)]


def narrow(rows, scored, five, table):
    """The five-candidate view of an eighteen-candidate scoring.

    `five[j]` gives the five scenario keys of question j in the order the bench
    used them, recovered by `vocab_bias.bench_five_candidates`. `table` maps a
    scenario key to the surface form this vocabulary writes it with.
    """
    out = []
    for j, (r, x) in enumerate(zip(rows, scored)):
        text, scen, gold = five[j]
        want = [table[s] for s in scen]
        opts = [str(o) for o in r["options"]]
        idx = [opts.index(w) for w in want]
        # The eighteen-candidate rows file names the right answer by its
        # scenario key, which is the key the bench carries too, and it writes
        # that answer in this vocabulary's own words among the options. The key
        # and the surface form have to agree, and the text has to be the
        # question the bench asked, or the five-candidate view is of some other
        # row.
        if str(r.get("text")) != text:
            raise SystemExit("row %d is not the question the bench asked" % j)
        if str(r.get("gold_label")) != gold:
            raise SystemExit("row %d: the rows file says the answer is %r, the bench says %r"
                             % (j, r.get("gold_label"), gold))
        if opts[int(r["answer"])] != table[gold]:
            raise SystemExit("row %d: the right answer reads %r, this vocabulary writes it %r"
                             % (j, opts[int(r["answer"])], table[gold]))
        out.append(([x["logits"][i] for i in idx], want, want.index(table[gold]), table[gold]))
    return out


# ----------------------------------------------------------------- controls

def controls():
    """Three checks on the instrument, before any number is read.

    A measurement of a new checkpoint is worth nothing until the path that
    produced it returns what the published path returned.
    """
    print("## Control on the instrument\n")
    print("| Check | Result |")
    print("|---|---|")

    # 1. the checkpoints that are already published, scored again here
    for label, sub, which, published in (
            ("news rubrics", "news", "рубрики новостей", "unseen/ladder"),
            ("emotions", "emotions", "эмоции", "unseen/ladder_emotions")):
        rows = [r for r in lines(os.path.join(RES, "unseen/unseen_rows.jsonl"))
                if r["set"] == which]
        worst, flips, n, done = 0.0, 0, 0, []
        for tag in (V13, V11, SEEDBASE, AUG):
            mine = os.path.join(SEED2, sub, tag + ".jsonl")
            theirs = os.path.join(RES, published,
                                  ("v1.3-ep3of4" if tag == V13 else tag) + ".jsonl")
            if not (os.path.exists(mine) and os.path.exists(theirs)):
                continue
            a, b = lines(mine), lines(theirs)
            if len(a) != len(rows) or len(b) != len(rows):
                continue
            done.append(tag)
            for x, y in zip(a, b):
                worst = max(worst, max(abs(p - q) for p, q in zip(x["logits"], y["logits"])))
                flips += int(x["pick"] != y["pick"])
                n += 1
        if done:
            print("| %s, %s rescored against the published dumps | %d rows, "
                  "%d answers differ, largest logit difference %.3g |"
                  % (label, " and ".join(done), n, flips, worst))

    # 2. five of eighteen against the published five-candidate bench
    pub = os.path.join(RES, "bench_rujev_v13.jsonl")
    mine = os.path.join(SEED2, "fix18", V13 + ".jsonl")
    if os.path.exists(pub) and os.path.exists(mine):
        allrows = lines(os.path.join(RES, "fix18", "rows.jsonl"))
        scored = lines(mine)
        if len(scored) == len(allrows):
            vocab = jload(os.path.join(RES, "fix18", "vocab.json"))
            five = vb.bench_five_candidates()
            by = {}
            for r in lines(pub):
                by.setdefault(r["vocab"], []).append(r)
            worst, flips, n = 0.0, 0, 0
            for v in BENCH_SETS:
                keep = [i for i, r in enumerate(allrows) if r["set"] == v]
                rows = [allrows[i] for i in keep]
                cut = [scored[i] for i in keep]
                got = narrow(rows, cut, five, vocab[v]["метки"])
                for (lg, _o, ans, _g), ref in zip(got, by.get(v, [])):
                    worst = max(worst, max(abs(p - q) for p, q in zip(lg, ref["logits"])))
                    flips += int(max(range(len(lg)), key=lambda k: lg[k]) != ref["pick"])
                    n += 1
                    if ans != int(ref["answer"]):
                        raise SystemExit("the five-candidate view of %s is not aligned" % v)
            print("| five of the eighteen logits against the published five-candidate "
                  "bench, v1.3 | %d rows, %d answers differ, largest logit difference %.3g |"
                  % (n, flips, worst))

    # 3. the two names the released v1.3 carries
    #
    # The matrix of the scientific rubrics was scored under the ladder's names,
    # where the released v1.3 appears as the third epoch of its run. Reading it
    # there as the released v1.3 rests on these three published directories,
    # which hold dumps under both names.
    for sub, label in (("unseen3/scibench", "the scientific rubrics"),
                       ("unseen/ladder", "the news rubrics"),
                       ("unseen/ladder_emotions", "the emotions")):
        a = os.path.join(RES, sub, "v1.3.jsonl")
        b = os.path.join(RES, sub, "v1.3-ep3of4.jsonl")
        if os.path.exists(a) and os.path.exists(b):
            x, y = lines(a), lines(b)
            worst = max(max(abs(p - q) for p, q in zip(u["logits"], v["logits"]))
                        for u, v in zip(x, y))
            flips = sum(1 for u, v in zip(x, y) if u["pick"] != v["pick"])
            print("| v1.3 against v1.3-ep3of4 in the published dumps, %s | "
                  "%d rows, %d answers differ, largest logit difference %.3g |"
                  % (label, len(x), flips, worst))

    # 4. the fourth epoch read from two directories
    for sub, label in (("fix18", "the wording bench"), ("news", "the news rubrics")):
        a = os.path.join(SEED2, sub, "v1.3-seed2-ep4of4.jsonl")
        b = os.path.join(SEED2, sub, "v1.3-seed2-final.jsonl")
        if os.path.exists(a) and os.path.exists(b):
            x, y = lines(a), lines(b)
            same = sum(1 for p, q in zip(x, y) if p["pick"] == q["pick"])
            print("| the fourth epoch of the new run from two directories, %s | "
                  "the same answer on %d of %d rows |" % (label, same, len(x)))
    print()


# ----------------------------------------------------------------- datasets

def build():
    """[(name, {tag: samples}, note)], in the order the tables print."""
    sets = []
    five = vb.bench_five_candidates()

    for label, sub, rows_path, keys in (
            ("news rubrics", "news", "unseen/unseen_rows.jsonl", "рубрики новостей"),
            ("emotions", "emotions", "unseen/unseen_rows.jsonl", "эмоции")):
        rows = [r for r in lines(os.path.join(RES, rows_path)) if r["set"] == keys]
        got, missing = dumps(os.path.join(SEED2, sub), rows, ALL_TAGS)
        if [t for t in CORE if t in missing]:
            print("## %s\n\nmissing %s, skipped\n"
                  % (label, ", ".join(t for t in CORE if t in missing)))
            continue
        sets.append((label, {t: samples(rows, got[t]) for t in got}, "%d rows, %d candidates"
                     % (len(rows), len(rows[0]["options"]))))

    # Kept under its own name: the loops below rebind `rows` per dataset, and
    # the twenty random tens of the rubrics are built from these rows after
    # `build` has returned.
    mrows = lines(os.path.join(RES, "unseen3", "matrix", "rows.jsonl"))
    labels = mrows[0]["options"]
    # The matrix was scored with the ladder's own names, where the released
    # v1.3 is the third epoch of the v1.3 run and is dumped as
    # `v1.3-ep3of4`; there is no file called `v1.3` in that directory. They are
    # one checkpoint, and the control section holds that against the published
    # dumps that carry both names.
    mat, missing = dumps(os.path.join(RES, "unseen3", "matrix"), mrows, ALL_TAGS,
                         alias={V13: "v1.3-ep3of4"})
    if [t for t in CORE if t in missing]:
        print("## scientific rubrics\n\nmissing %s, skipped\n"
              % ", ".join(t for t in CORE if t in missing))
        mat = None
    if mat:
        series = {t: [x["logits"] for x in mat[t]] for t in mat}
        built = ms.build_ten(mrows, labels, sorted(labels)[:ms.TEN_SIZE])
        sets.append(("scientific rubrics, the alphabetical ten",
                     {t: ms.samples_for(built, labels, series[t]) for t in series},
                     "%d rows, %d candidates" % (len(built), ms.CANDIDATES)))

    for name, sub, vocab_path, which in (("bench", "fix18", "fix18/vocab.json", BENCH_SETS),
                                         ("probe", "vocab3", "vocab3/vocab.json", PROBE_SETS)):
        allrows = lines(os.path.join(RES, sub, "rows.jsonl"))
        vocab = jload(os.path.join(RES, vocab_path))
        for s in which:
            rows = [r for r in allrows if r["set"] == s]
            keep = [i for i, r in enumerate(allrows) if r["set"] == s]
            got, missing = dumps(os.path.join(SEED2, sub), allrows, ALL_TAGS)
            if [t for t in CORE if t in missing]:
                print("## %s %s\n\nmissing %s, skipped\n"
                      % (name, s, ", ".join(t for t in CORE if t in missing)))
                break
            cut = {t: [got[t][i] for i in keep] for t in got}
            sets.append(("%s %s, 18 candidates" % (name, s),
                         {t: samples(rows, cut[t]) for t in cut}, "%d rows" % len(rows)))
            table = vocab[s]["метки"]
            sets.append(("%s %s, 5 candidates" % (name, s),
                         {t: narrow(rows, cut[t], five, table) for t in cut},
                         "%d rows, five of the eighteen" % len(rows)))
    return sets, (mrows if mat else None), mat


# ---------------------------------------------------------------- quantities

def quantities(fix):
    """The four per-row vectors the rule names, from tag -> per-row hits."""
    new, v13, v11, base, aug = (fix[NEW], fix[V13], fix[V11], fix[SEEDBASE], fix[AUG])
    n = len(new)
    return [
        ("spread of the v1.3 recipe", [new[i] - v13[i] for i in range(n)],
         "the new run minus the released v1.3"),
        ("spread of the v1.1 recipe", [base[i] - v11[i] for i in range(n)],
         "seed-base-s2 minus v1.1"),
        ("gap between the recipes",
         [(new[i] + v13[i]) / 2.0 - (base[i] + v11[i]) / 2.0 for i in range(n)],
         "the mean of the two v1.3 runs minus the mean of the two v1.1 runs"),
        ("what the forms are worth at one seed", [new[i] - aug[i] for i in range(n)],
         "the new run minus seed-aug-s2-ep3"),
    ]


def verdict(q):
    """The rule, on one dataset. q is name -> mean difference."""
    gap = abs(q["gap between the recipes"])
    return (gap > abs(q["spread of the v1.3 recipe"])
            and gap > abs(q["spread of the v1.1 recipe"]))


def main():
    print("# A second seed of the v1.3 recipe\n")
    print("The rule is in `results/v13_seed2/README.md`, committed before the run. "
          "%d half-splits, seeds 0 to %d, both halves scored; bootstrap %d draws, seed %d.\n"
          % (ub.SPLITS, ub.SPLITS - 1, ub.BOOT, ub.BOOT_SEED))

    controls()
    sets, mrows, mat = build()
    if not sets:
        print("nothing to read yet")
        return

    # ----------------------------------------------------- the accuracies
    print("## Accuracy of every checkpoint, out of fold with biases\n")
    print("| Dataset | | " + " | ".join(CORE) + " |")
    print("|" + "---|" * (len(CORE) + 2))
    fixes = {}
    for name, series, note in sets:
        fix = {t: ub.out_of_fold(series[t], True) for t in series}
        fixes[name] = fix
        raw = {t: ub.raw_hits(series[t]) for t in series}
        print("| %s | as published | %s |"
              % (name, " | ".join("%.3f" % ub.mean(raw[t]) for t in CORE)))
        print("| %s | with biases | %s |"
              % (note, " | ".join("%.3f" % ub.mean(fix[t]) for t in CORE)))

    # ------------------------------------------------------ the quantities
    print("\n## The four quantities, with intervals over rows and over labels\n")
    table = {}
    for name, series, _note in sets:
        print("\n### %s\n" % name)
        print("| Quantity | difference | 95% over rows | 95% over labels |")
        print("|---|---|---|---|")
        labels = [g for _lg, _o, _a, g in series[NEW]]
        means = {}
        for q, diff, _what in quantities(fixes[name]):
            lo_r, hi_r = ub.ci_rows(diff, ub.BOOT_SEED)
            lo_l, hi_l = ub.ci_labels(diff, labels, ub.BOOT_SEED)
            means[q] = ub.mean(diff)
            print("| %s | %+.3f | [%+.3f, %+.3f] | [%+.3f, %+.3f] |"
                  % (q, means[q], lo_r, hi_r, lo_l, hi_l))
        table[name] = means

    # ------------------------------------- the twenty tens of the rubrics
    tens = None
    if mat:
        labels = mrows[0]["options"]
        series = {t: [x["logits"] for x in mat[t]] for t in mat}
        print("\n## The scientific rubrics over twenty random tens\n")
        print("Seeds 0 to %d, built as `matrix_sensitivity.py` builds them, every "
              "quantity as the rule has it.\n" % (ms.TENS - 1))
        print("| Seed | spread v1.3 | spread v1.1 | gap | forms at one seed | rule holds |")
        print("|---|---|---|---|---|---|")
        import random as _random
        cols = {q: [] for q in QNAMES}
        held = 0
        for seed in range(ms.TENS):
            ten = _random.Random(seed).sample(sorted(labels), ms.TEN_SIZE)
            built = ms.build_ten(mrows, labels, ten)
            fix = {t: ub.out_of_fold(ms.samples_for(built, labels, series[t]), True)
                   for t in CORE}
            means = {q: ub.mean(d) for q, d, _w in quantities(fix)}
            ok = verdict(means)
            held += int(ok)
            for q in cols:
                cols[q].append(means[q])
            print("| %d | %+.3f | %+.3f | %+.3f | %+.3f | %s |"
                  % (seed, means["spread of the v1.3 recipe"],
                     means["spread of the v1.1 recipe"],
                     means["gap between the recipes"],
                     means["what the forms are worth at one seed"],
                     "yes" if ok else "no"))
        print("\n| Quantity | median | lowest | highest |")
        print("|---|---|---|---|")
        for q in ("spread of the v1.3 recipe", "spread of the v1.1 recipe",
                  "gap between the recipes", "what the forms are worth at one seed"):
            print("| %s | %+.3f | %+.3f | %+.3f |"
                  % (q, statistics.median(cols[q]), min(cols[q]), max(cols[q])))
        tens = {"median": {q: statistics.median(cols[q]) for q in cols}, "held": held}
        print("\nThe rule holds in %d of the %d tens." % (held, ms.TENS))

    # ---------------------------------------------------------- the verdict
    print("\n## The rule of results/v13_seed2/README.md\n")
    print("| Dataset | gap | spread v1.3 | spread v1.1 | the gap is larger than both |")
    print("|---|---|---|---|---|")
    for name, means in table.items():
        print("| %s | %+.3f | %+.3f | %+.3f | %s |"
              % (name, means["gap between the recipes"],
                 means["spread of the v1.3 recipe"], means["spread of the v1.1 recipe"],
                 "yes" if verdict(means) else "no"))
    if tens:
        m = tens["median"]
        print("| scientific rubrics, median over twenty tens | %+.3f | %+.3f | %+.3f | %s |"
              % (m["gap between the recipes"], m["spread of the v1.3 recipe"],
                 m["spread of the v1.1 recipe"],
                 "yes, in %d of %d tens" % (tens["held"], ms.TENS)
                 if verdict(m) else "no, in %d of %d tens" % (tens["held"], ms.TENS)))

    print("\nAt two runs per recipe this is a description and not a statistical test.")

    # --------------------------------------------------------- the epochs
    print("\n## The other epochs of the new run\n")
    print("The epoch that is read was fixed as the third of four before training. "
          "The others are here and carry no conclusion.\n")
    order = NEW_EPOCHS + ["v1.3-seed2-final"]
    print("| Dataset | " + " | ".join(t.split("seed2-")[-1] for t in order) + " |")
    print("|" + "---|" * (len(order) + 1))
    for name, series, _note in sets:
        cells = []
        for t in order:
            if t not in series:
                cells.append("not scored")
            elif t in fixes[name]:
                cells.append("%.3f" % ub.mean(fixes[name][t]))
            else:
                cells.append("%.3f" % ub.mean(ub.out_of_fold(series[t], True)))
        print("| %s | %s |" % (name, " | ".join(cells)))


if __name__ == "__main__":
    main()
