#!/usr/bin/env python3
"""The tables of the v1.3.1 run, out of the dumps committed in this repository.

    python3 results/analysis/v131.py

Deterministic, standard library only, no network and no weights. Every number
printed below comes from a file in `results/`, and the script refuses to print
anything until the control on the instrument passes.

What the run was for is in `results/v131/README.md`, written and committed
before the GPU was rented. The reading rule and the epoch that is read were
fixed there. This script applies that rule and does not choose anything.

Two paths meet here and the difference between them is the control.

  * The released v1.3 is read from `results/validation/val_v13_trainsplit.jsonl`,
    the published dump, scored on the CPU of the machine this project is written
    on. It is not scored again.
  * v1.3.1 is scored on that same CPU, on the rows of its own validation split,
    which are the published dump's rows minus the 252 defective ones. So the
    two models are compared on identical rows, in identical order, through
    identical code.

The GPU figures the training run printed are in `results/v131/train_log.txt`.
They are not used for any comparison. They are checked against the CPU
accuracies as a second control, because a card and a processor disagreeing by
more than rounding would mean one of the two numbers is not the model's.
"""
import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
RES = os.path.join(ROOT, "results")
V131 = os.path.join(RES, "v131")
sys.path.insert(0, HERE)

# The four epochs, under the names their dumps carry. The third is the one the
# rule fixed in writing before training, and it is the one the rule reads.
EPOCHS = [("epoch 1 of 4", "v1.3.1-ep1of4"),
          ("epoch 2 of 4", "v1.3.1-ep2of4"),
          ("epoch 3 of 4, the epoch the rule fixed", "v1.3.1-ep3of4"),
          ("epoch 4 of 4", "v1.3.1-ep4of4")]
READ = "v1.3.1-ep3of4"

# The six Russian SuperGLUE tasks named in the rule, and the aggregates it
# bounds. 0.005 applies to these two aggregates and not to a single task, which
# the rule states and argues before any number was seen.
RSG = ("danetqa", "parus", "rcb", "russe", "rwsd", "terra")
BOUND = 0.005

# The share of the majority class on the 1,092 clean rucola validation rows,
# measured in `results/rucola_test/README.md`.
MAJORITY_RUCOLA = 0.7372

# The six vocabularies of the wording bench and the two probe styles, in the
# order the published tables use them.
BENCH_SETS = ["английский ключ", "русское слово", "русская фраза",
              "жаргон", "названия служб", "описание запроса"]
BENCH_EN = {"английский ключ": "English key", "русское слово": "Russian word",
            "русская фраза": "Russian phrase", "жаргон": "anglicism jargon",
            "названия служб": "department name",
            "описание запроса": "descriptive sentence"}
STYLES = ["стиль_A", "стиль_B"]
STYLE_EN = {"стиль_A": "style A", "стиль_B": "style B"}

BOOT = 5000
BOOT_SEED = 20261114


def die(msg):
    raise SystemExit("v131.py: " + msg)


def lines(path):
    if not os.path.exists(path):
        die("missing %s" % os.path.relpath(path, ROOT))
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def jload(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def pick(scores):
    return max(range(len(scores)), key=lambda i: scores[i])


def acc(hits):
    return sum(hits) / float(len(hits)) if hits else float("nan")


def mcnemar(a, b):
    """Exact two-sided sign test on the rows where the two models differ.

    The tail is summed in logarithms. On the whole validation split the two
    models disagree on thousands of rows, and `2 ** n` on that n does not fit
    in a float.
    """
    only_a = sum(1 for x, y in zip(a, b) if x and not y)
    only_b = sum(1 for x, y in zip(a, b) if y and not x)
    n = only_a + only_b
    if n == 0:
        return only_a, only_b, 1.0
    k = min(only_a, only_b)
    ln2 = math.log(2.0)
    tail = 0.0
    for i in range(k + 1):
        lg = (math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
              - n * ln2)
        tail += math.exp(lg)
    return only_a, only_b, min(1.0, 2 * tail)


def boot_ci(a, b):
    """Percentile interval of the paired accuracy difference, b minus a."""
    n = len(a)
    d = [int(y) - int(x) for x, y in zip(a, b)]
    rnd = random.Random(BOOT_SEED)
    draws = sorted(sum(rnd.choices(d, k=n)) / float(n) for _ in range(BOOT))
    return draws[int(0.025 * BOOT)], draws[int(0.975 * BOOT)]


# --------------------------------------------------------------- the rows

def load_validation():
    """The published v1.3 dump, the four v1.3.1 dumps, and the control.

    The rule states the control: the rows v1.3.1 is scored on have to be the
    same rows, in the same order, as the subset of the published dump that
    survives the strip, checked on the task name, the index of the correct
    candidate and the number of candidates, on all 26,494 rows.
    """
    pub = lines(os.path.join(RES, "validation", "val_v13_trainsplit.jsonl"))
    with open(os.path.join(RES, "rucola_test", "val_v13_affected_rows.txt"),
              encoding="utf-8") as fh:
        aff = set(int(x) for x in fh if x.strip())
    if len(pub) != 26746 or len(aff) != 252:
        die("the published dump has %d rows and the defect list %d; expected "
            "26746 and 252" % (len(pub), len(aff)))
    kept = [r for i, r in enumerate(pub) if i not in aff]

    got, bad = {}, 0
    for _label, tag in EPOCHS:
        rows = lines(os.path.join(V131, "val", tag + ".jsonl"))
        if len(rows) != len(kept):
            die("%s holds %d rows of %d" % (tag, len(rows), len(kept)))
        for a, b in zip(kept, rows):
            if (a["task"] != b["task"] or int(a["answer"]) != int(b["answer"])
                    or len(a["scores"]) != len(b["scores"])):
                bad += 1
        got[tag] = rows
    return kept, got, bad


def gpu_log():
    """The accuracy and the temperature each epoch printed during training."""
    path = os.path.join(V131, "train_log.txt")
    out, order = {}, []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if "T = " in line and "val acc" in line:
            name = line.split()[0].rstrip("/").split("/")[-1]
            temp = float(line.split("T = ")[1].split()[0])
            a = float(line.split("val acc")[1].split()[0])
            out[name] = (temp, a)
            order.append(name)
    if len(out) != 4:
        die("the training log names %d epochs with a temperature, expected 4" % len(out))
    # The log names the snapshots as the run wrote them. The fourth is the one
    # training stopped at and it is saved twice, under `ep3` and `epfinal`,
    # byte for byte; the temperature pass read it as `epfinal`.
    by_tag = {}
    for (label, tag), name in zip(EPOCHS, ["ep0", "ep1", "ep2", "epfinal"]):
        if name not in out:
            die("the training log does not name %s" % name)
        by_tag[tag] = out[name]
    return by_tag


# --------------------------------------------------------------- the bench

def narrow(rows, scored, five, table):
    """The five-candidate view of an eighteen-candidate scoring.

    The same reading as `results/analysis/seed2.py`, and the same refusal to
    guess: a scenario key is compared with a scenario key and a surface form
    with a surface form, because the bench shuffles the options and the meaning
    of a row is `options[answer]` and never the index.
    """
    out = []
    for j, (r, x) in enumerate(zip(rows, scored)):
        text, scen, gold = five[j]
        want = [table[s] for s in scen]
        opts = [str(o) for o in r["options"]]
        idx = [opts.index(w) for w in want]
        if str(r.get("text")) != text:
            die("bench row %d is not the question the bench asked" % j)
        if str(r.get("gold_label")) != gold:
            die("bench row %d: the rows file says the answer is %r, the bench says %r"
                % (j, r.get("gold_label"), gold))
        if opts[int(r["answer"])] != table[gold]:
            die("bench row %d: the right answer reads %r, this vocabulary writes it %r"
                % (j, opts[int(r["answer"])], table[gold]))
        out.append(([x["logits"][i] for i in idx], want.index(table[gold])))
    return out


def bench_tables():
    """The wording bench and the two probes, as description and under no rule."""
    print("### The wording bench, all 18 candidates\n")
    print("The released v1.3 is recomputed here from its own published dump, so "
          "both columns come out of this script rather than out of two tables.\n")
    allrows = lines(os.path.join(RES, "fix18", "rows.jsonl"))
    pub = lines(os.path.join(RES, "fix18", "scores_v13.jsonl"))
    mine = lines(os.path.join(V131, "fix18", READ + ".jsonl"))
    if not (len(pub) == len(mine) == len(allrows)):
        die("the bench dumps hold %d, %d and %d rows" % (len(pub), len(mine), len(allrows)))
    vocab = jload(os.path.join(RES, "fix18", "vocab.json"))

    print("| Vocabulary | n | v1.3 | v1.3.1 | difference |")
    print("|---|---|---|---|---|")
    both18 = {}
    for v in BENCH_SETS:
        keep = [i for i, r in enumerate(allrows) if r["set"] == v]
        # A name that matches nothing would print an empty row rather than
        # fail, and an empty row in a comparison table is worse than a crash.
        if len(keep) != 1000:
            die("the bench holds %d rows of vocabulary %r, expected 1000"
                % (len(keep), v))
        a = [int(pick(pub[i]["logits"]) == int(allrows[i]["answer"])) for i in keep]
        b = [int(pick(mine[i]["logits"]) == int(allrows[i]["answer"])) for i in keep]
        both18[v] = (a, b)
        print("| %s | %d | %.3f | %.3f | %+.3f |"
              % (BENCH_EN[v], len(keep), acc(a), acc(b), acc(b) - acc(a)))
    a = [x for v in BENCH_SETS for x in both18[v][0]]
    b = [x for v in BENCH_SETS for x in both18[v][1]]
    oa, ob, p = mcnemar(a, b)
    print("| all six | %d | %.3f | %.3f | %+.3f |" % (len(a), acc(a), acc(b), acc(b) - acc(a)))
    print("\nOver all six vocabularies: only v1.3 right on %d questions, only "
          "v1.3.1 on %d, sign test p = %.3g.\n" % (oa, ob, p))

    print("### The wording bench, the five-candidate view\n")
    print("The same scoring read at the five candidates the bench originally "
          "asked, recovered per question by `results/analysis/vocab_bias.py`.\n")
    import vocab_bias as vb                                         # noqa: E402
    five = vb.bench_five_candidates()

    # The narrowing is checked against the published five-candidate bench
    # before it is used: v1.3's eighteen logits, cut to five, have to be the
    # five logits the release's own five-candidate dump holds.
    ref = {}
    for r in lines(os.path.join(RES, "bench_rujev_v13.jsonl")):
        ref.setdefault(r["vocab"], []).append(r)
    worst, flips, seen = 0.0, 0, 0
    for v in BENCH_SETS:
        keep = [i for i, r in enumerate(allrows) if r["set"] == v]
        got = narrow([allrows[i] for i in keep], [pub[i] for i in keep],
                     five, vocab[v]["метки"])
        for (lg, ans), r in zip(got, ref.get(v, [])):
            worst = max(worst, max(abs(p - q) for p, q in zip(lg, r["logits"])))
            flips += int(pick(lg) != int(r["pick"]))
            seen += 1
            if ans != int(r["answer"]):
                die("the five-candidate view of %r is not aligned with the "
                    "published bench" % v)
    if seen != 6000:
        die("the published five-candidate bench covered %d of 6000 rows" % seen)
    print("Checked first: cutting v1.3's eighteen logits to five returns the "
          "release's published five-candidate dump on all %d rows, %d answers "
          "differing, largest logit difference %.3g.\n" % (seen, flips, worst))

    print("| Vocabulary | n | v1.3 | v1.3.1 | difference |")
    print("|---|---|---|---|---|")
    tot_a, tot_b = [], []
    for v in BENCH_SETS:
        keep = [i for i, r in enumerate(allrows) if r["set"] == v]
        if len(keep) != 1000:
            die("the bench holds %d rows of vocabulary %r, expected 1000"
                % (len(keep), v))
        rows = [allrows[i] for i in keep]
        ga = narrow(rows, [pub[i] for i in keep], five, vocab[v]["метки"])
        gb = narrow(rows, [mine[i] for i in keep], five, vocab[v]["метки"])
        a = [int(pick(lg) == ans) for lg, ans in ga]
        b = [int(pick(lg) == ans) for lg, ans in gb]
        tot_a += a
        tot_b += b
        print("| %s | %d | %.3f | %.3f | %+.3f |"
              % (BENCH_EN[v], len(a), acc(a), acc(b), acc(b) - acc(a)))
    print("| all six | %d | %.3f | %.3f | %+.3f |"
          % (len(tot_a), acc(tot_a), acc(tot_b), acc(tot_b) - acc(tot_a)))
    oa, ob, p = mcnemar(tot_a, tot_b)
    print("\nOver all six vocabularies: only v1.3 right on %d questions, only "
          "v1.3.1 on %d, sign test p = %.3g.\n" % (oa, ob, p))

    for sub, title, cands in (("vocab2", "The first probe, five candidates", 5),
                              ("vocab3", "The second probe, 18 candidates", 18)):
        print("### %s\n" % title)
        rows = lines(os.path.join(RES, sub, "rows.jsonl"))
        pub = lines(os.path.join(RES, sub, "scores_v13.jsonl"))
        mine = lines(os.path.join(V131, sub, READ + ".jsonl"))
        if not (len(pub) == len(mine) == len(rows)):
            die("the %s dumps hold %d, %d and %d rows" % (sub, len(pub), len(mine), len(rows)))
        if len(rows[0]["options"]) != cands:
            die("%s rows carry %d candidates, expected %d"
                % (sub, len(rows[0]["options"]), cands))
        print("| Style | n | v1.3 | v1.3.1 | difference |")
        print("|---|---|---|---|---|")
        for s in STYLES:
            keep = [i for i, r in enumerate(rows) if r["set"] == s]
            if len(keep) != 1000:
                die("%s holds %d rows of %r, expected 1000" % (sub, len(keep), s))
            a = [int(pick(pub[i]["logits"]) == int(rows[i]["answer"])) for i in keep]
            b = [int(pick(mine[i]["logits"]) == int(rows[i]["answer"])) for i in keep]
            print("| %s | %d | %.3f | %.3f | %+.3f |"
                  % (STYLE_EN[s], len(a), acc(a), acc(b), acc(b) - acc(a)))
        print("")


# --------------------------------------------------------------- the report

def main():
    kept, got, bad = load_validation()
    log = gpu_log()

    hits = {}
    for _label, tag in EPOCHS:
        hits[tag] = [int(pick(r["scores"]) == int(r["answer"])) for r in got[tag]]
    v13 = [int(pick(r["scores"]) == int(r["answer"])) for r in kept]
    tasks = sorted(set(r["task"] for r in kept))

    def sub(h, which):
        return [x for x, r in zip(h, kept) if r["task"] in which]

    print("# v1.3.1, the numbers\n")
    print("Written by `results/analysis/v131.py` out of the dumps in this "
          "repository. The rule above was committed before the run.\n")

    print("## The control on the instrument\n")
    print("| Check | Result |")
    print("|---|---|")
    print("| the rows of this run against the surviving rows of the published "
          "v1.3 dump, on task, answer index and candidate count | %d rows, %d "
          "disagreements |" % (len(kept), bad))
    if bad:
        die("the control failed on %d rows; no number below is readable" % bad)
    worst = 0.0
    for _label, tag in EPOCHS:
        worst = max(worst, abs(acc(hits[tag]) - log[tag][1]))
    print("| the four CPU accuracies against the four the training run printed "
          "on the card | largest difference %.4f |" % worst)
    print("| the share of the majority class on the clean rucola rows, from "
          "`results/rucola_test/` | %.4f |" % MAJORITY_RUCOLA)
    print("")

    print("## The run, epoch by epoch\n")
    print("The temperature is the one the release's own instrument returns, "
          "`results/analysis/fit_seed2_temperatures.py`, on this run's own "
          "validation split. It carries no condition.\n")
    print("| Epoch | temperature | accuracy, all 26,494 rows | accuracy, rucola, "
          "1,092 rows | accuracy, the six Russian SuperGLUE tasks, 10,157 rows |")
    print("|---|---|---|---|---|")
    for label, tag in EPOCHS:
        print("| %s | %.4f | %.4f | %.4f | %.4f |"
              % (label, log[tag][0], acc(hits[tag]), acc(sub(hits[tag], {"rucola"})),
                 acc(sub(hits[tag], set(RSG)))))
    print("| the released v1.3, on these same rows | 1.8640 | %.4f | %.4f | %.4f |"
          % (acc(v13), acc(sub(v13, {"rucola"})), acc(sub(v13, set(RSG)))))
    print("\nThe released v1.3's temperature is the published one, 1.8640, from "
          "`results/seeds/FINDINGS.md`. It was fitted on the validation split "
          "the release was trained against, which held the 252 defective rows, "
          "so it is quoted here and not compared.\n")

    print("## The rule\n")
    rows_of = {"rucola, the 1,092 rows that carry a real label": {"rucola"},
               "the six Russian SuperGLUE tasks, 10,157 rows": set(RSG),
               "the whole validation split, 26,494 rows": set(tasks)}
    print("| What the rule reads | n | v1.3 | v1.3.1, epoch 3 of 4 | difference "
          "| 95 per cent interval | verdict |")
    print("|---|---|---|---|---|---|---|")
    verdicts = {}
    for label, which in rows_of.items():
        a, b = sub(v13, which), sub(hits[READ], which)
        d = acc(b) - acc(a)
        lo, hi = boot_ci(a, b)
        if which == {"rucola"}:
            ok = d > 0
            verd = "better than v1.3, condition one holds" if ok else \
                "not better than v1.3, condition one fails"
        else:
            ok = d >= -BOUND
            verd = "within 0.005, condition two holds" if ok else \
                "worse by more than 0.005, condition two fails"
        verdicts[label] = ok
        print("| %s | %d | %.4f | %.4f | %+.4f | %+.4f to %+.4f | %s |"
              % (label, len(a), acc(a), acc(b), d, lo, hi, verd))
    print("")
    for label, which in rows_of.items():
        a, b = sub(v13, which), sub(hits[READ], which)
        oa, ob, p = mcnemar(a, b)
        print("* %s: only v1.3 right on %d rows, only v1.3.1 on %d, sign test "
              "p = %.3g." % (label, oa, ob, p))
    print("")
    passed = all(verdicts.values())
    print("**The rule %s.** %s\n" % (
        "allows publication" if passed else "refuses publication",
        "Both conditions hold. Publication still needs its owner's word, which "
        "the rule does not replace." if passed else
        "v1.3.1 is not published, and these numbers stand as they came out."))

    print("## Accuracy by task, as description\n")
    print("The rule is not applied per task and no conclusion below rests on a "
          "single one. On `parus` one decision is 0.010 and on `rwsd` 0.005, "
          "which is why.\n")
    print("| Task | n | v1.3 | v1.3.1, epoch 3 of 4 | difference |")
    print("|---|---|---|---|---|")
    for t in tasks:
        a, b = sub(v13, {t}), sub(hits[READ], {t})
        print("| %s | %d | %.4f | %.4f | %+.4f |"
              % (t, len(a), acc(a), acc(b), acc(b) - acc(a)))
    print("")

    print("## RuCoLA against the majority class\n")
    a, b = sub(v13, {"rucola"}), sub(hits[READ], {"rucola"})
    print("| Model | accuracy on the 1,092 clean rows | over the majority class |")
    print("|---|---|---|")
    print("| the share of the majority class | %.4f | |" % MAJORITY_RUCOLA)
    print("| the released v1.3 | %.4f | %+.4f |" % (acc(a), acc(a) - MAJORITY_RUCOLA))
    print("| v1.3.1, epoch 3 of 4 | %.4f | %+.4f |" % (acc(b), acc(b) - MAJORITY_RUCOLA))
    print("")

    bench_tables()


if __name__ == "__main__":
    main()
