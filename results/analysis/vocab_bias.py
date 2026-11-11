#!/usr/bin/env python3
"""Per-option biases, per-label contributions and seed spread on the probe vocabularies.

    python3 -I results/analysis/vocab_bias.py

Three questions about the label vocabularies, asked of the published per-row
scores and of nothing else.

One. How much of the difference between the two releases on a vocabulary is one
label? The unseen news rubrics turned out to be almost entirely a lean towards
one rubric, which `results/unseen/bias.md` shows. The same question can be put
to every vocabulary and slice here, by taking each label's contribution to the
difference in questions and by recomputing the difference with that label left
out.

Two. What survives a per-option correction? Each release is allowed its own
additive bias per option, fitted out of fold by the machinery of
`unseen_bias.py`, and the difference is read again.

Three. How does the difference compare with the spread between seeds of one
recipe? Three runs of the v1.1 recipe are published, the release itself and two
replicates, and a difference smaller than the gap between them is not evidence
about a recipe.

## How the rows are assembled

The five-candidate option strings of the wording bench are not published: the
dump `robustness.py --dump` writes was a working file. They are recovered here
and the recovery is checked. `results/vocab2/rows.jsonl` keeps, per question,
the five scenario keys of that question, shuffled once more by the crc32 of the
text; inverting that shuffle gives the bench's own order back. The check is that
the position of the correct answer in the recovered order equals the `answer`
field stored in the bench score files, for all 1,000 questions and all six
vocabularies. The script stops if it does not.

Rows are matched by position within a vocabulary block, never by text: seven of
the thousand questions repeat verbatim, so text is not a key. Positional
alignment is verified field by field before anything is computed.

The five-candidate figures for the vocab3 probe are obtained by renormalising
its eighteen-candidate logits over the five options of that question, which is
exact for this model because every option is scored in its own forward pass.

## Where the biases are fitted, and why it matters

A slice keeps the questions whose correct label is in it and leaves the option
lists alone, so on a slice some options are never the right answer. Fitting the
biases on the slice itself then drives those options to minus infinity: on the
strict jargon slice seven of the eighteen options never occur as the answer and
their fitted biases run from -20 to the bound at -50, while the eleven that do
occur stay inside plus or minus 2.5. That is maximum likelihood doing its job,
and it is also the slice definition leaking into the correction, which a caller
could not reproduce because a caller does not know which labels are impossible.

So the headline figure for every slice is fitted on all 1,000 questions of that
vocabulary, where every label can be the answer, and only then read on the
slice. The figure from fitting on the slice itself is printed beside it, marked,
because it is the one a naive run produces and it is larger.

Deterministic, standard library only, no network, no weights.
"""
import json
import math
import os
import random
import sys
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)
import unseen_bias as ub                                          # noqa: E402

R = os.path.normpath(os.path.join(HERE, "..", ".."))
RES = os.path.join(R, "results")

BENCH = ["английский ключ", "русское слово", "русская фраза",
         "жаргон", "названия служб", "описание запроса"]

# label of a seed run -> its score file, for the five-candidate bench
SEEDS = [("v1.1 released", "bench_rujev_v11.jsonl"),
         ("seed-base-s2", "seeds/seed-base-s2_bench.jsonl"),
         ("seed-base-s3", "seeds/seed-base-s3_bench.jsonl")]


def L(p):
    return [json.loads(l) for l in open(os.path.join(RES, p), encoding="utf-8") if l.strip()]


def J(p):
    return json.load(open(os.path.join(RES, p), encoding="utf-8"))


def die(msg):
    raise SystemExit("vocab_bias: " + msg)


# ---------------------------------------------------------------- assembling

def unshuffle(seq, text):
    """Undo one random.Random(crc32(text)).shuffle of a list of len(seq)."""
    probe = list(range(len(seq)))
    random.Random(zlib.crc32(text.encode("utf-8"))).shuffle(probe)
    out = [None] * len(seq)
    for position, source in enumerate(probe):
        out[source] = seq[position]
    return out


def bench_five_candidates():
    """Per question, the five scenario keys in the order the bench used them."""
    v2a = [r for r in L("vocab2/rows.jsonl") if r["set"] == "стиль_A"]
    if len(v2a) != 1000:
        die("expected 1000 style A rows in vocab2, got %d" % len(v2a))
    return [(r["text"], unshuffle([o[:-4].lower() for o in r["options"]], r["text"]),
             r["gold_label"]) for r in v2a]


def blocks_of(rows):
    out = {}
    for r in rows:
        out.setdefault(r["vocab"], []).append(r)
    return out


def build():
    """cells[(vocabulary, k)] = {"rows": [(options, gold)], "models": {name: [logits]}}"""
    vocab_bench = {v: d["метки"] for v, d in J("fix18/vocab.json").items()}
    five = bench_five_candidates()

    # --- the five-candidate bench, every published run on it
    scored = {}
    for name, path in SEEDS:
        scored[name] = blocks_of(L(path))
    scored["v1.3 released"] = blocks_of(L("bench_rujev_v13.jsonl"))
    for extra in ("seed-aug-s2", "seed-aug-s2-ep3", "seed-aug-s3"):
        scored[extra] = blocks_of(L("seeds/%s_bench.jsonl" % extra))

    cells = {}
    for v in BENCH:
        table = vocab_bench[v]
        rows, models = [], {}
        for j, (text, scen, gold) in enumerate(five):
            opts = [table[s] for s in scen]
            rows.append((opts, gold, opts.index(table[gold])))
        for name, blk in scored.items():
            if v not in blk or len(blk[v]) != 1000:
                continue
            series = blk[v]
            for j, (_opts, _gold, ans) in enumerate(rows):
                if series[j]["answer"] != ans:
                    die("recovered option order disagrees with %s, %s row %d" % (name, v, j))
            models[name] = [r["logits"] for r in series]
        cells[(v, 5)] = {"rows": rows, "models": models}

    # --- the eighteen-candidate bench
    f18 = L("fix18/rows.jsonl")
    s11, s13 = L("fix18/scores_v11.jsonl"), L("fix18/scores_v13.jsonl")
    for a, b, c in zip(f18, s11, s13):
        if not (a["vocab"] == b["vocab"] == c["vocab"] and a["answer"] == b["answer"] == c["answer"]
                and a["gold_label"] == b["gold_label"] == c["gold_label"]):
            die("fix18 score files are not aligned with fix18/rows.jsonl")
    by = {}
    for r, x, y in zip(f18, s11, s13):
        by.setdefault(r["vocab"], []).append((r, x, y))
    for v in BENCH:
        rows = [(r["options"], r["gold_label"], r["answer"]) for r, _x, _y in by[v]]
        cells[(v, 18)] = {"rows": rows,
                          "models": {"v1.1 released": [x["logits"] for _r, x, _y in by[v]],
                                     "v1.3 released": [y["logits"] for _r, _x, y in by[v]]}}

    # --- the two probes
    probe_names = {("vocab2", "стиль_A"): "коды _SCN", ("vocab2", "стиль_B"): "аббревиатуры",
                   ("vocab3", "стиль_A"): "коды", ("vocab3", "стиль_B"): "сленг"}
    for name in ("vocab2", "vocab3"):
        rows_all = L("%s/rows.jsonl" % name)
        a11, a13 = L("%s/scores_v11.jsonl" % name), L("%s/scores_v13.jsonl" % name)
        for p, x, y in zip(rows_all, a11, a13):
            if not (p["vocab"] == x["vocab"] == y["vocab"] and p["answer"] == x["answer"] == y["answer"]
                    and p["gold_label"] == x["gold_label"] == y["gold_label"]):
                die("%s score files are not aligned with its rows" % name)
        grouped = {}
        for p, x, y in zip(rows_all, a11, a13):
            grouped.setdefault(p["vocab"], []).append((p, x, y))
        k = 5 if name == "vocab2" else 18
        for style, trio in grouped.items():
            label = probe_names[(name, style)]
            rows = [(p["options"], p["gold_label"], p["answer"]) for p, _x, _y in trio]
            cells[(label, k)] = {"rows": rows,
                                 "models": {"v1.1 released": [x["logits"] for _p, x, _y in trio],
                                            "v1.3 released": [y["logits"] for _p, _x, y in trio]}}
        if name == "vocab3":
            # five-candidate view, by renormalising over the five bench options
            table = {s: J("vocab3/vocab.json")[s]["метки"] for s in grouped}
            for style, trio in grouped.items():
                label = probe_names[(name, style)]
                lab = table[style]
                rows, keep = [], []
                for j, (p, _x, _y) in enumerate(trio):
                    _text, scen, gold = five[j]
                    want = [lab[s] for s in scen]
                    idx = [p["options"].index(w) for w in want]
                    rows.append((want, gold, want.index(lab[gold])))
                    keep.append(idx)
                cells[(label, 5)] = {
                    "rows": rows,
                    "models": {m: [[lg[i] for i in idx] for lg, idx in zip(series, keep)]
                               for m, series in cells[(label, 18)]["models"].items()}}
    return cells


def slices_of(vocabulary):
    """label -> membership of the all, clean and strict slices."""
    if vocabulary in BENCH:
        o = J("fix18/overlap_bench.json")[vocabulary]
    elif vocabulary in ("коды _SCN", "аббревиатуры"):
        style = "стиль_A" if vocabulary == "коды _SCN" else "стиль_B"
        o = J("vocab2/overlap.json")[style]
    else:
        style = "стиль_A" if vocabulary == "коды" else "стиль_B"
        o = J("vocab3/overlap.json")[style]
    allk = set(o)
    clean = {k for k, v in o.items()
             if not any("точное" in kinds for kinds in v["совпадения"].values())}
    strict = {k for k, v in o.items() if not v["совпадения"]}
    return {"all": allk, "clean": clean, "strict": strict}


# ---------------------------------------------------------------- statistics

def hits(logits, rows, idx):
    return [int(max(range(len(logits[i])), key=lambda k: logits[i][k]) == rows[i][2]) for i in idx]


def samples(logits, rows, idx):
    return [(logits[i], [str(o) for o in rows[i][0]], rows[i][2], rows[i][1]) for i in idx]


def mcnemar(b, c):
    """Exact two-sided binomial test on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    m = min(b, c)
    tail = sum(math.comb(n, i) for i in range(m + 1)) * (0.5 ** n)
    return min(1.0, 2 * tail)


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    cells = build()
    print("Per-option biases, per-label contributions and seed spread on the vocabularies")
    print("Out of fold: %d half-splits, seeds 0 to %d, both halves scored. "
          "Bootstrap %d draws, seed %d.\n" % (ub.SPLITS, ub.SPLITS - 1, ub.BOOT, ub.BOOT_SEED))

    order = [(v, k) for v in BENCH for k in (5, 18)]
    order += [("коды _SCN", 5), ("аббревиатуры", 5),
              ("коды", 5), ("коды", 18), ("сленг", 5), ("сленг", 18)]

    summary = []
    FITS = {}
    for v, k in order:
        if (v, k) not in cells:
            continue
        cell = cells[(v, k)]
        rows = cell["rows"]
        sl = slices_of(v)
        print("\n## %s, %d candidates\n" % (v, k))

        # fitted once, on every question of this vocabulary, then read on a slice
        whole = list(range(len(rows)))
        want = {"v1.1 released", "v1.3 released"}
        if k == 5:
            want |= {name for name, _p in SEEDS}
        fit_all = {m: ub.out_of_fold(samples(series, rows, whole), True)
                   for m, series in cell["models"].items() if m in want}
        FITS[(v, k)] = fit_all

        for sname in ("all", "clean", "strict"):
            keep = sl[sname]
            idx = [i for i, r in enumerate(rows) if r[1] in keep]
            if not idx or (sname != "all" and len(keep) == len(sl["all"])):
                continue
            if len(keep) < 2:
                print("### %s: %d labels, skipped\n" % (sname, len(keep)))
                continue
            print("### %s: %d labels, %d questions\n" % (sname, len(keep), len(idx)))

            # 1. accuracy as published, every run available on this cell
            print("| Run | accuracy |")
            print("|---|---|")
            acc = {}
            for m, series in cell["models"].items():
                h = hits(series, rows, idx)
                acc[m] = mean(h)
            for m in sorted(acc, key=lambda m: -acc[m]):
                print("| %s | %.3f |" % (m, acc[m]))

            if "v1.1 released" not in cell["models"] or "v1.3 released" not in cell["models"]:
                print()
                continue
            h11 = hits(cell["models"]["v1.1 released"], rows, idx)
            h13 = hits(cell["models"]["v1.3 released"], rows, idx)
            raw = mean(h13) - mean(h11)

            # 2. the difference, raw and after the biases
            on11 = ub.out_of_fold(samples(cell["models"]["v1.1 released"], rows, idx), True)
            on13 = ub.out_of_fold(samples(cell["models"]["v1.3 released"], rows, idx), True)
            f11 = [fit_all["v1.1 released"][i] for i in idx]
            f13 = [fit_all["v1.3 released"][i] for i in idx]
            labels = [rows[i][1] for i in idx]
            diff = [x - y for y, x in zip(f11, f13)]
            lo_r, hi_r = ub.ci_rows(diff, ub.BOOT_SEED)
            lo_l, hi_l = ub.ci_labels(diff, labels, ub.BOOT_SEED)
            raw_diff = [x - y for y, x in zip(h11, h13)]
            rlo_r, rhi_r = ub.ci_rows(raw_diff, ub.BOOT_SEED)
            rlo_l, rhi_l = ub.ci_labels(raw_diff, labels, ub.BOOT_SEED)
            print("\n| v1.3 minus v1.1 | difference | 95% over rows | 95% over labels |")
            print("|---|---|---|---|")
            print("| as published | %+.3f | [%+.3f, %+.3f] | [%+.3f, %+.3f] |"
                  % (raw, rlo_r, rhi_r, rlo_l, rhi_l))
            print("| after biases and T | %+.3f | [%+.3f, %+.3f] | [%+.3f, %+.3f] |"
                  % (mean(diff), lo_r, hi_r, lo_l, hi_l))
            print("| v1.1 with biases | %.3f | | |" % mean(f11))
            print("| v1.3 with biases | %.3f | | |" % mean(f13))
            if sname != "all":
                print("| after biases fitted on the slice itself | %+.3f | | |"
                      % (mean(on13) - mean(on11)))
                print("\nThe biases behind the headline row were fitted on all %d questions of "
                      "this vocabulary and then read on the slice. Fitting them on the slice "
                      "itself gives %.3f against %.3f, which is larger because on this slice "
                      "some options are never the answer and the fit removes them."
                      % (len(rows), mean(on13), mean(on11)))

            # 3. per-label contribution and leave-one-label-out
            per = {}
            for g in sorted(keep):
                gi = [j for j, i in enumerate(idx) if rows[i][1] == g]
                if not gi:
                    continue
                net = sum(h13[j] - h11[j] for j in gi)
                per[g] = (len(gi), net)
            loo = {}
            for g in per:
                gi = [j for j, i in enumerate(idx) if rows[i][1] != g]
                loo[g] = mean([h13[j] for j in gi]) - mean([h11[j] for j in gi])
            top = sorted(per, key=lambda g: -abs(per[g][1]))[:3]
            print("\n| Label | n | net questions, v1.3 minus v1.1 | difference without it |")
            print("|---|---|---|---|")
            for g in top:
                print("| %s | %d | %+d | %+.3f |" % (g, per[g][0], per[g][1], loo[g]))
            if "qa" in per and "qa" not in top:
                print("| qa | %d | %+d | %+.3f |" % (per["qa"][0], per["qa"][1], loo["qa"]))
            worst = min(loo, key=lambda g: loo[g])
            best = max(loo, key=lambda g: loo[g])
            print("\nLeave one label out: lowest %+.3f without `%s`, highest %+.3f without `%s`, "
                  "against %+.3f on every label." % (loo[worst], worst, loo[best], best, raw))
            if "qa" in per:
                up = sum(1 for g in per if per[g][1] > 0)
                down = sum(1 for g in per if per[g][1] < 0)
                print("By label v1.3 is ahead on %d and behind on %d of %d." % (up, down, len(per)))

            summary.append({"vocab": v, "k": k, "slice": sname, "n": len(idx),
                            "raw": raw, "fixed": mean(diff),
                            "on_slice": mean(on13) - mean(on11),
                            "acc": dict(acc), "qa": per.get("qa")})
            print()

    # ---------------------------------------------------------- label picks
    print("\n## How often each release answers `qa`, 18 candidates\n")
    print("| Vocabulary | label for qa | questions that are qa | v1.1 picks it | v1.3 picks it |")
    print("|---|---|---|---|---|")
    for v, k in order:
        if k != 18 or (v, k) not in cells:
            continue
        cell = cells[(v, k)]
        rows = cell["rows"]
        gold_rows = [i for i, r in enumerate(rows) if r[1] == "qa"]
        if not gold_rows:
            continue
        lab = rows[gold_rows[0]][0][rows[gold_rows[0]][2]]
        counts = {}
        for m in ("v1.1 released", "v1.3 released"):
            series = cell["models"][m]
            counts[m] = sum(1 for i, r in enumerate(rows)
                            if r[0][max(range(len(series[i])), key=lambda t: series[i][t])] == lab)
        print("| %s | %s | %d | %d | %d |"
              % (v, lab, len(gold_rows), counts["v1.1 released"], counts["v1.3 released"]))

    # ---------------------------------------------------------------- seeds
    print("\n## The v1.1 recipe across seeds, five candidates\n")
    runs = ["v1.1 released", "seed-base-s2", "seed-base-s3"]
    print("| Vocabulary | slice | %s | spread | v1.3 | v1.3 minus best | v1.3 minus mean |"
          % " | ".join(runs))
    print("|" + "---|" * (len(runs) + 6))
    seed_spread = {}
    with_biases = []
    for v in BENCH:
        cell = cells[(v, 5)]
        rows = cell["rows"]
        sl = slices_of(v)
        for sname in ("all", "clean", "strict"):
            keep = sl[sname]
            idx = [i for i, r in enumerate(rows) if r[1] in keep]
            if not idx or len(keep) < 2 or (sname != "all" and len(keep) == len(sl["all"])):
                continue
            a = {m: mean(hits(cell["models"][m], rows, idx)) for m in runs if m in cell["models"]}
            if len(a) < len(runs):
                continue
            spread = max(a.values()) - min(a.values())
            v13 = mean(hits(cell["models"]["v1.3 released"], rows, idx))
            seed_spread[(v, sname)] = spread
            print("| %s | %s | %s | %.3f | %.3f | %+.3f | %+.3f |"
                  % (v, sname, " | ".join("%.3f" % a[m] for m in runs), spread, v13,
                     v13 - max(a.values()), v13 - mean(list(a.values()))))
            fits = FITS.get((v, 5), {})
            if all(m in fits for m in runs + ["v1.3 released"]):
                b = {m: mean([fits[m][i] for i in idx]) for m in runs + ["v1.3 released"]}
                with_biases.append((v, sname, b, max(b[m] for m in runs) - min(b[m] for m in runs)))

    if with_biases:
        print("\nThe same runs after per-option biases, fitted on all 1,000 questions of the "
              "vocabulary and read on the slice:\n")
        print("| Vocabulary | slice | %s | spread | v1.3 |" % " | ".join(runs))
        print("|" + "---|" * (len(runs) + 4))
        for v, sname, b, spread in with_biases:
            print("| %s | %s | %s | %.3f | %.3f |"
                  % (v, sname, " | ".join("%.3f" % b[m] for m in runs), spread,
                     b["v1.3 released"]))

    print("\nExact McNemar, `seed-base-s2` against the released v1.1, five candidates:\n")
    print("| Vocabulary | slice | b (v1.1 only) | c (seed only) | p |")
    print("|---|---|---|---|---|")
    for v in BENCH:
        cell = cells[(v, 5)]
        rows = cell["rows"]
        sl = slices_of(v)
        for sname in ("all", "strict"):
            keep = sl[sname]
            idx = [i for i, r in enumerate(rows) if r[1] in keep]
            if not idx or len(keep) < 2 or (sname != "all" and len(keep) == len(sl["all"])):
                continue
            h1 = hits(cell["models"]["v1.1 released"], rows, idx)
            h2 = hits(cell["models"]["seed-base-s2"], rows, idx)
            b = sum(1 for x, y in zip(h1, h2) if x and not y)
            c = sum(1 for x, y in zip(h1, h2) if y and not x)
            print("| %s | %s | %d | %d | %.3g |" % (v, sname, b, c, mcnemar(b, c)))

    print("\n## Cells where the difference between the releases is no larger than the seed spread\n")
    print("| Vocabulary | slice | v1.3 minus v1.1 | seed spread |")
    print("|---|---|---|---|")
    found = 0
    for s in summary:
        if s["k"] != 5:
            continue
        sp = seed_spread.get((s["vocab"], s["slice"]))
        if sp is None:
            continue
        if abs(s["raw"]) <= sp:
            found += 1
            print("| %s | %s | %+.3f | %.3f |" % (s["vocab"], s["slice"], s["raw"], sp))
    if not found:
        print("| none | | | |")

    print("\n## After the biases, every cell\n")
    print("| Vocabulary | candidates | slice | n | as published | after biases | "
          "after biases fitted on the slice |")
    print("|---|---|---|---|---|---|---|")
    for s in summary:
        on = "%+.3f" % s["on_slice"] if s["slice"] != "all" else "same"
        print("| %s | %d | %s | %d | %+.3f | %+.3f | %s |"
              % (s["vocab"], s["k"], s["slice"], s["n"], s["raw"], s["fixed"], on))


if __name__ == "__main__":
    main()
