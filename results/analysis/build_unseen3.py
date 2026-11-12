#!/usr/bin/env python3
"""Build the rows of a further unseen dataset, by the rule in results/unseen3/README.md.

    python3 -I results/analysis/build_unseen3.py --test <test.jsonl> \\
        --set "научные рубрики" --licence mit --out results/unseen3/scibench/rows.jsonl

The test split is downloaded by hand and passed in, so that this script needs no
network and the revision used is recorded rather than resolved at run time. For
the dataset used here that is

    https://huggingface.co/datasets/ai-forever/ru-scibench-grnti-classification
    revision a984f8a65b2e9fae7a347bf7761b0eaad79c73a1, file test.jsonl

The selection follows the rule that was committed before any row was seen: the
test split only, seed 20261105, stratified by label, 1,000 rows or all of them
if there are fewer, the ten most frequent classes when there are more than ten,
and then five candidates per question, the correct label and four drawn from the
remaining nine.

One thing the rule did not anticipate. In this dataset every one of the 28
classes holds exactly 99 test rows, so "the ten most frequent" is a complete
tie. It is broken by name, ascending, which is deterministic and was fixed
without looking at any score. The ten classes it selects are therefore the first
ten in Russian alphabetical order, which has nothing to do with how hard they
are, and nothing to do with the result either way.
"""
import argparse
import collections
import json
import random

SEED = 20261105
MAX_ROWS = 1000
MIN_ROWS = 300
MAX_CLASSES = 10
CANDIDATES = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True, help="the downloaded test split, JSON lines")
    ap.add_argument("--set", dest="which", required=True, help='the value of the "set" field')
    ap.add_argument("--licence", required=True)
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--label-field", default="label_text")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.test, encoding="utf-8") if l.strip()]
    counts = collections.Counter(r[a.label_field] for r in rows)
    classes = [k for k, _n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    if len(classes) > MAX_CLASSES:
        classes = classes[:MAX_CLASSES]
    keep = [r for r in rows if r[a.label_field] in classes]
    # a canonical order before anything random, so the file order of the source
    # cannot reach the sample
    keep.sort(key=lambda r: (r[a.label_field], str(r.get("id", "")), r[a.text_field]))

    rng = random.Random(SEED)
    if len(keep) > MAX_ROWS:
        by = collections.defaultdict(list)
        for r in keep:
            by[r[a.label_field]].append(r)
        per = MAX_ROWS // len(classes)
        picked = []
        for c in sorted(by):
            picked.extend(rng.sample(by[c], min(per, len(by[c]))))
        picked.sort(key=lambda r: (r[a.label_field], str(r.get("id", "")), r[a.text_field]))
        keep = picked
    if len(keep) < MIN_ROWS:
        raise SystemExit("only %d rows after filtering, the rule needs at least %d"
                         % (len(keep), MIN_ROWS))

    out = []
    for r in keep:
        gold = r[a.label_field]
        others = [c for c in classes if c != gold]
        if len(classes) > CANDIDATES:
            opts = [gold] + rng.sample(others, CANDIDATES - 1)
        else:
            opts = list(classes)
        order = list(range(len(opts)))
        rng.shuffle(order)
        opts = [opts[i] for i in order]
        out.append({"set": a.which, "licence": a.licence, "text": r[a.text_field],
                    "gold": gold, "options": opts, "answer": opts.index(gold)})

    with open(a.out, "w", encoding="utf-8") as fh:
        for rec in out:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("rows: %d | classes: %d | candidates: %d | chance %.3f"
          % (len(out), len(classes), len(out[0]["options"]), 1 / len(out[0]["options"])))
    print("answer positions:", dict(sorted(collections.Counter(r["answer"] for r in out).items())))


if __name__ == "__main__":
    main()
