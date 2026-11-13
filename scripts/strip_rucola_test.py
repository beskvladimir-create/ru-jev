#!/usr/bin/env python3
"""Remove the unlabelled RuCoLA test rows from a built mixture.

    python3 scripts/strip_rucola_test.py --in model/data-v1.3 --out model/data-v1.3.1

RuCoLA's test split carries no label. The builder used to pool every split of a
source and map the -1 placeholder onto a valid option index, so 2,788 sentences
entered the mixture carrying "неприемлемо", a label they do not have.
`scripts/build_data.py` no longer admits them, but a build made before the fix
still holds them, and rebuilding with the fixed builder draws a different
train/validation carve, because the pool it shuffles is smaller.

This takes the defect out of an existing build and changes nothing else. Every
remaining row keeps its wording, its option order and its position in the file,
so a model trained on the output differs from one trained on the input by the
removed rows and by nothing else. That is what makes the two comparable.

Deterministic, standard library only, no network. The sentences are identified
by the sha256 of their stripped text, read from
`results/rucola_test/test_only_sha256.txt`, which holds the hashes of the
sentences that occur in RuCoLA's test split and in neither labelled split. The
sentences themselves are not in this repository, see `DATASETS.md`.
"""
import argparse
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
HASHES = os.path.join(ROOT, "results", "rucola_test", "test_only_sha256.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True, help="a built mixture directory")
    ap.add_argument("--out", required=True, help="where the cleaned build goes")
    ap.add_argument("--hashes", default=HASHES)
    a = ap.parse_args()

    drop = set(l.strip() for l in open(a.hashes, encoding="utf-8") if l.strip())
    os.makedirs(a.out, exist_ok=True)
    total = 0
    for name in ("train", "val"):
        src = os.path.join(a.src, name + ".jsonl")
        kept, removed = [], 0
        for line in open(src, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("task") == "rucola":
                d = hashlib.sha256(r["input"].strip().encode("utf-8")).hexdigest()
                if d in drop:
                    removed += 1
                    continue
            kept.append(line)
        with open(os.path.join(a.out, name + ".jsonl"), "w", encoding="utf-8") as f:
            f.writelines(kept)
        total += removed
        print("%-6s %d rows in, %d removed, %d written" % (name, len(kept) + removed, removed, len(kept)))
    if total != len(drop):
        print("note: %d rows removed against %d hashes; a build caps its pools, so "
              "not every test sentence has to be in it" % (total, len(drop)))


if __name__ == "__main__":
    main()
