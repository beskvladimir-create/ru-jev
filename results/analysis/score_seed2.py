#!/usr/bin/env python3
"""Score the checkpoints of the second seed of the v1.3 recipe, and the
checkpoints it has to be compared with, over one file of rows.

    python results/analysis/score_seed2.py --models ../model \
        --rows results/fix18/rows.jsonl --out results/v13_seed2/fix18

Why this exists beside `score_ladder.py`. That script produced the published
ladder dumps and carries the ladder as a fixed list, and it scores the rows of
one `set` at a time, which suits the unseen datasets where one file holds
several. The wording bench holds six vocabularies in one file and all of them
are wanted, and the checkpoints here are not the ladder. Rather than change a
script whose output is already published, this one takes the rows file whole and
carries its own list.

It does no scoring of its own: it calls `scripts/score_rujev.py` once per
checkpoint, which is the scorer every other dump in `results/` came from.
Running it needs the weights.

The weights of the new run are not published, only their scores on these rows,
the same way the intermediate checkpoints of the ladder are handled.

Resumable: each checkpoint is written to a temporary file and renamed only once
it holds as many rows as the input, so an interrupted run leaves nothing
half-written and the next run picks up where it stopped. A finished dump with
the wrong number of rows is scored again rather than trusted.
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))

# (directory under --models, tag written into the dump and used as the file name)
#
# The four epochs of the new run, then the four checkpoints the rule in
# results/v13_seed2/README.md compares it with. `v1.3-seed2-ep4of4` and
# `v1.3-seed2-final` are the same weights read from two directories, the fourth
# epoch before and after the temperature was fitted, and both are scored as a
# control on the scoring path, as `v1.3` and `v1.3-ep3of4` are in the ladder.
CHECKPOINTS = [
    ("v13-seed2/ep0", "v1.3-seed2-ep1of4"),
    ("v13-seed2/ep1", "v1.3-seed2-ep2of4"),
    ("v13-seed2/ep2", "v1.3-seed2-ep3of4"),
    ("v13-seed2/ep3", "v1.3-seed2-ep4of4"),
    ("v13-seed2/final", "v1.3-seed2-final"),
    ("ru-jev-v1.3/ship", "v1.3"),
    ("ru-jev-v1.1", "v1.1"),
    ("seeds/seed-base-s2", "seed-base-s2"),
    ("seeds/seed-aug-s2-ep3", "seed-aug-s2-ep3"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--models", required=True, help="the directory holding the checkpoints")
    ap.add_argument("--out", required=True, help="where the per-checkpoint dumps go")
    ap.add_argument("--set", dest="which",
                    help='score only the rows whose "set" field is this; all rows by default')
    ap.add_argument("--only", help="comma separated tags, to score part of the list")
    ap.add_argument("--python", default=sys.executable,
                    help="the interpreter that has torch and transformers")
    a = ap.parse_args()

    rows = [l for l in open(a.rows, encoding="utf-8") if l.strip()]
    if a.which:
        rows = [l for l in rows if json.loads(l).get("set") == a.which]
        if not rows:
            raise SystemExit("no rows with set == %r in %s" % (a.which, a.rows))
    os.makedirs(a.out, exist_ok=True)

    subset = os.path.join(a.out, "_rows.jsonl")
    with open(subset, "w", encoding="utf-8") as fh:
        fh.writelines(rows)

    wanted = set(a.only.split(",")) if a.only else None
    scorer = os.path.join(ROOT, "scripts", "score_rujev.py")
    for sub, tag in CHECKPOINTS:
        if wanted is not None and tag not in wanted:
            continue
        model = os.path.join(a.models, sub)
        if not os.path.isdir(model):
            print("skip %s, no weights at %s" % (tag, model), flush=True)
            continue
        out = os.path.join(a.out, tag + ".jsonl")
        if os.path.exists(out):
            have = sum(1 for _l in open(out, encoding="utf-8"))
            if have == len(rows):
                print("skip %s, already scored" % tag, flush=True)
                continue
            print("rescoring %s, it holds %d rows of %d" % (tag, have, len(rows)), flush=True)
        print("== %s" % tag, flush=True)
        part = out + ".partial"
        subprocess.run([a.python, scorer, "--model", model, "--rows", subset,
                        "--out", part, "--tag", tag], check=True)
        have = sum(1 for _l in open(part, encoding="utf-8"))
        if have != len(rows):
            raise SystemExit("%s wrote %d rows of %d" % (tag, have, len(rows)))
        os.replace(part, out)

    print("done", flush=True)


if __name__ == "__main__":
    main()
