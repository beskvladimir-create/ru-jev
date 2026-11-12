#!/usr/bin/env python3
"""Score every checkpoint of the ladder on one unseen dataset.

    python results/analysis/score_ladder.py --set "рубрики новостей" \
        --models ../model --out results/unseen/ladder
    python results/analysis/score_ladder.py --set "эмоции" \
        --models ../model --out results/unseen/ladder_emotions

This is the script that produced `results/unseen/ladder/*.jsonl` and
`results/unseen/ladder_emotions/*.jsonl`. It does no scoring of its own: it
selects the rows of one dataset and then calls `scripts/score_rujev.py` once per
checkpoint, which is the same scorer every other dump in `results/` came from.
Running it therefore needs the weights, unlike the analysis scripts next to it,
which need only the dumps.

The intermediate checkpoints are not published, only their scores on these rows.
The two released ones are on Hugging Face. `ru-jev-v1.3/ep2` and
`ru-jev-v1.3/ship` are the same weights read from two directories, and both are
scored, as a control on the scoring path.

Deterministic: the option order of each row is fixed in the row file, and the
scorer applies no temperature.

Resumable: each checkpoint is written to a temporary file and renamed only when
it holds as many rows as the input, so a run that is interrupted leaves no
half-written dump behind and the next run picks up where it stopped. A finished
dump with the wrong number of rows is scored again rather than trusted.
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))

# (directory under --models, tag written into the dump and used as the file name)
LADDER = [
    ("ru-jev-v1.1", "v1.1"),
    ("seeds/seed-base-s2", "seed-base-s2"),
    ("ru-jev-v1.2", "v1.2"),
    ("seeds/seed-aug-s2", "seed-aug-s2"),
    ("seeds/seed-aug-s2-ep3", "seed-aug-s2-ep3"),
    ("seeds/seed-aug-s3", "seed-aug-s3"),
    ("ru-jev-v1.3/ep0", "v1.3-ep1of4"),
    ("ru-jev-v1.3/ep1", "v1.3-ep2of4"),
    ("ru-jev-v1.3/ep2", "v1.3-ep3of4"),
    ("ru-jev-v1.3/ship", "v1.3"),
    ("ru-jev-v1.3", "v1.3-ep4of4"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="which", required=True,
                    help='the value of the "set" field to score, for example "эмоции"')
    ap.add_argument("--rows", default=os.path.join(ROOT, "results/unseen/unseen_rows.jsonl"))
    ap.add_argument("--models", required=True, help="the directory holding the checkpoints")
    ap.add_argument("--out", required=True, help="where the per-checkpoint dumps go")
    ap.add_argument("--python", default=sys.executable,
                    help="the interpreter that has torch and transformers")
    ap.add_argument("--only", help="comma separated tags, to score part of the ladder")
    a = ap.parse_args()

    rows = [l for l in open(a.rows, encoding="utf-8") if l.strip()
            and json.loads(l)["set"] == a.which]
    if not rows:
        raise SystemExit("no rows with set == %r in %s" % (a.which, a.rows))
    os.makedirs(a.out, exist_ok=True)
    subset = os.path.join(a.out, "_rows.jsonl")
    with open(subset, "w", encoding="utf-8") as fh:
        fh.writelines(rows)
    print("%d rows with set == %r" % (len(rows), a.which))

    wanted = set(a.only.split(",")) if a.only else None
    scorer = os.path.join(ROOT, "scripts", "score_rujev.py")
    for directory, tag in LADDER:
        if wanted is not None and tag not in wanted:
            continue
        out = os.path.join(a.out, tag + ".jsonl")
        if os.path.exists(out):
            have = sum(1 for _l in open(out, encoding="utf-8"))
            if have == len(rows):
                print("skip %s, already scored" % tag, flush=True)
                continue
            print("rescoring %s, it holds %d rows of %d" % (tag, have, len(rows)), flush=True)
        model = os.path.join(a.models, directory)
        if not os.path.isdir(model):
            print("missing %s, skipped" % model, flush=True)
            continue
        print("== %s" % tag, flush=True)
        part = out + ".partial"
        subprocess.run([a.python, scorer, "--model", model, "--rows", subset,
                        "--out", part, "--tag", tag], check=True)
        have = sum(1 for _l in open(part, encoding="utf-8"))
        if have != len(rows):
            raise SystemExit("%s wrote %d rows of %d" % (tag, have, len(rows)))
        os.replace(part, out)
    os.remove(subset)
    print("done")


if __name__ == "__main__":
    main()
