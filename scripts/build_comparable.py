#!/usr/bin/env python3
"""The six comparable tasks: identical rows with identical option wordings in
the v1.1 and v1.3 mixtures.

These tasks carry fixed option sets that the builder does not rewrite into
styles, so their rows are the same in both mixtures. The check is built in: the
script fails if they diverge.

The rows are put in a canonical order, by (task, text, options), so that the
per-row predictions of two models can be lined up by row number.
"""
import argparse, json, collections

SIX = ["russe", "danetqa", "terra", "parus", "rcb", "rwsd"]

ap = argparse.ArgumentParser()
ap.add_argument("--val-v13", required=True)
ap.add_argument("--val-v11", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()


def load(p):
    d = collections.defaultdict(list)
    for l in open(p, encoding="utf-8"):
        r = json.loads(l)
        if r["task"] in SIX:
            d[r["task"]].append(r)
    return d


A, B = load(a.val_v13), load(a.val_v11)
rows = []
for t in SIX:
    key = lambda r: (r["task"], r["input"], tuple(r["options"]), r["answer"])
    ka = collections.Counter(key(r) for r in A[t])
    kb = collections.Counter(key(r) for r in B[t])
    assert ka == kb, f"{t}: the mixtures' rows diverge, no comparison is possible"
    rows += sorted(A[t], key=key)

with open(a.out, "w", encoding="utf-8") as fh:
    for i, r in enumerate(rows):
        fh.write(json.dumps({"row_id": i, "task": r["task"], "input": r["input"],
                             "options": r["options"], "answer": r["answer"],
                             "decision_type": r.get("decision_type")},
                            ensure_ascii=False) + "\n")
print(f"rows: {len(rows)}")
print("per task:", dict(collections.Counter(r["task"] for r in rows)))
