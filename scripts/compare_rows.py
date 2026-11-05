#!/usr/bin/env python3
"""Per-row comparison of two releases on the comparable rows.

This answers the suspicion that an identical count of correct answers means one
checkpoint was scored twice: it prints the hashes of the weight files, the
number of disagreements in each direction, and writes a per-row table so the
count can be checked by hand.
"""
import argparse, collections, hashlib, json, os

ap = argparse.ArgumentParser()
ap.add_argument("--rows", required=True)
ap.add_argument("--a", required=True, help="name=path to the first model's scores")
ap.add_argument("--b", required=True)
ap.add_argument("--weights", action="append", default=[], help="name=path to a weight file")
ap.add_argument("--out", required=True)
a = ap.parse_args()


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


for spec in a.weights:
    name, path = spec.split("=", 1)
    print(f"{name}: sha256 {sha(path)}  ({os.path.getsize(path)} bytes)  {path}")

rows = [json.loads(l) for l in open(a.rows, encoding="utf-8")]
na, pa = a.a.split("=", 1)
nb, pb = a.b.split("=", 1)
A = [json.loads(l) for l in open(pa, encoding="utf-8")]
B = [json.loads(l) for l in open(pb, encoding="utf-8")]
assert len(A) == len(B) == len(rows), (len(A), len(B), len(rows))

ca = cb = both = neither = 0
by_task = collections.defaultdict(lambda: [0, 0, 0, 0])
with open(a.out, "w", encoding="utf-8") as fh:
    for r, x, y in zip(rows, A, B):
        assert x["answer"] == y["answer"] == r["answer"]
        ok_a = x["pick"] == x["answer"]
        ok_b = y["pick"] == y["answer"]
        if ok_a and ok_b: both += 1; by_task[r["task"]][0] += 1
        elif ok_a: ca += 1; by_task[r["task"]][1] += 1
        elif ok_b: cb += 1; by_task[r["task"]][2] += 1
        else: neither += 1; by_task[r["task"]][3] += 1
        fh.write(json.dumps({"row_id": r["row_id"], "task": r["task"],
                             "answer": r["answer"],
                             f"pick_{na}": x["pick"], f"pick_{nb}": y["pick"],
                             f"ok_{na}": ok_a, f"ok_{nb}": ok_b},
                            ensure_ascii=False) + "\n")

n = len(rows)
print(f"\nrows {n}")
print(f"both correct: {both}")
print(f"only {na} correct: {ca}")
print(f"only {nb} correct: {cb}")
print(f"neither correct: {neither}")
print(f"total correct: {na} {both+ca}, {nb} {both+cb}")
print(f"disagreements: {ca+cb}")
print(f"\n| Task | both correct | only {na} | only {nb} | neither | n |")
print("|---|---|---|---|---|---|")
for t, v in by_task.items():
    print(f"| {t} | {v[0]} | {v[1]} | {v[2]} | {v[3]} | {sum(v)} |")
print(f"\nper-row dump: {a.out}")
