#!/usr/bin/env python3
"""How much of a rebuilt validation split was in the earlier training split.

Rebuilding the mixture does not recover the split a released checkpoint was
trained on. For some tasks the carve is stable and for others it is not, and
where it is not, a rebuilt validation row can be a row the checkpoint already
trained on. Scoring a release on such a rebuild reports an inflated accuracy
and refits its temperature to the wrong value.

This script measures that directly. A row is identified by its task and its
context string, not by the option wordings, because the option wordings are
exactly what changes between builds; two rows with the same context are the
same underlying example whatever the candidates are called.

Three numbers per task:

  kept        rebuilt validation rows whose context was also in the earlier
              validation split: the share of the carve that survived
  leaked      rebuilt validation rows whose context was in the earlier
              TRAINING split: the share that is contaminated
  new         rebuilt validation rows whose context was in neither, which
              happens when the source dataset revision itself moved

Usage:
  python scripts/split_overlap.py --earlier <dir> --rebuilt <dir> \
      [--json out.json]

Each directory holds train.jsonl and val.jsonl as the builder writes them.
"""
import argparse
import json
import os
from collections import defaultdict


def contexts(path):
    """Map task -> set of context strings, and the per-row list in file order."""
    by_task = defaultdict(set)
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            key = r["input"]
            by_task[r["task"]].add(key)
            rows.append((r["task"], key))
    return by_task, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--earlier", required=True, help="directory of the earlier build")
    ap.add_argument("--rebuilt", required=True, help="directory of the rebuilt mixture")
    ap.add_argument("--json", help="also write the table as JSON")
    a = ap.parse_args()

    e_train, _ = contexts(os.path.join(a.earlier, "train.jsonl"))
    e_val, _ = contexts(os.path.join(a.earlier, "val.jsonl"))
    _, r_val_rows = contexts(os.path.join(a.rebuilt, "val.jsonl"))

    stat = defaultdict(lambda: {"n": 0, "kept": 0, "leaked": 0, "new": 0})
    for task, key in r_val_rows:
        s = stat[task]
        s["n"] += 1
        if key in e_val.get(task, ()):
            s["kept"] += 1
        elif key in e_train.get(task, ()):
            s["leaked"] += 1
        else:
            s["new"] += 1

    order = sorted(stat, key=lambda t: (stat[t]["leaked"] / stat[t]["n"], t))
    print(f"earlier build: {a.earlier}")
    print(f"rebuilt build: {a.rebuilt}")
    print(f"rows in the rebuilt validation split: {sum(s['n'] for s in stat.values())}\n")
    print("| task | rows | kept | leaked into validation | neither |")
    print("|---|---|---|---|---|")
    for t in order:
        s = stat[t]
        n = s["n"]
        print(f"| {t} | {n} | {s['kept']/n:.1%} | {s['leaked']/n:.1%} | {s['new']/n:.1%} |")
    tot_n = sum(s["n"] for s in stat.values())
    tot_l = sum(s["leaked"] for s in stat.values())
    tot_k = sum(s["kept"] for s in stat.values())
    print(f"| all | {tot_n} | {tot_k/tot_n:.1%} | {tot_l/tot_n:.1%} | "
          f"{(tot_n-tot_k-tot_l)/tot_n:.1%} |")

    if a.json:
        out = {"earlier": a.earlier, "rebuilt": a.rebuilt,
               "identity": "task + context string",
               "tasks": {t: dict(stat[t]) for t in order},
               "total": {"n": tot_n, "kept": tot_k, "leaked": tot_l,
                         "new": tot_n - tot_k - tot_l}}
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=1)
        print(f"\nwritten: {a.json}")


if __name__ == "__main__":
    main()
