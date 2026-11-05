#!/usr/bin/env python3
"""Temperature and ECE over half-splits, reproducibly.

What was wrong before. In aggregate.py and unseen_table.py the index list was
shuffled IN PLACE and never reset between splits:

    idx = list(range(len(rows)))
    for s in range(SPLITS):
        random.Random(s).shuffle(idx)     # idx is already shuffled by the previous step

Split s was therefore Random(s) applied to the permutation left by s-1: the
splits ran as a chain rather than as independent draws. The figures were still
deterministic, but they depended on the order of rows in the file and changed
whenever that order did, which is why they could not be reproduced.

Here the indices are reset on every split, each split's seed is derived from a
recorded SPLIT_SEED, and the rows are first put in a canonical order by their
own content, so the order of rows in the file cannot affect the result at all.

The canonical key of a row is (slice, text, index of the correct answer). It is
present in every score dump and is the same across models, so two models being
compared get the same splits row for row and the difference in their fitted
temperatures carries no split noise.
"""
import argparse, hashlib, json, math, random, statistics

SPLIT_SEED = 20261105        # recorded; change it only together with a new set of figures
SPLITS = 25
BINS = 15


def row_key(r):
    s = r.get("set") or r.get("vocab") or r.get("task") or ""
    t = r.get("text") or r.get("input") or ""
    return f"{s}\x00{t}\x00{r.get('row_id', '')}\x00{r['answer']}"


def canonical(rows):
    """Order by the sha256 of the key: independent of the order in the file."""
    return sorted(rows, key=lambda r: hashlib.sha256(
        row_key(r).encode("utf-8")).hexdigest())


def logprobs(r, T):
    lg = r.get("logits")
    if lg is not None:
        return [x / T for x in lg]
    return [math.log(max(x, 1e-12)) / T for x in r["probs"]]


def nll(rows, T):
    s = 0.0
    for r in rows:
        p = logprobs(r, T)
        m = max(p)
        s -= (p[r["answer"]] - m - math.log(sum(math.exp(x - m) for x in p)))
    return s / max(len(rows), 1)


def fitT(rows, lo=0.2, hi=8.0, steps=300):
    best, bt = None, 1.0
    for i in range(steps + 1):
        T = lo + (hi - lo) * i / steps
        v = nll(rows, T)
        if best is None or v < best:
            best, bt = v, T
    return bt


def probs_at(r, T):
    p = logprobs(r, T)
    m = max(p)
    e = [math.exp(x - m) for x in p]
    z = sum(e)
    return [x / z for x in e]


def ece(rows, T):
    b = [[] for _ in range(BINS)]
    for r in rows:
        p = probs_at(r, T)
        j = max(range(len(p)), key=lambda k: p[k])
        b[min(int(p[j] * BINS), BINS - 1)].append((p[j], int(j == r["answer"])))
    n = sum(len(x) for x in b)
    return sum(len(x) / n * abs(sum(c for c, _ in x) / len(x)
                                - sum(h for _, h in x) / len(x))
               for x in b if x)


def calibration(rows):
    rows = canonical(rows)
    n = len(rows)
    ts, es = [], []
    for s in range(SPLITS):
        idx = list(range(n))                      # reset on every split
        random.Random(SPLIT_SEED + s).shuffle(idx)
        h = n // 2
        T = fitT([rows[i] for i in idx[:h]])
        ts.append(T)
        es.append(ece([rows[i] for i in idx[h:]], T))
    return {"T_mean": statistics.mean(ts), "T_min": min(ts), "T_max": max(ts),
            "ECE_own_T": statistics.mean(es), "ECE_raw": ece(rows, 1.0),
            "n": n, "splits": SPLITS, "split_seed": SPLIT_SEED}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", action="append", required=True, help="name=path")
    ap.add_argument("--by-set", action="store_true",
                    help="report separately per set or vocab field")
    ap.add_argument("--json-out")
    a = ap.parse_args()
    data = {}
    for spec in a.scores:
        name, path = spec.split("=", 1)
        data[name] = [json.loads(l) for l in open(path, encoding="utf-8")]
    out = {}
    groups = [None]
    if a.by_set:
        groups = []
        for rows in data.values():
            for r in rows:
                g = r.get("set") or r.get("vocab")
                if g and g not in groups:
                    groups.append(g)
    print(f"SPLIT_SEED = {SPLIT_SEED}, splits {SPLITS}, ECE bins {BINS}\n")
    for g in groups:
        if g:
            print(f"## {g}\n")
        print("| Model | n | Fitted T | T min | T max | ECE as is | ECE after its own T |")
        print("|---|---|---|---|---|---|---|")
        for name, rows in data.items():
            rs = rows if g is None else [r for r in rows
                                         if (r.get("set") or r.get("vocab")) == g]
            if not rs:
                continue
            c = calibration(rs)
            out[f"{g or 'all'}|{name}"] = c
            print(f"| {name} | {c['n']} | {c['T_mean']:.4f} | {c['T_min']:.3f} | "
                  f"{c['T_max']:.3f} | {c['ECE_raw']:.4f} | {c['ECE_own_T']:.4f} |")
        print()
    if a.json_out:
        json.dump(out, open(a.json_out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print(f"figures: {a.json_out}")


if __name__ == "__main__":
    main()
