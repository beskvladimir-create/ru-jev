#!/usr/bin/env python3
"""Independent probe summary: three slices per style, paired McNemar test.

Slices:
  all 18   every label of the vocabulary
  clean    without labels that have an EXACT match among the training options
  strict   without any relation at all, including prefixes in either direction

McNemar. Both releases answer the same questions, so the comparison is paired.
b counts questions where v1.1 is right and v1.3 is wrong, c the reverse. The
p-value comes from the exact two-sided binomial test: with few disagreements a
chi-square approximation is wrong.

Confidence is computed at each model's OWN temperature: the released one for
Ru-Jev, 1.0 for Laya. Accuracy does not depend on the temperature, because the
argmax is invariant to it.
"""
import argparse, json, math, statistics

OWN_T = {"ru-jev-v1.3": 1.8640, "ru-jev-v1.1": 2.2754}
THR = 0.8

# Style keys in the data are Russian; these are the names used in the paper.
STYLE_EN = {"стиль_A": "style A", "стиль_B": "style B"}


ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="results/vocab3")
a = ap.parse_args()
R = a.dir

ov = json.load(open(f"{R}/overlap.json", encoding="utf-8"))
rows = [json.loads(l) for l in open(f"{R}/rows.jsonl", encoding="utf-8")]
data = {n: [json.loads(l) for l in open(f"{R}/scores_{f}.jsonl", encoding="utf-8")]
        for n, f in [("ru-jev-v1.1", "v11"), ("ru-jev-v1.3", "v13"),
                     ("laya instr A", "laya_instrA")]}


def probs_at(r, T):
    lg = r.get("logits")
    p = [x / T for x in lg] if lg is not None else \
        [math.log(max(x, 1e-12)) / T for x in r["probs"]]
    m = max(p)
    e = [math.exp(x - m) for x in p]
    z = sum(e)
    return [x / z for x in e]


def mcnemar(b, c):
    """Exact two-sided binomial test on the disagreements."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    return min(1.0, 2.0 * tail / (2 ** n))


for style in ov:
    labs = set(ov[style])
    exact = {s for s, v in ov[style].items()
             if any("точное" in k for k in v["совпадения"].values())}
    touched = {s for s, v in ov[style].items() if v["совпадения"]}
    slices = [("all 18", labs),
              ("clean by exact match", labs - exact),
              ("strict, clean by both criteria", labs - touched)]
    print(f"\n## {STYLE_EN.get(style, style)}\n")
    for sname, keep in slices:
        idx = [i for i, r in enumerate(rows)
               if r["vocab"] == style and r["gold_label"] in keep]
        n = len(idx)
        print(f"### {sname}: {len(keep)} labels, {n} questions\n")
        print("| Model | Accuracy | Mean confidence | Gap | "
              "Confident errors per 1000 |")
        print("|---|---|---|---|---|")
        hit = {}
        for name, rs_all in data.items():
            T = OWN_T.get(name, 1.0)
            rs = [rs_all[i] for i in idx]
            hit[name] = [r["pick"] == r["answer"] for r in rs]
            acc = sum(hit[name]) / n
            pk = [probs_at(r, T)[r["pick"]] for r in rs]
            conf = statistics.mean(pk)
            over = sum(1 for r, p in zip(rs, pk)
                       if r["pick"] != r["answer"] and p > THR)
            print(f"| {name} | {acc:.3f} | {conf:.3f} | {conf-acc:+.3f} | "
                  f"{round(over / n * 1000)} |")
        h1, h3 = hit["ru-jev-v1.1"], hit["ru-jev-v1.3"]
        b = sum(1 for x, y in zip(h1, h3) if x and not y)
        c = sum(1 for x, y in zip(h1, h3) if y and not x)
        p = mcnemar(b, c)
        d = (sum(h3) - sum(h1)) / n
        print(f"\nMcNemar, v1.3 against v1.1: difference {d:+.3f}, "
              f"only v1.1 correct {b}, only v1.3 correct {c}, "
              f"disagreements {b+c}, p = {p:.4g}\n")
    k = len(rows[0]["options"])
    print(f"Candidates per question {k}, chance {1/k:.3f}.")
