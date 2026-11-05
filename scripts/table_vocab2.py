#!/usr/bin/env python3
"""Summary of the two rule-derived label vocabularies.

The method is the one in aggregate.py, so the figures stay comparable with the
wording bench:
  accuracy          share of correct answers, invariant to the temperature
  confidence        mean probability of the chosen candidate at each model's OWN
                    temperature: the released one for Ru-Jev, 1.0 for Laya
  gap               confidence minus accuracy
  confident errors  share of errors with p_pick > 0.8, per 1,000 questions

"Clean" means labels with no EXACT match among the training options, according
to overlap.json. Prefix relations are left out of the headline figure: they
touch every label of both styles, so they cannot serve as an exclusion rule.
They are reported separately by overlap.py.
"""
import argparse, json, math, statistics

OWN_T = {"ru-jev-v1.3": 1.8640, "ru-jev-v1.1": 2.2754}
THR = 0.8

# Style keys in the data are Russian; these are the names used in the paper.
STYLE_EN = {"стиль_A": "style A", "стиль_B": "style B"}


ap = argparse.ArgumentParser()
ap.add_argument("--overlap", required=True)
ap.add_argument("--scores", action="append", required=True, help="name=path")
a = ap.parse_args()

ov = json.load(open(a.overlap, encoding="utf-8"))


def probs_at(r, T):
    lg = r.get("logits")
    p = [x / T for x in lg] if lg is not None else \
        [math.log(max(x, 1e-12)) / T for x in r["probs"]]
    m = max(p)
    e = [math.exp(x - m) for x in p]
    z = sum(e)
    return [x / z for x in e]


data = {}
for spec in a.scores:
    name, path = spec.split("=", 1)
    data[name] = [json.loads(l) for l in open(path, encoding="utf-8")]

for style in ov:
    clean = {s for s, v in ov[style].items()
             if not any("точное" in k for k in v["совпадения"].values())}
    dirty = sorted(set(ov[style]) - clean)
    print(f"\n### {STYLE_EN.get(style, style)}: {len(clean)} clean labels "
          f"of {len(ov[style])}")
    if dirty:
        print(f"\nExcluded from the clean figure (exact match in training): "
              f"{', '.join(dirty)}\n")
    print("| Model | all 18 | clean | Mean confidence | Gap | Confident errors per 1000 |")
    print("|---|---|---|---|---|---|")
    for name, rows in data.items():
        rs = [r for r in rows if r.get("set") == style]
        if not rs:
            continue
        T = OWN_T.get(name, 1.0)
        acc = sum(r["pick"] == r["answer"] for r in rs) / len(rs)
        cl = [r for r in rs if r.get("gold_label") in clean]
        acc_c = sum(r["pick"] == r["answer"] for r in cl) / len(cl)
        pk = [probs_at(r, T)[r["pick"]] for r in rs]
        conf = statistics.mean(pk)
        over = sum(1 for r, p in zip(rs, pk) if r["pick"] != r["answer"] and p > THR)
        print(f"| {name} | {acc:.3f} | {acc_c:.3f} | {conf:.3f} | {conf-acc:+.3f} | "
              f"{round(over / len(rs) * 1000)} |")
    k = len(rs[0].get("logits") or rs[0]["probs"])
    print(f"\nQuestions {len(rs)}, candidates {k}, chance {1/k:.3f}.")
