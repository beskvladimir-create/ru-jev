#!/usr/bin/env python3
"""Accuracy per label, and where the errors go.

Rows are ordered by the difference v1.3 minus v1.1, so it is visible where the
recipe helped and where it hurt. The directory is given as the first argument.
"""
import json, collections, sys
R = sys.argv[1] if len(sys.argv) > 1 else "results/vocab3"

# Style keys in the data are Russian; these are the names used in the paper.
STYLE_EN = {"стиль_A": "style A", "стиль_B": "style B"}

rows = [json.loads(l) for l in open(f"{R}/rows.jsonl", encoding="utf-8")]
voc = json.load(open(f"{R}/vocab.json", encoding="utf-8"))
ov = json.load(open(f"{R}/overlap.json", encoding="utf-8"))
sc = {n: [json.loads(l) for l in open(f"{R}/scores_{f}.jsonl", encoding="utf-8")]
      for n, f in [("v1.1", "v11"), ("v1.3", "v13"), ("laya", "laya_instrA")]}
for style in voc:
    inv = {v: k for k, v in voc[style]["метки"].items()}
    exact = {s for s, v in ov[style].items()
             if any("точное" in k for k in v["совпадения"].values())}
    idx = [i for i, r in enumerate(rows) if r["vocab"] == style]
    print(f"\n### {STYLE_EN.get(style, style)}: accuracy per label\n")
    print("| Label | surface form | n | v1.1 | v1.3 | laya | v1.3 minus v1.1 |")
    print("|---|---|---|---|---|---|---|")
    out = []
    for L in sorted({rows[i]["gold_label"] for i in idx}):
        ii = [i for i in idx if rows[i]["gold_label"] == L]
        v = [sum(sc[n][i]["pick"] == sc[n][i]["answer"] for i in ii) / len(ii)
             for n in ("v1.1", "v1.3", "laya")]
        out.append((v[1] - v[0], L, len(ii), v))
    for d, L, n, v in sorted(out):
        star = " (exact match in training)" if L in exact else ""
        print(f"| {L}{star} | {voc[style]['метки'][L]} | {n} | "
              + " | ".join(f"{x:.3f}" for x in v) + f" | {d:+.3f} |")
    att = collections.Counter()
    for i in idx:
        r = sc["v1.3"][i]
        if r["pick"] != r["answer"]:
            att[inv[rows[i]["options"][r["pick"]]]] += 1
    print("\nWhere v1.3's errors go:", ", ".join(f"{k} {v}" for k, v in att.most_common(6)))
