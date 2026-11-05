#!/usr/bin/env python3
"""Summary of the recomputed bench: all 18 candidates in every question.

The columns are the ones the model card uses: accuracy, mean confidence at each
model's own temperature, the gap between them, and the number of confident
errors (p_pick > 0.8) per 1,000 questions. For jargon the clean subset of 15
labels is reported as well, as in the card: exactly audio, news and weather
coincide with the output of the transliteration rule.

The figures are NOT comparable with the card, which puts 5 candidates out of 18
in each question against 18 here, so chance is 0.056 rather than 0.200.
"""
import argparse, json, math, statistics, collections

OWN_T = {"ru-jev-v1.1": 2.2754, "ru-jev-v1.3": 1.8640}
THR = 0.8
LEAKED = {"audio", "news", "weather"}      # coincide with the transliteration rule

# The vocabulary keys in the data are Russian; these are the English names used
# in the paper, printed alongside so the table reads without them.
VOCAB_EN = {"русское слово": "Russian word", "английский ключ": "English key",
            "описание запроса": "Descriptive sentence",
            "русская фраза": "Russian phrase",
            "названия служб": "Department name",
            "жаргон": "Anglicism jargon"}


ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="results/fix18")
a = ap.parse_args()
R = a.dir

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


vocabs = []
for r in rows:
    if r["vocab"] not in vocabs:
        vocabs.append(r["vocab"])

print("## Recomputed bench, all 18 candidates in every question\n")
print("| Vocabulary | Model | Accuracy | Mean confidence | Gap | "
      "Confident errors per 1000 |")
print("|---|---|---|---|---|---|")
for v in vocabs:
    idx = [i for i, r in enumerate(rows) if r["vocab"] == v]
    for name, all_rows in data.items():
        T = OWN_T.get(name, 1.0)
        rs = [all_rows[i] for i in idx]
        acc = sum(r["pick"] == r["answer"] for r in rs) / len(rs)
        pk = [probs_at(r, T)[r["pick"]] for r in rs]
        conf = statistics.mean(pk)
        over = sum(1 for r, p in zip(rs, pk)
                   if r["pick"] != r["answer"] and p > THR)
        print(f"| {VOCAB_EN.get(v, v)} | {name} | {acc:.3f} | {conf:.3f} | {conf-acc:+.3f} | "
              f"{round(over / len(rs) * 1000)} |")

print("\n### Anglicism jargon, the clean subset of 15 labels "
      "(without audio, news, weather)\n")
print("| Model | accuracy, all 18 | accuracy, clean 15 | gap, clean 15 |")
print("|---|---|---|---|")
idx = [i for i, r in enumerate(rows) if r["vocab"] == "жаргон"]
idc = [i for i in idx if rows[i]["gold_label"] not in LEAKED]
for name, all_rows in data.items():
    T = OWN_T.get(name, 1.0)
    f = lambda ii: sum(all_rows[i]["pick"] == all_rows[i]["answer"] for i in ii) / len(ii)
    acc_c = f(idc)
    conf_c = statistics.mean(probs_at(all_rows[i], T)[all_rows[i]["pick"]] for i in idc)
    print(f"| {name} | {f(idx):.3f} | {acc_c:.3f} | {conf_c-acc_c:+.3f} |")
print(f"\nClean questions {len(idc)} of {len(idx)}.")

print("\n## Accuracy per label, by vocabulary\n")
for v in vocabs:
    idx = [i for i, r in enumerate(rows) if r["vocab"] == v]
    print(f"### {VOCAB_EN.get(v, v)}\n")
    print("| Label | surface form | n | v1.1 | v1.3 | laya | v1.3 minus v1.1 |")
    print("|---|---|---|---|---|---|---|")
    out = []
    for L in sorted({rows[i]["gold_label"] for i in idx}):
        ii = [i for i in idx if rows[i]["gold_label"] == L]
        vv = [sum(data[n][i]["pick"] == data[n][i]["answer"] for i in ii) / len(ii)
              for n in ("ru-jev-v1.1", "ru-jev-v1.3", "laya instr A")]
        out.append((vv[1] - vv[0], L, len(ii), vv, rows[ii[0]]["options"][rows[ii[0]]["answer"]]))
    for d, L, n, vv, w in sorted(out):
        print(f"| {L} | {w} | {n} | " + " | ".join(f"{x:.3f}" for x in vv) + f" | {d:+.3f} |")
    print()
k = len(rows[0]["options"])
print(f"Candidates per question {k}, chance {1/k:.3f}.")
