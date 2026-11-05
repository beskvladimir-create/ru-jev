#!/usr/bin/env python3
"""Bench rows for the two rule-derived label vocabularies.

The questions, the candidate sets and the correct answers come from the existing
wording bench, so the difficulty of the questions is unchanged. ONLY the surface
form of the label differs. The order of candidates is derived from the crc32 of
the text, as in the unseen-dataset block: Ru-Jev is indifferent to it, scoring
each pair in its own pass, while Laya reads all options in one pass and order is
an input to it, so the order has to be the same for every model and both styles.

The vocabulary is read from vocab.json and is not changed once the measurement
has been taken.
"""
import argparse, json, random, zlib, collections

ap = argparse.ArgumentParser()
ap.add_argument("--bench", required=True, help="bench_rows.jsonl of the wording bench")
ap.add_argument("--vocab", required=True, help="vocab.json with the fixed styles")
ap.add_argument("--out", required=True)
a = ap.parse_args()

rows = [json.loads(l) for l in open(a.bench, encoding="utf-8")]
voc = json.load(open(a.vocab, encoding="utf-8"))

# option string -> scenario, recovered from the correct answer of each row
inv = collections.defaultdict(dict)
for r in rows:
    lab = r.get("gold_label") or r.get("gold")
    inv[r["vocab"]][r["options"][r["answer"]]] = lab

base = [r for r in rows if r["vocab"] == "русское слово"]
assert len(base) == 1000, len(base)
m = inv["русское слово"]


def ordered(items, text):
    o = list(items)
    random.Random(zlib.crc32(text.encode("utf-8"))).shuffle(o)
    return o


n = 0
with open(a.out, "w", encoding="utf-8") as fh:
    for r in base:
        scen = [m[o] for o in r["options"]]
        gold = m[r["options"][r["answer"]]]
        order = ordered(scen, r["text"])          # the same order for both styles
        for style, d in voc.items():
            opts = [d["метки"][s] for s in order]
            fh.write(json.dumps({
                "text": r["text"], "options": opts,
                "answer": order.index(gold),
                "vocab": style, "gold_label": gold,
                "set": style,
            }, ensure_ascii=False) + "\n")
            n += 1
print(f"rows: {n} ({len(base)} questions x {len(voc)} styles)")
