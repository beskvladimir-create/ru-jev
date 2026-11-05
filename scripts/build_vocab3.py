#!/usr/bin/env python3
"""Bench rows for the independent probe: all 18 categories in every question.

How this differs from the wording bench and from the rule-derived probe. Those
put 5 candidates out of 18 categories in a question; here all 18 are present.
Chance falls from 0.200 to 0.056, so the task is harder and figures from here are
NOT comparable with the model card or with the rule-derived probe.

The questions and the correct answers come from the existing bench, so the
questions themselves are unchanged. The order of candidates is derived from the
crc32 of the text and is the same for both styles: the permutation depends only
on the text and the length of the list, so position i holds the same scenario in
both styles. Ru-Jev is indifferent to order, scoring each pair in its own pass,
while Laya reads all options in one pass and order is an input to it.
"""
import argparse, json, random, zlib, collections

ap = argparse.ArgumentParser()
ap.add_argument("--bench", required=True)
ap.add_argument("--vocab", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

rows = [json.loads(l) for l in open(a.bench, encoding="utf-8")]
voc = json.load(open(a.vocab, encoding="utf-8"))

inv = {}
for r in rows:
    if r["vocab"] == "русское слово":
        inv[r["options"][r["answer"]]] = r.get("gold_label") or r.get("gold")

base = [r for r in rows if r["vocab"] == "русское слово"]
assert len(base) == 1000, len(base)

scenarios = sorted(next(iter(voc.values()))["метки"])
assert len(scenarios) == 18, len(scenarios)


def ordered(items, text):
    o = list(items)
    random.Random(zlib.crc32(text.encode("utf-8"))).shuffle(o)
    return o


n = 0
with open(a.out, "w", encoding="utf-8") as fh:
    for r in base:
        gold = inv[r["options"][r["answer"]]]
        order = ordered(scenarios, r["text"])     # the same order for both styles
        for style, d in voc.items():
            fh.write(json.dumps({
                "text": r["text"],
                "options": [d["метки"][s] for s in order],
                "answer": order.index(gold),
                "vocab": style, "gold_label": gold, "set": style,
            }, ensure_ascii=False) + "\n")
            n += 1
print(f"rows: {n} ({len(base)} questions x {len(voc)} styles, {len(scenarios)} candidates each)")
