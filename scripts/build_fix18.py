#!/usr/bin/env python3
"""The wording bench again, but with ALL 18 candidates in every question.

The published bench puts 5 candidates out of 18 categories in each question.
Here there are 18, as in the independent probe. Chance falls from 0.200 to
0.056, so absolute figures from this file are NOT comparable with the model
card. Only comparisons inside this file are: every model and every vocabulary is
scored in the same setting.

The questions, the categories and the correct answers come from the existing
bench. The order of candidates is derived from the crc32 of the text and is the
same for all six vocabularies and all models, because the permutation depends
only on the text and the length of the list.
"""
import argparse, json, random, zlib, collections

ap = argparse.ArgumentParser()
ap.add_argument("--bench", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--vocab-out", required=True)
a = ap.parse_args()

rows = [json.loads(l) for l in open(a.bench, encoding="utf-8")]

# vocabulary -> {scenario: surface form}, recovered from the correct answer of each row
voc = collections.defaultdict(dict)
for r in rows:
    lab = r.get("gold_label") or r.get("gold")
    voc[r["vocab"]][lab] = r["options"][r["answer"]]
for v, d in voc.items():
    assert len(d) == 18 and len(set(d.values())) == 18, (v, len(d))

# gold_clean is carried over from the bench: it marks the 15 jargon labels that
# do not coincide with the transliteration rule
clean = {}
for r in rows:
    if "gold_clean" in r:
        clean[(r["vocab"], r.get("gold_label") or r.get("gold"))] = r["gold_clean"]

base = [r for r in rows if r["vocab"] == "русское слово"]
assert len(base) == 1000, len(base)
inv_ru = {w: s for s, w in voc["русское слово"].items()}
scen = sorted(voc["русское слово"])


def ordered(items, text):
    o = list(items)
    random.Random(zlib.crc32(text.encode("utf-8"))).shuffle(o)
    return o


n = 0
with open(a.out, "w", encoding="utf-8") as fh:
    for r in base:
        gold = inv_ru[r["options"][r["answer"]]]
        order = ordered(scen, r["text"])      # the same order for every vocabulary
        for v, d in voc.items():
            rec = {"text": r["text"], "options": [d[s] for s in order],
                   "answer": order.index(gold), "vocab": v,
                   "gold_label": gold, "set": v}
            if (v, gold) in clean:
                rec["gold_clean"] = clean[(v, gold)]
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
json.dump({v: {"описание": f"wording bench vocabulary: {v}", "метки": d}
           for v, d in voc.items()},
          open(a.vocab_out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"rows: {n} ({len(base)} questions x {len(voc)} vocabularies, {len(scen)} candidates each)")
