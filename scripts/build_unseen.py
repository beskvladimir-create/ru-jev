#!/usr/bin/env python3
"""Rows of two datasets that neither compared model trained on.

Why these are kept apart from the wording bench. That bench is built on MASSIVE,
whose training split has been in our mixture since version 1, so it is home
ground and a gap in accuracy there proves nothing. Neither dataset here is in
our mixture, nor, by its model card, in Laya's training.

The category names are taken from the source rather than invented: for news
topics from the label_text field of ai-forever/headline-classification, for
emotions from ClassLabel(names=...) in sagteam/cedr_v1. Ru-Jev reads the option
text, so an invented name would corrupt the measurement.

The order of candidates is derived from the crc32 of the text. Ru-Jev is
indifferent to it, scoring each pair in its own pass, while Laya reads all
options in one pass and order is an input to it. The order therefore has to be
the same for both and reproducible.
"""
import argparse, json, random, zlib

ap = argparse.ArgumentParser()
ap.add_argument("--headline", required=True,
                help="test jsonl of ai-forever/headline-classification with a label_text field")
ap.add_argument("--out", required=True)
ap.add_argument("--n-per-set", type=int, default=1000)
_a = ap.parse_args()
OUT = _a.out
N_PER_SET = _a.n_per_set

RUBRICS = ["спорт", "происшествия", "политика", "наука", "культура", "экономика"]
EMO_RU = {"joy": "радость", "sadness": "грусть", "surprise": "удивление",
          "fear": "страх", "anger": "злость"}
EMO_ORDER = ["joy", "sadness", "surprise", "fear", "anger"]   # indices 0..4


def ordered(options, text):
    o = list(options)
    random.Random(zlib.crc32(text.encode("utf-8"))).shuffle(o)
    return o


rows = []

# News topics. The source file is downloaded beforehand; it holds 12,000 rows
# with a label_text field.
src = [json.loads(l) for l in open(_a.headline, encoding="utf-8")]
rnd = random.Random("headline")
rnd.shuffle(src)
for r in src[:N_PER_SET]:
    gold = r["label_text"]
    assert gold in RUBRICS, gold
    opts = ordered(RUBRICS, r["text"])
    rows.append({"set": "рубрики новостей", "licence": "mit",
                 "text": r["text"], "gold": gold,
                 "options": opts, "answer": opts.index(gold)})

# Emotions. Only rows with exactly one label are used: multi-label rows do not
# reduce to choosing one candidate, and an empty label means absence of emotion,
# a separate class that is not among the five.
from datasets import load_dataset
cd = load_dataset("sagteam/cedr_v1")
sp = "test" if "test" in cd else list(cd)[0]
single = [r for r in cd[sp] if len(r["labels"]) == 1]
rnd = random.Random("cedr")
rnd.shuffle(single)
opts_all = [EMO_RU[e] for e in EMO_ORDER]
for r in single[:N_PER_SET]:
    gold = EMO_RU[EMO_ORDER[r["labels"][0]]]
    opts = ordered(opts_all, r["text"])
    rows.append({"set": "эмоции", "licence": "apache-2.0",
                 "text": r["text"], "gold": gold,
                 "options": opts, "answer": opts.index(gold)})

with open(OUT, "w", encoding="utf-8") as fh:
    for r in rows:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")

import collections
print(f"written {len(rows)} rows to {OUT}")
for s, c in collections.Counter(r["set"] for r in rows).items():
    sub = [r for r in rows if r["set"] == s]
    print(f"  {s}: {c} rows, candidates {len(sub[0]['options'])}, "
          f"answer distribution {dict(collections.Counter(r['gold'] for r in sub))}")
