#!/usr/bin/env python3
"""Audit a probe vocabulary against EVERY option string in the training mixtures.

A label that coincides is not replaced: the vocabulary is fixed before the
measurement is taken. It is excluded from the headline figure and reported
separately, the way 15 of 18 is reported for jargon.

Three levels, with the direction of the relation named, because it matters:
  точное        the label is literally among the training options (case-insensitive)
  обуч_с_метки  some training option starts with the label. The risk for short
                abbreviations: КАЛ is the start of календарь, which is in training
  метка_с_обуч  the label starts with a training option. The risk for codes:
                ALARM_SCN starts with the key alarm, which is in training

The keys of the JSON report are kept in Russian ("метки" for the label map,
"совпадения" for the hits, "чистая" for a label with no relation at all) because
the table and analysis scripts read them by those names.
"""
import argparse, json

ap = argparse.ArgumentParser()
ap.add_argument("--vocab", required=True)
ap.add_argument("--mix", action="append", required=True, help="name=path to train.jsonl")
ap.add_argument("--out", required=True)
ap.add_argument("--min-len", type=int, default=3, help="prefixes shorter than this are ignored")
a = ap.parse_args()

voc = json.load(open(a.vocab, encoding="utf-8"))
mixes = {}
for spec in a.mix:
    name, path = spec.split("=", 1)
    opts = set()
    for l in open(path, encoding="utf-8"):
        for o in json.loads(l).get("options", []):
            opts.add(str(o).lower())
    mixes[name] = opts
    print(f"{name}: unique option strings in training {len(opts)}")

report = {}
for style, d in voc.items():
    rec = {}
    for scen, lab in d["метки"].items():
        low = lab.lower()
        hits = {}
        for name, opts in mixes.items():
            kinds = []
            if low in opts:
                kinds.append("точное")
            if len(low) >= a.min_len and any(o.startswith(low) and o != low for o in opts):
                kinds.append("обуч_с_метки")
            if any(len(o) >= a.min_len and low.startswith(o) and o != low for o in opts):
                kinds.append("метка_с_обуч")
            if kinds:
                hits[name] = kinds
        rec[scen] = {"метка": lab, "совпадения": hits, "чистая": not hits}
    report[style] = rec
    clean = [s for s in rec if rec[s]["чистая"]]
    print(f"\n{style}: clean {len(clean)} of {len(rec)}")
    for s in sorted(rec):
        v = rec[s]
        mark = "clean" if v["чистая"] else str(v["совпадения"])
        print(f"   {s:16s} {v['метка']:20s} {mark}")
json.dump(report, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"\nreport: {a.out}")
