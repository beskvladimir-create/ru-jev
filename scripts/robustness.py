#!/usr/bin/env python3
"""How much the answer depends on how the options are worded.

A caller names their categories in their own words, and those words are not the
ones in the training data. This renders the same MASSIVE routing questions under
several vocabularies and reports accuracy for each. Each vocabulary is marked with the version whose training mix first contains
it, so a column that a checkpoint never trained on measures generalisation
rather than recall.

Usage: python scripts/robustness.py --model model/ru-jev-v1.1 [--n 600]
"""
import argparse
import json
import os
import random
import sys
import zlib

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_data import (SCENARIO_PHRASE, SCENARIO_RU, jargon_translit,  # noqa: E402
                        load_any)

# Wordings a real caller might use, none of which appear in any training mix.
SCENARIO_JARGON = {
    "alarm": "алармы", "audio": "аудио", "calendar": "календарь-ивенты",
    "cooking": "фудтех", "datetime": "таймзоны", "email": "мейлы",
    "general": "смолток", "iot": "ай-оу-ти", "lists": "листы",
    "music": "плейлисты", "news": "ньюс", "play": "плеер",
    "qa": "факт-чек", "recommendation": "реко", "social": "сошиал",
    "takeaway": "фуд-доставка", "transport": "логистика", "weather": "вэзер",
}
SCENARIO_DEPARTMENT = {
    "alarm": "служба напоминаний", "audio": "отдел звука",
    "calendar": "отдел планирования", "cooking": "кулинарная редакция",
    "datetime": "служба времени", "email": "почтовая служба",
    "general": "общая линия", "iot": "инженерная служба",
    "lists": "отдел списков", "music": "музыкальная редакция",
    "news": "новостная редакция", "play": "служба воспроизведения",
    "qa": "справочная служба", "recommendation": "отдел рекомендаций",
    "social": "отдел соцсетей", "takeaway": "служба доставки",
    "transport": "транспортная служба", "weather": "метеослужба",
}
SCENARIO_QUESTION = {
    k: f"человек спрашивает про {v}" for k, v in SCENARIO_RU.items()
}

# Third element: from which release this style appears in the training mixture.
# v1 and v1.1 saw only the English key and the Russian word; everything else is
# unfamiliar to them. Without this marker the comparison between releases reads
# wrongly, because a trained style measures recall and not generalisation.
VOCABS = [
    ("английский ключ", None, "v1"),
    ("русское слово", SCENARIO_RU, "v1.1"),
    ("русская фраза", SCENARIO_PHRASE, "v1.2"),
    ("жаргон", SCENARIO_JARGON, "нигде"),
    ("названия служб", SCENARIO_DEPARTMENT, "нигде"),
    ("описание запроса", SCENARIO_QUESTION, "нигде"),
]


# v1.3 trains on transliterations of intent keys, and an intent key begins with
# its scenario name, so some of the scenario jargon labels reach training as a
# first word. For those labels the jargon column measures recall rather than
# generalisation. They are counted separately and the report gives the clean
# subset as well.
LEAKED = sorted(l for l, w in SCENARIO_JARGON.items() if w == jargon_translit(l))
CLEAN = sorted(l for l in SCENARIO_JARGON if l not in LEAKED)


class Scorer(nn.Module):
    def __init__(self, path):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(path)
        self.head = nn.Linear(self.backbone.config.hidden_size, 1)
        self.head.load_state_dict(torch.load(os.path.join(path, "head.pt"), map_location="cpu"))

    def forward(self, ids, am):
        return self.head(self.backbone(input_ids=ids, attention_mask=am)
                         .last_hidden_state[:, 0]).squeeze(-1)


def build_cases(n, n_options):
    """The bench rows, identical for every vocabulary and every model."""
    ds = load_any("mteb/amazon_massive_scenario", "ru")
    split = "validation" if "validation" in ds else list(ds)[0]
    rows = [r for r in ds[split] if r.get("label") in SCENARIO_RU]
    random.seed(13)
    random.shuffle(rows)
    rows = rows[:n]
    labels = sorted(SCENARIO_RU)
    cases = []
    for r in rows:
        gold = r["label"]
        others = [l for l in labels if l != gold]
        cases.append((str(r["text"]), gold,
                      random.sample(others, min(n_options - 1, len(others)))))
    return cases


def dump_rows(args):
    """Write every vocabulary's rendered options so another model can be scored
    on exactly these rows. The option order is seeded from a stable checksum of
    the text, not from hash(), which is salted per process."""
    cases = build_cases(args.n, args.options)
    written = 0
    with open(args.dump, "w", encoding="utf-8") as fh:
        for name, table, seen in VOCABS:
            def surface(lab):
                return table[lab] if table else lab.replace("_", " ")
            for text, gold, distract in cases:
                opts = [surface(gold)] + [surface(d) for d in distract]
                order = list(range(len(opts)))
                random.Random(zlib.crc32(text.encode("utf-8"))).shuffle(order)
                fh.write(json.dumps({
                    "vocab": name, "seen_from": seen, "text": text,
                    "gold_label": gold, "gold_clean": gold in CLEAN,
                    "options": [opts[i] for i in order],
                    "answer": order.index(0),
                }, ensure_ascii=False) + "\n")
                written += 1
    print(f"rows written: {written} to {args.dump}")
    print(f"vocabularies: {len(VOCABS)}  questions per vocabulary: {len(cases)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=600, help="questions per vocabulary")
    ap.add_argument("--options", type=int, default=5)
    ap.add_argument("--dump", help="write the rendered rows to this JSONL and exit; "
                                   "lets another model be scored on byte-identical rows")
    args = ap.parse_args()

    if args.dump:
        dump_rows(args)
        return

    calib = json.load(open(os.path.join(args.model, "calibration.json")))
    T = float(calib.get("temperature", 1.0))
    max_len = int(calib.get("max_len", 256))
    tok = AutoTokenizer.from_pretrained(args.model)
    model = Scorer(args.model).eval()
    torch.set_num_threads(os.cpu_count())

    # Same questions and same distractors for every vocabulary, so the only
    # thing that changes between columns is the wording. Built in one place so
    # that --dump and the scorer below can never drift apart.
    cases = build_cases(args.n, args.options)

    # The vocabulary names and the "seen from" marks are data values: they are
    # written into every dump as the fields "vocab" and "seen_from", and every
    # published table is keyed by them. CODEBOOK.md glosses each one.
    print(f"model: {args.model}  T={T}  questions per vocabulary: {len(cases)}\n")
    print("| vocabulary | in training since | correct | mean confidence | confidently wrong |")
    print("|---|---|---|---|---|")
    for name, table, seen in VOCABS:
        def surface(lab):
            return table[lab] if table else lab.replace("_", " ")
        ok = 0
        confs = []
        overconfident = 0
        hits = []
        with torch.no_grad():
            for text, gold, distract in cases:
                opts = [surface(gold)] + [surface(d) for d in distract]
                order = list(range(len(opts)))
                random.Random(zlib.crc32(text.encode("utf-8"))).shuffle(order)
                shuffled = [opts[i] for i in order]
                answer = order.index(0)
                enc = tok([text] * len(shuffled), shuffled, truncation=True,
                          max_length=max_len, padding=True, return_tensors="pt")
                p = torch.softmax(model(enc["input_ids"], enc["attention_mask"]) / max(T, 1e-2), dim=0)
                pick = int(torch.argmax(p))
                confs.append(float(p[pick]))
                hits.append(int(pick == answer))
                if pick == answer:
                    ok += 1
                elif float(p[pick]) > 0.8:
                    overconfident += 1
        print(f"| {name} | {seen} | {ok}/{len(cases)} = {ok/len(cases):.3f} | "
              f"{sum(confs)/len(confs):.3f} | {overconfident} |")
        if name == "жаргон":
            cl = [i for i, (_, gold, _) in enumerate(cases) if gold in CLEAN]
            n_cl = sum(hits[i] for i in cl)
            print(f"| жаргон, clean subset | нигде | {n_cl}/{len(cl)} = "
                  f"{n_cl/max(len(cl), 1):.3f} | | |")


if __name__ == "__main__":
    main()
