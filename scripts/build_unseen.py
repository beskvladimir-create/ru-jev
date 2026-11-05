#!/usr/bin/env python3
"""Строки двух наборов, которых не видела ни одна из сравниваемых моделей.

Зачем отдельно от стенда формулировок. Тот стенд построен на MASSIVE, чей
обучающий сплит лежит в нашей смеси с версии 1, то есть это домашнее поле, и
разрыв в точности там ничего не доказывает. Здесь оба набора не входят ни в
нашу смесь, ни, по карточке, в обучение Laya.

Названия категорий берутся из источника, а не придумываются: у рубрик новостей
это поле label_text в ai-forever/headline-classification, у эмоций это
ClassLabel(names=...) в sagteam/cedr_v1. Ру-Жев читает текст варианта, поэтому
выдуманное название испортило бы замер.

Порядок вариантов детерминирован от crc32 текста. Ру-Жеву он безразличен, он
считает каждую пару отдельным проходом, а Laya читает все варианты за один
проход, и ей порядок это вход. Значит он обязан быть одинаковым и
воспроизводимым.
"""
import argparse, json, random, zlib

ap = argparse.ArgumentParser()
ap.add_argument("--headline", required=True,
                help="jsonl теста ai-forever/headline-classification с полем label_text")
ap.add_argument("--out", required=True)
ap.add_argument("--n-per-set", type=int, default=1000)
_a = ap.parse_args()
OUT = _a.out
N_PER_SET = _a.n_per_set

RUBRICS = ["спорт", "происшествия", "политика", "наука", "культура", "экономика"]
EMO_RU = {"joy": "радость", "sadness": "грусть", "surprise": "удивление",
          "fear": "страх", "anger": "злость"}
EMO_ORDER = ["joy", "sadness", "surprise", "fear", "anger"]   # индексы 0..4


def ordered(options, text):
    o = list(options)
    random.Random(zlib.crc32(text.encode("utf-8"))).shuffle(o)
    return o


rows = []

# Рубрики новостей. Файл источника уже скачан, в нём 12 000 строк с label_text.
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

# Эмоции. Берём только строки ровно с одной меткой: многометочные к выбору
# одного варианта не сводятся, а пустые это отсутствие эмоции, отдельный класс,
# которого в списке пяти нет.
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
print(f"записано {len(rows)} строк в {OUT}")
for s, c in collections.Counter(r["set"] for r in rows).items():
    sub = [r for r in rows if r["set"] == s]
    print(f"  {s}: {c} строк, вариантов {len(sub[0]['options'])}, "
          f"распределение ответов {dict(collections.Counter(r['gold'] for r in sub))}")
