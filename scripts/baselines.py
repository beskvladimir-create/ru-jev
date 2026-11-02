#!/usr/bin/env python3
"""What a cheap baseline gets on the same data.

A number like 0.96 means nothing on its own. This trains the two things a
practitioner would reach for first, on the same training rows and scored on the
same validation rows, so every claim about the model has something to stand
against: predicting the majority class, and TF-IDF character n-grams with
logistic regression.

Tasks with a fixed label set are framed as plain classification. Yes/no tasks
are mapped to a canonical true/false label first, because the training mix
varies the wording ("да/нет", "верно/неверно", "true/false") and a classifier
predicting a surface form would be judged on words the example never offered.

Usage: python scripts/baselines.py data/train.jsonl data/val.jsonl
"""
import json
import sys
from collections import defaultdict

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

TRUE = {"да", "верно", "истина", "yes", "true", "приемлемо"}
FALSE = {"нет", "неверно", "ложь", "no", "false", "неприемлемо"}

# Ru-Jev v1.1, held-out half of the validation set, temperature fitted on the
# other half. Reproduce with scripts/eval.py.
RUJEV = {"massive_intent": 0.965, "massive_scenario": 0.954, "georeview": 0.629,
         "rureviews": 0.772, "toxicity": 0.961, "danetqa": 0.742, "russe": 0.822,
         "terra": 0.775, "rucola": 0.650, "rwsd": 0.564}
BOOLEAN = {"danetqa", "russe", "terra", "rucola", "rwsd"}


def read(path):
    out = defaultdict(list)
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        task = r["task"]
        if task not in RUJEV:
            continue
        opts = [str(o).strip() for o in r["options"]]
        gold = opts[r["answer"]]
        if task in BOOLEAN:
            low = gold.lower()
            if len(opts) != 2 or low not in TRUE | FALSE:
                continue
            gold = "T" if low in TRUE else "F"
        out[task].append((r["input"], gold))
    return out


def main():
    tr, va = read(sys.argv[1]), read(sys.argv[2])
    print("| task | train n | eval n | majority class | TF-IDF + logreg | Ru-Jev v1.1 |")
    print("| --- | --- | --- | --- | --- | --- |")
    for task in RUJEV:
        if len(tr[task]) < 50 or len(va[task]) < 50:
            print(f"| {task} | too few usable rows | | | | |")
            continue
        xtr, ytr = zip(*tr[task])
        xva, yva = zip(*va[task])
        majority = DummyClassifier(strategy="most_frequent").fit(xtr, ytr).score(xva, yva)
        clf = make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                            max_features=300_000, sublinear_tf=True),
            LogisticRegression(max_iter=2000, C=4.0)).fit(xtr, ytr)
        print(f"| {task} | {len(xtr)} | {len(xva)} | {majority:.3f} | "
              f"{clf.score(xva, yva):.3f} | {RUJEV[task]:.3f} |")


if __name__ == "__main__":
    main()
