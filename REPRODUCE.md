# Reproducing the paper

Every number in the paper is produced by a script in this repository from the
released weights. This file says which command produces which table.

## System requirements

CPU is enough for every measurement; only training needs a GPU.

| Item | Requirement |
|---|---|
| Python | 3.11 or newer |
| Packages | `pip install -r requirements.txt` |
| Disk | about 1.5 GB for the weights, 40 MB for the per-row scores here |
| Weights | downloaded from Hugging Face, see below |

The exact versions, the training hardware and the GPU time are in
`ENVIRONMENT.md`. The dataset revisions every figure comes from are pinned in
`DATASETS.md`.

Laya, the comparison model, pins versions of its own and must live in a separate
environment:

    python -m venv peer && peer/bin/pip install laya

## Getting the weights

    huggingface-cli download besktech/ru-jev --local-dir weights/v1.3
    huggingface-cli download besktech/ru-jev --revision v1.1 --local-dir weights/v1.1
    shasum -a 256 -c SHA256SUMS.weights   # paths need adjusting to the layout above

The files in this repository check with:

    shasum -a 256 -c SHA256SUMS

## Integrity of the inputs

Before any table, the per-row scores and the probe vocabularies can be verified
against the hashes recorded when they were made. The reading rule of the
independent probe was written down before its numbers were seen; its hashes are
in `results/vocab3/README.md` and in `SHA256SUMS`.

## Table by table

Paths assume this directory as the working directory and the released weights at
`weights/v1.3` and `weights/v1.1`.

### Training trajectory (Table 1)

Not re-run here: it is the log of the training run, quoted in `ENVIRONMENT.md`
with the loss of each epoch. To retrain:

    python scripts/build_data.py
    python scripts/train.py --config configs/v1_3.yaml

The released checkpoint is the third epoch of four. The learning-rate schedule
is set from the configured epoch count, so epochs 1 to 3 are identical whether
or not the fourth runs.

### Which wordings the augmentation helps with (Table 2)

    python -I results/analysis/final.py results

Writes the table that `results/analysis/final.txt` already holds: accuracy of
both releases in both candidate-set sizes, the exact McNemar counts, the
Holm-adjusted p over the 23 held-out cells, the label-level bootstrap interval,
AURC and confident errors. The bootstrap seed is 20261105.

Per-label detail, which is where the slang result shows as a reshuffle rather
than a lift:

    python scripts/table_fix18.py                  # six bench vocabularies, 18 candidates
    python scripts/table_vocab3.py                 # independent probe, three slices
    python scripts/table_vocab2.py --overlap results/vocab2/overlap.json \
        --scores "ru-jev-v1.1=results/vocab2/scores_v11.jsonl" \
        --scores "ru-jev-v1.3=results/vocab2/scores_v13.jsonl" \
        --scores "laya instr A=results/vocab2/scores_laya_instrA.jsonl"
    python scripts/bylabel.py results/vocab3

The five-candidate wording bench as the model card reports it:

    python scripts/aggregate.py \
        "ru-jev-v1.1=results/bench_rujev_v11.jsonl" \
        "ru-jev-v1.3=results/bench_rujev_v13.jsonl" \
        "laya instr A=results/bench_laya_instrA.jsonl" \
        "laya instr B=results/bench_laya_instrB.jsonl"

### Overlap audit of every probe vocabulary

    python scripts/build_data.py        # rebuilds data/ for the v1.3 mixture
    python scripts/overlap.py --vocab results/vocab3/vocab.json \
        --mix "v1.3=data/train.jsonl" --mix "v1.1=<v1.1 mixture>/train.jsonl" \
        --out /tmp/overlap.json

Reports, for every label, whether it is an exact match among the training
options and whether it is related to one by a prefix in either direction. This
is what marks the English key and the Russian word as training styles rather
than held-out wordings.

### Unseen datasets (Table 3)

    python scripts/build_unseen.py --headline <headline test jsonl> --out /tmp/unseen_rows.jsonl
    python scripts/score_rows.py --model weights/v1.3 --rows /tmp/unseen_rows.jsonl \
        --out /tmp/unseen_v13.jsonl --tag ru-jev-v1.3
    peer/bin/python scripts/score_laya.py --rows /tmp/unseen_rows.jsonl \
        --out /tmp/unseen_laya.jsonl --instr-id A \
        --instr "К какой категории относится запрос пользователя?"
    python scripts/unseen_table.py \
        "ru-jev-v1.1=results/unseen/unseen_v11.jsonl" \
        "ru-jev-v1.3=results/unseen/unseen_v13.jsonl"

### Identical rows for both releases (Table 4)

    python scripts/build_comparable.py --val-v13 data/val.jsonl \
        --val-v11 <v1.1 mixture>/val.jsonl --out /tmp/comparable_rows.jsonl
    python scripts/score_rows.py --model weights/v1.1 --rows /tmp/comparable_rows.jsonl \
        --out /tmp/comparable_v11.jsonl --tag v11
    python scripts/score_rows.py --model weights/v1.3 --rows /tmp/comparable_rows.jsonl \
        --out /tmp/comparable_v13.jsonl --tag v13
    python scripts/compare_rows.py --rows results/fix18/comparable_rows.jsonl \
        --a "v11=results/fix18/comparable_v11.jsonl" \
        --b "v13=results/fix18/comparable_v13.jsonl" \
        --weights "v1.1=weights/v1.1/model.safetensors" \
        --weights "v1.3=weights/v1.3/model.safetensors" \
        --out /tmp/byrow.jsonl

Prints the hashes of the weight files alongside the counts, so the identical
total of 8,120 correct answers cannot be mistaken for one checkpoint scored
twice. The temperature and ECE rows of that table:

    python scripts/calib.py \
        "ru-jev-v1.1=results/fix18/comparable_v11.jsonl" \
        "ru-jev-v1.3=results/fix18/comparable_v13.jsonl"

### Coverage against accuracy (Table 5)

    python scripts/coverage.py results/validation/val_v13_trainsplit.jsonl

Reproduces the table to the last digit. The temperature is fitted on one half and
the curve is built on the other; the split seed defaults to 17, the value the
release was prepared with, and it is part of the published result.

Do not rebuild the mixture to redo this. Six of the thirteen tasks reproduce
their train/validation split exactly and seven do not, so a rebuilt validation
row can be a row the release trained on: scoring the released checkpoint on a
rebuilt split gives 0.8540 instead of 0.8208 and refits the temperature to
1.5222 instead of 1.8640. `results/validation/README.md` has the per-task
numbers.

### Latency (Table 6)

Run the two engines one after the other, with nothing else on the machine:

    python scripts/latency.py --engine rujev --model weights/v1.3 \
        --rows results/fix18/rows.jsonl --out /tmp/latency_rujev.json
    peer/bin/python scripts/latency.py --engine laya \
        --rows results/fix18/rows.jsonl --out /tmp/latency_laya.json

Median of 25 runs after 5 discarded warm-up runs, at 2, 5 and 18 candidates.

### The released temperature

    python scripts/verify_temperature.py --model weights/v1.3 \
        --scores results/validation/val_v13_trainsplit.jsonl

Refits the temperature and compares it with the 1.8640 in `calibration.json`.
The difference is 0.0000. Two defaults make that exact: the half-split seed 17,
and the 301-point grid `0.2 + 7.8 i / 300`, of which the released value is a
point. `--grid trainer` uses the narrowing search in `train.py` instead and
lands 0.005 away.

### Fitting a temperature on your own labels

    python scripts/fit_temperature.py --model besktech/ru-jev \
        --rows your_decisions.jsonl --out your_calibration.json

Needed if you invent your own label vocabulary: the paper shows the released
temperature does not carry to unfamiliar wording or to a new domain.

## Calibration splits

`scripts/calib.py` holds the half-split machinery used everywhere, with the seed
recorded as `SPLIT_SEED = 20261105`. An earlier implementation shuffled its index
list in place without resetting it between splits, so each split was applied to
the permutation left by the previous one and the result depended on the order of
rows in the file. `aggregate.py` and `unseen_table.py` now call `calib.py` rather
than carrying their own copy.

## What is not here

The per-row scores of the seed-variation experiment are not included: they come
to 35 MB and nothing in the paper rests on them. What they showed, and the
mistake that made the first version of that experiment void, is written up in
`results/seeds/НАЙДЕНО.md` (in Russian).

The one-off scripts that drove the rented GPU and the releases are not here
either. Fourteen of them read secrets by name and carry machine access details,
and all of them are single-run work.

## Licence and citation

Code and weights are MIT. The training data keep the licences of their sources,
listed per source in `DATASETS.md`; none of the ten datasets ships a LICENSE
file, so every licence there is the `license` field of the repository's README
metadata. Citation details are in the paper.
