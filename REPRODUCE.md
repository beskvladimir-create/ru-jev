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
    python scripts/score_rujev.py --model weights/v1.3 --rows /tmp/unseen_rows.jsonl \
        --out /tmp/unseen_v13.jsonl --tag ru-jev-v1.3
    peer/bin/python scripts/score_laya.py --rows /tmp/unseen_rows.jsonl \
        --out /tmp/unseen_laya.jsonl --instr-id A \
        --instr "К какой категории относится запрос пользователя?"
    python scripts/unseen_table.py \
        "v1.1=results/unseen/unseen_v11.jsonl" \
        "v1.3=results/unseen/unseen_v13.jsonl"

The tags matter here. `unseen_table.py` looks each model's released temperature
up by its tag, so `v1.1=` and `v1.3=` give the mean confidence after the
released temperature, which is what `results/unseen/table.md` reports. Any other
tag falls through to a temperature of 1 and the confidence column comes out raw,
0.855 instead of 0.680 for v1.1. Accuracy and ECE are unaffected.

### Identical rows for both releases (Table 4)

    python scripts/build_comparable.py --val-v13 data/val.jsonl \
        --val-v11 <v1.1 mixture>/val.jsonl --out /tmp/comparable_rows.jsonl
    python scripts/score_rujev.py --model weights/v1.1 --rows /tmp/comparable_rows.jsonl \
        --out /tmp/comparable_v11.jsonl --tag v11
    python scripts/score_rujev.py --model weights/v1.3 --rows /tmp/comparable_rows.jsonl \
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

### One bias per option

    python scripts/fit_temperature.py --model besktech/ru-jev \
        --rows your_decisions.jsonl --out your_calibration.json --biases

A temperature cannot change which option comes out, so it cannot repair a model
that leans towards one of your labels whatever the input says. This fits one
additive bias per option together with the temperature, by maximum likelihood on
the same labelled rows, and writes both into the calibration file. The answer
then becomes `softmax(s / T + b[option])`. `serve/app.py` and `scripts/eval.py`
read the biases from that file and apply them. `--no-temperature` holds the
temperature at 1 and fits the biases alone. The script reports the accuracy out
of fold, because unlike a temperature these can be fitted to noise.

    python3 -I scripts/test_calibration.py

Checks the fit on scores built with a known lean, and checks that with the
biases switched off it agrees with the grid search every published temperature
came out of. No weights and no network.

### Where the economy and politics boundary moves

    python3 -I results/analysis/unseen_ladder.py results/unseen

Writes the table that `results/unseen/ladder.md` holds: eleven checkpoints
between v1.1 and v1.3 on the 1,000 news headlines, with the AUC of
`logit(экономика) - logit(политика)` over the 340 headlines of those two
rubrics, a paired bootstrap of that AUC against v1.1 (2,000 draws, seed
20261105), the median margin on each of the two rubrics, and recall for all six.
It reads the per-checkpoint logits from `results/unseen/ladder/`. The
intermediate weights are not published, only their scores on these rows.
Standard library only, no network.

### Biases, per-label contributions and seed spread on the vocabularies

    python3 -I results/analysis/vocab_bias.py

Writes the tables that `results/vocab_bias/README.md` holds: for every label
vocabulary, every slice and both candidate counts, the accuracy of each
published run, the difference between the releases before and after a
per-option correction, each label's contribution to that difference in
questions, the difference with any one label left out, how often each release
answers `qa`, the spread between three seeds of the v1.1 recipe with an exact
McNemar test, and the cells where the difference between the releases is no
larger than that spread. It reads only the per-row score files. Standard library
only, no network, no weights.

### Do the rankings of the ladder agree across unseen datasets

    python3 -I results/analysis/ladder_rank.py

Writes the table that `results/unseen3/ladder_rank.md` holds: the ten distinct
checkpoints on every unseen dataset, as published and with per-option biases,
the Spearman rank correlation between every pair of datasets, the six fixed
pairs of checkpoints with bootstrap intervals, and the spread between two runs
of the v1.1 recipe. The rule for reading it is in `results/unseen3/README.md`
and was committed before any row was downloaded.

The rows of the third dataset are built by

    python3 -I results/analysis/build_unseen3.py --test <test.jsonl> \
        --set "научные рубрики" --licence mit --out results/unseen3/scibench/rows.jsonl

from the test split of `ai-forever/ru-scibench-grnti-classification`, revision
`a984f8a65b2e`, downloaded by hand. The per-checkpoint logits come from
`results/analysis/score_ladder.py`, which needs the weights.

### The full score matrix and the checks it allows

    python3 -I results/analysis/score_ladder.py \
        --set "научные рубрики, все 28" --rows results/unseen3/matrix/rows.jsonl \
        --models ../model --out results/unseen3/matrix

Scores all ten checkpoints on all 2,772 test texts of the GRNTI rubrics against
all 28 rubric labels, 776,160 pairs, and writes one file per checkpoint into
`results/unseen3/matrix/`. It needs the weights, and it saves per checkpoint, so
an interrupted run continues where it stopped. On one CPU it takes about twelve
hours.

    python3 -I results/analysis/matrix_sensitivity.py

Writes the sections that `results/unseen3/ladder_rank.md` carries after its
verdict: the assembled alphabetical ten held against the dump that was scored
directly, twenty random tens of the 28 rubrics with seeds 0 to 19, and all 28
rubrics offered at once. Everything it computes was decided after the rule was
written and after the verdict was known, and it is marked as such in the output.
Standard library only, no network, no weights.

    python3 -I results/analysis/matrix_prefix.py > results/unseen3/prefix.md

Asks whether v1.3 stays ahead of the second v1.1 seed when the four rubrics
whose names start with a training option (Биология, Геология, Горное дело,
Математика) are left out: on the alphabetical ten without them, then over twenty
random tens of all 28 rubrics and twenty random tens of the 24 rubrics without a
prefix match, seeds 0 to 19, with intervals over rows and over labels. Added
after the reading rule, in answer to a question on the Hugging Face thread.
Standard library only, no network, no weights.

### Biases on every checkpoint of the ladder

    python3 -I results/analysis/ladder_bias.py results/unseen

Writes the table that `results/unseen/ladder_bias.md` holds: for each of the two
unseen datasets and each of the eleven checkpoints, accuracy as published and
accuracy out of fold once that checkpoint is allowed its own per-option biases,
followed by six paired differences with bootstrap intervals over rows and over
labels. It uses the functions of `unseen_bias.py`, so the splits, the seeds and
the number of draws are the same. Standard library only, no network.

The per-checkpoint logits it reads were produced by

    python results/analysis/score_ladder.py --set "рубрики новостей" \
        --models ../model --out results/unseen/ladder
    python results/analysis/score_ladder.py --set "эмоции" \
        --models ../model --out results/unseen/ladder_emotions

which needs the weights, unlike everything else in `results/analysis/`. It calls
`scripts/score_rujev.py` once per checkpoint, so the dumps come from the same
scorer as every other score file in `results/`.

### Biases on the two unseen datasets

    python3 -I results/analysis/unseen_bias.py results/unseen

Writes the table that `results/unseen/bias.md` holds: accuracy of both releases
on the news rubrics and on CEDR emotions, as published and after biases fitted
out of fold, with the paired difference and bootstrap intervals over rows and
over labels. Splits are seeded 0 to 24, the bootstrap seed is 20261105 with
5,000 draws. Standard library only, no network.

## Calibration splits

`scripts/calib.py` holds the half-split machinery used everywhere, with the seed
recorded as `SPLIT_SEED = 20261105`. An earlier implementation shuffled its index
list in place without resetting it between splits, so each split was applied to
the permutation left by the previous one and the result depended on the order of
rows in the file. `aggregate.py` and `unseen_table.py` now call `calib.py` rather
than carrying their own copy.

## Seed variation

    results/seeds/*_bench.jsonl

Five extra training runs, 6000 bench rows each, raw logits. Two are valid second
and third seeds of the v1.1 recipe and give the seed spread quoted in the
Limitations section; three were trained on a stale mixture and are not seeds of
v1.3. `results/seeds/README.md` has the per-vocabulary table and says which is
which. Their validation scores are not included, 35 MB of a mixture no release
was built from; the mistake that made the experiment void is written up in
`results/seeds/FINDINGS.md`.

## A second seed of the v1.3 recipe

The rule this run is read by was committed before the training started, in
`results/v13_seed2/README.md`, and the run itself is recorded in the same file
after it. The weights are not published; the scores are.

The mixture was rebuilt with the builder the released run was trained with,
commit `96fb67f`, because a later commit changed the split. The hashes of the
rebuilt `train.jsonl` and `val.jsonl`, and what a check of the build found, are
in that README. No copy of the mixture the release was actually trained on
survives, so a byte comparison against it is not possible.

    python3 scripts/train.py --config configs/v1_3_seed2.yaml

The config is `configs/v1_3_seed2.yaml`, which differs from `configs/v1_3.yaml`
in the seed and the output directory and in nothing else. Four epochs, 9,000
steps each, batch 16, about five hours on one rented card. `scripts/train.py`
writes every epoch into the same output directory, so an external snapshotter
copied each epoch out as it was saved; keeping four epochs needs that, or four
directories.

Training fits a temperature once, for the epoch it stops at. The other three
were fitted afterwards with the same instrument:

    python3 results/analysis/fit_seed2_temperatures.py \
        --config configs/v1_3_seed2.yaml --rows data/val.jsonl \
        <snapshot of each epoch>

On the card this took a few minutes per epoch; it runs on a CPU as well, slowly.
The fourth epoch is fitted again by the same command as a control and has to
return the figure the training already wrote into the release. It did: 3.7980,
with the validation loss and accuracy of all four epochs identical to the
training log.

Then the nine checkpoints are scored over every dataset the rule reads:

    python3 results/analysis/score_seed2.py --models <weights> \
        --rows results/unseen/unseen_rows.jsonl --set "рубрики новостей" \
        --out results/v13_seed2/news
    python3 results/analysis/score_seed2.py --models <weights> \
        --rows results/unseen/unseen_rows.jsonl --set "эмоции" \
        --out results/v13_seed2/emotions
    python3 results/analysis/score_seed2.py --models <weights> \
        --rows results/unseen3/matrix/rows.jsonl --out results/unseen3/matrix \
        --only v1.3-seed2-ep1of4,v1.3-seed2-ep2of4,v1.3-seed2-ep3of4,v1.3-seed2-ep4of4,v1.3-seed2-final
    python3 results/analysis/score_seed2.py --models <weights> \
        --rows results/vocab3/rows.jsonl --out results/v13_seed2/vocab3
    python3 results/analysis/score_seed2.py --models <weights> \
        --rows results/fix18/rows.jsonl --out results/v13_seed2/fix18

The wording bench and the style probe are scored once with all eighteen
candidates, and the five-candidate views are taken from those logits by choosing
columns, because each pair of a context and a candidate is scored on its own.
Nothing is scored twice. On one CPU the five commands together take about eight
hours; they are resumable, and a dump is renamed into place only when it holds
every row.

The scoring runs on the CPU on purpose. A card would do it in an hour, but then
the control against the published dumps would be comparing two machines instead
of one path against another, and it would stop meaning anything. On this CPU the
four already published checkpoints come back with a largest logit difference of
exactly zero.

    python3 -I results/analysis/seed2.py

Prints every table in `results/v13_seed2/README.md` after the rule: the control
on the instrument, the accuracy of each checkpoint out of fold, the four
quantities with intervals over rows and over labels, and the verdict the rule
gives per dataset.

## The unlabelled RuCoLA test rows

Recomputes every published figure that rests on the released validation split,
with the 2,789 unlabelled RuCoLA test rows and without them. Reads only files in
this repository, needs no weights and no network:

    python3 -I results/analysis/rucola_test.py

Tables in `results/rucola_test/README.md`.

To rebuild the index of affected rows instead of trusting the committed one, you
need the dataset and the mixture, neither of which is in this repository. With
both in hand the index is every position `i` of
`results/validation/val_v13_trainsplit.jsonl` whose row is `rucola` and whose
sentence, stripped and hashed with sha256, is one of the 2,789 in
`results/rucola_test/test_sentence_sha256.txt` and is not also in RuCoLA's
labelled splits. The result must equal `val_v13_affected_rows.txt`.

The builder no longer admits those rows. `rows_simple` drops a negative raw
label before `answer_from` maps it, so a RuCoLA build is now 9,591 training and
1,065 validation rows rather than 12,101 and 1,344, and the composition check
expects the smaller figure. The other three callers of `rows_simple`,
`rureviews`, `toxicity` and `georeview`, hold no negative label in any split and
lose nothing. No checkpoint was retrained, so every published number stays as
published and the recomputation above is what the numbers would have been.

## Split stability of a rebuilt mixture

    python scripts/split_overlap.py --earlier <earlier build> --rebuilt data \
        --json results/validation/split_overlap.json

Prints, per task, how much of a rebuilt validation split was in the earlier
training split. `results/validation/split_overlap.md` holds the run behind the
table in the paper, with the hashes of both builds and the check that the earlier
build is the split the release was evaluated on.

## What is not here

The one-off scripts that drove the rented GPU and the releases are not here
either. Fourteen of them read secrets by name and carry machine access details,
and all of them are single-run work.

## Licence and citation

Code and weights are MIT. The training data keep the licences of their sources,
listed per source in `DATASETS.md`; none of the ten datasets ships a LICENSE
file, so every licence there is the `license` field of the repository's README
metadata. How to cite the paper and the archived release is in `README.md`.
