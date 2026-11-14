# Unlabelled RuCoLA test rows in the training mixture

RuCoLA ships three splits and the test split carries no label. The csv behind it
holds `id, sentence` and nothing else, and the loading script fills the label
with -1 so that the column exists.

`rows_simple` in `scripts/build_data.py` pooled every split of a source, and the
only guard it had read the answer index after `answer_from` had mapped the raw
label:

    ans = answer_from(lab)
    if ans < 0 or ans >= len(options):
        continue

For RuCoLA the mapping is `0 if int(lab) == 1 else 1`. It turned -1 into 1, a
perfectly valid index, so the guard caught nothing and 2,789 sentences entered
the mixture carrying the option `неприемлемо` as their right answer. They do not
have that label. They do not have any label.

The sizes agree with that reading exactly. The build holds 12,101 rucola rows in
training and 1,344 in validation, 13,445 together, which is RuCoLA's train 7,869
plus dev 2,787 plus test 2,789.

## Where the rows went

Counted by sha256 of the stripped sentence, against the split each build keeps
on disk. One sentence occurs in both the test split and the labelled splits of
the source, so it is excluded from every count below: it has a real label and is
legitimate training data. It occurs twice in the training half of every build,
which is what makes the exclusion worth stating: counted against the wider list
of 2,789 hashes the three training figures below read 2,509, 2,538 and 2,495,
and a build's two splits then add up to 2,790 rather than 2,788. The counts
below are against `test_only_sha256.txt`, and every build's two figures sum to
2,788.

| Build | Split | rucola rows | Of them from the test split |
|---|---|---|---|
| v1.1 | training | 12,101 | 2,507 |
| v1.1 | validation | 1,344 | 281 |
| v1.3 | training | 12,101 | 2,536 |
| v1.3 | validation | 1,344 | 252 |
| rebuilt with the current builder | training | 12,101 | 2,493 |
| rebuilt with the current builder | validation | 1,344 | 295 |

Every affected row in every build carries `неприемлемо` as its right answer, in
whichever order the two options were shuffled into. The v1.3 rows are counted on
`model/data-stale-prejargon`, which `split_overlap.md` establishes as the carve
the release was measured on, row for row against `val_v13_trainsplit.jsonl`,
26,746 of 26,746.

In the v1.3 training mixture the affected rows are 2,536 of 143,991, which is
1.76 per cent. In its validation split they are 252 of 26,746, 0.94 per cent.

## What it did to the published figures

`results/analysis/rucola_test.py` recomputes each of them from the published
per-row scores, with the affected rows and without.

### How many rows

| Where | Rows | Of them from the RuCoLA test split |
|---|---|---|
| the validation split, all tasks | 26746 | 252 |
| the validation split, rucola only | 1344 | 252 |

Every one of them carries the option "неприемлемо" as its right answer, which is the label the loader invented.

### Accuracy

| Set | Rows | As published | Without the test rows | Change |
|---|---|---|---|---|
| rucola, validation | 1344 | 0.6473 | 0.6886 | +0.0413 |
| every task, validation | 26746 | 0.8208 | 0.8242 | +0.0034 |

On the affected rows alone the released checkpoint agrees with the invented label on 0.4683 of 252, which is about what a coin gives on a two-way decision.

### Temperature and ECE

Fitted on one half of the rows with seed 17 on the 301-point grid, read on the other half, which is the path the released 1.8640 came from.

| Set | Rows | Fitted T | ECE on the other half |
|---|---|---|---|
| as published | 26746 | 1.8640 | 0.0183 |
| without the test rows | 26494 | 1.8380 | 0.0171 |

The ECE difference is smaller than the gap between two honest ways of computing ECE on the same rows, so it carries nothing. The temperature moves by 0.0260.

### Coverage

Built on the half the temperature was not fitted on, as `scripts/coverage.py` builds it.

| Share answered | Threshold, as published | Accuracy, as published | Threshold, without | Accuracy, without |
|---|---|---|---|---|
| 30% | 0.98 | 0.9928 | 0.98 | 0.9937 |
| 40% | 0.93 | 0.9847 | 0.93 | 0.9858 |
| 50% | 0.85 | 0.9666 | 0.85 | 0.9674 |
| 60% | 0.77 | 0.9470 | 0.77 | 0.9478 |
| 70% | 0.69 | 0.9214 | 0.69 | 0.9233 |
| 80% | 0.61 | 0.8918 | 0.61 | 0.8956 |
| 90% | 0.54 | 0.8592 | 0.54 | 0.8621 |
| 100% | 0.23 | 0.8234 | 0.25 | 0.8244 |

### Which epoch the run would have shipped

The released checkpoint is the third of four. The trainer's own evaluation is over all 26746 rows, so the count below subtracts, from each epoch, the rows it answered as the mixture labelled them.

| Epoch | Accuracy in the log | Right on the 252 affected | Accuracy without them |
|---|---|---|---|
| epoch 1 of 4 | 0.7536 | 145 | 0.7553 |
| epoch 2 of 4 | 0.7920 | 131 | 0.7946 |
| epoch 3 of 4, released | 0.8181 | 118 | 0.8214 |
| epoch 4 of 4 | 0.8139 | 126 | 0.8169 |

The epoch that wins without the affected rows is **epoch 3 of 4, released**, which is the epoch that was shipped. The choice does not move.

The two counts come from different code paths, the trainer's batched evaluation and the standalone scorer, which disagree by 72 decisions over the whole split and so by about one over 252. The gap between the third epoch and the fourth is 112 decisions, so that mismatch cannot turn the comparison over.


## The consequence that matters most

The invented rows are not spread evenly over the two options. Every one of them
says `неприемлемо`, so removing them changes what a model has to beat on this
task. On the validation split the release was measured on:

| | Rows | Share of the majority class | v1.3 |
|---|---|---|---|
| rucola as published | 1,344 | 0.5990, `приемлемо` 805 against `неприемлемо` 539 | 0.6473 |
| rucola without the test rows | 1,092 | 0.7372, `приемлемо` 805 against `неприемлемо` 287 | 0.6886 |

Read as published, the release sits five points above a constant answer. Read on
the rows that have a real label, it sits five points below one. The invented
rows made the task look harder than it is and the model look better on it than
it is, and both effects point the same way.

This is computed from the mixture on disk, not from a file in this repository,
so it is stated here rather than printed by the script. It also means the
baseline table in the repository README, where rucola reads 0.587 for the
majority class and 0.609 for TF-IDF against 0.650 for v1.1, rests on a
contaminated split and needs recomputing before it is quoted again. That rerun
needs `scripts/baselines.py` over a clean build and is not done here.

## What is not affected

The comparison of the two releases on the six Russian SuperGLUE tasks does not
touch RuCoLA. Those six are `danetqa`, `parus`, `rcb`, `russe`, `rwsd` and
`terra`, 10,157 validation rows, on which the released v1.3 answers 8,120
correctly, reproduced here from the same dump. RuCoLA is not among them, so the
figure stands as published.

## What cannot be recomputed here

The released v1.1 has no per-row dump on this validation split, only the released
v1.3 does. The three figures of the first hazard in the paper are therefore not
all recoverable from what is in this repository: 0.647 is v1.3 on its own
validation split and is recomputed above as 0.6473, but 0.909 and 0.650 are v1.1
figures and would need that checkpoint scored again.

## Why the sentences are not here

`DATASETS.md` records the position for RuCoLA: Apache-2.0 covers its labels and
its code, the sentences are drawn from linguistic publications and carry their
own authorship, and the corpus is not redistributed from this repository.
`test_sentence_sha256.txt` holds the sha256 of each stripped test sentence
instead, 2,789 of them, sorted. A reader with the dataset can rebuild the index
of affected rows from it and compare against `val_v13_affected_rows.txt`, which
is committed here and is what the analysis reads.

## One thing found on the way, not part of the fix

The same pooling of every split feeds the test splits of three further sources
into training: `rureviews` 2,048 rows, `georeview` 2,048 and `toxicity` 10,000.
Those rows carry real labels, so nothing is invented and they are ordinary
training data. They are worth knowing about only because a later evaluation on
any of those benchmarks' own test splits would be scoring on rows the model
trained on. Checked for the same defect as RuCoLA: none of the three holds a
negative label in any split, so none of them loses a row to the fix.

## Files

| File | What it is |
|---|---|
| `README.md` | this file |
| `test_sentence_sha256.txt` | sha256 of each of the 2,789 stripped RuCoLA test sentences, sorted |
| `val_v13_affected_rows.txt` | the 252 positions, zero based, of the affected rows in `results/validation/val_v13_trainsplit.jsonl` |
| `epoch_hits_on_affected.json` | how many of those 252 rows each epoch of the v1.3 run answers as the mixture labelled them |
