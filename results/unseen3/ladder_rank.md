# Do the rankings of the ladder agree across unseen datasets?

Reproduce with:

    python3 -I results/analysis/ladder_rank.py

The rule for reading this was written down in `README.md` in this directory and
committed before any row was downloaded and before anything was scored. That
commit is `4d23c2d`, timed 2026-11-11T09:17:23+04:00.

## The question

`results/unseen/ladder_bias.md` ranks the ten distinct checkpoints of the ladder
on the news rubrics and on the CEDR emotions, once each checkpoint is allowed
its own additive bias per option. The two rankings disagree, Spearman -0.17.
With two datasets there is no way to tell whether either of them travels, so a
third dataset was added, far from the training tasks, and the verdict rule was
fixed first: a ranking survives a new dataset if its Spearman with that dataset,
with biases, is at least 0.50.

## The third dataset

Scientific rubrics of the GRNTI classifier,
`ai-forever/ru-scibench-grnti-classification`, revision
`a984f8a65b2e9fae7a347bf7761b0eaad79c73a1`, test split, licence `mit` in the
card metadata. No task in the training mixture is about scientific
classification.

990 rows, ten classes, five candidates per question, chance 0.200. The test
split holds 28 classes with exactly 99 rows each, so "the ten most frequent"
is a complete tie; it was broken by name, ascending, which is deterministic and
was fixed without looking at any score. `results/analysis/build_unseen3.py`
builds the rows and `rows.jsonl` holds them, sha256
`f660d399e6cf5c45a9e5813db50491a35f973b4042443b6e609a996c84c91f5c`.

Checks, all run before scoring:

- **Text overlap with training: none.** All 990 selected texts were compared
  with the 143,991 training rows of both mixtures, by exact match after
  normalising whitespace and case. Zero hits.
- **Label overlap with training: no exact match.** No training option is an
  exact match of a class name and no training option begins with one. Four
  class names begin with a training option: `Биология` with `биолог`,
  `Геология` with `геолог`, `Горное дело` with `гор`, `Математика` with `м`.
  The first two are real words from the reasoning tasks; the last two are
  artefacts of answer options one to three characters long, which are a prefix
  of almost anything. None of the four is excluded, and the figures below are on
  all ten classes.
- **Labels as the model reads them**, the Russian `label_text` field of the
  dataset, unchanged.

The fourth dataset, sentiment of film reviews, is not here. Its MTEB mirror
carries `license: unknown`, which the rule treats as unclear, and its first
source gives the classes only in English, `Bad`, `Neutral`, `Good`, which the
rule also stops on. The decision is with the author.

Do the rankings of the ladder agree across unseen datasets?
25 half-splits, seeds 0 to 24, both halves scored; bootstrap 5000 draws, seed 20261105.


## news rubrics, 1000 rows, 6 candidates

| Checkpoint | as published | with biases | change |
|---|---|---|---|
| v1.1 | 0.652 | 0.673 | +0.021 |
| seed-base-s2 | 0.601 | 0.636 | +0.035 |
| v1.2 | 0.577 | 0.664 | +0.087 |
| seed-aug-s2 | 0.600 | 0.651 | +0.051 |
| seed-aug-s2-ep3 | 0.601 | 0.681 | +0.080 |
| seed-aug-s3 | 0.556 | 0.629 | +0.073 |
| v1.3-ep1of4 | 0.526 | 0.645 | +0.119 |
| v1.3-ep2of4 | 0.554 | 0.693 | +0.139 |
| v1.3-ep3of4 | 0.568 | 0.640 | +0.072 |
| v1.3-ep4of4 | 0.551 | 0.625 | +0.074 |

Best as published `v1.1` at 0.652, best with biases `v1.3-ep2of4` at 0.693. Control, v1.3 against v1.3-ep3of4: the same answer on every row.

Within this dataset: the gain from the biases against where a checkpoint ends up, Spearman +0.31; the ranking as published against the ranking with biases, +0.35.

## emotions, 1000 rows, 5 candidates

| Checkpoint | as published | with biases | change |
|---|---|---|---|
| v1.1 | 0.455 | 0.556 | +0.101 |
| seed-base-s2 | 0.545 | 0.577 | +0.032 |
| v1.2 | 0.536 | 0.591 | +0.055 |
| seed-aug-s2 | 0.418 | 0.521 | +0.103 |
| seed-aug-s2-ep3 | 0.436 | 0.525 | +0.089 |
| seed-aug-s3 | 0.452 | 0.525 | +0.073 |
| v1.3-ep1of4 | 0.509 | 0.587 | +0.078 |
| v1.3-ep2of4 | 0.471 | 0.563 | +0.092 |
| v1.3-ep3of4 | 0.511 | 0.571 | +0.060 |
| v1.3-ep4of4 | 0.490 | 0.565 | +0.075 |

Best as published `seed-base-s2` at 0.545, best with biases `v1.2` at 0.591. Control, v1.3 against v1.3-ep3of4: the same answer on every row.

Within this dataset: the gain from the biases against where a checkpoint ends up, Spearman -0.68; the ranking as published against the ranking with biases, +0.94.

## scientific rubrics, 990 rows, 5 candidates

| Checkpoint | as published | with biases | change |
|---|---|---|---|
| v1.1 | 0.472 | 0.566 | +0.094 |
| seed-base-s2 | 0.536 | 0.612 | +0.076 |
| v1.2 | 0.539 | 0.555 | +0.016 |
| seed-aug-s2 | 0.544 | 0.600 | +0.056 |
| seed-aug-s2-ep3 | 0.546 | 0.624 | +0.077 |
| seed-aug-s3 | 0.622 | 0.684 | +0.062 |
| v1.3-ep1of4 | 0.507 | 0.580 | +0.073 |
| v1.3-ep2of4 | 0.603 | 0.662 | +0.059 |
| v1.3-ep3of4 | 0.601 | 0.657 | +0.056 |
| v1.3-ep4of4 | 0.572 | 0.617 | +0.045 |

Best as published `seed-aug-s3` at 0.622, best with biases `seed-aug-s3` at 0.684. Control, v1.3 against v1.3-ep3of4: the same answer on every row.

Within this dataset: the gain from the biases against where a checkpoint ends up, Spearman +0.04; the ranking as published against the ranking with biases, +0.89.

## Spearman between datasets, over the ten checkpoints


with biases:

| | news rubrics | emotions | scientific rubrics |
|---|---|---|---|
| news rubrics | 1.00 | -0.17 | -0.18 |
| emotions | -0.17 | 1.00 | -0.41 |
| scientific rubrics | -0.18 | -0.41 | 1.00 |

as published:

| | news rubrics | emotions | scientific rubrics |
|---|---|---|---|
| news rubrics | 1.00 | -0.16 | -0.41 |
| emotions | -0.16 | 1.00 | -0.25 |
| scientific rubrics | -0.41 | -0.25 | 1.00 |

## The six fixed pairs, after the biases, out of fold


### news rubrics

| Pair | difference | 95% over rows | 95% over labels | What it asks |
|---|---|---|---|---|
| seed-base-s2 minus v1.1 | -0.037 | [-0.062, -0.011] | [-0.093, +0.018] | a second seed of the v1.1 recipe against the release |
| v1.3-ep3of4 minus v1.1 | -0.034 | [-0.060, -0.006] | [-0.067, +0.001] | the two released checkpoints |
| v1.3-ep3of4 minus seed-base-s2 | +0.004 | [-0.024, +0.030] | [-0.021, +0.030] | the later release against that seed |
| v1.3-ep2of4 minus v1.3-ep3of4 | +0.053 | [+0.032, +0.075] | [+0.030, +0.079] | the epoch before the released one against it |
| v1.3-ep2of4 minus v1.1 | +0.020 | [-0.007, +0.047] | [-0.013, +0.050] | that earlier epoch against the earlier release |
| seed-aug-s2-ep3 minus seed-aug-s2 | +0.030 | [+0.012, +0.047] | [+0.016, +0.043] | early against late epoch in another run |

### emotions

| Pair | difference | 95% over rows | 95% over labels | What it asks |
|---|---|---|---|---|
| seed-base-s2 minus v1.1 | +0.020 | [-0.008, +0.048] | [-0.012, +0.038] | a second seed of the v1.1 recipe against the release |
| v1.3-ep3of4 minus v1.1 | +0.014 | [-0.016, +0.044] | [-0.055, +0.050] | the two released checkpoints |
| v1.3-ep3of4 minus seed-base-s2 | -0.006 | [-0.036, +0.025] | [-0.066, +0.020] | the later release against that seed |
| v1.3-ep2of4 minus v1.3-ep3of4 | -0.008 | [-0.030, +0.014] | [-0.110, +0.040] | the epoch before the released one against it |
| v1.3-ep2of4 minus v1.1 | +0.006 | [-0.024, +0.037] | [-0.158, +0.084] | that earlier epoch against the earlier release |
| seed-aug-s2-ep3 minus seed-aug-s2 | +0.003 | [-0.012, +0.019] | [-0.022, +0.022] | early against late epoch in another run |

### scientific rubrics

| Pair | difference | 95% over rows | 95% over labels | What it asks |
|---|---|---|---|---|
| seed-base-s2 minus v1.1 | +0.046 | [+0.013, +0.080] | [-0.034, +0.133] | a second seed of the v1.1 recipe against the release |
| v1.3-ep3of4 minus v1.1 | +0.091 | [+0.059, +0.125] | [+0.010, +0.184] | the two released checkpoints |
| v1.3-ep3of4 minus seed-base-s2 | +0.045 | [+0.016, +0.074] | [+0.010, +0.077] | the later release against that seed |
| v1.3-ep2of4 minus v1.3-ep3of4 | +0.004 | [-0.016, +0.026] | [-0.052, +0.046] | the epoch before the released one against it |
| v1.3-ep2of4 minus v1.1 | +0.096 | [+0.063, +0.129] | [+0.027, +0.176] | that earlier epoch against the earlier release |
| seed-aug-s2-ep3 minus seed-aug-s2 | +0.023 | [+0.004, +0.042] | [-0.005, +0.052] | early against late epoch in another run |

## The v1.1 recipe across two runs

The third run of this recipe has no surviving weights, so this is two runs, not three.

| Dataset | v1.1 | seed-base-s2 | spread | v1.3 | v1.3 minus v1.1 | within the spread |
|---|---|---|---|---|---|---|
| **as published** | | | | | | |
| news rubrics | 0.652 | 0.601 | 0.051 | 0.568 | -0.084 | no |
| emotions | 0.455 | 0.545 | 0.090 | 0.511 | +0.056 | yes |
| scientific rubrics | 0.472 | 0.536 | 0.065 | 0.601 | +0.129 | no |
| **with biases** | | | | | | |
| news rubrics | 0.673 | 0.636 | 0.037 | 0.640 | -0.034 | yes |
| emotions | 0.556 | 0.577 | 0.020 | 0.571 | +0.014 | yes |
| scientific rubrics | 0.566 | 0.612 | 0.046 | 0.657 | +0.091 | no |

## The reading rule of results/unseen3/README.md

- The **news rubrics** ranking does not survive **scientific rubrics**: Spearman -0.18, the rule asks for at least 0.50.
- The **emotions** ranking does not survive **scientific rubrics**: Spearman -0.41, the rule asks for at least 0.50.
- **film reviews**: not available, no verdict.

At ten checkpoints this is a description and not a statistical test.

## What this says

**Neither ranking survives.** The news-rubric ranking correlates with the
scientific rubrics at -0.18 and the emotion ranking at -0.41, against the 0.50
the rule asks for. All three pairwise correlations between the three datasets
are negative, with biases and without. A checkpoint that is good on one of these
datasets is, if anything, slightly worse on the others.

**The three datasets do not even agree on who wins.** As published the best is
v1.1 on the news rubrics, `seed-base-s2` on the emotions and `seed-aug-s3` on
the scientific rubrics. With biases it is `v1.3-ep2of4`, `v1.2` and
`seed-aug-s3`. No checkpoint is first anywhere twice, and the released v1.3 is
first nowhere.

**The correction does not create the disagreement, and it does not cure it
either.** The ranking as published and the ranking with biases agree within a
dataset, +0.35 on the news rubrics, +0.94 on the emotions and +0.89 on the
scientific rubrics. Across datasets both rankings disagree about equally,
-0.17 and -0.16 for the first pair. What the biases change is the level, not
the order.

**The one thing that does repeat is that the released epoch is not the best
epoch of its own run.** On the scientific rubrics `v1.3-ep2of4` and
`v1.3-ep3of4` are level, +0.004 with the interval covering zero, where on the
news rubrics the earlier epoch was ahead by +0.053. So the earlier epoch is
never worse on any of the three.

**On the scientific rubrics the two releases do separate, and in v1.3's
favour.** +0.091 with biases, interval over rows [+0.059, +0.125] and over
labels [+0.010, +0.184], both clear of zero, and the same holds against
`seed-base-s2`, +0.045. This is the only one of the three datasets where the
difference between the releases is larger than the spread between two runs of
the v1.1 recipe.

## What this does not say

Ten checkpoints is ten points. A rank correlation over ten points is a
description, and none of the numbers above carries a p-value.

The spread between runs of the v1.1 recipe is over two runs here, the released
v1.1 and `seed-base-s2`, because the third run has no surviving weights. Two
runs give a range, not a distribution.

The verdict concerns the rankings, not the checkpoints. That a ranking does not
travel does not mean the accuracies are wrong; it means that choosing a
checkpoint by one of these datasets tells you nothing about the others.
