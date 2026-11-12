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

## The full score matrix

Everything from here on was decided after the rule in `README.md` was written
and after its verdict was known. The verdict above stands on the alphabetical
ten and nothing below changes it.

The model scores each (text, label) pair on its own, so the ten checkpoints
were run once more over the whole test split against all of its labels: 2,772
texts by 28 rubrics by 10 checkpoints, 776,160 pairs. The logits are in
`matrix/`, one file per checkpoint, with `matrix/rows.jsonl` holding the texts
and the 28 labels. From that matrix any set of candidate labels can be
assembled without touching the weights again.

Reproduce the matrix with:

    python3 -I results/analysis/score_ladder.py \
        --set "научные рубрики, все 28" --rows results/unseen3/matrix/rows.jsonl \
        --models ../model --out results/unseen3/matrix

and everything below it with:

    python3 -I results/analysis/matrix_sensitivity.py

### The matrix against the direct scoring

The published ten was scored directly, from its own rows file. Assembling the
same ten out of the matrix has to give the same thing, and it does.

| Check | Result |
|---|---|
| the rows are the same, text, options and answer | yes |
| largest disagreement between a matrix logit and a directly scored one | 0.000519 |
| rows where the answer is the same | 9,900 of 9,900 |

| Checkpoint | from the matrix | published above |
|---|---|---|
| v1.1 | 0.472 as published, 0.566 with biases | 0.472 as published, 0.566 with biases |
| seed-base-s2 | 0.536 as published, 0.612 with biases | 0.536 as published, 0.612 with biases |
| v1.2 | 0.539 as published, 0.555 with biases | 0.539 as published, 0.555 with biases |
| seed-aug-s2 | 0.544 as published, 0.600 with biases | 0.544 as published, 0.600 with biases |
| seed-aug-s2-ep3 | 0.546 as published, 0.624 with biases | 0.546 as published, 0.624 with biases |
| seed-aug-s3 | 0.622 as published, 0.684 with biases | 0.622 as published, 0.684 with biases |
| v1.3-ep1of4 | 0.507 as published, 0.580 with biases | 0.507 as published, 0.580 with biases |
| v1.3-ep2of4 | 0.603 as published, 0.662 with biases | 0.603 as published, 0.662 with biases |
| v1.3-ep3of4 | 0.601 as published, 0.657 with biases | 0.601 as published, 0.657 with biases |
| v1.3-ep4of4 | 0.572 as published, 0.617 with biases | 0.572 as published, 0.617 with biases |

The remaining difference of 5.2e-4 in a logit is the arithmetic of two separate
runs on the same weights, and it moves no answer on any of the 9,900 rows.

## Sensitivity check, added after the reading rule

The ten rubrics of the published set were chosen by an alphabetical tie-break,
because all 28 classes hold exactly 99 test rows. The tie-break was fixed
before any score was seen, but it is still one draw out of many, so the matrix
is used to ask what another draw would have given.

Twenty random tens of the 28 rubrics, seeds 0 to 19. Everything else is as the
rule has it: every row of the chosen ten, five candidates per question drawn
with seed 20261105, biases fitted out of fold over 25 half-splits seeded 0 to
24, each used twice. Every ten is 990 rows, because every class holds 99.

| Seed | n | Spearman with news | Spearman with emotions | v1.3 minus v1.1 | v1.3 minus seed-base-s2 |
|---|---|---|---|---|---|
| 0 | 990 | -0.24 | -0.27 | +0.139 | +0.090 |
| 1 | 990 | -0.03 | -0.34 | +0.149 | +0.048 |
| 2 | 990 | -0.26 | -0.29 | +0.093 | +0.041 |
| 3 | 990 | -0.07 | -0.16 | +0.123 | +0.085 |
| 4 | 990 | -0.18 | -0.16 | +0.118 | +0.064 |
| 5 | 990 | -0.30 | +0.03 | +0.183 | +0.048 |
| 6 | 990 | -0.04 | -0.26 | +0.071 | +0.058 |
| 7 | 990 | -0.22 | +0.04 | +0.211 | +0.079 |
| 8 | 990 | -0.19 | +0.09 | +0.112 | +0.058 |
| 9 | 990 | -0.16 | -0.21 | +0.195 | +0.053 |
| 10 | 990 | -0.14 | -0.20 | +0.153 | +0.076 |
| 11 | 990 | -0.15 | -0.10 | +0.135 | +0.058 |
| 12 | 990 | -0.20 | -0.02 | +0.236 | +0.101 |
| 13 | 990 | -0.19 | -0.27 | +0.134 | +0.054 |
| 14 | 990 | -0.30 | -0.30 | +0.135 | +0.071 |
| 15 | 990 | -0.14 | -0.18 | +0.091 | +0.074 |
| 16 | 990 | -0.14 | -0.20 | +0.120 | +0.086 |
| 17 | 990 | -0.22 | -0.40 | +0.119 | +0.066 |
| 18 | 990 | -0.37 | -0.18 | +0.119 | +0.037 |
| 19 | 990 | -0.30 | -0.35 | +0.185 | +0.075 |

| Quantity | median | lowest | highest | tens at or above 0.50 |
|---|---|---|---|---|
| Spearman with news rubrics | -0.188 | -0.370 | -0.030 | 0 of 20 |
| Spearman with emotions | -0.201 | -0.401 | +0.085 | 0 of 20 |
| v1.3 minus v1.1 | +0.134 | +0.071 | +0.236 | |
| v1.3 minus seed-base-s2 | +0.065 | +0.037 | +0.101 | |

**The choice of the ten does not carry the verdict.** Not one of the twenty
draws reaches the 0.50 the rule asks for, against either of the earlier
datasets, and not one of the forty correlations is even positive beyond +0.09.
The published ten gave -0.18 against the news rubrics and -0.41 against the
emotions; both sit inside the range the twenty draws cover.

**The one difference that does hold on this dataset holds on every draw.**
v1.3 is ahead of v1.1 with biases on all twenty tens, between +0.071 and
+0.236, and ahead of `seed-base-s2` on all twenty, between +0.037 and +0.101.
The lowest of those twenty margins against v1.1 is still larger than the
spread between the two runs of the v1.1 recipe on this dataset, 0.046.

## All 28 rubrics offered at once, added after the reading rule

The same matrix answers a harder question than the rule asked: every one of the
2,772 test texts with all 28 rubrics as candidates, where chance is 0.036.

| Checkpoint | as published | with biases | change |
|---|---|---|---|
| v1.1 | 0.206 | 0.288 | +0.082 |
| seed-base-s2 | 0.255 | 0.347 | +0.092 |
| v1.2 | 0.270 | 0.311 | +0.042 |
| seed-aug-s2 | 0.275 | 0.339 | +0.064 |
| seed-aug-s2-ep3 | 0.295 | 0.376 | +0.081 |
| seed-aug-s3 | 0.322 | 0.399 | +0.077 |
| v1.3-ep1of4 | 0.240 | 0.344 | +0.104 |
| v1.3-ep2of4 | 0.342 | 0.404 | +0.062 |
| v1.3-ep3of4 | 0.337 | 0.403 | +0.067 |
| v1.3-ep4of4 | 0.316 | 0.372 | +0.055 |

Every checkpoint is far above chance with 28 candidates on a task no part of
the training mixture covers, and the order is close to the order on the ten:
`seed-aug-s3` and the middle epochs of v1.3 at the top, v1.1 last. The biases
help every checkpoint here, by between +0.042 and +0.104, which is what the
28-way case should do, since there are 27 of them to fit rather than 4.

Nothing in these two sections was part of the rule, and none of it is offered
as a test. The verdict of this file remains the one stated above, on the ten
rubrics the rule selected.
