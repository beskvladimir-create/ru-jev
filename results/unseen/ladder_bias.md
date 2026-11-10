# Every checkpoint of the ladder, once it is allowed its own biases

Reproduce with:

    python3 -I results/analysis/ladder_bias.py results/unseen

`bias.md` fits one additive bias per option on the two released checkpoints.
This file does the same for all eleven checkpoints of the ladder, on both unseen
datasets, and then asks which checkpoint reads them best once each one is
allowed its own correction. That is a different question from accuracy as
published, and the two do not give the same answer.

Method, taken from `unseen_bias.py` and identical to it: 25 half-splits seeded 0
to 24, each used twice, so every row is answered 25 times by a fit that never
saw it; bootstrap intervals from seed 20261105 with 5,000 draws, over rows and
over labels. Nothing is fitted and scored on the same rows. The six pairs are
fixed in the script before any of this was run.

The per-checkpoint logits are in `ladder/` and `ladder_emotions/`, produced by
`results/analysis/score_ladder.py`. The intermediate weights are not published.

Per-option biases on every checkpoint of the ladder
25 half-splits, seeds 0 to 24, both halves scored; bootstrap 5000 draws, seed 20261105

## рубрики новостей, 1000 rows, 6 options

| Checkpoint | as published | with biases and T | change |
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
| v1.3 | 0.568 | 0.640 | +0.072 |
| v1.3-ep4of4 | 0.551 | 0.625 | +0.074 |

Best with biases: `v1.3-ep2of4` at 0.693. Best as published: `v1.1` at 0.652.

Paired differences after the biases, out of fold:

| Pair | difference | 95% over rows | 95% over labels | What it asks |
|---|---|---|---|---|
| seed-base-s2 minus v1.1 | -0.037 | [-0.062, -0.011] | [-0.093, +0.018] | a second seed of the v1.1 recipe against the release |
| v1.3 minus v1.1 | -0.034 | [-0.060, -0.006] | [-0.067, +0.001] | the two released checkpoints |
| v1.3 minus seed-base-s2 | +0.004 | [-0.024, +0.030] | [-0.021, +0.030] | the later release against that seed rather than against v1.1 |
| v1.3-ep2of4 minus v1.3-ep3of4 | +0.053 | [+0.032, +0.075] | [+0.030, +0.079] | the epoch before the released one against it |
| v1.3-ep2of4 minus v1.1 | +0.020 | [-0.007, +0.047] | [-0.013, +0.050] | that earlier epoch against the earlier release |
| seed-aug-s2-ep3 minus seed-aug-s2 | +0.030 | [+0.012, +0.047] | [+0.016, +0.043] | early against late epoch, the same question in another run |

## эмоции, 1000 rows, 5 options

| Checkpoint | as published | with biases and T | change |
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
| v1.3 | 0.511 | 0.571 | +0.060 |
| v1.3-ep4of4 | 0.490 | 0.565 | +0.075 |

Best with biases: `v1.2` at 0.591. Best as published: `seed-base-s2` at 0.545.

Paired differences after the biases, out of fold:

| Pair | difference | 95% over rows | 95% over labels | What it asks |
|---|---|---|---|---|
| seed-base-s2 minus v1.1 | +0.020 | [-0.008, +0.048] | [-0.012, +0.038] | a second seed of the v1.1 recipe against the release |
| v1.3 minus v1.1 | +0.014 | [-0.016, +0.044] | [-0.055, +0.050] | the two released checkpoints |
| v1.3 minus seed-base-s2 | -0.006 | [-0.036, +0.025] | [-0.066, +0.020] | the later release against that seed rather than against v1.1 |
| v1.3-ep2of4 minus v1.3-ep3of4 | -0.008 | [-0.030, +0.014] | [-0.110, +0.040] | the epoch before the released one against it |
| v1.3-ep2of4 minus v1.1 | +0.006 | [-0.024, +0.037] | [-0.158, +0.084] | that earlier epoch against the earlier release |
| seed-aug-s2-ep3 minus seed-aug-s2 | +0.003 | [-0.012, +0.019] | [-0.022, +0.022] | early against late epoch, the same question in another run |

## What this says

**On the news rubrics the epoch before the released one wins, and clearly.**
`v1.3-ep2of4` goes from ninth of eleven as published, 0.554, to first with
biases, 0.693. Against the released third epoch that is +0.053, with both
intervals clear of zero, over rows [+0.032, +0.075] and over labels [+0.030,
+0.079]. It also edges past v1.1, +0.020, though that interval covers zero.

**On the emotions it does not.** `v1.3-ep2of4` is 0.563 against 0.571 for the
released epoch, a difference of -0.008 whose intervals cover zero on both
resamplings. Whatever makes the second epoch the better reader of news rubrics
once its lean is removed does not carry to the other dataset. The answer to the
question that prompted this file is therefore no: the second epoch does not lead
in general, it leads on one dataset.

**The size of the correction is itself a measurement.** How much a checkpoint
gains from the biases is how hard it was leaning. On the news rubrics the gain
runs from +0.021 for v1.1 to +0.139 for `v1.3-ep2of4`: the checkpoint that
needed the most correction is the one that reads best after it. Accuracy as
published cannot separate those two things, and the released epoch was chosen on
accuracy.

**The two rankings disagree on both datasets.** As published the best on news is
v1.1 at 0.652 and the best on emotions is `seed-base-s2` at 0.545. With biases
the best on news is `v1.3-ep2of4` at 0.693 and on emotions `v1.2` at 0.591.
The released v1.1 is first only on the news rubrics as published; with biases,
neither released checkpoint is first on either dataset.

**Seed variation is the same size as the difference between releases.** A second
seed of the v1.1 recipe differs from v1.1 by -0.037 on news and +0.020 on
emotions, against -0.034 and +0.014 for v1.3 against v1.1. Taking v1.3 against
that seed instead of against the release leaves +0.004 and -0.006, both
intervals covering zero on both datasets. One run per recipe cannot separate a
recipe from a seed, which is what `results/seeds/FINDINGS.md` says from the
other side.

## What this does not say

The accuracies are out of fold, but picking the best of eleven checkpoints on
the same 1,000 rows is a selection, and the figure of the winner is optimistic
by an unknown amount. The six pairs in the table are not: they were fixed before
the fit was run, and they are the ones to read. "Best with biases" is a
description of this table, not an estimate of how that checkpoint would do on
new rows.

Neither dataset was trained on by any checkpoint, and neither is large. One
thousand rows, six labels on one and five on the other. The intervals over
labels are wide for that reason, and on the emotions they are wide enough that
most of these differences are not distinguishable from zero.

## A note on reproducibility

The logits here were recomputed on 2026-11-10 and they are not bit-identical to
the published `unseen_v11.jsonl` and `unseen_v13.jsonl`. On the emotions, v1.1
agrees to 7.6e-05 at worst, and v1.3 agrees to 1e-4 on 998 of its 1,000 rows
with two rows at 4.0e-4. Every one of the 2,000 picks is the same and both
accuracies match the published ones to four decimals. The smallest margin
between the top two candidates anywhere in that dump is 0.0038, ten times the
largest disagreement, so no row is near flipping. This is floating-point noise
between library builds, and every statistic above is on differences of order one.
