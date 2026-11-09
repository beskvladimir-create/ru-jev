# Where the economy and politics boundary moves

Reproduce with:

    python3 -I results/analysis/unseen_ladder.py results/unseen

A reader of the model card pointed out that after per-option biases are fitted,
v1.3 still draws a worse boundary between `экономика` and `политика` than v1.1
does. `bias.md` confirms it from the other side: those two rubrics carry -0.028
of the -0.034 that is left of the gap. The question here is at which step of the
training line the boundary moves, so that the answer names a step rather than a
release.

What is read. For every checkpoint, the difference
`logit(экономика) - logit(политика)` on each of the 1,000 headlines. It is the
boundary itself, taken before any softmax, so no temperature and no bias can
touch it. Over the 340 headlines whose true label is one of those two rubrics,
its AUC is the probability that a random economic headline is scored above a
random political one.

The checkpoints are not published, only their scores on these rows, which are in
`ladder/`. The two released ones are on Hugging Face. `v1.3-ep3of4` and `v1.3`
are the same weights read from two directories, `ru-jev-v1.3/ep2` and
`ru-jev-v1.3/ship`, and they agree on every figure, which is the control on the
scoring path.

A note on reproducibility. The logits here were recomputed on 2026-11-09 and
they are not bit-identical to the published `unseen_v11.jsonl` and
`unseen_v13.jsonl`: the largest disagreement is 1.1e-4 on a logit of order 8,
with the same option picked on all 1,000 rows and the same accuracy, 0.6520 and
0.5680. That is floating-point noise between library builds. Every statistic
below is on differences of order one, so it is unaffected.

Where the economy and politics boundary moves
1000 headlines, 340 of them экономика or политика; bootstrap 2000 draws, seed 20261105

| Checkpoint | Accuracy | AUC экономика vs политика | AUC minus v1.1 | 95% over headlines |
|---|---|---|---|---|
| v1.1 | 0.652 | 0.928 |  |  |
| seed-base-s2 | 0.601 | 0.909 | -0.019 | [-0.044, +0.005] |
| v1.2 | 0.577 | 0.929 | +0.000 | [-0.024, +0.023] |
| seed-aug-s2 | 0.600 | 0.911 | -0.018 | [-0.041, +0.005] |
| seed-aug-s2-ep3 | 0.601 | 0.921 | -0.007 | [-0.028, +0.013] |
| seed-aug-s3 | 0.556 | 0.834 | -0.094 | [-0.133, -0.059] |
| v1.3-ep1of4 | 0.526 | 0.921 | -0.007 | [-0.025, +0.011] |
| v1.3-ep2of4 | 0.554 | 0.938 | +0.010 | [-0.008, +0.029] |
| v1.3-ep3of4 | 0.568 | 0.895 | -0.033 | [-0.061, -0.008] |
| v1.3 | 0.568 | 0.895 | -0.033 | [-0.061, -0.008] |
| v1.3-ep4of4 | 0.551 | 0.875 | -0.053 | [-0.088, -0.023] |

| Checkpoint | median margin, economic | median margin, political | median paired change vs v1.1, economic | the same, political | economic headlines lost to политика | answers политика |
|---|---|---|---|---|---|---|
| v1.1 | +3.51 | -5.90 | +0.00 | +0.00 | 39 | 232 |
| seed-base-s2 | +1.85 | -5.44 | -0.79 | +0.55 | 58 | 268 |
| v1.2 | +0.53 | -6.42 | -2.33 | -0.69 | 76 | 313 |
| seed-aug-s2 | +2.71 | -13.45 | +0.10 | -6.73 | 60 | 344 |
| seed-aug-s2-ep3 | +1.78 | -6.72 | -0.68 | -1.00 | 62 | 333 |
| seed-aug-s3 | -0.75 | -5.08 | -3.97 | +0.58 | 107 | 365 |
| v1.3-ep1of4 | +0.72 | -1.13 | -2.61 | +4.74 | 39 | 223 |
| v1.3-ep2of4 | +0.37 | -3.16 | -2.65 | +2.82 | 66 | 285 |
| v1.3-ep3of4 | +0.11 | -6.12 | -2.66 | -0.09 | 84 | 318 |
| v1.3 | +0.11 | -6.12 | -2.66 | -0.09 | 84 | 318 |
| v1.3-ep4of4 | -0.07 | -10.99 | -3.40 | -4.97 | 88 | 313 |

| Checkpoint | культура | наука | политика | происшествия | спорт | экономика |
|---|---|---|---|---|---|---|
| v1.1 | 0.348 | 0.437 | 0.771 | 0.851 | 0.750 | 0.730 |
| seed-base-s2 | 0.280 | 0.487 | 0.735 | 0.752 | 0.728 | 0.603 |
| v1.2 | 0.360 | 0.348 | 0.813 | 0.727 | 0.733 | 0.460 |
| seed-aug-s2 | 0.360 | 0.310 | 0.849 | 0.758 | 0.706 | 0.592 |
| seed-aug-s2-ep3 | 0.348 | 0.329 | 0.843 | 0.789 | 0.672 | 0.603 |
| seed-aug-s3 | 0.366 | 0.386 | 0.801 | 0.826 | 0.628 | 0.328 |
| v1.3-ep1of4 | 0.149 | 0.462 | 0.639 | 0.832 | 0.383 | 0.690 |
| v1.3-ep2of4 | 0.236 | 0.304 | 0.705 | 0.876 | 0.656 | 0.529 |
| v1.3-ep3of4 | 0.230 | 0.430 | 0.771 | 0.776 | 0.733 | 0.448 |
| v1.3 | 0.230 | 0.430 | 0.771 | 0.776 | 0.733 | 0.448 |
| v1.3-ep4of4 | 0.230 | 0.418 | 0.753 | 0.727 | 0.733 | 0.425 |

What each checkpoint is:

- `v1.1`: the published v1.1
- `seed-base-s2`: a second seed of the v1.1 recipe
- `v1.2`: v1.2, phrase wordings added
- `seed-aug-s2`: v1.3 schedule on a v1.2-level mixture, seed 2
- `seed-aug-s2-ep3`: the same run at epoch 3
- `seed-aug-s3`: v1.3 schedule on a v1.2-level mixture, seed 3
- `v1.3-ep1of4`: v1.3 training, after epoch 1
- `v1.3-ep2of4`: v1.3 training, after epoch 2
- `v1.3-ep3of4`: v1.3 training, after epoch 3, the released weights
- `v1.3`: the published v1.3, scored from the ship directory
- `v1.3-ep4of4`: v1.3 training, after epoch 4, overtrained

## What this says

**The margin moves twice, and neither time is the transliteration lesson.**

The first move is v1.2, the release that added phrase wordings and nothing else:
the median economic headline loses 2.33 of the eventual 2.66, and recall on
`экономика` falls from 0.730 to 0.460. The second is the first epoch of v1.3
training, which arrives at -2.61 and then stays there: -2.65 after epoch 2,
-2.66 after epoch 3. Whatever moves the boundary is learned at once, not
accumulated, and it is already present in a mixture that has no transliterated
labels in it at all.

**Separability holds much longer than the margin does.** The AUC is 0.928 for
v1.1, 0.929 for v1.2, 0.921 after the first epoch and 0.938 after the second,
none of which clears zero against v1.1. It falls only at the third epoch, to
0.895, where the interval is [-0.061, -0.008] and excludes zero, and further at
the fourth, to 0.875. So the distribution shifts early and the ordering decays
late, in the epoch that was released.

**The three runs on a v1.2-level mixture rule the transliterations out.**
`seed-aug-s2`, `seed-aug-s2-ep3` and `seed-aug-s3` ran the v1.3 schedule on a
mixture that never saw a transliterated label, and the worst AUC of the whole
ladder is one of them, `seed-aug-s3` at 0.834 with an interval of [-0.133,
-0.059]. A lesson that is not in the data cannot be what damaged the boundary.
The spread between those three, 0.834 to 0.921, is also a reminder of how much
of this is seed and how little is recipe.

**What the released checkpoint is.** Of the four v1.3 epochs, the third is both
the most accurate overall (0.568 against 0.526, 0.554 and 0.551) and the one
where the boundary first degrades significantly. The epoch that was chosen on
accuracy is not the epoch that reads this pair of rubrics best; by AUC the
second epoch is, at 0.938, and it is three points worse on accuracy.

## What this does not say

The rubrics are an evaluation-only dataset, 1,000 headlines, and no release
trained on it. One pair of labels out of fifteen possible pairs is examined here
because that is the pair the gap sits on; the other pairs were not looked at, so
nothing here generalises to the model's behaviour elsewhere. And the ladder is
one run per step, not an average over seeds, except where a seed replicate is
listed as its own row.
