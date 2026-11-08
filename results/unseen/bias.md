# One bias per option on the unseen datasets

Reproduce with:

    python3 -I results/analysis/unseen_bias.py results/unseen

A temperature cannot change which option comes out: it divides every score by
the same number and the highest score stays the highest. So when a model is
wrong because it leans towards one label whatever the input says, no
temperature will repair it. One additive bias per option will, because the
biases are added after the division and do move the argmax:

    p = softmax(s / T + b[option])

On the news rubrics v1.3 answers `политика` 318 times where 166 headlines are
about politics, against 232 answers for v1.1, and its recall on `экономика`
falls from 0.730 to 0.448. That is the shape a bias can correct.

Nothing below is fitted and scored on the same rows. Each of 25 half-splits is
used twice, once in each direction, so every row is answered 25 times by a fit
that never saw it, and the accuracy is the mean of those answers over all rows.
Splits are seeded 0 to 24. The bootstrap draws 5,000 times from seed 20261105.

## Accuracy

| Dataset | Model | as published | biases and T | biases, T held at 1 |
|---|---|---|---|---|
| news rubrics, 1000 rows, 6 options | ru-jev-v1.1 | 0.652 | 0.673 | 0.670 |
| news rubrics, 1000 rows, 6 options | ru-jev-v1.3 | 0.568 | 0.640 | 0.639 |
| CEDR emotions, 1000 rows, 5 options | ru-jev-v1.1 | 0.455 | 0.556 | 0.558 |
| CEDR emotions, 1000 rows, 5 options | ru-jev-v1.3 | 0.511 | 0.571 | 0.576 |

Most of the v1.3 deficit on news rubrics is a lean, not a misreading: 0.568
becomes 0.640 with six numbers fitted on held-out rows, and the model itself is
untouched. On emotions both releases gain about ten points, which says the
released temperature's label set and this one are simply far apart.

## Paired difference, v1.3 minus v1.1

| Dataset | Setting | difference | 95% over rows | 95% over labels |
|---|---|---|---|---|
| news rubrics | as published | -0.084 | [-0.113, -0.055] | [-0.172, -0.017] |
| news rubrics | after biases and T | -0.034 | [-0.060, -0.006] | [-0.067, +0.001] |
| news rubrics | after biases, T held at 1 | -0.031 | [-0.058, -0.004] | [-0.066, +0.005] |
| CEDR emotions | as published | +0.056 | [+0.020, +0.093] | [-0.099, +0.221] |
| CEDR emotions | after biases and T | +0.014 | [-0.016, +0.044] | [-0.055, +0.050] |
| CEDR emotions | after biases, T held at 1 | +0.018 | [-0.014, +0.050] | [-0.091, +0.079] |

The interval over rows asks whether the gap is real for this sample of
headlines. The interval over labels resamples whole labels and asks whether it
survives a different choice of rubrics, which is the harder question. After the
biases the news gap is still there over rows and no longer clears zero over
labels.

## The biases themselves

Fitted on all 1,000 rows of each set. For reading only: none of the accuracy
above uses them. Only differences between biases matter, so one option is held
at zero; both releases hold the same one, which is what makes the last column
readable on its own.

News rubrics, `культура` held at zero:

| Option | ru-jev-v1.1 | ru-jev-v1.3 | difference |
|---|---|---|---|
| политика | -1.535 | -1.939 | -0.404 |
| наука | -0.150 | -0.460 | -0.310 |
| культура | +0.000 | +0.000 | +0.000 |
| происшествия | -1.847 | -1.841 | +0.006 |
| спорт | -0.861 | -0.804 | +0.058 |
| экономика | -1.242 | -0.667 | +0.575 |

Temperature fitted with them: v1.1 2.291, v1.3 1.901.

CEDR emotions, `грусть` held at zero:

| Option | ru-jev-v1.1 | ru-jev-v1.3 | difference |
|---|---|---|---|
| радость | +0.775 | -0.126 | -0.901 |
| удивление | -0.777 | -1.377 | -0.599 |
| грусть | +0.000 | +0.000 | +0.000 |
| страх | -1.530 | -1.326 | +0.205 |
| злость | -1.545 | -1.029 | +0.517 |

Temperature fitted with them: v1.1 2.329, v1.3 1.379.

On emotions the biases leave nothing between the two releases. After them the
difference is +0.014, and the interval covers zero both over rows and over
labels.

The two columns that carry the news story are the first and the last. The later
release has to be pushed 0.404 further off `политика` and 0.575 further towards
`экономика` than the earlier one, which is the same thing the per-label counts
say: of the 153 headlines only v1.1 gets right, 41 are economy read as politics.

## What this is not

The biases are fitted on the caller's labelled rows and belong to that label
set. Carrying them to a different set of labels means nothing, and they are not
shipped with the weights. `scripts/fit_temperature.py --biases` writes them
into the same `calibration.json` as the temperature, and the serving API and
`scripts/eval.py` apply whatever that file carries.
