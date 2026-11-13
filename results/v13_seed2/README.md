# A second seed of the v1.3 recipe: the rule, written before the training run

This file is committed before the GPU is rented and before anything is trained.
Nothing below was chosen after seeing a number. The commit that introduces it
carries the time.

## Why

On the scientific rubrics of `results/unseen3/`, with per-option biases, the
released v1.3 is ahead of `seed-base-s2` by +0.045 and ahead of the released
v1.1 by +0.091, while the two published runs of the v1.1 recipe, v1.1 and
`seed-base-s2`, sit 0.046 apart. The difference between the recipes is therefore
of the same size as the spread inside one of them, and with one run of the v1.3
recipe there is no way to separate the two.

There is no valid second run of the v1.3 recipe. `seed-aug-s2` and
`seed-aug-s2-ep3` are not one: they trained on a mixture built before commit
`96fb67f`, that is, without the transliterated and clipped forms of the intent
labels, which are the whole content of v1.3. `results/seeds/FINDINGS.md` holds
that in full. Without a second run the gap between the recipes on the scientific
rubrics cannot be placed, and it matters whether it is nearer +0.025 or +0.065.

## The run

`configs/v1_3_seed2.yaml` is `configs/v1_3.yaml` with two fields changed, the
seed and the output directory. The seed is 1337, the seed of
`configs/seed_aug_s2.yaml`, so that two pairs are clean at once: this run
against the released v1.3 differs in the seed alone on one mixture, and this run
against `seed-aug-s2-ep3` differs in the data alone at one seed.

## The epoch

**The epoch that is read is fixed here, before training: the third of four**, as
the release is. It is not chosen by result. The other three epochs are kept and
scored, and they appear in the tables, but no conclusion rests on them. If the
third epoch of this run turns out to be a poor checkpoint, that is a finding
about the recipe and it is reported as one, not repaired by reading a different
epoch.

## The data

The mixture is rebuilt with `scripts/build_data.py` as it stood at commit
`96fb67f`, which is the builder the released v1.3 used. The builder changed
three times afterwards, and one of those changes, `44826b6`, changed how the
split is drawn, so the current builder does not produce the v1.3 mixture.

The hashes of the v1.3 mixture were never recorded, not in `SHA256SUMS` and not
in `results/training/v1_3_train.log`, so a byte comparison against the original
is not available. The build is instead held against everything the released run
did record, before anything is trained:

- 198 distinct surface forms of the intent labels, four styles over 60 intents,
  which is 60 English keys, 60 Russian words, 60 transliterations and 18 clipped
  forms, less 16 that coincide across styles. The rule as first written asked
  for 240. That was 60 times 4, an expectation rather than a measurement, and it
  was corrected to 198 in commit `08c03ee` on `main`, in
  `results/seeds/FINDINGS.md` and `configs/SEED_RUNS.md`, from the style
  functions of the builder at `96fb67f`. The check below was read against 198;
  nothing else in this rule changed, and the paragraph recording what the check
  found is left as it was written;
- 143,991 training and 26,746 validation decisions;
- 13 tasks, with the builder's own completeness check passing.

If any of those does not match, the work stops and the mismatch is reported
rather than trained through. This is the trap the earlier seed runs fell into:
the row counts and the task composition matched there too, because new label
styles change the wording inside a row and not the number of rows, and only the
count of surface forms would have caught it.

The sha256 of the mixture that is actually trained on is recorded below, so that
this run, unlike the one it repeats, can be checked.

## What is computed

Per-option additive biases and the temperature are fitted out of fold with the
functions of `results/analysis/unseen_bias.py`: 25 half-splits seeded 0 to 24,
each used twice, so every row is answered by a fit that never saw it. Intervals
are bootstrap over rows and over labels, seed 20261105, 5,000 draws.

The datasets are the news rubrics, the CEDR emotions, the scientific rubrics
both as the alphabetical ten and as the twenty random tens of
`results/analysis/matrix_sensitivity.py`, and the wording bench and the two
probes at 5 candidates and at 18.

On each of them, four quantities:

| Quantity | Definition | What it asks |
|---|---|---|
| spread of the v1.3 recipe | this run minus the released v1.3 | how far two runs of one recipe sit apart |
| spread of the v1.1 recipe | `seed-base-s2` minus v1.1 | the same for the earlier recipe |
| gap between the recipes | the mean of the two v1.3 runs minus the mean of the two v1.1 runs | what the recipe change is worth |
| what the forms are worth at one seed | this run minus `seed-aug-s2-ep3` | the data change alone, seed held at 1337 |

## The rule

**On a dataset the gap between the recipes counts as established if it is larger
in absolute value than both within-recipe spreads. Otherwise it lies inside the
spread between seeds and is reported as such.**

On the scientific rubrics the rule is applied to the median over the twenty
random tens, and the number of tens in which it holds is stated beside it.

With two runs per recipe this is a description and not a statistical test. Two
runs give a range, not a distribution, and no p-value is attached to any of
these numbers.

## A control on the instrument

The released v1.3 is scored again through the same path as this run, and it has
to return the published logits and the same answer on every row. A measurement
of a new checkpoint is worth nothing until that holds.

## The build that was trained on

Built 2026-11-12 with `scripts/build_data.py` as it stood at commit `96fb67f`,
run with its defaults, which is `--include-rucola` and `--cap 30000`.

| File | sha256 | rows | bytes |
|---|---|---|---|
| `train.jsonl` | `39ad58ba24818f078f9d1bbb64cfe62f514abb5e1bff153eff9f43fbae327fcf` | 143,991 | 68,987,169 |
| `val.jsonl` | `8b39de0a87b46a5aff61f894b0e2cfa48a2b2862e4c930a20920f7e457398b4f` | 26,746 | 13,002,155 |

### What the check found

| Check | Reference | This build | |
|---|---|---|---|
| training decisions | 143,991 | 143,991 | matches |
| validation decisions | 26,746 | 26,746 | matches |
| tasks | 13, completeness check passing | 13, passing | matches |
| rows per task | the builder's own `EXPECTED` table | every task equal to it | matches |
| distinct surface forms of the intent labels | 240 | 198 | see below |

**The 240 is a nominal figure and the measured number for this builder is 198.**
Four styles over 60 intents is 240 only if no two of them ever agree. The
clipped style, `jargon_clip`, keeps the first word of the label and cuts it to
five characters, so the 60 intents collapse onto 18 values: every label that
starts with `email` becomes `емэйл`, every one that starts with `calendar`
becomes `кален`. 60 English keys, 60 Russian words, 60 transliterations and 18
clipped forms, less 16 forms that two styles produce alike, is 198. The style
functions give that number on their own, without any data.

The count is of surface forms that appear as an option anywhere in the mixture,
and on the two mixtures that are still on disk it returns exactly what
`results/seeds/FINDINGS.md` recorded for them: 120 for the stale mixture the
earlier seed runs used, where the only single Cyrillic words are `посчитать` and
`приветствие`, and 60 for the v1.1 mixture. This build has 198 forms and 20
single Cyrillic words, `аларм`, `вэзер`, `иот`, `кукинг`, `ньюс`, `плэй` and
the rest, which are the transliterated and clipped forms. The jargon lesson is
in this mixture and was not in theirs. `FINDINGS.md` says "with four styles
there would have been 240"; that sentence states an expectation, not a
measurement, and 198 is the figure a measurement returns.

### Two further notes on the build

`RussianNLP/rucola` is no longer served by the hub; `datasets` reported that it
could not be found and used the local cache of 2026-11-02, which is the copy
the released run was built from. The task is present and complete, 12,101
training and 1,344 validation rows, but a rebuild on a machine without that
cache will not get it.

Train and validation share 372 input texts, 0.26 percent, almost all of them in
the two MASSIVE tasks where the same utterance occurs in both source splits.
The current builder gives 396 on the same sources, so this is a property of the
sources rather than of the split, and it is the same in the released run.

A byte comparison with the mixture v1.3 actually trained on remains impossible:
no copy of it survives on disk, only the v1.1 mixture and the stale one, and its
hashes were never recorded.

## The run as it happened

Everything below is a record of the run, written after it finished. The rule
above was committed before the training started and is not touched by it.

| Epoch | Mean training loss | Validation loss | Validation accuracy | Fitted temperature |
|---|---|---|---|---|
| 1 of 4 | 0.5975 | 0.4984 | 0.7631 | 0.9274 |
| 2 of 4 | 0.4249 | 0.4255 | 0.8137 | 1.2811 |
| **3 of 4, the epoch the rule reads** | **0.2529** | **0.4849** | **0.8220** | **2.0116** |
| 4 of 4 | 0.0759 | 0.7946 | 0.8227 | 3.7980 |

Validation is the 26,746 rows of `val.jsonl` of the rebuilt mixture, every epoch
on the same rows. The full training log is in
[`train_log.txt`](train_log.txt), verbatim, including the warnings the loaders
printed.

Where the temperatures come from. The run itself fits a temperature only once,
at the end, on the fourth epoch, and wrote 3.7980 into the release folder. The
other three were fitted afterwards with the same instrument: the validation pass
and the `fit_temperature` of `scripts/train.py`, the same collate with the
option order left alone, the same batch size out of the same config, on the same
rows. `scripts/fit_temperature.py` was not used for this. That script is the one
shipped for a reader to calibrate on their own decisions, it runs on the CPU, and
it is not the path the released temperature came from. As a control the fourth
epoch was fitted again through the same path and returned 3.7980, the figure the
training had already written, and the validation loss and accuracy of all four
epochs came back identical to the log.

The temperature rises with every epoch, from 0.93 to 3.80, while the validation
loss turns at the third epoch and the accuracy stops moving. That is the same
shape the released run has, and it is the reason the epoch is fixed in advance
rather than chosen afterwards: the fourth epoch is the most accurate by a
hundredth and the worst calibrated by a factor of two.

### The weights, and what they are

The four epochs and the calibrated release are on local disk, outside this
repository, in `model/v13-seed2/{ep0,ep1,ep2,ep3,final}`. They are not
published. What this repository carries is the scores they produced.

Two of those folders hold one and the same model. The external snapshotter takes
its folder name from the `epoch` field of `calibration.json`, and
`scripts/train.py` labels its last save `final` rather than `3`, so the fourth
epoch arrived as `epfinal` and is stored here as `ep3`. The release folder is
that same save with the temperature filled in. Their weights agree byte for
byte, sha256
`1a0258ce3e92b9a6773cd0e5c555bd7f57df52b7b9008f1e06d00dcd274b6d57`, and they
differ only in `calibration.json`. Both are scored, separately, as the control
on the scoring path that the rule above asks for: the same weights read from two
directories have to answer identically.

### The rental

| | |
|---|---|
| Card | NVIDIA RTX PRO 4000 Blackwell, 24 GB |
| Rented | 2026-11-12 15:30:46 UTC to 21:31:46 UTC, 6 hours 1 minute |
| Rate | 0.576 USD per hour |
| Cost | 3.47 USD |
| Training | 15:37:18 UTC to about 20:46 UTC, 9,000 steps per epoch, batch 16 |

The hour after the training went to fitting the temperatures of the first three
epochs and to taking the weights down. Each epoch was taken off the card as soon
as it was snapped rather than at the end, so that a card that dies, or a balance
that runs out, costs at most the epoch in flight. Every transfer was checked by
comparing the sha256 of the weights on both sides, and the folder was put in
place only after the two agreed.

## What the numbers say

Everything in this section is the output of `results/analysis/seed2.py`, which
computes what the rule above asks for and nothing else. The rule was not
touched after the numbers were seen, and the epoch that is read is still the
third of four.

### The short of it

The rule answers yes on the scientific rubrics, which is the dataset this run
was trained to settle, and on four of the six vocabularies of the wording bench.
It answers no on the news rubrics and on the CEDR emotions, where the sign is
against the v1.3 recipe rather than for it.

The largest effect in the whole table is on the jargon vocabulary: the gap
between the recipes is +0.182 at eighteen candidates and +0.159 at five, against
within-recipe spreads of 0.074 and 0.004. At one seed, with the data change
alone, the label forms are worth +0.262 and +0.201 there. That is the one place
where the content of v1.3, the transliterated and clipped surface forms of the
intent labels, is doing visible work, and it is also the place where it would be
expected to.

Three things in this section argue against reading the result as a clean win,
and they are stated here rather than left to a reader to find.

First, on the scientific rubrics the margin is thin. The rule is applied to the
median over the twenty random tens, as the pre-registration requires: the gap is
+0.078 and the two within-recipe spreads are -0.039 and +0.070. The gap clears
the larger of them by 0.008. Taken ten by ten rather than at the median, the
rule holds in 11 of the 20 tens and fails in 9. The reason is visible in the
same table: the v1.1 recipe moves +0.070 between its own two seeds on this
dataset, which is nearly the whole size of the gap being measured. The
instability that made this run necessary is still the dominant term.

Second, on the scientific rubrics the recipe gap is not explained by the label
forms. The median of what the forms are worth at one seed is -0.007, that is
nothing, while the gap between the recipes is +0.078. On the alphabetical ten
the same quantity reads +0.038, and the two readings disagree, which is itself a
statement about how much one draw of ten rubrics out of 28 can be trusted. The
honest version is that on this dataset the recipe gap is carried by something
other than the forms, and this run does not say what.

Third, the epoch that was fixed in advance is not the best epoch of the run on
two datasets. On the news rubrics the first epoch reads 0.702 against 0.626 for
the third, and on the emotions 0.590 against 0.484. The pre-registration says
this is reported as a finding about the recipe rather than repaired by reading
another epoch, and that is what is done here. The four epochs are in the last
table.

The style probe carries little either way. It sits near chance at eighteen
candidates, where the out of fold bias fit has almost nothing to work with and
in places lowers accuracy below the raw score, and no reading in this section
rests on it.

### The control, and its one nonzero

Seven of the eight checks in the control table return a largest logit
difference of exactly zero: the four comparison checkpoints rescored on this
machine against their published dumps, the released v1.3 read under both of the
names it carries, and the fourth epoch of the new run read from two
directories.

The eighth is not zero. Five of the eighteen logits of a bench question, held
against the published five-candidate dump of the same question, differ by at
most 0.000889 over 6,000 rows, with no answer changed on any row. The model
scores each (text, option) pair on its own, so these are arithmetically the same
numbers; the difference is the float noise of a different batch composition,
since the eighteen-candidate call groups and pads its work differently from the
five-candidate call. It is four orders of magnitude below the smallest
difference any reading here rests on, and it is reported rather than rounded
away.

### The tables

### Control on the instrument

| Check | Result |
|---|---|
| news rubrics, v1.3 and v1.1 and seed-base-s2 and seed-aug-s2-ep3 rescored against the published dumps | 4000 rows, 0 answers differ, largest logit difference 0 |
| emotions, v1.3 and v1.1 and seed-base-s2 and seed-aug-s2-ep3 rescored against the published dumps | 4000 rows, 0 answers differ, largest logit difference 0 |
| five of the eighteen logits against the published five-candidate bench, v1.3 | 6000 rows, 0 answers differ, largest logit difference 0.000889 |
| v1.3 against v1.3-ep3of4 in the published dumps, the scientific rubrics | 990 rows, 0 answers differ, largest logit difference 0 |
| v1.3 against v1.3-ep3of4 in the published dumps, the news rubrics | 1000 rows, 0 answers differ, largest logit difference 0 |
| v1.3 against v1.3-ep3of4 in the published dumps, the emotions | 1000 rows, 0 answers differ, largest logit difference 0 |
| the fourth epoch of the new run from two directories, the wording bench | the same answer on 6000 of 6000 rows |
| the fourth epoch of the new run from two directories, the news rubrics | the same answer on 1000 of 1000 rows |

### Accuracy of every checkpoint, out of fold with biases

| Dataset | | v1.3-seed2-ep3of4 | v1.3 | v1.1 | seed-base-s2 | seed-aug-s2-ep3 |
|---|---|---|---|---|---|---|
| news rubrics | as published | 0.549 | 0.568 | 0.652 | 0.601 | 0.601 |
| 1000 rows, 6 candidates | with biases | 0.626 | 0.640 | 0.673 | 0.636 | 0.681 |
| emotions | as published | 0.417 | 0.511 | 0.455 | 0.545 | 0.436 |
| 1000 rows, 5 candidates | with biases | 0.484 | 0.571 | 0.556 | 0.577 | 0.525 |
| scientific rubrics, the alphabetical ten | as published | 0.585 | 0.601 | 0.472 | 0.536 | 0.546 |
| 990 rows, 5 candidates | with biases | 0.662 | 0.657 | 0.566 | 0.612 | 0.624 |
| bench английский ключ, 18 candidates | as published | 0.890 | 0.900 | 0.894 | 0.880 | 0.878 |
| 1000 rows | with biases | 0.884 | 0.891 | 0.891 | 0.879 | 0.881 |
| bench английский ключ, 5 candidates | as published | 0.955 | 0.962 | 0.957 | 0.951 | 0.955 |
| 1000 rows, five of the eighteen | with biases | 0.943 | 0.952 | 0.950 | 0.947 | 0.949 |
| bench русское слово, 18 candidates | as published | 0.878 | 0.892 | 0.883 | 0.883 | 0.888 |
| 1000 rows | with biases | 0.878 | 0.887 | 0.881 | 0.878 | 0.883 |
| bench русское слово, 5 candidates | as published | 0.954 | 0.960 | 0.956 | 0.953 | 0.958 |
| 1000 rows, five of the eighteen | with biases | 0.944 | 0.955 | 0.951 | 0.953 | 0.950 |
| bench русская фраза, 18 candidates | as published | 0.883 | 0.885 | 0.703 | 0.689 | 0.874 |
| 1000 rows | with biases | 0.881 | 0.880 | 0.795 | 0.788 | 0.877 |
| bench русская фраза, 5 candidates | as published | 0.955 | 0.957 | 0.881 | 0.902 | 0.950 |
| 1000 rows, five of the eighteen | with biases | 0.952 | 0.958 | 0.906 | 0.921 | 0.948 |
| bench жаргон, 18 candidates | as published | 0.623 | 0.640 | 0.431 | 0.471 | 0.415 |
| 1000 rows | with biases | 0.682 | 0.755 | 0.538 | 0.534 | 0.420 |
| bench жаргон, 5 candidates | as published | 0.823 | 0.855 | 0.633 | 0.658 | 0.611 |
| 1000 rows, five of the eighteen | with biases | 0.844 | 0.890 | 0.708 | 0.707 | 0.643 |
| bench названия служб, 18 candidates | as published | 0.730 | 0.733 | 0.632 | 0.644 | 0.753 |
| 1000 rows | with biases | 0.756 | 0.746 | 0.677 | 0.667 | 0.750 |
| bench названия служб, 5 candidates | as published | 0.874 | 0.869 | 0.835 | 0.828 | 0.879 |
| 1000 rows, five of the eighteen | with biases | 0.892 | 0.893 | 0.849 | 0.851 | 0.895 |
| bench описание запроса, 18 candidates | as published | 0.867 | 0.878 | 0.848 | 0.855 | 0.878 |
| 1000 rows | with biases | 0.870 | 0.880 | 0.857 | 0.859 | 0.876 |
| bench описание запроса, 5 candidates | as published | 0.955 | 0.958 | 0.939 | 0.939 | 0.954 |
| 1000 rows, five of the eighteen | with biases | 0.944 | 0.952 | 0.943 | 0.940 | 0.948 |
| probe стиль_A, 18 candidates | as published | 0.123 | 0.146 | 0.104 | 0.080 | 0.086 |
| 1000 rows | with biases | 0.096 | 0.039 | 0.063 | 0.078 | 0.089 |
| probe стиль_A, 5 candidates | as published | 0.295 | 0.270 | 0.265 | 0.226 | 0.311 |
| 1000 rows, five of the eighteen | with biases | 0.393 | 0.360 | 0.373 | 0.344 | 0.374 |
| probe стиль_B, 18 candidates | as published | 0.464 | 0.476 | 0.428 | 0.393 | 0.516 |
| 1000 rows | with biases | 0.501 | 0.515 | 0.478 | 0.432 | 0.525 |
| probe стиль_B, 5 candidates | as published | 0.689 | 0.780 | 0.648 | 0.626 | 0.727 |
| 1000 rows, five of the eighteen | with biases | 0.736 | 0.795 | 0.729 | 0.666 | 0.755 |

### The four quantities, with intervals over rows and over labels


#### news rubrics

| Quantity | difference | 95% over rows | 95% over labels |
|---|---|---|---|
| spread of the v1.3 recipe | -0.014 | [-0.041, +0.014] | [-0.026, -0.002] |
| spread of the v1.1 recipe | -0.037 | [-0.062, -0.011] | [-0.093, +0.018] |
| gap between the recipes | -0.022 | [-0.041, -0.002] | [-0.037, -0.009] |
| what the forms are worth at one seed | -0.054 | [-0.081, -0.027] | [-0.083, -0.009] |

#### emotions

| Quantity | difference | 95% over rows | 95% over labels |
|---|---|---|---|
| spread of the v1.3 recipe | -0.086 | [-0.116, -0.055] | [-0.166, -0.023] |
| spread of the v1.1 recipe | +0.020 | [-0.008, +0.048] | [-0.012, +0.038] |
| gap between the recipes | -0.039 | [-0.063, -0.013] | [-0.132, +0.009] |
| what the forms are worth at one seed | -0.041 | [-0.069, -0.010] | [-0.111, +0.020] |

#### scientific rubrics, the alphabetical ten

| Quantity | difference | 95% over rows | 95% over labels |
|---|---|---|---|
| spread of the v1.3 recipe | +0.004 | [-0.024, +0.031] | [-0.040, +0.046] |
| spread of the v1.1 recipe | +0.046 | [+0.013, +0.080] | [-0.034, +0.133] |
| gap between the recipes | +0.070 | [+0.048, +0.093] | [+0.025, +0.120] |
| what the forms are worth at one seed | +0.038 | [+0.008, +0.068] | [-0.014, +0.087] |

#### bench английский ключ, 18 candidates

| Quantity | difference | 95% over rows | 95% over labels |
|---|---|---|---|
| spread of the v1.3 recipe | -0.007 | [-0.020, +0.005] | [-0.023, +0.005] |
| spread of the v1.1 recipe | -0.012 | [-0.023, +0.000] | [-0.027, +0.001] |
| gap between the recipes | +0.002 | [-0.008, +0.012] | [-0.005, +0.011] |
| what the forms are worth at one seed | +0.003 | [-0.009, +0.015] | [-0.006, +0.011] |

#### bench английский ключ, 5 candidates

| Quantity | difference | 95% over rows | 95% over labels |
|---|---|---|---|
| spread of the v1.3 recipe | -0.009 | [-0.018, -0.000] | [-0.020, -0.001] |
| spread of the v1.1 recipe | -0.004 | [-0.012, +0.005] | [-0.010, +0.004] |
| gap between the recipes | -0.001 | [-0.008, +0.007] | [-0.006, +0.006] |
| what the forms are worth at one seed | -0.006 | [-0.015, +0.003] | [-0.014, +0.000] |

#### bench русское слово, 18 candidates

| Quantity | difference | 95% over rows | 95% over labels |
|---|---|---|---|
| spread of the v1.3 recipe | -0.009 | [-0.021, +0.003] | [-0.027, +0.005] |
| spread of the v1.1 recipe | -0.002 | [-0.015, +0.010] | [-0.012, +0.006] |
| gap between the recipes | +0.003 | [-0.009, +0.015] | [-0.008, +0.018] |
| what the forms are worth at one seed | -0.005 | [-0.017, +0.008] | [-0.018, +0.008] |

#### bench русское слово, 5 candidates

| Quantity | difference | 95% over rows | 95% over labels |
|---|---|---|---|
| spread of the v1.3 recipe | -0.011 | [-0.020, -0.002] | [-0.024, -0.001] |
| spread of the v1.1 recipe | +0.002 | [-0.005, +0.010] | [-0.004, +0.009] |
| gap between the recipes | -0.002 | [-0.010, +0.006] | [-0.009, +0.005] |
| what the forms are worth at one seed | -0.006 | [-0.015, +0.003] | [-0.016, +0.003] |

#### bench русская фраза, 18 candidates

| Quantity | difference | 95% over rows | 95% over labels |
|---|---|---|---|
| spread of the v1.3 recipe | +0.001 | [-0.012, +0.013] | [-0.010, +0.011] |
| spread of the v1.1 recipe | -0.006 | [-0.026, +0.013] | [-0.027, +0.012] |
| gap between the recipes | +0.089 | [+0.072, +0.107] | [+0.046, +0.139] |
| what the forms are worth at one seed | +0.004 | [-0.008, +0.017] | [-0.006, +0.015] |

#### bench русская фраза, 5 candidates

| Quantity | difference | 95% over rows | 95% over labels |
|---|---|---|---|
| spread of the v1.3 recipe | -0.006 | [-0.014, +0.001] | [-0.012, -0.001] |
| spread of the v1.1 recipe | +0.015 | [+0.001, +0.029] | [+0.006, +0.025] |
| gap between the recipes | +0.042 | [+0.030, +0.054] | [+0.023, +0.063] |
| what the forms are worth at one seed | +0.004 | [-0.006, +0.013] | [-0.003, +0.010] |

#### bench жаргон, 18 candidates

| Quantity | difference | 95% over rows | 95% over labels |
|---|---|---|---|
| spread of the v1.3 recipe | -0.074 | [-0.101, -0.047] | [-0.148, -0.010] |
| spread of the v1.1 recipe | -0.004 | [-0.026, +0.018] | [-0.076, +0.045] |
| gap between the recipes | +0.182 | [+0.156, +0.209] | [+0.061, +0.347] |
| what the forms are worth at one seed | +0.262 | [+0.228, +0.295] | [+0.091, +0.457] |

#### bench жаргон, 5 candidates

| Quantity | difference | 95% over rows | 95% over labels |
|---|---|---|---|
| spread of the v1.3 recipe | -0.046 | [-0.067, -0.026] | [-0.092, -0.006] |
| spread of the v1.1 recipe | -0.001 | [-0.025, +0.023] | [-0.037, +0.032] |
| gap between the recipes | +0.159 | [+0.137, +0.183] | [+0.068, +0.284] |
| what the forms are worth at one seed | +0.201 | [+0.171, +0.232] | [+0.101, +0.326] |

#### bench названия служб, 18 candidates

| Quantity | difference | 95% over rows | 95% over labels |
|---|---|---|---|
| spread of the v1.3 recipe | +0.010 | [-0.008, +0.029] | [-0.003, +0.026] |
| spread of the v1.1 recipe | -0.011 | [-0.030, +0.009] | [-0.061, +0.027] |
| gap between the recipes | +0.079 | [+0.060, +0.099] | [+0.029, +0.141] |
| what the forms are worth at one seed | +0.007 | [-0.012, +0.025] | [-0.019, +0.036] |

#### bench названия служб, 5 candidates

| Quantity | difference | 95% over rows | 95% over labels |
|---|---|---|---|
| spread of the v1.3 recipe | -0.001 | [-0.015, +0.013] | [-0.013, +0.011] |
| spread of the v1.1 recipe | +0.002 | [-0.014, +0.018] | [-0.021, +0.022] |
| gap between the recipes | +0.042 | [+0.028, +0.057] | [+0.016, +0.076] |
| what the forms are worth at one seed | -0.004 | [-0.019, +0.012] | [-0.021, +0.016] |

#### bench описание запроса, 18 candidates

| Quantity | difference | 95% over rows | 95% over labels |
|---|---|---|---|
| spread of the v1.3 recipe | -0.011 | [-0.024, +0.003] | [-0.036, +0.008] |
| spread of the v1.1 recipe | +0.003 | [-0.013, +0.018] | [-0.012, +0.018] |
| gap between the recipes | +0.017 | [+0.005, +0.029] | [+0.004, +0.030] |
| what the forms are worth at one seed | -0.006 | [-0.020, +0.007] | [-0.020, +0.005] |

#### bench описание запроса, 5 candidates

| Quantity | difference | 95% over rows | 95% over labels |
|---|---|---|---|
| spread of the v1.3 recipe | -0.008 | [-0.019, +0.002] | [-0.019, +0.001] |
| spread of the v1.1 recipe | -0.003 | [-0.014, +0.008] | [-0.014, +0.005] |
| gap between the recipes | +0.007 | [-0.002, +0.016] | [-0.002, +0.016] |
| what the forms are worth at one seed | -0.004 | [-0.014, +0.006] | [-0.012, +0.003] |

#### probe стиль_A, 18 candidates

| Quantity | difference | 95% over rows | 95% over labels |
|---|---|---|---|
| spread of the v1.3 recipe | +0.057 | [+0.042, +0.073] | [+0.011, +0.111] |
| spread of the v1.1 recipe | +0.015 | [+0.002, +0.029] | [-0.043, +0.091] |
| gap between the recipes | -0.003 | [-0.014, +0.009] | [-0.056, +0.045] |
| what the forms are worth at one seed | +0.007 | [-0.012, +0.026] | [-0.059, +0.070] |

#### probe стиль_A, 5 candidates

| Quantity | difference | 95% over rows | 95% over labels |
|---|---|---|---|
| spread of the v1.3 recipe | +0.033 | [+0.012, +0.054] | [-0.021, +0.092] |
| spread of the v1.1 recipe | -0.029 | [-0.045, -0.013] | [-0.060, +0.004] |
| gap between the recipes | +0.018 | [+0.002, +0.034] | [-0.030, +0.077] |
| what the forms are worth at one seed | +0.018 | [-0.005, +0.041] | [-0.044, +0.073] |

#### probe стиль_B, 18 candidates

| Quantity | difference | 95% over rows | 95% over labels |
|---|---|---|---|
| spread of the v1.3 recipe | -0.013 | [-0.040, +0.013] | [-0.085, +0.060] |
| spread of the v1.1 recipe | -0.046 | [-0.074, -0.018] | [-0.128, +0.028] |
| gap between the recipes | +0.053 | [+0.029, +0.077] | [-0.046, +0.168] |
| what the forms are worth at one seed | -0.024 | [-0.049, +0.002] | [-0.074, +0.033] |

#### probe стиль_B, 5 candidates

| Quantity | difference | 95% over rows | 95% over labels |
|---|---|---|---|
| spread of the v1.3 recipe | -0.059 | [-0.083, -0.034] | [-0.128, -0.008] |
| spread of the v1.1 recipe | -0.063 | [-0.089, -0.037] | [-0.128, -0.013] |
| gap between the recipes | +0.069 | [+0.046, +0.091] | [+0.016, +0.134] |
| what the forms are worth at one seed | -0.019 | [-0.042, +0.005] | [-0.062, +0.023] |

### The scientific rubrics over twenty random tens

Seeds 0 to 19, built as `matrix_sensitivity.py` builds them, every quantity as the rule has it.

| Seed | spread v1.3 | spread v1.1 | gap | forms at one seed | rule holds |
|---|---|---|---|---|---|
| 0 | -0.065 | +0.049 | +0.082 | -0.021 | yes |
| 1 | -0.019 | +0.101 | +0.089 | -0.005 | no |
| 2 | -0.010 | +0.052 | +0.062 | +0.017 | yes |
| 3 | -0.057 | +0.038 | +0.076 | -0.014 | yes |
| 4 | -0.036 | +0.053 | +0.073 | +0.003 | yes |
| 5 | -0.040 | +0.135 | +0.095 | +0.014 | no |
| 6 | +0.011 | +0.013 | +0.070 | +0.025 | yes |
| 7 | -0.084 | +0.132 | +0.103 | -0.024 | no |
| 8 | -0.035 | +0.054 | +0.068 | +0.018 | yes |
| 9 | -0.026 | +0.141 | +0.111 | +0.013 | no |
| 10 | -0.033 | +0.078 | +0.098 | -0.008 | yes |
| 11 | -0.050 | +0.077 | +0.071 | -0.018 | no |
| 12 | -0.082 | +0.135 | +0.127 | -0.000 | no |
| 13 | -0.073 | +0.080 | +0.058 | -0.044 | no |
| 14 | -0.047 | +0.064 | +0.080 | -0.015 | yes |
| 15 | -0.034 | +0.018 | +0.065 | +0.002 | yes |
| 16 | -0.038 | +0.035 | +0.084 | +0.014 | yes |
| 17 | -0.049 | +0.053 | +0.068 | -0.008 | yes |
| 18 | -0.036 | +0.082 | +0.060 | -0.018 | no |
| 19 | -0.079 | +0.111 | +0.091 | -0.040 | no |

| Quantity | median | lowest | highest |
|---|---|---|---|
| spread of the v1.3 recipe | -0.039 | -0.084 | +0.011 |
| spread of the v1.1 recipe | +0.070 | +0.013 | +0.141 |
| gap between the recipes | +0.078 | +0.058 | +0.127 |
| what the forms are worth at one seed | -0.007 | -0.044 | +0.025 |

The rule holds in 11 of the 20 tens.

### The rule of results/v13_seed2/README.md

| Dataset | gap | spread v1.3 | spread v1.1 | the gap is larger than both |
|---|---|---|---|---|
| news rubrics | -0.022 | -0.014 | -0.037 | no |
| emotions | -0.039 | -0.086 | +0.020 | no |
| scientific rubrics, the alphabetical ten | +0.070 | +0.004 | +0.046 | yes |
| bench английский ключ, 18 candidates | +0.002 | -0.007 | -0.012 | no |
| bench английский ключ, 5 candidates | -0.001 | -0.009 | -0.004 | no |
| bench русское слово, 18 candidates | +0.003 | -0.009 | -0.002 | no |
| bench русское слово, 5 candidates | -0.002 | -0.011 | +0.002 | no |
| bench русская фраза, 18 candidates | +0.089 | +0.001 | -0.006 | yes |
| bench русская фраза, 5 candidates | +0.042 | -0.006 | +0.015 | yes |
| bench жаргон, 18 candidates | +0.182 | -0.074 | -0.004 | yes |
| bench жаргон, 5 candidates | +0.159 | -0.046 | -0.001 | yes |
| bench названия служб, 18 candidates | +0.079 | +0.010 | -0.011 | yes |
| bench названия служб, 5 candidates | +0.042 | -0.001 | +0.002 | yes |
| bench описание запроса, 18 candidates | +0.017 | -0.011 | +0.003 | yes |
| bench описание запроса, 5 candidates | +0.007 | -0.008 | -0.003 | no |
| probe стиль_A, 18 candidates | -0.003 | +0.057 | +0.015 | no |
| probe стиль_A, 5 candidates | +0.018 | +0.033 | -0.029 | no |
| probe стиль_B, 18 candidates | +0.053 | -0.013 | -0.046 | yes |
| probe стиль_B, 5 candidates | +0.069 | -0.059 | -0.063 | yes |
| scientific rubrics, median over twenty tens | +0.078 | -0.039 | +0.070 | yes, in 11 of 20 tens |

At two runs per recipe this is a description and not a statistical test.

### The other epochs of the new run

The epoch that is read was fixed as the third of four before training. The others are here and carry no conclusion.

| Dataset | ep1of4 | ep2of4 | ep3of4 | ep4of4 | final |
|---|---|---|---|---|---|
| news rubrics | 0.702 | 0.688 | 0.626 | 0.608 | 0.608 |
| emotions | 0.590 | 0.563 | 0.484 | 0.507 | 0.507 |
| scientific rubrics, the alphabetical ten | 0.694 | 0.710 | 0.662 | 0.646 | 0.646 |
| bench английский ключ, 18 candidates | 0.869 | 0.881 | 0.884 | 0.880 | 0.880 |
| bench английский ключ, 5 candidates | 0.943 | 0.948 | 0.943 | 0.943 | 0.943 |
| bench русское слово, 18 candidates | 0.875 | 0.878 | 0.878 | 0.877 | 0.877 |
| bench русское слово, 5 candidates | 0.943 | 0.950 | 0.944 | 0.944 | 0.944 |
| bench русская фраза, 18 candidates | 0.868 | 0.878 | 0.881 | 0.879 | 0.879 |
| bench русская фраза, 5 candidates | 0.940 | 0.951 | 0.952 | 0.953 | 0.953 |
| bench жаргон, 18 candidates | 0.751 | 0.722 | 0.682 | 0.681 | 0.681 |
| bench жаргон, 5 candidates | 0.872 | 0.862 | 0.844 | 0.844 | 0.844 |
| bench названия служб, 18 candidates | 0.726 | 0.731 | 0.756 | 0.747 | 0.747 |
| bench названия служб, 5 candidates | 0.874 | 0.873 | 0.892 | 0.884 | 0.884 |
| bench описание запроса, 18 candidates | 0.871 | 0.869 | 0.870 | 0.865 | 0.865 |
| bench описание запроса, 5 candidates | 0.942 | 0.941 | 0.944 | 0.943 | 0.943 |
| probe стиль_A, 18 candidates | 0.092 | 0.095 | 0.096 | 0.079 | 0.079 |
| probe стиль_A, 5 candidates | 0.392 | 0.399 | 0.393 | 0.396 | 0.396 |
| probe стиль_B, 18 candidates | 0.527 | 0.492 | 0.501 | 0.443 | 0.443 |
| probe стиль_B, 5 candidates | 0.730 | 0.706 | 0.736 | 0.685 | 0.685 |
