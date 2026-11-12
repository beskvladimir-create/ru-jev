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
