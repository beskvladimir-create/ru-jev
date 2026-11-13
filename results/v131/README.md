# v1.3.1, the rule, written before the training run

This file is committed before the GPU is rented and before anything is trained.
Nothing below was chosen after seeing a number. The commit that introduces it
carries the time.

## Why

RuCoLA's test split carries no label. The builder mapped its -1 placeholder onto
a valid option index, so 2,788 sentences entered the v1.3 mixture carrying
`неприемлемо`, a label they do not have: 2,536 in training and 252 in the
validation split the release was measured on. `results/rucola_test/` holds what
they did to every published figure.

The release was not retrained for that, and every published number stands as
published. This run asks a narrower question: what the released checkpoint would
have been without the defect.

## The data, and why they are not rebuilt

The obvious move is to rebuild with the fixed builder. It is the wrong one. The
fix makes the RuCoLA pool smaller, the per-task shuffle therefore lands
differently, and the build draws a different train and validation carve. Against
the released v1.3 that carve leaks: measured with `scripts/split_overlap.py`,
89.4 per cent of the rebuilt validation rows of `rucola` had been in v1.3's
training split, and the rest of the mixture moves too. Scoring the release on
rows it trained on is the hazard this project has already been caught by twice,
and it would land on the one task this run is about.

So the mixture is not rebuilt. It is the v1.3 mixture with the defective rows
taken out and nothing else touched:

1. `scripts/build_data.py` at commit `96fb67f`, the builder v1.3 used, run with
   its defaults, reproduces the v1.3 mixture byte for byte. Checked against the
   sha256 recorded in `results/v13_seed2/README.md` before this file was written:
   `39ad58ba24818f078f9d1bbb64cfe62f514abb5e1bff153eff9f43fbae327fcf` for
   `train.jsonl` and `8b39de0a87b46a5aff61f894b0e2cfa48a2b2862e4c930a20920f7e457398b4f`
   for `val.jsonl`, 143,991 and 26,746 rows. Both matched.
2. `scripts/strip_rucola_test.py` removes the 2,536 defective training rows and
   the 252 defective validation rows, by the sha256 of the sentence, and leaves
   every other row with its wording, its option order and its position.

| File | sha256 | rows |
|---|---|---|
| `train.jsonl` | `eb2abeee4b82ccde36d836ad07eaf7174ccf1bccdfdc4deecfd13dcffdde2549` | 141,455 |
| `val.jsonl` | `784ae64fc3bc09b01d298c4190c65568d64d72710a96c85c2cc98e5d67ff8cd6` | 26,494 |

The consequence that makes the comparison worth running: the validation split of
this run is a subset of v1.3's, row for row, so the released checkpoint does not
have to be scored again. Its answers on these rows are already published in
`results/validation/val_v13_trainsplit.jsonl`, and the two runs are compared on
identical rows with no leakage anywhere.

## The config

`configs/v1_3_1.yaml` is `configs/v1_3.yaml` with the output directory changed
and nothing else. The seed stays 42, the seed the release was trained with, so
this run against the released v1.3 differs in the removed rows alone.

## The epoch

**The epoch that is read is fixed here, before training: the third of four**, as
the release is. It is not chosen by result. The other three are kept and scored
and appear in the tables, and no conclusion rests on them.

## The rule

v1.3.1 goes to Hugging Face only if both of these hold.

**One.** On RuCoLA it is better than the released v1.3. The rows are the 1,092
validation rows of `rucola` that carry a real label, which are validation rows
in both runs.

**Two.** It is not worse than the released v1.3 by more than 0.005, on each of:

- the six Russian SuperGLUE tasks, `danetqa`, `parus`, `rcb`, `russe`, `rwsd`
  and `terra`, 10,157 rows, which this build and v1.3's carve hold identically;
- the whole validation split, 26,494 rows.

Otherwise it is not published and the result is written here as it came out.

A note on how the bound is read, decided here rather than afterwards. The 0.005
is applied to those two aggregates, not to each task on its own. On `parus` one
decision is 0.010 and on `rwsd` 0.005, so a per-task bound of 0.005 would be
finer than a single answer and any coin flip would break it. The per-task table
is reported in full, as description.

Publication also needs its owner's word, which this rule does not replace. The
rule can only refuse.

## What else is measured, as description and carrying no condition

- the temperature of each of the four epochs, on this run's own validation split;
- validation accuracy by task;
- RuCoLA against the released v1.3 and against the share of the majority class,
  which on the clean rows is 0.7372;
- the wording bench and the two probes at 5 and at 18 candidates, against the
  released v1.3.

## The control on the instrument

The released v1.3 is read from the published dump rather than scored again, so
the control is the other way round: the rows this run is scored on have to be
the same rows, in the same order, as the subset of
`results/validation/val_v13_trainsplit.jsonl` that survives the strip. That is
checked on the task name, the index of the correct candidate and the number of
candidates, on all 26,494 rows, before any number is read.

## Results

Filled in after the run, below this line, in a later commit.
