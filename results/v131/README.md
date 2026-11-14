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

The run trained on 14 November 2026 on one RTX 4090 of 24,564 MiB, the card the
release was trained on, under torch 2.8.0+cu129, four epochs at about 43 minutes
each. Three hours and five minutes of rented time, 2.85 USD. The pod was shut
off as soon as the five snapshots were on disk and each one had been checked for
its 306,939,648 parameters, its head and its calibration file. `ep3` and
`epfinal` are the same checkpoint saved twice, byte for byte.

Everything below is regenerated by

    python3 results/analysis/v131.py

out of the dumps committed in `results/v131/`, with no network, no weights and
no card.

### The verdict

**The rule refuses publication.** Condition one holds and condition two fails on
both aggregates, so v1.3.1 is not published and the numbers stand as they came
out.

Taking the defective rows out does to RuCoLA exactly what `results/rucola_test/`
said it would. The released checkpoint scores 0.6886 on the 1,092 clean rows,
below the 0.7372 of the majority class. This run scores 0.7546, above it. That
is the one place where the release is beaten, and it is beaten by a margin no
reading of the noise explains.

It costs 0.0072 over the whole validation split and 0.0123 over the six Russian
SuperGLUE tasks, both outside the 0.005 the rule allowed. The loss is not spread
evenly: `massive_intent` falls 0.0267 on 3,000 rows and `russe` 0.0140 on 8,505,
while `massive_scenario` and `toxicity` do not move. On the held-out wording
bench the five-candidate reading falls 0.015 over 6,000 questions, and the
jargon vocabulary alone falls 0.048.

### What this run cannot answer

It cannot separate two causes, and it was never able to. Removing the defect
means removing rows, 2,536 of 143,991, which is 1.8 per cent of the training
mixture. The drop on the other tasks may come from the labels being honest or
from the data being smaller, and this design cannot tell the two apart. The
experiment that would separate them is a run where the defective rows keep their
place and carry their true labels, and RuCoLA's test split has no true labels to
give. The question is closed by the data and not by the budget.

A second thing the run does not establish. The epoch was fixed in writing
before training, the third of four, and it is the epoch read above. The second
epoch happens to score higher on RuCoLA, 0.7637 against 0.7546, and the table
reports it. It is not read and no conclusion rests on it, because choosing it
after seeing it would be choosing by result. Its accuracy over the whole split
is 0.8014, far below the released 0.8242, so the choice would not have rescued
condition two in any case.

### What is kept, and what is not

The four snapshots are on local disk and are not published. The mixture itself
is not committed and not redistributed, which `DATASETS.md` requires; it is
reproducible from the two commands in `REPRODUCE.md` and verifiable against the
two sha256 above. The per-step training log went down with the pod.
`results/v131/train_log.txt` holds what the local watcher kept, which is every
line carrying a number or a verdict.

### The control on the instrument

| Check | Result |
|---|---|
| the rows of this run against the surviving rows of the published v1.3 dump, on task, answer index and candidate count | 26494 rows, 0 disagreements |
| the four CPU accuracies against the four the training run printed on the card | largest difference 0.0000 |
| the share of the majority class on the clean rucola rows, from `results/rucola_test/` | 0.7372 |

### The run, epoch by epoch

The temperature is the one the release's own instrument returns, `results/analysis/fit_seed2_temperatures.py`, on this run's own validation split. It carries no condition.

| Epoch | temperature | accuracy, all 26,494 rows | accuracy, rucola, 1,092 rows | accuracy, the six Russian SuperGLUE tasks, 10,157 rows |
|---|---|---|---|---|
| epoch 1 of 4 | 1.1370 | 0.7646 | 0.7491 | 0.6675 |
| epoch 2 of 4 | 1.1726 | 0.8014 | 0.7637 | 0.7438 |
| epoch 3 of 4, the epoch the rule fixed | 1.8729 | 0.8170 | 0.7546 | 0.7871 |
| epoch 4 of 4 | 3.6754 | 0.8174 | 0.7262 | 0.7978 |
| the released v1.3, on these same rows | 1.8640 | 0.8242 | 0.6886 | 0.7994 |

The released v1.3's temperature is the published one, 1.8640, from `results/seeds/FINDINGS.md`. It was fitted on the validation split the release was trained against, which held the 252 defective rows, so it is quoted here and not compared.

### The rule

| What the rule reads | n | v1.3 | v1.3.1, epoch 3 of 4 | difference | 95 per cent interval | verdict |
|---|---|---|---|---|---|---|
| rucola, the 1,092 rows that carry a real label | 1092 | 0.6886 | 0.7546 | +0.0659 | +0.0375 to +0.0943 | better than v1.3, condition one holds |
| the six Russian SuperGLUE tasks, 10,157 rows | 10157 | 0.7994 | 0.7871 | -0.0123 | -0.0204 to -0.0041 | worse by more than 0.005, condition two fails |
| the whole validation split, 26,494 rows | 26494 | 0.8242 | 0.8170 | -0.0072 | -0.0114 to -0.0029 | worse by more than 0.005, condition two fails |

* rucola, the 1,092 rows that carry a real label: only v1.3 right on 96 rows, only v1.3.1 on 168, sign test p = 1.11e-05.
* the six Russian SuperGLUE tasks, 10,157 rows: only v1.3 right on 946 rows, only v1.3.1 on 821, sign test p = 0.00317.
* the whole validation split, 26,494 rows: only v1.3 right on 1754 rows, only v1.3.1 on 1563, sign test p = 0.000968.

**The rule refuses publication.** v1.3.1 is not published, and these numbers stand as they came out.

### Accuracy by task, as description

The rule is not applied per task and no conclusion below rests on a single one. On `parus` one decision is 0.010 and on `rwsd` 0.005, which is why.

| Task | n | v1.3 | v1.3.1, epoch 3 of 4 | difference |
|---|---|---|---|---|
| danetqa | 821 | 0.7527 | 0.7442 | -0.0085 |
| georeview | 3000 | 0.6310 | 0.6223 | -0.0087 |
| massive_intent | 3000 | 0.9597 | 0.9330 | -0.0267 |
| massive_scenario | 3000 | 0.9547 | 0.9550 | +0.0003 |
| mera_qa | 245 | 0.5959 | 0.4857 | -0.1102 |
| parus | 100 | 0.5900 | 0.6300 | +0.0400 |
| rcb | 220 | 0.5091 | 0.4409 | -0.0682 |
| rucola | 1092 | 0.6886 | 0.7546 | +0.0659 |
| rureviews | 3000 | 0.7737 | 0.7697 | -0.0040 |
| russe | 8505 | 0.8240 | 0.8100 | -0.0140 |
| rwsd | 204 | 0.4804 | 0.5098 | +0.0294 |
| terra | 307 | 0.7329 | 0.7524 | +0.0195 |
| toxicity | 3000 | 0.9537 | 0.9557 | +0.0020 |

### RuCoLA against the majority class

| Model | accuracy on the 1,092 clean rows | over the majority class |
|---|---|---|
| the share of the majority class | 0.7372 | |
| the released v1.3 | 0.6886 | -0.0486 |
| v1.3.1, epoch 3 of 4 | 0.7546 | +0.0174 |

#### The wording bench, all 18 candidates

The released v1.3 is recomputed here from its own published dump, so both columns come out of this script rather than out of two tables.

| Vocabulary | n | v1.3 | v1.3.1 | difference |
|---|---|---|---|---|
| English key | 1000 | 0.900 | 0.885 | -0.015 |
| Russian word | 1000 | 0.892 | 0.896 | +0.004 |
| Russian phrase | 1000 | 0.885 | 0.884 | -0.001 |
| anglicism jargon | 1000 | 0.640 | 0.617 | -0.023 |
| department name | 1000 | 0.733 | 0.750 | +0.017 |
| descriptive sentence | 1000 | 0.878 | 0.877 | -0.001 |
| all six | 6000 | 0.821 | 0.818 | -0.003 |

Over all six vocabularies: only v1.3 right on 270 questions, only v1.3.1 on 251, sign test p = 0.43.

#### The wording bench, the five-candidate view

The same scoring read at the five candidates the bench originally asked, recovered per question by `results/analysis/vocab_bias.py`.

Checked first: cutting v1.3's eighteen logits to five returns the release's published five-candidate dump on all 6000 rows, 0 answers differing, largest logit difference 0.000107.

| Vocabulary | n | v1.3 | v1.3.1 | difference |
|---|---|---|---|---|
| English key | 1000 | 0.962 | 0.954 | -0.008 |
| Russian word | 1000 | 0.960 | 0.956 | -0.004 |
| Russian phrase | 1000 | 0.957 | 0.945 | -0.012 |
| anglicism jargon | 1000 | 0.855 | 0.807 | -0.048 |
| department name | 1000 | 0.869 | 0.864 | -0.005 |
| descriptive sentence | 1000 | 0.958 | 0.944 | -0.014 |
| all six | 6000 | 0.927 | 0.912 | -0.015 |

Over all six vocabularies: only v1.3 right on 218 questions, only v1.3.1 on 127, sign test p = 1.1e-06.

#### The first probe, five candidates

| Style | n | v1.3 | v1.3.1 | difference |
|---|---|---|---|---|
| style A | 1000 | 0.889 | 0.888 | -0.001 |
| style B | 1000 | 0.379 | 0.400 | +0.021 |

#### The second probe, 18 candidates

| Style | n | v1.3 | v1.3.1 | difference |
|---|---|---|---|---|
| style A | 1000 | 0.146 | 0.106 | -0.040 |
| style B | 1000 | 0.476 | 0.425 | -0.051 |

