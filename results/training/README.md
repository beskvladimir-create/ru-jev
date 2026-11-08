# The training run of v1.3

## Where these figures come from

`v1_3_train.log` is the standard output of the v1.3 training run, copied here
verbatim. It was produced by `scripts/train.py --config configs/v1_3.yaml` on
the rented GPU described in `ENVIRONMENT.md`, and it was captured on 2026-11-03
at 02:42 UTC, when the run finished. It is the only primary record of the
per-epoch figures: the checkpoints themselves carry only
`{"epoch": n, "temperature": ...}` in their `calibration.json` and no metrics,
and no other log of that run survived.

Table 1 of the paper and the trajectory table of the model card are transcribed
from this file, with the epochs renumbered from 1 rather than from 0.

One line of the log is in Russian, `состав полный: все 13 задач на месте`,
meaning "the composition is complete: all 13 tasks are present". It is the
builder's check that no task was silently skipped, which is the check that was
added after v1.2 trained without `rucola` and `terra`. The log is kept exactly
as it was written.

## The trajectory

| Epoch | Training loss | Validation loss | Validation accuracy |
|---|---|---|---|
| 1 | 0.6029 | 0.5054 | 0.7536 |
| 2 | 0.4336 | 0.4576 | 0.7920 |
| 3 | 0.2714 | 0.4831 | **0.8181** |
| 4 | 0.0849 | **0.8204** | 0.8139 |

Validation is the full held-out split of 26,746 decisions. The released weights
are the third epoch. In the fourth the training loss fell to 0.0849 while the
validation loss rose from 0.4831 to 0.8204, and a temperature fitted on that
epoch comes out at 3.9204, which is what overconfidence looks like as a number.
Had only the last checkpoint been kept, the released model would have been the
worse one and nothing in the artefacts would have shown it.

The three `[retry]` lines at the top are the loader failing to fetch `rucola`
through the dataset script and falling back to the parquet branch, which is why
`configs/v1_3.yaml` carries `include_rucola: true` and why the builder stops
when a task is missing.

## What is not here

The per-epoch validation loss and accuracy cannot be recomputed from this
repository. The epoch checkpoints are not published, only the third one is, and
recomputing a validation loss needs the checkpoint of each epoch. The figures in
the table above are therefore evidence of the kind a log is, not of the kind a
rerun is.
