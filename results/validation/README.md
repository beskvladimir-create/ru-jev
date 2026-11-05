# Validation scores of the released checkpoint

`val_v13_trainsplit.jsonl` holds the raw scores of the released v1.3 checkpoint
on all 26,746 rows of the validation split that the training run itself carved,
one JSON object per decision with the task name, the index of the correct
candidate and the raw scores before any temperature. This is the file the
released temperature and the coverage table were produced from.

## What it reproduces

    python scripts/verify_temperature.py --model <weights> \
        --scores results/validation/val_v13_trainsplit.jsonl
    python scripts/coverage.py results/validation/val_v13_trainsplit.jsonl

The first prints a refitted temperature of 1.8640 against the 1.8640 recorded in
`calibration.json`, a difference of 0.0000. The second reproduces the coverage
table of the paper to the last digit: 0.85 and 0.9666 at half coverage, 0.69 and
0.9214 at 70%, 0.54 and 0.8592 at 90%, 0.23 and 0.8234 at full coverage.

Two things have to match for that, and both are defaults in the scripts. The
half-split seed is 17, the value the release was prepared with. The temperature
search is the 301-point grid `0.2 + 7.8 i / 300`; the released 1.8640 is a point
of exactly that grid, and the narrowing search in `train.py` lands 0.005 away.

## Three accuracy figures on the same checkpoint

They are not inconsistent, they count different things, and an earlier version of
the repository README gave one of them under another's row count.

| Figure | Rows | Where it comes from |
|---|---|---|
| 0.8181 | 26,746 | the training run's own evaluation at the end of epoch 3 |
| 0.8208 | 26,746 | this standalone scoring pass over the same rows |
| 0.8234 | 13,373 | the half held out from temperature fitting, seed 17 |

The first two differ by 72 decisions out of 26,746. The trainer batches sixteen
examples and pads every candidate of the batch to its longest sequence, while the
scoring pass pads only within one decision, so the two paths are numerically
slightly different. 528 of the rows here, two percent, have less than 0.1
separating the top two candidates, which is where 72 flips sit comfortably. The
third figure is higher because it is half the rows, and it is the one the coverage
table is built on, since the temperature has to come from rows the curve is not
scored on.

## Why the mixture cannot simply be rebuilt

This dump is kept because rebuilding the data does not recover the split it was
made on. Rebuilding with the current builder and the current `datasets`, then
scoring the same released checkpoint, gives 0.8540 on 26,746 rows and refits the
temperature to 1.5222 instead of 1.8640. The reason is contamination, not a
better model: a rebuilt validation row can be a row the release trained on.

Per task, measured by `scripts/split_overlap.py` and recorded in full with the
hashes of both builds in `split_overlap.md`. A row is identified by its task and
its context string, never by the option wordings, since those are what the
builder changes between versions.

| Task | Rows in the rebuilt validation | Same rows as before | Rebuilt validation rows that were in the earlier training set |
|---|---|---|---|
| danetqa | 821 | 100.0% | 0.0% |
| parus | 100 | 100.0% | 0.0% |
| rcb | 220 | 100.0% | 0.0% |
| russe | 8,505 | 100.0% | 0.0% |
| rwsd | 204 | 100.0% | 0.0% |
| terra | 307 | 100.0% | 0.0% |
| massive_intent | 3,000 | 62.8% | 1.8% |
| massive_scenario | 3,000 | 63.2% | 2.0% |
| toxicity | 3,000 | 1.9% | 17.4% |
| rureviews | 3,000 | 5.6% | 43.8% |
| georeview | 3,000 | 5.3% | 46.1% |
| rucola | 1,344 | 9.9% | 90.1% |
| mera_qa | 245 | 9.0% | 91.0% |
| all | 26,746 | 54.1% | 17.8% |

The rows that are in neither earlier split, 28.0% overall, are rows the source
dataset revision itself moved.

An earlier version of this file gave this table from a one-off computation with a
different row identity, which reported massive at 6% leaked and mera_qa at 92%.
The numbers above come from the published script and are the ones to use.

The six tasks that reproduce exactly are the six the paper compares the two
releases on, which is why that comparison reproduces to the row. The other seven
are the ones whose option wordings are drawn per example and whose pools are
capped, and their split does not survive a change of dataset revision or of the
`datasets` version. The builder's per-task seeding keeps a wording change in one
task from moving another task's split, which it does; it cannot keep the split
fixed when the source rows underneath it move.

That the earlier build really does hold the split the release was measured on is
checked, not assumed: its 26,746 validation rows match this dump row for row on
task, correct index and candidate count, and rescoring the first 60 of them with
the released checkpoint reproduces the published scores to 0.000017. The proof is
in `split_overlap.md`.

The practical consequence: for anything measured on the model's own validation
split, use this dump. A rebuild is fine for the six comparable tasks and for the
wording bench, which is built from MASSIVE rows that are listed explicitly, and
not for the rest.

The rebuilt dump is not included. It would be 25 MB of numbers that are wrong for
the published figures and right for nothing, and the only thing worth keeping
from it is the table above.
