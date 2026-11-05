# How much of a rebuilt validation split was already trained on

Two episodes in this project turned on the same question: a validation split was
rebuilt, a release scored higher on it, and the gain was contamination rather
than quality. Both intersections are below, from the same script, so the two are
not confused again.

A row is identified by its task and its context string, never by the option
wordings, because the option wordings are what the builder changes between
versions. Two rows with the same context are the same underlying example
whatever its candidates are called.

`kept` is the share of rebuilt validation rows whose context was also in the
earlier validation split. `leaked` is the share whose context was in the earlier
TRAINING split: scoring the earlier release on those rows measures it on its own
training data. `neither` is the share that was in no earlier split at all, which
happens when the source dataset revision moved underneath the builder.

## The builds

None of them is in this repository: the mixture is assembled from thirteen
sources that keep their own licences (see `DATASETS.md`). The hashes are here so
that a rebuild can be compared against the ones these numbers come from.

| Build | File | SHA-256 |
|---|---|---|
| v1.1 train | model/data-v1.1/train.jsonl | acb5d1631a15b702895d6ea1ee2a11d7fe62974973117e9ec7947c585872a8f0 |
| v1.1 val | model/data-v1.1/val.jsonl | 5af888542515ca969fffdec8dbf30bed796ff7190f20d946c8ea29c460182903 |
| v1.3-era train | model/data-stale-prejargon/train.jsonl | d55c84078a718456727ac7701eaa05a82fae272fdc58c72b2f4a8557ff45cc4f |
| v1.3-era val | model/data-stale-prejargon/val.jsonl | 110c8ae9d606f4e63037b840c3b9ef244f6f6f10bca67c80509d6d4f2895ae6a |
| rebuilt train | data/train.jsonl | a0073e63e33fd1e5ecbac686d0d98258b9582d19f151b88c503385b815a75e76 |
| rebuilt val | data/val.jsonl | 48f1370eea49b74d649fbc4a244b10ab873f1746cb6071aac8c876eed218a399 |

The middle pair is named after a stale mixture, and the name is about option
wordings, not the split. Its validation carve is the carve the released v1.3 was
measured on, which is checked rather than assumed at the end of this file.

## Episode one: splits coupled through a shared random stream

This is the intersection behind the first hazard in the paper. The builder carved
each task's validation slice from the module-level random generator, so adding
label styles to one task shifted every later task's partition, and v1.1's
training rows ended up inside the next build's validation split.

    python scripts/split_overlap.py \
        --earlier model/data-v1.1 \
        --rebuilt model/data-stale-prejargon \
        --json results/validation/split_overlap_v11.json

| task | rows | kept | leaked into validation | neither |
|---|---|---|---|---|
| danetqa | 821 | 100.0% | 0.0% | 0.0% |
| parus | 100 | 100.0% | 0.0% | 0.0% |
| rcb | 220 | 100.0% | 0.0% | 0.0% |
| rureviews | 3000 | 100.0% | 0.0% | 0.0% |
| russe | 8505 | 100.0% | 0.0% | 0.0% |
| rwsd | 204 | 100.0% | 0.0% | 0.0% |
| terra | 307 | 100.0% | 0.0% | 0.0% |
| massive_intent | 3000 | 63.0% | 1.8% | 35.2% |
| massive_scenario | 3000 | 63.1% | 1.9% | 35.1% |
| toxicity | 3000 | 2.4% | 21.6% | 76.0% |
| georeview | 3000 | 5.3% | 47.2% | 47.5% |
| mera_qa | 245 | 10.6% | 89.4% | 0.0% |
| rucola | 1344 | 9.7% | 90.3% | 0.0% |
| all | 26746 | 64.8% | 13.5% | 21.7% |

The paper reports 90.3% for acceptability, 89.8% for reasoning, 47.2% for review
rating and 21.7% for toxicity. Acceptability (`rucola`) and review rating
(`georeview`) match to the decimal. Reasoning (`mera_qa`) gives 89.4% here and
toxicity 21.6%, 0.4 and 0.1 away from the figures in the paper, which were
computed before this script existed and matched rows by a slightly different
identity. The numbers above are the ones this script produces.

Note what the leak did to a reported figure: v1.1's accuracy on acceptability
read 0.909 on those rows and 0.650 on genuinely held-out ones.

## Episode two: a rebuild does not recover the split v1.3 was measured on

    python scripts/split_overlap.py \
        --earlier model/data-stale-prejargon \
        --rebuilt data \
        --json results/validation/split_overlap_v13.json

| task | rows | kept | leaked into validation | neither |
|---|---|---|---|---|
| danetqa | 821 | 100.0% | 0.0% | 0.0% |
| parus | 100 | 100.0% | 0.0% | 0.0% |
| rcb | 220 | 100.0% | 0.0% | 0.0% |
| russe | 8505 | 100.0% | 0.0% | 0.0% |
| rwsd | 204 | 100.0% | 0.0% | 0.0% |
| terra | 307 | 100.0% | 0.0% | 0.0% |
| massive_intent | 3000 | 62.8% | 1.8% | 35.4% |
| massive_scenario | 3000 | 63.2% | 2.0% | 34.7% |
| toxicity | 3000 | 1.9% | 17.4% | 80.7% |
| rureviews | 3000 | 5.6% | 43.8% | 50.6% |
| georeview | 3000 | 5.3% | 46.1% | 48.6% |
| rucola | 1344 | 9.9% | 90.1% | 0.0% |
| mera_qa | 245 | 9.0% | 91.0% | 0.0% |
| all | 26746 | 54.1% | 17.8% | 28.0% |

Six tasks reproduce their carve exactly and leak nothing. They are the six the
paper compares the two releases on, which is why that comparison reproduces to
the row. The other seven draw option wordings per example and cap their pools,
and their carve does not survive a change of dataset revision or of the
`datasets` version. The builder's per-task seeding keeps a wording change in one
task from moving another task's split, which it does; it cannot keep a split
fixed when the source rows underneath it move.

The consequence in one number: the released checkpoint scores 0.8208 on its own
validation split and 0.8540 on a rebuilt one, and refits its temperature to
1.5222 instead of 1.8640. The second pair is contamination, not a better model.

The rebuilt scores are not published. They are 25 MB of numbers that are wrong
for the released figures and right for nothing; the table above is the only thing
worth keeping from that run.

## Why the earlier build is the split v1.3 was measured on

Checked twice rather than assumed.

The 26,746 rows of `model/data-stale-prejargon/val.jsonl` match
`val_v13_trainsplit.jsonl` row for row, in file order, on task name, index of the
correct candidate and number of candidates: 26,746 of 26,746. The rebuilt
validation file matches the same dump on only 15,078, so the check discriminates.

Scoring the first 60 of those rows with the released checkpoint then reproduces
the published scores to 0.000017, which is the rounding of the dump itself. The
rows are the same rows, text and candidates included, not merely the same shape.
