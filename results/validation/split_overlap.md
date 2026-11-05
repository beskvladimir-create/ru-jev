# How much of a rebuilt validation split was already trained on

Rebuilding the training mixture does not recover the validation split the
released v1.3 checkpoint was carved against. This file measures the damage, and
it is the reason `val_v13_trainsplit.jsonl` in this directory is the artifact of
record for anything measured on the model's own validation split.

Produced by:

    python scripts/split_overlap.py \
        --earlier model/data-stale-prejargon \
        --rebuilt data \
        --json results/validation/split_overlap.json

## The two builds compared

| Role | File | SHA-256 |
|---|---|---|
| earlier train | model/data-stale-prejargon/train.jsonl | d55c84078a718456727ac7701eaa05a82fae272fdc58c72b2f4a8557ff45cc4f |
| earlier val | model/data-stale-prejargon/val.jsonl | 110c8ae9d606f4e63037b840c3b9ef244f6f6f10bca67c80509d6d4f2895ae6a |
| rebuilt train | data/train.jsonl | a0073e63e33fd1e5ecbac686d0d98258b9582d19f151b88c503385b815a75e76 |
| rebuilt val | data/val.jsonl | 48f1370eea49b74d649fbc4a244b10ab873f1746cb6071aac8c876eed218a399 |

Neither build is in this repository: the mixture is assembled from thirteen
sources that keep their own licences (see `DATASETS.md`). The hashes are here so
that a rebuild can be compared against the one these numbers come from.

## The earlier build holds the split v1.3 was evaluated on

The directory is named after a stale mixture, and the name is about the option
wordings, not the split. Its validation carve is the carve the release was
measured on, and that is checked twice rather than assumed.

First, the 26,746 rows of `model/data-stale-prejargon/val.jsonl` match
`val_v13_trainsplit.jsonl` row for row, in file order, on task name, index of the
correct candidate and number of candidates: 26,746 of 26,746. The rebuilt
validation file matches the same dump on only 15,078 of 26,746, so the check
discriminates.

Second, scoring the first 60 of those rows with the released checkpoint
reproduces the published scores to 0.000017, which is the rounding of the dump
itself. The rows are the same rows, text and candidates included, not merely the
same shape.

## Result

A row is identified by its task and its context string, never by the option
wordings, because the option wordings are what the builder changes between
versions. Two rows with the same context are the same underlying example
whatever its candidates are called.

`kept` is the share of rebuilt validation rows whose context was also in the
earlier validation split. `leaked` is the share whose context was in the earlier
TRAINING split: those rows are contaminated, and scoring the release on them
reports accuracy on its own training data. `neither` is the share that was in no
earlier split at all, which happens when the source dataset revision moved
underneath the builder.

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
1.5222 instead of 1.8640. The second pair of numbers is contamination, not a
better model.

The rebuilt scores are not published. They are 25 MB of numbers that are wrong
for the released figures and right for nothing; the table above is the only thing
worth keeping from that run.
