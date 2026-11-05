# Seed-variation runs

Five extra training runs made to estimate how much of the difference between the
two releases is seed noise. They are published because the paper quotes numbers
from them, not because they answer the question they were built for. Four of
them do not: they were trained on the wrong mixture, and this directory is the
record of that.

## What each run is

| Run | Recipe | Trained on | Epoch scored |
|---|---|---|---|
| seed-base-s2 | v1.1 | `model/data-v1.1`, the v1.1 mixture | last |
| seed-base-s3 | v1.1 | `model/data-v1.1`, the v1.1 mixture | last |
| seed-aug-s2 | v1.3 schedule | a stale pre-jargon mixture | last |
| seed-aug-s2-ep3 | v1.3 schedule | a stale pre-jargon mixture | third |
| seed-aug-s3 | v1.3 schedule | a stale pre-jargon mixture | last |

The two base runs are valid second and third seeds of the v1.1 recipe. The three
augmented runs are not seeds of v1.3. The mixture they were given was built on
2026-11-02 at 19:49, eight hours before the commit that added the two
transliterated label styles which are the whole content of v1.3, so they were
trained without the lesson that was being tested. Row counts could not catch it:
the new styles change option strings inside a row, not the number of rows. What
caught it was counting distinct option strings per task, 120 for intents in the
stale copy against 198 after rebuilding.

## Accuracy on the wording bench, five candidates per question

Argmax over raw logits, so none of this depends on a temperature.

| Run | English key | Russian word | Russian phrase | jargon | department names | descriptive |
|---|---|---|---|---|---|---|
| seed-base-s2 | 0.951 | 0.953 | 0.902 | **0.658** | 0.828 | 0.939 |
| seed-base-s3 | 0.960 | 0.957 | 0.893 | **0.622** | 0.828 | 0.951 |
| seed-aug-s2 | 0.947 | 0.958 | 0.957 | 0.619 | 0.883 | 0.952 |
| seed-aug-s2-ep3 | 0.955 | 0.958 | 0.950 | 0.611 | 0.879 | 0.954 |
| seed-aug-s3 | 0.959 | 0.959 | 0.950 | 0.588 | 0.885 | 0.952 |
| released v1.1 | 0.957 | 0.956 | 0.881 | 0.633 | 0.835 | 0.939 |
| released v1.3 | 0.962 | 0.960 | 0.957 | 0.855 | 0.869 | 0.958 |

On the clean fifteen labels of the jargon column, the three whose generated form
coincides with an ordinary Russian word removed, the two base runs give 0.645 and
0.599 against 0.624 for the released v1.1.

## What the numbers support

**Seed spread of the v1.1 recipe.** Two replicates, 0.658 and 0.622 on jargon
with all labels, 0.645 and 0.599 on the clean slice. The released v1.1 sits
between them in both cases, at 0.633 and 0.624. The spread is 0.036 and 0.046.
The v1.3 gain over v1.1 on the same column is 0.222 and 0.214, five to six times
that spread, which is the only thing these runs are evidence for.

**Nothing about v1.3.** No valid replicate of v1.3 exists. The augmented runs
land in the v1.1 and v1.2 band, which is what a run without the jargon lesson is
expected to do, and says nothing about the stability of 0.855.

## Row format

`<run>_bench.jsonl`, 6000 rows each: the 1000 bench questions under each of the
six label vocabularies. One JSON object per question with the raw logits, the
model's pick, the index of the correct candidate, the vocabulary name, the gold
label and whether its wording is clean of the training mixture. No temperature is
applied anywhere in these files.

The per-row validation scores of these runs are not published. They are scores of
checkpoints trained on a mixture that no release was built from, and nothing in
the paper is measured on them.

`НАЙДЕНО.md` is the working record of the mistake, in Russian, kept as written.
