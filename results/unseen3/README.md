# Two further unseen datasets: the rule, written before the data

This file is committed before any row is downloaded and before anything is
scored. Nothing below was chosen after seeing a number. The commit that
introduces it carries the time.

## Why

After per-option biases are fitted, the ranking of the ten distinct checkpoints
of the ladder on the news rubrics and on the CEDR emotions does not agree: the
Spearman correlation between the two datasets is about -0.17 with biases and
-0.16 without them. Two rankings that disagree cannot both describe the
checkpoints, and with two datasets there is no way to tell which of them, if
either, is the one that travels. Two further datasets are added: one far from
the training tasks and one close to them.

## The datasets

**A, scientific rubrics.** `ai-forever/ru-scibench-grnti-classification`, GRNTI
rubrics of scientific abstracts, or its MTEB mirror if the first source has no
usable test split. Far from the training mixture: no task in it is about
scientific classification.

**B, sentiment of film reviews.** `mteb/KinopoiskClassification`, or its first
source. Close to the training mixture, which holds two sentiment tasks,
`RuReviewsClassification` and `GeoreviewClassification`.

Each dataset has to pass the checks below before it is used at all. A dataset
that fails is not used, and the reason is written here. Failing one does not
stop the other.

**B was not used.** Its MTEB mirror, `mteb/KinopoiskClassification`, carries
`license: unknown` in its card metadata, and the rule above treats an unclear
licence as a failure. Its first source,
`ai-forever/kinopoisk-sentiment-classification`, is MIT and passes on the
licence, but it gives the three classes only in English, `Bad`, `Neutral` and
`Good`, and the rule stops on classes that are given only in English. Both
failures were found before anything was scored. The natural Russian rendering of
those classes would be `негативный`, `нейтральный` and `позитивный`, which are
exactly the option strings the training mixture uses for `RuReviewsClassification`,
so translating them would have replaced one failure with another: the labels
would no longer be unseen. The dataset is therefore absent from the results and
no verdict is given for it.

## The checks a dataset has to pass

1. **Licence.** Commercial use has to be permitted by the card of the dataset
   and by its first source. If it is not permitted, or if it is unclear, the
   dataset is not used and the reason is recorded.
2. **No text overlap with training.** Every selected row is compared with the
   texts of the training mixture, by exact match after normalising whitespace
   and case. The sources of the mixture are those listed in `DATASETS.md`.
3. **No label overlap with training.** Every class name is compared with every
   option string of the training mixture, by exact match and by prefix in both
   directions, the way `scripts/overlap.py` does it.
4. **Labels as the model reads them.** The class names are taken in the form
   the dataset gives them, in Russian. If the classes are given only as codes or
   only in English, the work stops and the options are put to the author.

## How the rows are chosen

- The test split only.
- Seed 20261105, stratified by label.
- 1,000 rows per dataset. If the test split holds fewer, every row is taken, and
  if that is fewer than 300 rows the dataset is not used.
- If a dataset has more than ten classes, the ten most frequent classes of the
  test split are kept. Each question then offers five candidates: the correct
  one and four drawn at random from the remaining nine, with the same seed, in a
  random order.
- If a dataset has ten classes or fewer, every class is offered as a candidate.

## The checkpoints

Ten distinct checkpoints of the ladder. `v1.3` and `v1.3-ep3of4` are the same
weights read from two directories; they are counted once, as `v1.3-ep3of4`, and
`v1.3` is scored as well only as a control on the scoring path.

v1.1, `seed-base-s2`, v1.2, `seed-aug-s2`, `seed-aug-s2-ep3`, `seed-aug-s3`,
v1.3-ep1of4, v1.3-ep2of4, v1.3-ep3of4, v1.3-ep4of4.

## What is computed

1. For each dataset, the accuracy of each checkpoint as published and out of
   fold once that checkpoint is allowed its own additive bias per option, by the
   functions of `results/analysis/unseen_bias.py`: 25 half-splits seeded 0 to
   24, each used twice.
2. The Spearman rank correlation of those ten checkpoints between every pair of
   the four datasets, the news rubrics, the emotions, A and B, separately with
   biases and without them.
3. The same six pairs of checkpoints as in `results/unseen/ladder_bias.md`, with
   bootstrap intervals over rows and over labels, seed 20261105, 5,000 draws.
4. The spread between the two published runs of the v1.1 recipe, the released
   v1.1 and `seed-base-s2`, and a mark on every cell where the difference
   between v1.3 and v1.1 is no larger than that spread.

On the spread: the third run of that recipe, `seed-base-s3`, has no surviving
weights, only its bench logits, so it cannot be scored on a new dataset. The
spread on the new datasets is therefore over two runs, not three, and it is
written as such everywhere it appears.

## The reading rule

**A ranking of the ten checkpoints, the one from the news rubrics or the one
from the emotions, survives a new dataset if its Spearman correlation with that
dataset, with biases, is at least 0.5. Below 0.5 it does not survive.**

The verdict is given separately for A and for B.

At n = 10 this is a description and not a statistical test. No p-value is
attached to it and none should be read into it.
