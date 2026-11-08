# A different kind of label: the v1.3 gain does not carry over, and on abbreviations it turns negative

State as of the morning of 2026-11-05. This is a cost-free check on the laptop,
with no training. The model card, the landing page and the text of the paper
were not changed.

## Why

The model card explains the v1.3 gain by saying that the model generalises the
form of a label rather than its words. The test: take two ways of writing the
labels that are not in training, and see whether the gain carries over to a new
kind of form.

## The vocabulary, fixed before the measurement

`vocab.json`, sha256
`5f4738805d51fa044fdf092afe1bf57538301e04ae8d7fc14053366443db4652`. It was built
by a rule from the bench vocabularies that already existed, with no manual
selection, and it was not changed after the results were seen.

**Style A**: the English scenario key in capital Latin letters with the suffix
`_SCN`, like an enum constant in code. `ALARM_SCN`, `IOT_SCN`.

**Style B**: the first three letters of the Russian scenario name in capital
Cyrillic, like an internal abbreviation. `БУД`, `УМН`.

All 36 labels are distinct, with no collisions. The full table is in
`vocab.json`, and `CODEBOOK.md` glosses every label.

## The check against training

The check was made against ALL options of the training mixtures, over all
thirteen tasks, not only the intents and scenarios. The v1.3 mixture had to be
rebuilt with the current builder for this, because the local copy was out of
date; the details are in `../seeds/FINDINGS.md`.

By exact match: in style A 18 labels of 18 are clean, in style B 17 of 18. `РЕК`
is excluded, because it occurs literally among the training options in both
mixtures. The headline "clean" figure is computed by this rule, the same way the
model card gives 15 of 18 for jargon.

**A separate caveat, and it matters for reading the result.** These
vocabularies cannot be called fully independent of training. All 36 labels have
a relation to it, but in opposite directions:

| Style | Relation | Example |
|---|---|---|
| A | the label starts with a trained option | `ALARM_SCN` starts with the key `alarm` |
| B | a trained option starts with the label | `КАЛ` is the start of the word `календарь` |

A prefix will not do as a rule of exclusion: it touches all 36 labels and would
have left no headline figure at all. It does explain the difference between the
styles, see below.

## Result

The same 1,000 bench questions, the same candidate sets, the option order from
the crc32 of the text and identical for both styles and all models. Chance is
0.200.

### Style A, a code identifier

| Model | All 18 | Clean | Mean confidence | Gap | Confident errors per 1000 |
|---|---|---|---|---|---|
| ru-jev-v1.1 | 0.856 | 0.856 | 0.852 | -0.004 | 28 |
| ru-jev-v1.3 | 0.889 | 0.889 | 0.882 | -0.007 | 36 |
| laya instr A | 0.557 | 0.557 | 0.671 | +0.114 | 59 |

### Style B, a three-letter abbreviation

| Model | All 18 | Clean | Mean confidence | Gap | Confident errors per 1000 |
|---|---|---|---|---|---|
| ru-jev-v1.1 | 0.403 | 0.420 | 0.580 | +0.177 | 76 |
| ru-jev-v1.3 | 0.379 | 0.360 | 0.643 | +0.264 | 114 |
| laya instr A | 0.322 | 0.331 | 0.502 | +0.180 | 40 |

Each model uses its own temperature: for Ru-Jev the released ones, 2.2754 and
1.8640, and for Laya one, since it returns probabilities itself.

## What follows from this

**The v1.3 gain is barely visible on a new kind of form.** On style A it is
there but small: 0.889 against 0.856, plus 0.033. On the wording bench the gap
was plus 0.06 on average and plus 0.22 on jargon. So the generalisation of label
form works, but only in the direction it was taught.

**On abbreviations the recipe turns negative.** 0.379 against 0.403 for v1.1 on
all eighteen, and 0.360 against 0.420 on the clean ones. v1.3 is worse than its
predecessor here, and this is not noise: at n of about a thousand and an
accuracy of about 0.4 the standard deviation is 0.015, and a difference of 0.06
is four deviations.

**And worse on calibration.** On style B the gap for v1.3 is +0.264 against
+0.177 for v1.1, and the confident errors are 114 per thousand against 76. On an
unfamiliar kind of form v1.3 does not merely err more often, it errs
confidently. That is exactly the property the model card praises v1.3 for not
having on jargon.

**The fall is not even, it is a reshuffle.** In style B v1.3 goes to almost
nothing where v1.1 worked: `play` 0.200 to 0.007, `lists` 0.552 to 0.017,
`music` 0.650 to 0.050, `qa` 0.689 to 0.198. And conversely it wins on `alarm`,
0.065 to 0.935, and `recommendation`, 0.000 to 0.829. Both models are near
chance overall, but on different labels. Which abbreviations are read is not
decided by the recipe.

**Why style A came out easier than style B.** In style A the label contains the
trained English key in full and only adds case and a suffix. In style B three
letters are left of the trained word. This is visible inside style A as well:
its weakest label is `QA_SCN`, at 0.547 and 0.538, and that is the only case
where the hint is just two characters long. Both models lean on a matching start
of the string rather than on the meaning of the label.

**Laya loses on both styles**, 0.557 and 0.322 against our 0.889 and 0.379, and
on style A it is noticeably overconfident, a gap of +0.114 against our near
zero. But style B is a floor for all three, and ranking models there is
meaningless.

## What this measurement does not say

This is not a test of robustness to unfamiliar categories; the categories are
the same eighteen. It is a test of robustness to an unfamiliar way of writing
the same categories. And it is one vocabulary per kind, built by one rule: a
different abbreviation rule could have given different numbers.

## Files

`vocab.json` the vocabulary, `rows.jsonl` the bench rows, `overlap.json` the
report of the check against training, `scores_v11.jsonl` and `scores_v13.jsonl`
Ru-Jev's raw logits with no temperature, `scores_laya_instrA.jsonl` Laya's
probabilities as it returns them, `table.md` the summary and the per-label
breakdown.

Scripts: `build_vocab2.py` builds the vocabulary and the rows, `overlap.py`
checks them against training, `table_vocab2.py` builds the summary, `bylabel.py`
the per-label breakdown. All four are in `scripts/` and reproduce the numbers in
this file.
