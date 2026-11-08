# vocab3: by the rule recorded in advance the transfer to slang is confirmed, but the gain does not rest on robustness

State as of the morning of 2026-11-05. A cost-free measurement on the laptop,
with no training. The reading rule was recorded before the figures, see
`README.md`, 08:53:38 +04, sha256
`0455e9ad86137008878dac2b7a65a1f14e65b2ceae057b6c56770a8ecc15d3ef`. The model
card, the landing page and the text of the paper were not changed.

## The setting, and why these numbers cannot be placed next to the model card

The same 1,000 bench questions, but with **all 18 categories offered in every
question** rather than 5 of 18 as in the model card and in vocab2. Chance is
0.056 against 0.200. The task is harder, so the absolute numbers here are
comparable only with each other. Comparing v1.3 with v1.1 inside vocab3 is
valid: both were scored in the same setting, on the same rows, with the same
option order.

The vocabulary was supplied verbatim, `vocab.json`, sha256 `ae8b9463...`. Style
B is slang, style A is codes.

## The main slice: style B, strict, 14 labels, 773 questions

| Model | Accuracy | Mean confidence | Gap | Confident errors per 1000 |
|---|---|---|---|---|
| ru-jev-v1.1 | 0.369 | 0.624 | +0.255 | 151 |
| ru-jev-v1.3 | 0.419 | 0.694 | +0.275 | 191 |
| laya instr A | 0.171 | 0.588 | +0.417 | 203 |

McNemar: the difference is +0.050, with 106 questions right for v1.1 only, 145
right for v1.3 only, 251 disagreements, p = 0.016.

**By the recorded rule this is the first branch: v1.3 is higher and p < 0.05, so
the robustness carries over to slang.** The rule is kept and the conclusion is
recorded.

## But the gain does not rest on what the word "robustness" suggests

Three things that have to be read together with the headline figure.

**This is a reshuffle, not a lift.** 251 questions of 773 disagree, that is a
third of the slice, and the movement goes both ways in large pieces. v1.3 goes
to almost nothing where v1.1 worked: `перечни` 0.534 to 0.086, `сводка` 0.846 to
0.436, `ящик` 0.849 to 0.616, `пластинки` 0.550 to 0.350. And it wins heavily on
others: `справочное бюро` 0.000 to 0.585, `синоптик` 0.140 to 0.649, `болталка`
0.074 to 0.353, `ежедневник` 0.291 to 0.488. The final +0.050 is what is left of
shifts in opposite directions, not an even gain. Robustness as a property would
mean that nothing got worse anywhere. Things did, and by a lot.

**Calibration moves the other way.** On the main slice the gap for v1.3 is
+0.275 against +0.255 for v1.1, and the confident errors are 191 per thousand
against 151. So v1.3 is more accurate and at the same time errs confidently more
often. That is exactly the property the model card praises v1.3 for not having
on jargon, and here it does not hold.

**The lower bound settles nothing.** Style A, strict slice: v1.1 0.018, v1.3
0.061, p = 0.0002. Formally v1.3 is higher and significantly so, but chance is
0.056, so v1.3 is exactly at the level of a coin and v1.1 is below it. A
significant lead on the floor is not a skill. Codes of this kind are read by
neither release, and by the rule the lower bound gives no right to any claim.

## All slices

### Style A, codes

| Slice | Labels | Questions | v1.1 | v1.3 | Difference | b | c | p |
|---|---|---|---|---|---|---|---|---|
| all 18 | 18 | 1000 | 0.104 | 0.146 | +0.042 | 35 | 77 | 9.0e-05 |
| clean by exact match | 17 | 865 | 0.120 | 0.169 | +0.049 | 35 | 77 | 9.0e-05 |
| strict | 9 | 512 | 0.018 | 0.061 | +0.043 | 6 | 28 | 1.95e-04 |

| Slice | Model | Gap | Confident errors per 1000 |
|---|---|---|---|
| all 18 | v1.1 | +0.257 | 60 |
| all 18 | v1.3 | +0.265 | 39 |
| all 18 | laya | +0.357 | 83 |
| clean | v1.1 | +0.244 | 60 |
| clean | v1.3 | +0.243 | 43 |
| clean | laya | +0.348 | 81 |
| strict | v1.1 | +0.335 | 84 |
| strict | v1.3 | +0.266 | 31 |
| strict | laya | +0.341 | 88 |

The p-value on the "all 18" and "clean" slices is the same, and not by accident:
`МЕД`, the only label with an exact match, is never guessed by either release on
any of its 135 questions, so removing it changes the number of questions and not
a single disagreement.

### Style B, slang

| Slice | Labels | Questions | v1.1 | v1.3 | Difference | b | c | p |
|---|---|---|---|---|---|---|---|---|
| all 18 | 18 | 1000 | 0.428 | 0.476 | +0.048 | 141 | 189 | 0.0096 |
| clean by exact match | 18 | 1000 | 0.428 | 0.476 | +0.048 | 141 | 189 | 0.0096 |
| strict | 14 | 773 | 0.369 | 0.419 | +0.050 | 106 | 145 | 0.016 |

| Slice | Model | Gap | Confident errors per 1000 |
|---|---|---|---|
| all 18 | v1.1 | +0.224 | 130 |
| all 18 | v1.3 | +0.207 | 154 |
| all 18 | laya | +0.361 | 181 |
| strict | v1.1 | +0.255 | 151 |
| strict | v1.3 | +0.275 | 191 |
| strict | laya | +0.417 | 203 |

The "all 18" and "clean" slices are identical row for row: style B has no exact
match with training at all.

## About Laya

It loses on both styles and on every slice: 0.171 against our 0.369 and 0.419 on
the main slice. And it stays the most overconfident, with a gap of +0.417. This
agrees with the home bench and with the unseen news rubrics, but not with the
emotions, where its overconfidence disappeared. The claim about Laya being
overconfident does not hold everywhere, see `../unseen/FINDINGS.md`.

## What this measurement does not say

The categories are the same eighteen; only the way they are written changes.
This is a test of robustness to unfamiliar wording, not to unfamiliar
categories. There is one vocabulary per style: a different set of slang could
have given different numbers, and the spread by label inside a slice shows that
directly.

A comparison with the jargon gap from the model card is impossible here: there a
question offers 5 options out of 18, here it offers 18, and the number of
candidates changes everything.

## Files

`vocab.json` the vocabulary, `rows.jsonl` the rows, `overlap.json` the report of
the check against training, `scores_v11.jsonl` and `scores_v13.jsonl` Ru-Jev's
raw logits with no temperature, `scores_laya_instrA.jsonl` Laya's probabilities
as it returns them, `table.md` the summary with the slices, the McNemar counts
and the per-label breakdown, `README.md` the reading rule.

Scripts: `build_vocab3.py` builds the rows, `overlap.py` checks them against
training, `table_vocab3.py` computes the slices and McNemar, `bylabel.py` the
per-label breakdown. All of them reproduce the numbers in this file.
