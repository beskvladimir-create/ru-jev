# The unseen datasets: no right to say "we are more accurate", a narrow claim about confidence

State as of the night of 2026-11-05. This is block B.2 of the plan. The model
card, the landing page and the text of the paper were not changed.

## Why this block exists

The wording bench is built on MASSIVE, whose training split has been in our
mixture since version 1. We are far ahead of Laya there, but that is home
ground, and the gap cannot be presented as superiority. Here are two datasets
neither model has seen: not our mixture of thirteen tasks, and, by its own card,
not Laya's training either.

## The datasets and their licences

| Dataset | Source | Rows | Candidates | Licence |
|---|---|---|---|---|
| рубрики новостей (news rubrics) | ai-forever/headline-classification, the same corpus as mteb/HeadlineClassification | 1000 | 6 | mit |
| эмоции (CEDR emotions) | sagteam/cedr_v1, the same corpus as mteb/CEDRClassification | 1000 | 5 | apache-2.0 |

**A caveat about the licences that has to stay in the text.** The plan required
the licences to be checked in a file. There is no licence file in any of the
four repositories, neither in the mteb mirrors nor in the sources. The licence
is declared only by the `license` field in the README metadata: `mit` and
`apache-2.0`. That is a statement by the repository owner, not a separate
licence file. It is not a legal opinion.

The category names are taken from the source rather than invented. For the
rubrics they are the `label_text` field, and the mapping is unambiguous: 0
спорт, 1 происшествия, 2 политика, 3 наука, 4 культура, 5 экономика. For the
emotions it is `ClassLabel(names=['joy','sadness','surprise','fear','anger'])`
in `sagteam/cedr_v1`, and the first row there matches the first row of the mteb
mirror word for word, so the order of the indices is established rather than
guessed. The Russian words used are радость, грусть, удивление, страх, злость.

Only rows with exactly one label are taken from the emotions set. The set is
multi-label, and multi-label does not reduce to choosing one option. An empty
label means no emotion, a separate class that is not among the five.

The option order is deterministic, from the crc32 of the row text. Ru-Jev is
indifferent to it, because it scores each pair in a separate pass. Laya reads
all options in one pass, and for it the order is part of the input, so it has to
be the same for both. The rows of the dataset: sha256
`dd50deccb8be40dde7453dfdf3ae671ba917d5168c5607bb0962e881e5c38569`.

## Result

### News rubrics, 1,000 rows, 6 candidates, chance 0.167

| Model | Accuracy | Mean confidence | Gap | ECE as is | Fitted T | ECE after its own T |
|---|---|---|---|---|---|---|
| v1.1 | 0.652 | 0.680 | +0.028 | 0.2037 | 2.759 | 0.0674 |
| v1.3 | 0.568 | 0.663 | +0.095 | 0.2452 | 2.462 | 0.0599 |
| laya | 0.610 | 0.852 | +0.242 | 0.2422 | 2.237 | 0.0545 |

### CEDR emotions, 1,000 rows, 5 candidates, chance 0.200

| Model | Accuracy | Mean confidence | Gap | ECE as is | Fitted T | ECE after its own T |
|---|---|---|---|---|---|---|
| v1.1 | 0.455 | 0.540 | +0.085 | 0.2868 | 3.430 | 0.0564 |
| v1.3 | 0.511 | 0.447 | -0.064 | 0.0815 | 1.546 | 0.0587 |
| laya | 0.605 | 0.661 | +0.056 | 0.0708 | 1.381 | 0.0766 |

For the gap and the mean confidence each model uses its own temperature: for
Ru-Jev the released ones, 1.8640 for v1.3 and 2.2754 for v1.1, and for Laya its
own, since it returns probabilities itself. The ECE and the fitted T are
averages over 25 random half-splits, fitted on one half and measured on the
other, which is the method the model card uses.

The fitted temperature and the residual ECE in these two tables were refreshed
on 2026-11-09 with the current `scripts/unseen_table.py`, which takes its
half-splits from `calib.py` with the recorded `SPLIT_SEED = 20261105`. Accuracy,
mean confidence, the gap and the raw ECE did not move. `table.md` in this
directory lists the old and the new value of every figure that did.

## What can be said from this and what cannot

**There is nothing to say about accuracy, and that has to be accepted.** On the
rubrics v1.1 leads (0.652), Laya is second (0.610) and v1.3 is last (0.568). On
the emotions Laya leads (0.605), v1.3 is second (0.511) and v1.1 is last
(0.455). No model is better than another on both datasets. Off home ground we
have no advantage in accuracy.

**There is a claim about confidence, but it is narrow.** On the rubrics Laya
reports a mean confidence of 0.852 at an accuracy of 0.610, a gap of +0.242. On
the same dataset ours are +0.028 and +0.095. That is the same picture as on the
home bench, where Laya was at +0.23, so it is a property of the model rather
than of the dataset. On the emotions, though, the gap disappears: Laya is at
+0.056, and v1.3 there is underconfident, at -0.064. So the sentence "Laya is
overconfident" is wrong in general, and the claim holds only for a particular
dataset.

**After calibration nobody is better.** With each model's own fitted
temperature the ECE of all three is in one corridor: 0.055 against 0.057 against
0.066 on the rubrics, and 0.054 against 0.054 against 0.081 on the emotions. The
residual calibration is therefore not our advantage. Our advantage is only in
the raw scores before a temperature, and only on one of the two datasets.

**Latency, and it runs against us.** The numbers in this paragraph were wrong
and were corrected on 2026-11-05 from a clean measurement, see `../fix18/AUDIT.md`,
section 4. The earlier claim of "1,000 decisions in 20 seconds against 2,000 in
about four minutes" came from runs that shared the processor with other work, at
different candidate counts, and with model loading included. A clean measurement
on one and the same M4, median of 25 runs: at two candidates 18.5 ms against
15.5, at five 32.8 against 17.8, at eighteen 73.8 against 30.5. So Ru-Jev is 1.2
times slower at two candidates and 2.4 times slower at eighteen, not slower by
an order of magnitude. The reason is structural: Laya reads all options in one
pass, Ru-Jev makes a separate pass for each (text, option) pair. Our price for
being independent of the option order is a multiple of the time per decision
when there are many candidates. No advantage in speed can be claimed.

## Files

`unseen_rows.jsonl` holds the rows themselves, with their options and answers.
`unseen_v11.jsonl` and `unseen_v13.jsonl` hold Ru-Jev's raw logits with no
temperature applied. `unseen_laya_headline.jsonl` and `unseen_laya_cedr.jsonl`
hold Laya's probabilities as it returns them; the rounding to four decimal
places is its own. `table.md` is the summary, rebuilt by `unseen_table.py`.

The build and scoring scripts were one-off scripts at the time:
`build_unseen.py`, `score_one.py`, `score_laya.py`, `unseen_table.py`. They are
now in `scripts/` in this repository, where `score_one.py` is published under
its later name `score_rujev.py`.
