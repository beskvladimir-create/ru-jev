# fix18: audit of the bench against training, dates, toxicity, calibration

State as of the morning of 2026-11-05. Inference on the laptop only, no
training. The model card was changed in exactly two places, items 9 and 11
below. Nothing else in it was touched.

Every check is made against REBUILT mixtures: `code/data/train.jsonl` was built
with the current builder and carries four styles of intent label, as v1.3 did.
The stale copy that the seed runs were trained on by mistake is kept separately
in `model/data-stale-prejargon/`. The details are in `../seeds/FINDINGS.md`.

## 2. The six bench vocabularies against the training mixtures

The main point first. **Two of the main bench vocabularies are literally
training styles.** `scripts/robustness.py` imports them from the builder:

```
from build_data import SCENARIO_PHRASE, SCENARIO_RU, jargon_translit, load_any
```

A check by value confirms complete identity:

| Bench vocabulary | Training style | Matches |
|---|---|---|
| русское слово | `SCENARIO_RU` | 18 of 18 |
| русская фраза | `SCENARIO_PHRASE` | 18 of 18 |

Next, which of them are in which mixture, as the label of the correct answer and
as a distractor, in the `massive_scenario` rows:

| Bench vocabulary | v1.3, correct | v1.3, distractor | v1.1, correct | v1.1, distractor |
|---|---|---|---|---|
| английский ключ | 18/18 | 18/18 | 18/18 | 18/18 |
| русское слово | 18/18 | 18/18 | 18/18 | 18/18 |
| русская фраза | 18/18 | 18/18 | **1/18** | **1/18** |
| жаргон | 0/18 | 0/18 | 0/18 | 0/18 |
| названия служб | 0/18 | 0/18 | 0/18 | 0/18 |
| описание запроса | 0/18 | 0/18 | 0/18 | 0/18 |

From this follows something that has to be held to in every claim.

**Three of the six bench columns are not a held-out test for either release.**
The English key and the Russian word are in the training of both releases in
full, both as correct labels and as distractors. The figures 0.956 and 0.960 on
the Russian word, and 0.957 and 0.962 on the English key, are recall of a
trained wording, not generalisation.

**The "русская фраза" column compares different things.** For v1.3 all 18
phrases are in training; for v1.1 one of 18 is. So the pair 0.881 against 0.957
is v1.1 on a wording it did not train on against v1.3 on a wording it did. The
gain of +0.076 here is expected by construction and measures nothing about
robustness.

**Only three columns are honestly held out for both releases:** jargon,
department names and the descriptive sentence. The corresponding pairs are 0.633
against 0.855, 0.835 against 0.869, and 0.939 against 0.958. Only these speak
about generalisation to an unfamiliar wording.

Prefix relations were counted in both directions, as in vocab2 and vocab3, and
the report is in `overlap_bench.json`.

## 3. Toxicity 0.722 against 0.954: yes, the renamed labels were in v1.3 training

The answer is **yes**, with per-row evidence from the mixture.

The builder holds five wordings of toxicity, in `build_data.py`:

| Style | Labels |
|---|---|
| 0 | токсичный / нетоксичный |
| 1 | оскорбительный / вежливый |
| 2 | хамство / нормальная речь |
| 3 | грубый / спокойный |
| 4 | toxic / clean |

How many rows of each style are in training, out of the 27,000 toxicity rows:

| Style | v1.3 | v1.1 |
|---|---|---|
| токсичный / нетоксичный | 5423 | **27000** |
| оскорбительный / вежливый | 5311 | 0 |
| хамство / нормальная речь | 5473 | 0 |
| грубый / спокойный | 5305 | 0 |
| toxic / clean | 5488 | 0 |

So v1.3 trained on all five wordings, including every "renamed" one, and v1.1 on
one only. The pair 0.722 against 0.954 is v1.1 on a wording it had not seen
against v1.3 on a wording it trained on.

The model card already withdraws this figure, at lines 140 to 145: "It also read
toxicity 0.722 against 0.954 as a gain in capability, when v1.1 scores 0.961 on
toxicity against the wordings it trained on". The withdrawal is correct, and it
now has per-row evidence under it rather than reasoning alone.

## 6. Dates: the test vocabulary existed before the transliteration rules

The exact answer: **earlier by 6 hours and 34 minutes**.

| What | Commit | Time |
|---|---|---|
| the `SCENARIO_JARGON` vocabulary in `scripts/robustness.py` | `54799b6` | 2026-11-02 21:08:01 +0400 |
| the `jargon_translit` and `jargon_clip` rules in `scripts/build_data.py` | `96fb67f` | 2026-11-03 03:42:22 +0400 |

Checked two ways. By content: `git log -S 'вэзер'` returns the single commit
`54799b6`, so the jargon labels were introduced there and never appeared again.
By diff: `54799b6..96fb67f` over `robustness.py` contains not one line of the
vocabulary, so the commit that added the rules only added the computation of the
leaked and clean subsets to that file and did not touch the vocabulary itself.

What can be claimed from this and what cannot. It can be claimed that the test
vocabulary was fixed before the rules were written, so the vocabulary was not
tuned to the rules. It cannot be claimed that the rules were written without
sight of the vocabulary: it had been in the repository for six and a half hours
by the time they were written. The model card discloses this anyway, by
reporting that three labels of eighteen coincided and by giving a separate
figure for the other fifteen.

## 8. Calibration with a recorded split seed

The cause of the irreproducibility was found. In `aggregate.py` and
`unseen_table.py` the list of indices was shuffled IN PLACE and never reset
between splits, so split number s was applied to the permutation left by s-1.
The numbers are deterministic even so, but they depend on the order of the rows
in the file.

The new `calib.py` resets the indices on every split, derives each split's seed
from a recorded `SPLIT_SEED = 20261105`, and first puts the rows in a canonical
order by the sha256 of their key, so the order of rows in the file cannot affect
the result at all.

The shift caused by the fix is small and overturns no published claim:

| Quantity | As in the model card | After the fix |
|---|---|---|
| whole bench, v1.1, ECE as is | 0.0867 | 0.0867 |
| whole bench, v1.1, fitted T | 3.009 | 2.9955 |
| whole bench, v1.1, ECE after its own T | 0.0171 | 0.0182 |

The raw ECE does not depend on the splits and matched exactly. The temperature
moved by 0.014 and the residual ECE by 0.001.

The full tables, for the whole bench, for the six vocabularies and for the
unseen datasets, are in `calib.md`, with the figures in `calib_bench.json` and
`calib_unseen.json`.

## 9 and 11. Changes to the model card

Made locally, in `model/ru-jev-v1.3/README.md` and in the template
`README.tmpl.md`, identically in both.

Licence. It read "MIT, weights and code." It now reads that MIT covers the code
and the weights, and that the data keep the licences of their sources, listed
per task in `DATASETS.md`.

The head. `head.pt` was converted to `head.safetensors` for v1.1 and for
v1.3/ship, and the contents were verified by reading them back. The example in
the card now reads `head.safetensors` through `safetensors.torch.load_file`
instead of `torch.load`.

| File | sha256 |
|---|---|
| `model/ru-jev-v1.1/head.safetensors` | `f5632bb843eefd055a9e62460e3b247b90d83843efcba5cac05054bffd6b6740` |
| `model/ru-jev-v1.3/ship/head.safetensors` | `724740455841c6c4c14ff53489b8c9e3956952d2cbf32ab1884b1f0855508bcf` |

**The example in the card only works once `head.safetensors` is on the hub.** At
the time this was written the file on the hub was still the old one. It was
uploaded together with the card afterwards.

`code/scripts/fit_temperature.py` was added: it fits a temperature on the
caller's own decisions, prints the spread over 25 half-splits and warns when
there are too few rows. Checked on 300 rows: a temperature of 1.769, a spread
from 1.23 to 2.10, and the script said by itself that a spread of 0.87 is wide
and more rows are needed.

## 7. The weights being compared are different

The suspicion that "8120 = 8120 means one checkpoint was scored twice" is
removed by the hashes:

| File | sha256 |
|---|---|
| `model/ru-jev-v1.1/model.safetensors` | `8aca6840cad16019ea7039efd24ab9a9974090207dd1c011bdb16d4d61794c5d` |
| `model/ru-jev-v1.3/ship/model.safetensors` | `7e07664cf8bb677a9285a034c73920cc96853c0aebdc885e10e3b98812e3596d` |
| `model/ru-jev-v1.1/head.pt` | `5ac456679ae1cdaf358c2acadb180f83d3143dde40404ecb78c7312d15e05d52` |
| `model/ru-jev-v1.3/ship/head.pt` | `95d7f5f7217a440868c8838c33a64770e7ac2fccd158063227c7a2bbeaa67121` |

The rows for the per-row comparison were built as `comparable_rows.jsonl`,
10,157 rows, sha256
`e3d54b5f38e96150b83adbc2220e0df27799d66efc0af7bc3aa986e1c407a6fb`. They were
checked to be the same sets in both mixtures over all six tasks, and the build
fails otherwise. The per-row predictions are computed in a separate section
below.

## 1. The bench recomputed with all 18 candidates in every question

The setting differs from the published one: there a question offers 5 options
out of 18 categories, here it offers all 18. Chance is 0.056 against 0.200. The
absolute numbers are therefore not comparable, while the gaps between releases
inside one setting are.

| Vocabulary | v1.1, 5 cand. | v1.1, 18 cand. | v1.3, 5 cand. | v1.3, 18 cand. |
|---|---|---|---|---|
| английский ключ | 0.957 | 0.894 | 0.962 | 0.900 |
| русское слово | 0.956 | 0.883 | 0.960 | 0.892 |
| русская фраза | 0.881 | 0.703 | 0.957 | 0.885 |
| жаргон, all 18 | 0.633 | 0.431 | 0.855 | 0.640 |
| жаргон, clean 15 | 0.624 | 0.427 | 0.838 | 0.595 |
| названия служб | 0.835 | 0.632 | 0.869 | 0.733 |
| описание запроса | 0.939 | 0.848 | 0.958 | 0.878 |

The main point: the gaps did not collapse, and on the held-out vocabularies they
even grew.

| Vocabulary | Status of the vocabulary | Gap at 5 cand. | Gap at 18 cand. |
|---|---|---|---|
| английский ключ | in the training of both | +0.005 | +0.006 |
| русское слово | in the training of both | +0.004 | +0.009 |
| русская фраза | in the training of v1.3 only | +0.076 | +0.182 |
| жаргон, clean 15 | held out for both | +0.214 | +0.168 |
| названия служб | held out for both | +0.034 | +0.101 |
| описание запроса | held out for both | +0.019 | +0.030 |

This reads as follows, and it is in favour of the release's claim. On the two
vocabularies both releases saw in training, the gap is near zero in either
setting, as it should be. On the three honestly held out, the gap survives at
eighteen candidates, and on two of the three it grows. So the card's conclusion
about dependence on wording is robust to the number of candidates and is not an
artefact of the easy five-candidate setting. On "русская фраза" the gap grows as
well, but there it is expected by construction: v1.3 trained on those phrases
and v1.1 did not.

**Calibration on jargon behaves ambiguously, and that has to be said.** The gap
for v1.3 is smaller, +0.132 against +0.182, so by the gap v1.3 is the more
honest. But its confident errors are MORE numerous, 109 per thousand against 79.
The reason is that v1.3 is more confident overall, so a larger share of its
errors crosses the 0.8 threshold. One calibration metric praises v1.3 and
another faults it, and neither can be picked alone.

Laya loses on all six vocabularies and stays the most overconfident: on jargon
its gap is +0.480 at an accuracy of 0.150.

The full tables, including a breakdown by each of the 18 labels for every
vocabulary, are in `table.md`.

### Checking the "3 of 18" figure from the model card

Recomputed independently with the card's own criterion, which is in
`scripts/robustness.py`: a label counts as leaked if the jargon vocabulary
coincides with the output of the `jargon_translit` rule. Exactly three coincide:
`audio` is `аудио`, `news` is `ньюс`, `weather` is `вэзер`. Fifteen are clean.
The card's figure is correct.

A caveat. The wider prefix criterion touches seven of the eighteen jargon labels
rather than three: it catches any label that some training option starts with,
not only the exact output of the rule. The card's criterion is stricter and
documented, so it stays the main one, but a claim that "fifteen labels have no
relation to training at all" would be stronger than the data allow. The report
under both criteria is in `overlap_bench.json`.

## 4. Latency on one and the same M4

Median of 25 runs, five warm-up runs discarded, the measurements taken one after
another and alone: nothing else was computing on the processor at the time. What
is measured is the full time of a decision, that is tokenisation plus the
forward pass plus the softmax.

| Candidates | Ru-Jev v1.3 | Laya | Ratio |
|---|---|---|---|
| 2 | 18.5 ms | 15.5 ms | 1.19 |
| 5 | 32.8 ms | 17.8 ms | 1.84 |
| 18 | 73.8 ms | 30.5 ms | 2.42 |

Spreads: Ru-Jev from 16.2 to 22.8 at k=2, from 18.3 to 38.1 at k=5, from 43.5 to
88.7 at k=18. Laya from 14.6 to 95.7 at k=2 (one outlier at the start), from
15.7 to 32.1 at k=5, from 24.0 to 38.9 at k=18.

The reason for the difference is structural, not a matter of implementation
quality. Ru-Jev scores each (context, option) pair in a separate pass of the
encoder, so the cost of a decision grows linearly with the number of candidates:
roughly 4 ms per candidate plus a constant part. Laya reads all candidates in
one pass, so it barely depends on k. This is the price of being independent of
the option order.

### Correction to an earlier write-up

`../unseen/FINDINGS.md` said that Laya "computes 1,000 decisions in 20 seconds
and Ru-Jev 2,000 decisions in about four minutes", and concluded that the loss
was by a multiple. The clean measurement does not support that. The earlier
numbers came from runs that shared the processor with other work, at different
candidate counts, and with model loading included. The correct statement: Ru-Jev
is 1.2 times slower at two candidates and 2.4 times slower at eighteen. No
advantage in speed can be claimed, but "slower by an order of magnitude" is
wrong as well.

The figures are in `latency_rujev.json` and `latency_laya.json`, from the script
`latency.py`.

## 7. Per-row predictions on the 10,157 comparable rows

The suspicion that "8120 = 8120 means one checkpoint was scored twice" is
removed.

The weights the scores came from are different, and that is checkable:

| File | sha256 | Bytes |
|---|---|---|
| v1.1 backbone | `8aca6840cad16019ea7039efd24ab9a9974090207dd1c011bdb16d4d61794c5d` | 1227771776 |
| v1.3 ship backbone | `7e07664cf8bb677a9285a034c73920cc96853c0aebdc885e10e3b98812e3596d` | 1227771776 |
| v1.1 head | `f5632bb843eefd055a9e62460e3b247b90d83843efcba5cac05054bffd6b6740` | 3244 |
| v1.3 ship head | `724740455841c6c4c14ff53489b8c9e3956952d2cbf32ab1884b1f0855508bcf` | 3244 |

The result of the independent recomputation:

| | Value |
|---|---|
| rows | 10157 |
| both correct | 7280 |
| only v1.1 correct | 840 |
| only v1.3 correct | 840 |
| neither correct | 1197 |
| total correct | 8120 and 8120 |
| disagreements | 1680 |

This matched the model card to the unit, including the symmetry of 840 against
840. By task:

| Task | both correct | only v1.1 | only v1.3 | neither | n |
|---|---|---|---|---|---|
| russe | 6341 | 650 | 667 | 847 | 8505 |
| danetqa | 551 | 58 | 67 | 145 | 821 |
| terra | 199 | 39 | 26 | 43 | 307 |
| parus | 46 | 22 | 13 | 19 | 100 |
| rcb | 76 | 23 | 36 | 85 | 220 |
| rwsd | 67 | 48 | 31 | 58 | 204 |

The per-row dump is in `comparable_byrow.jsonl`: for every row its number, the
task, the correct answer, each release's choice and a correctness flag. The
equality of 8120 can be checked by hand against it.

### Calibration on these rows with a recorded seed

| Quantity | v1.1 | v1.3 |
|---|---|---|
| fitted T, mean over 25 splits | 2.0054 | 1.5593 |
| T, range | 1.942 to 2.098 | 1.500 to 1.604 |
| ECE after its own T | 0.0190 | 0.0191 |

The temperature of v1.3 is lower in 25 of the 25 splits and the ranges do not
overlap. The difference in residual ECE: mean +0.0001, from -0.0063 to +0.0109,
in favour of v1.3 in 14 of the 25 splits, that is, none.

All four of the card's claims in this section were confirmed. The card gives T
of 1.995 and 1.541 against 2.0054 and 1.5593 here, and ECE of 0.0185 and 0.0188
against 0.0190 and 0.0191 here; "lower in 25 of 25" and the non-overlapping
ranges are confirmed, and so is the absence of a difference in residual ECE.
