# The seed arm: the runs trained on the wrong mixture, and the conclusion is withdrawn

State as of the morning of 2026-11-05. During the night this file said that the
jargon gap does not reproduce on other seeds. That was a mistake in the setup,
not a property of the recipe. The earlier version of this file was never
published and is not in this repository. The model card, the landing page and
the text of the paper were not changed, then or now.

## What the mistake was

The seed runs trained on a mixture built BEFORE the commit that is the content
of v1.3.

Commit `96fb67f` of 2026-11-03 03:42 added two further ways of writing intent
labels to the data builder: the English key in Cyrillic letters, and a shortened
form. At `scripts/build_data.py:502` the intents carry
`label_styles=[None, INTENT_RU, jargon_translit, jargon_clip]`, that is four
styles. The first v1.3 checkpoint was written on 2026-11-03 at 04:32, fifty
minutes after that commit, so v1.3 trained on a mixture that had those styles.

The file `code/data/train.jsonl` on the laptop was created on 2026-11-02 at
19:49, eight hours before the commit. In it the intents have 120 distinct
surface forms: 60 in Latin script and 60 as Russian words. That is two styles,
`None` and `INTENT_RU`. With four styles there would have been 240. There are no
jargon forms in that file at all: the only single Cyrillic words present are
`посчитать` and `приветствие`, ordinary Russian words.

That mixture is the one that was sent to the rented machine. So the seed runs
did not repeat v1.3. They repeated a v1.2-level mixture on the v1.3 schedule,
that is, runs without the very change that was being tested.

## Why the checks did not catch it

The mixture on the rented machine was compared with the local one by sha256 and
matched. The match was correct and the reference was wrong: the local copy was
out of date. This is the same trap as the weights folder, applied to the data.

The row counts matched as well, 143,991 and 26,746, exactly as in the model
card. They could not have failed to match: new styles change the wording of the
options inside a row, not the number of rows. The check on the task composition
passed for the same reason: 13 tasks, with the same counts in each.

What should have stopped the work: all three seed runs, the augmented ones and
the baseline, gave 0.588, 0.611 and 0.658 on jargon, that is, they landed in the
band of v1.1 and v1.2, which are at 0.633 and 0.608. A run without the jargon
lesson shows the result of a run without the jargon lesson. This was read as
instability of the recipe instead of raising a suspicion about the data.

## What from that night's work still stands

**A control on the instrument.** The new scorer on the released weights gives
0.9268 over all 6,000 bench rows and agrees with the published table on all six
vocabularies to three decimal places.

**The weights folder is not what was published.** In
`model/ru-jev-v1.3/calibration.json` the fields are `"epoch": "final"` and a
temperature of 3.9204, that is the fourth, overtrained epoch. The published
weights are in `model/ru-jev-v1.3/ship/`, where the fields are `"epoch": 2` and
a temperature of 1.8640, which matches Hugging Face. A byte comparison confirms
it: `ship/model.safetensors` and `ep2/model.safetensors` are the same file. Any
measurement "on v1.3" has to go through `ship`.

**The overtraining of the fourth epoch is confirmed on a second seed.** The
temperature at the third epoch is 1.8711 and at the fourth 3.6450, against 1.8640
and 3.92 for the published run.

**The v1.3 trajectory on jargon, new and sound.** Accuracy on the 1,000 jargon
rows by epoch of the published run: 0.857, 0.844, 0.855, 0.837. A flat line from
the first epoch. This is consistent with the jargon lesson being in the data: it
is learned at once rather than accumulating towards the end of training.

**There is no test leakage.** None of the 18 scenario jargon labels of the bench
(`аудио`, `смолток`, `плеер`, `ньюс`, `листы`, `ай-оу-ти`, `реко` and the rest)
occurs among the options of the training mixture. The jargon styles are applied
to intents only; `build_data.py:499` leaves the scenarios with
`[None, SCENARIO_RU, SCENARIO_PHRASE]`. That is exactly what the commit and the
config state.

## What it would take to close the seed question

Rebuild the mixture with the current builder and run two augmented seeds on it.
Without that, nothing can be claimed about the stability of 0.838 in either
direction: the night's figures speak only about a mixture with no jargon lesson.

The baseline seed runs remain a sound data point on their own: `seed_base_s2` on
the `model/data-v1.1` mixture gave jargon 0.658 against 0.633 for the published
v1.1, so the baseline branch reproduces to within 0.025.

## Files

Raw per-row logits, with no temperature applied: `<name>_bench.jsonl` over the
6,000 bench rows and `<name>_val.jsonl` over the validation split. The files are
sound. The only thing that was wrong is the mixture these checkpoints trained on.

Scripts: `score_one.py` scores a checkpoint, `fit_and_enrich.py` fits the
temperature and adds `p_pick`, `aggregate.py` builds the table. They were one-off
scripts at the time and are now in `scripts/` in this repository, where
`score_one.py` is published under its later name `score_rujev.py`.
