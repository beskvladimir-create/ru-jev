# The wording bench: Ru-Jev against Laya

Measured 2026-11-04. Code: `beskvladimir-create/ru-jev`, commit `80d75f7` plus
the fix to `scripts/robustness.py` from the commit after it, described below.

## What is here

| File | Contents |
|---|---|
| `bench_rujev_v11.jsonl` | Ru-Jev v1.1, per row: probabilities, logits, the option picked |
| `bench_rujev_v13.jsonl` | Ru-Jev v1.3, the released weights, `model/ru-jev-v1.3/ship` |
| `bench_laya_instrA.jsonl` | Laya, instruction "К какой категории относится запрос пользователя?" |
| `bench_laya_instrB.jsonl` | Laya, instruction "Какая служба должна обработать этот запрос?" |
| `table.md` | The summary built from the four files above |

6,000 rows in each: six label vocabularies over the same 1,000 questions.

The two Laya instructions translate as "Which category does the user request
belong to?" and "Which service should handle this request?". They are kept in
Russian because they are the input the model was given.

## How to reproduce

```bash
# 1. dump the bench rows, deterministically
python scripts/robustness.py --model model/ru-jev-v1.3/ship --n 1000 --dump bench_rows.jsonl

# 2. our releases
python scripts/score_rujev.py --model model/ru-jev-v1.3/ship --rows bench_rows.jsonl \
  --out bench_rujev_v13.jsonl --tag ru-jev-v1.3

# 3. the comparison model (pip install laya)
python scripts/score_laya.py --rows bench_rows.jsonl --out bench_laya_instrA.jsonl \
  --instr "К какой категории относится запрос пользователя?" --instr-id A

# 4. the summary
python scripts/aggregate.py "ru-jev-v1.1=..." "ru-jev-v1.3=..." "laya instr A=..." ...
```

`score_rujev.py`, `score_laya.py` and `aggregate.py` were one-off scripts when
these numbers were produced. They are in `scripts/` in this repository.

## What to know before reading the table

**This is Ru-Jev's home ground, and nearly all of the accuracy gap is that.**
The bench is built on `mteb/amazon_massive_scenario` (ru). The training part of
MASSIVE has been in Ru-Jev's training mixture since v1. Laya has not seen this
task and runs zero-shot here, and its own model card states "near chance on
typed-decisions zero-shot, 0.342". The accuracy column cannot be read as "our
model is better". A fair comparison needs datasets neither model has seen, which
is the next measurement.

**What is comparable without the home ground.** The gap between confidence and
accuracy, and the residual ECE after fitting each model its own temperature.
Here Laya behaves the way our v1.1 did before the fix, only worse: a gap of
+0.23 against +0.12 for v1.1 and +0.03 for v1.3, and 110 to 125 confidently
wrong answers per thousand against 44.

**An asymmetry in the input that runs against us.** Ru-Jev is given only the
pair (text, option), with no instruction. Laya by construction requires a
question, so it gets a prompt Ru-Jev does not have.

**Laya's option keys are neutral** (`o0`..`o4`). Only the value carries meaning,
and that value is exactly the string Ru-Jev sees. Otherwise Laya would read a
meaningful key on top of the option text.

**Two instructions are used to show the spread over question wording.** It is
small, 0.394 against 0.419 on jargon, so Laya's result is not explained by an
unlucky prompt.

**Laya reports probabilities to four decimal places.** In 0 and 2 rows out of
6,000 the probability of the correct option came back as zero. Those rows are
excluded from temperature fitting and do not affect ECE.

## The fix to the bench, without which the comparison would be invalid

In `scripts/robustness.py` the option order was drawn with
`random.Random(hash(text) & 0xffff)`. Python salts string hashes per process, so
the option order did not reproduce between runs. It was replaced with
`zlib.crc32`.

This does not affect Ru-Jev's numbers at all: it scores each (text, option) pair
in a separate pass, the softmax is taken over independent scores, and the order
does not enter the result. Verified: all twelve published numbers over the six
vocabularies reproduced to three decimal places.

It does affect Laya: it reads all options in one pass, and its card declares a
positional bias. Without a fixed order the two models would have been shown
different inputs.
