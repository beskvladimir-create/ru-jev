# Ru-Jev (Ружев)

Training and serving code for Ru-Jev, an open Russian-first typed-decision model.

The model does not generate text. Given a context and a set of candidate options, it
returns a calibrated probability over the options and picks one. It covers three
decision shapes: yes/no, choice from options, and ordinal scale. The backbone is a
compact encoder (mmBERT-base, 307M), so it runs on CPU and can be deployed inside a
closed perimeter where sending data to an external API is not allowed.

- Weights and model card: https://huggingface.co/besktech/ru-jev
- Project page: https://ru-jev.besk.tech
- Preprint: https://arxiv.org/abs/2611.08439
- Archived release: https://doi.org/10.5281/zenodo.23986846

**Not affiliated with TypeSafe AI. Not trained on Jev outputs.** Ru-Jev is trained
independently. The scoring head is trained from random initialisation and the
mmBERT-base backbone is fine-tuned along with it, on open, human-labelled
datasets only. The name is a play on the category,
not a claim of compatibility or endorsement.

## Reproducing the paper

`REPRODUCE.md` says which command produces which table, `ENVIRONMENT.md`
records the hardware and the package versions, and `DATASETS.md` pins the
dataset revisions every figure comes from. The per-row scores of every
experiment are in `results/`.

## How it works

Every example is reduced to a context, a list of candidate options and the index of
the correct one. The model scores each (context, option) pair with a linear head over
the first-token representation, and a softmax across the options gives the decision
distribution. One objective therefore covers yes/no, multiple choice and ordinal
scales at once. After training, a single temperature is fitted on the validation set,
which is what makes the returned confidence usable as a routing threshold.

## Results

v1.3, measured on its own held-out validation split. Two accuracy figures,
because they cover different numbers of rows and an earlier version of this
table gave one of them under the other's row count.

| Metric | Rows | Value |
| --- | --- | --- |
| Accuracy (micro), whole validation split, as the trainer measured it | 26,746 | 0.8181 |
| Accuracy (micro), whole validation split, standalone scoring pass | 26,746 | 0.8208 |
| Accuracy (micro), half held out from temperature fitting | 13,373 | 0.8234 |
| ECE, temperature fitted on the disjoint half | 13,373 | 0.0185 |
| ECE, temperature fitted on the same rows | 13,373 | 0.0159 |
| Released temperature | | 1.8640 |

Three accuracy figures on one checkpoint, counting different things. 0.8181 is
what the training run itself reported at the end of epoch 3 and is the figure in
the trajectory table of the model card and in the paper. 0.8208 is a separate
scoring pass over the same rows: it differs by 72 decisions out of 26,746,
because the trainer pads every candidate of a batch of sixteen to the longest
sequence in it while the scoring pass pads within one decision, and two percent
of these rows have less than 0.1 between their top two candidates. 0.8234 is the
half held out from temperature fitting, and it is the figure the coverage table
is built on, because the temperature has to come from rows the curve is not
scored on.

`results/validation/README.md` sets all three out, and also records why the
mixture cannot simply be rebuilt to recheck them: six of the thirteen tasks
reproduce their split exactly and seven do not, so a rebuilt validation row can
be a row the release trained on.

The earlier table here reported an ECE of 0.0054, which was fitted and scored on
the same rows. That figure is in-sample and optimistic, and it should not be
compared with the ones above.

`scripts/verify_temperature.py` refits the released temperature from the
released weights and the validation split and checks it against the 1.8640 in
`calibration.json`, so the figure can be rechecked rather than taken on trust.

The temperature cannot change which option comes out. It divides every score by
the same number, so the highest score stays the highest and accuracy is the same
at any temperature. If on your own data the model leans towards one of your
labels whatever the input says, that is not something a temperature can repair.
Fit one additive bias per option instead, with
`scripts/fit_temperature.py --biases` on the same labelled rows. The answer
becomes `softmax(s / T + b[option])`, the biases are written into the same
`calibration.json`, and the serving API and `scripts/eval.py` apply them from
there. On the news rubrics that no release trained on, this lifts v1.3 from
0.568 to 0.640, because the release leans towards `политика`.
`results/unseen/bias.md` has the figures and
`results/analysis/unseen_bias.py` reproduces them. The biases belong to the
label set they were fitted on and do not carry over to a different one.

Per task:

| Task | Accuracy | n |
| --- | --- | --- |
| massive_intent (60 labels) | 0.960 | 3000 |
| massive_scenario (18 labels) | 0.955 | 3000 |
| toxicity | 0.954 | 3000 |
| russe (word sense) | 0.824 | 8505 |
| rureviews (sentiment) | 0.774 | 3000 |
| danetqa (yes/no) | 0.753 | 821 |
| terra (entailment) | 0.733 | 307 |
| rucola (acceptability) | 0.647 | 1344 |
| georeview (5 levels) | 0.631 | 3000 |
| mera_qa (reasoning) | 0.596 | 245 |
| parus (choice) | 0.590 | 100 |
| rcb (3-way NLI) | 0.509 | 220 |
| rwsd (coreference) | 0.480 | 204 |

These are not comparable with the per-task numbers of earlier releases. The
validation slice moved between builds and so did the wording of the options;
`model/ru-jev-v1.3/README.md` sets out which tasks can be compared and which
cannot, and what the difference cost in retracted claims.

The overall figure is a micro average, so the large `russe` split weighs on it.
Reasoning-heavy tasks (rwsd, rcb, parus) are the weak points for an encoder of this
size.

A second version was trained on xlm-roberta-large (560M), configured in
`configs/v2.yaml`. It did not win: on the dev splits that are directly
comparable it is worse on the reasoning tasks and on `russe`, so the larger
backbone was dropped and the released line stayed on mmBERT-base.

## Layout

```
configs/base.yaml      v1: mmBERT-base, 3 epochs
configs/v1_1.yaml      v1.1: same backbone, broader data mix
configs/v1_3.yaml      v1.3: released; label wordings varied per example, 4 epochs
configs/v2.yaml        v2: xlm-roberta-large, 4 epochs (trained, not published)
scripts/build_data.py  builds the unified decision dataset into data/*.jsonl
scripts/train.py       scoring head + temperature calibration
scripts/eval.py        per-task accuracy and ECE
scripts/probe.py       capability probe on realistic requests, compares two checkpoints
scripts/robustness.py  the wording bench: one vocabulary of labels per column
scripts/score_rujev.py  scores a checkpoint over a file of decisions, writes raw logits
scripts/score_laya.py  the same for the comparison model
scripts/calib.py       temperature and ECE over half-splits, with a recorded seed
scripts/coverage.py    coverage against accuracy, threshold on confidence
scripts/latency.py     latency per decision at 2, 5 and 18 candidates
scripts/overlap.py     audits a probe vocabulary against the training mixtures
scripts/split_overlap.py  how much of a rebuilt validation split was trained on
scripts/build_*.py     builds the rows of each probe
scripts/table_*.py     the summary tables of each probe
scripts/aggregate.py   the five-candidate wording bench as the model card reports it
scripts/unseen_table.py  the two datasets no model trained on
scripts/compare_rows.py  per-row comparison of two releases, with weight hashes
scripts/verify_temperature.py  refits the released temperature and checks it
scripts/fit_temperature.py     fits a temperature, and with --biases one bias per option
scripts/test_calibration.py    checks the calibration fit on synthetic scores
scripts/run_pipeline.sh  build, train, evaluate, probe
serve/app.py           FastAPI service with a typed-decision endpoint
results/               per-row scores and tables of every experiment in the paper
results/analysis/unseen_bias.py  per-option biases on the two unseen datasets
REPRODUCE.md           which command produces which table
ENVIRONMENT.md         hardware, package versions, GPU time
DATASETS.md            sources, licences, pinned revisions and what is excluded
SHA256SUMS             hashes of the files here; SHA256SUMS.weights for the weights
```

During the work two scripts carried other names, which appear in the logs and in
earlier notes: `score_one.py` was `scripts/score_rujev.py` as it ran on the rented
machine, and `score_rows.py` was the same file after it was moved here. There is
one implementation and it is `scripts/score_rujev.py`.

## Training

Run on a GPU box with python 3.11. Torch has no 3.15 wheels at the time of
writing. Apple silicon works too: the scripts pick up MPS when no CUDA device is
present.

```bash
bash scripts/run_pipeline.sh                 # v1.1
CONFIG=configs/base.yaml bash scripts/run_pipeline.sh   # reproduce v1
```

That creates a virtualenv, builds `data/train.jsonl` and `data/val.jsonl` from the
Hugging Face sources, trains, and writes the checkpoint plus `calibration.json` into
`out/ru-jev-v1`. The full v1 run takes one to three GPU hours. Each data source is
wrapped so that a schema change on Hugging Face skips that source with a warning
instead of failing the whole build.

To run the steps separately:

```bash
python scripts/build_data.py --cap 30000
python scripts/train.py --config configs/v1_1.yaml
python scripts/eval.py --model out/ru-jev-v1.1
python scripts/probe.py --model out/ru-jev-v1.1 --model-b model/ru-jev-v1
```

`eval.py` reports accuracy per task and the calibration error on the held-out
split. `probe.py` is a different question: it runs the kinds of request a caller
of the serving API actually sends, such as routing a ticket to a department or
asking a free-form yes/no question, and compares two checkpoints on the same
cases. The dev splits can look healthy while those requests do not work.

Russian SuperGLUE and RuCoLA are script-based datasets and stopped opening in
`datasets` 4.0. The builder falls back to their auto-converted parquet branch, so
the build runs on both old and current versions of the library.

## Serving

```bash
pip install -r serve/requirements.txt
uvicorn serve.app:app --host 0.0.0.0 --port 8080
```

With nothing configured the service downloads `besktech/ru-jev` from the Hub on
first request, so a clone of this repository is enough to start it. It prefers a
local `out/ru-jev-v1.3` or `../model/ru-jev-v1.3/ship` when either exists. Set
`RU_JEV_MODEL` to a directory or a Hub repo id to override. Decisions are computed
locally, so the text never leaves the host.

`POST /v1/systemone` takes `{model, state, questions}` and returns
`{model, answers, usage, providerMetadata}`, with confidence per question under
`providerMetadata`. Question types are `boolean`, `choice` and `score`. The request
and response shape follows the typed-decision format, so a client written against a
closed decision API moves over by changing the base URL.

```bash
curl -s localhost:8080/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Клиента списали дважды, он просит вернуть деньги.",
  "questions": {
    "refund": {"type": "boolean", "instructions": "Клиент просит возврат?"},
    "team":   {"type": "choice", "criteria": {"billing": "оплата и возвраты",
                                               "support": "общие вопросы"}}
  }
}'
```

`GET /health` reports the loaded model directory, device, max length and temperature.

## What a cheap baseline gets

A number means nothing without something to compare it to. `scripts/baselines.py`
trains the two things a practitioner reaches for first, on the same training
rows, scored on the same validation rows.

| Task | Majority class | TF-IDF + logreg | Ru-Jev v1.1 |
| --- | --- | --- | --- |
| massive_scenario (18 labels) | 0.066 | 0.435 | 0.954 |
| massive_intent (60 labels) | 0.063 | 0.820 | 0.965 |
| terra (entailment) | 0.498 | 0.394 | 0.775 |
| danetqa (yes/no) | 0.502 | 0.568 | 0.742 |
| russe (word sense) | 0.631 | 0.701 | 0.822 |
| georeview (5 levels) | 0.199 | 0.544 | 0.629 |
| rureviews (sentiment) | 0.324 | 0.720 | 0.772 |
| rucola (acceptability) | 0.587 | 0.609 | 0.650 |
| toxicity | 0.766 | 0.943 | 0.961 |
| rwsd (coreference) | 0.554 | 0.554 | 0.564 |

Read it honestly. The model earns its keep on routing, on entailment and on
yes/no questions, where surface statistics either fail outright or fall far
short. On toxicity it is 1.8 points over TF-IDF and on sentiment 5.2 points: if
that is all you need, a linear model on character n-grams is cheaper to run and
easier to debug. On `rwsd` the model is level with predicting the majority
class, so it does not do coreference at all; that row is in the table because
leaving it out would be dishonest.

The other thing a baseline cannot give you is a calibrated probability, which
is what makes a threshold mean the same thing from one task to the next.

## Withholding an answer

A decision is only useful if you know when not to trust it. Pass `threshold`
and the service withholds any answer below that confidence and lists the
question under `providerMetadata.ru_jev.review`:

```json
{
  "threshold": 0.8,
  "state": {"text": "Списали деньги дважды за один заказ"},
  "questions": {"route": {"type": "choice", "criteria": {"billing": "оплата, счета и возвраты", "support": "техническая проблема"}}}
}
```

Answers below the threshold carry `"abstain": true`. Omit `threshold` and
nothing is withheld, which is how the service behaved before.

What a threshold buys, measured on the held-out half of the validation set
(13,373 decisions, v1.1):

| Share sent through automatically | Threshold | Accuracy on those | Errors per 1000 |
| --- | --- | --- | --- |
| 30% | 0.98 | 99.1% | 9 |
| 50% | 0.85 | 96.6% | 34 |
| 60% | 0.77 | 94.5% | 55 |
| 70% | 0.69 | 91.9% | 81 |
| 80% | 0.61 | 89.0% | 110 |
| 90% | 0.53 | 85.7% | 143 |
| 100% | none | 82.1% | 179 |

How much of each decision shape can go through while accuracy on what went
through stays at 95% or better:

| Decision shape | n | Share answered | Threshold |
| --- | --- | --- | --- |
| choice | 6565 | 94% | 0.57 |
| boolean | 14181 | 49% | 0.82 |
| scale | 6000 | 24% | 0.83 |

Routing over many options is the shape that clears, because naming one option
out of many is a question the model answers well. Ordinal scales are the shape
that does not: a five-level review score is a genuinely harder judgement and
three quarters of it still needs a person at this accuracy bar.

## Training data

Open datasets only, all permitting commercial use. Non-commercial sets are excluded,
and nothing is trained on the outputs of a closed decision API. See `DATASETS.md` for
the licence of each source and the reasoning behind the exclusions.

- Russian SuperGLUE (MIT): terra, danetqa, russe, rwsd, parus, rcb
- RuReviews (Apache-2.0): sentiment
- RuDetoxifier data (MIT): toxicity
- RuCoLA (Apache-2.0 labels): acceptability

## Citation

The preprint is [arXiv:2611.08439](https://arxiv.org/abs/2611.08439). This
repository is archived at Zenodo under
[10.5281/zenodo.23986846](https://doi.org/10.5281/zenodo.23986846), which always
resolves to the latest version. Cite 10.5281/zenodo.23986847 for the exact
version the paper was written against, tag `v1.3.0`.

```bibtex
@misc{beskorovainyi2026rujev,
  title         = {Label-Wording Augmentation in a Typed-Decision Encoder: Gains on Unfamiliar Words, Not on Abbreviations or Codes},
  author        = {Beskorovainyi, Vladimir},
  year          = {2026},
  eprint        = {2611.08439},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2611.08439}
}

@software{beskorovainyi2026rujevcode,
  title     = {Ru-Jev: open typed-decision model (v1.3) — code, per-row results and evaluation scripts},
  author    = {Beskorovainyi, Vladimir},
  year      = {2026},
  version   = {1.3.0},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.23986846},
  url       = {https://doi.org/10.5281/zenodo.23986846}
}
```

## License

MIT. Author: Vladimir Beskorovainyi, Besk Tech (besk.tech).
