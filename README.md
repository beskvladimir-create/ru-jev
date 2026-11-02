# Ru-Jev (Ружев)

Training and serving code for Ru-Jev, an open Russian-first typed-decision model.

The model does not generate text. Given a context and a set of candidate options, it
returns a calibrated probability over the options and picks one. It covers three
decision shapes: yes/no, choice from options, and ordinal scale. The backbone is a
compact encoder (mmBERT-base, 307M), so it runs on CPU and can be deployed inside a
closed perimeter where sending data to an external API is not allowed.

- Weights and model card: https://huggingface.co/besktech/ru-jev
- Project page: https://ru-jev.besk.tech

**Not affiliated with TypeSafe AI. Not trained on Jev outputs.** Ru-Jev is trained
from scratch on open, human-labelled datasets. The name is a play on the category,
not a claim of compatibility or endorsement.

## How it works

Every example is reduced to a context, a list of candidate options and the index of
the correct one. The model scores each (context, option) pair with a linear head over
the first-token representation, and a softmax across the options gives the decision
distribution. One objective therefore covers yes/no, multiple choice and ordinal
scales at once. After training, a single temperature is fitted on the validation set,
which is what makes the returned confidence usable as a routing threshold.

## Results

Measured on the held-out development split, 17,501 items.

| Metric | Value |
| --- | --- |
| ECE (calibration error) | 0.0054 |
| Temperature | 1.82 |
| Overall accuracy (micro) | 0.824 |

Per task:

| Task | Accuracy | n |
| --- | --- | --- |
| toxicity | 0.964 | 3000 |
| russe | 0.846 | 8505 |
| rureviews (sentiment) | 0.774 | 3000 |
| terra (entailment) | 0.762 | 307 |
| danetqa (yes/no) | 0.711 | 821 |
| rucola (acceptability) | 0.686 | 1344 |
| parus (choice) | 0.600 | 100 |
| rcb (3-way NLI) | 0.518 | 220 |
| rwsd (coreference) | 0.490 | 204 |

The overall figure is a micro average, so the large `russe` split weighs on it.
Reasoning-heavy tasks (rwsd, rcb, parus) are the weak points for an encoder of this
size.

A second version was trained on xlm-roberta-large (560M), configured in
`configs/v2.yaml`. It did not win. On the dev splits that are directly comparable it
is worse on the reasoning tasks and on `russe`, so the published checkpoint is v1.

## Layout

```
configs/base.yaml      v1: mmBERT-base, 3 epochs (the published checkpoint)
configs/v1_1.yaml      v1.1: same backbone, broader data mix
configs/v2.yaml        v2: xlm-roberta-large, 4 epochs (trained, not published)
scripts/build_data.py  builds the unified decision dataset into data/*.jsonl
scripts/train.py       scoring head + temperature calibration
scripts/eval.py        per-task accuracy and ECE
scripts/probe.py       capability probe on realistic requests, compares two checkpoints
scripts/run_pipeline.sh  build, train, evaluate, probe
serve/app.py           FastAPI service with a typed-decision endpoint
DATASETS.md            sources, licences and what is excluded
```

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

The service looks for `out/ru-jev-v1`, then for released weights in
`../model/ru-jev-v1`. Set `RU_JEV_MODEL` to point it anywhere else. Decisions are
computed locally, so the text never leaves the host.

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
| 30% | 0.98 | 99.3% | 7 |
| 50% | 0.85 | 97.1% | 29 |
| 60% | 0.77 | 95.1% | 49 |
| 80% | 0.60 | 89.5% | 105 |
| 100% | none | 82.5% | 175 |

Per decision shape, at 95% accuracy or better: routing over many options clears
the whole stream, while yes/no and small choices clear about half of it.

## Training data

Open datasets only, all permitting commercial use. Non-commercial sets are excluded,
and nothing is trained on the outputs of a closed decision API. See `DATASETS.md` for
the licence of each source and the reasoning behind the exclusions.

- Russian SuperGLUE (MIT): terra, danetqa, russe, rwsd, parus, rcb
- RuReviews (Apache-2.0): sentiment
- RuDetoxifier data (MIT): toxicity
- RuCoLA (Apache-2.0 labels): acceptability

## License

MIT. Author: Vladimir Beskorovainyi, Besk Tech (besk.tech).
