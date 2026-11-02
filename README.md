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
configs/base.yaml      v1: mmBERT-base, 3 epochs
configs/v2.yaml        v2: xlm-roberta-large, 4 epochs (trained, not published)
scripts/build_data.py  builds the unified decision dataset into data/*.jsonl
scripts/train.py       scoring head + temperature calibration
scripts/eval.py        per-task accuracy and ECE
scripts/run_pipeline.sh  the three steps above, end to end
serve/app.py           FastAPI service with a typed-decision endpoint
DATASETS.md            sources, licences and what is excluded
```

## Training

Run on a GPU box with python 3.11. Torch has no 3.15 wheels at the time of writing.

```bash
bash scripts/run_pipeline.sh
```

That creates a virtualenv, builds `data/train.jsonl` and `data/val.jsonl` from the
Hugging Face sources, trains, and writes the checkpoint plus `calibration.json` into
`out/ru-jev-v1`. The full v1 run takes one to three GPU hours. Each data source is
wrapped so that a schema change on Hugging Face skips that source with a warning
instead of failing the whole build.

To run the steps separately:

```bash
python scripts/build_data.py --cap 30000
python scripts/train.py --config configs/base.yaml
python scripts/eval.py --model out/ru-jev-v1
```

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
