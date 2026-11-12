# Ru-Jev

## Title

Ru-Jev: an open Russian-first typed-decision model. Training code, serving code,
evaluation scripts and the per-row results behind the paper.

- Weights and model card: https://huggingface.co/besktech/ru-jev
- Project page: https://ru-jev.besk.tech
- Preprint: https://arxiv.org/abs/2611.08439
- Archived release: https://doi.org/10.5281/zenodo.23986846

## Description

The model does not generate text. Given a context and a set of candidate options, it
returns a calibrated probability over the options and picks one. It covers three
decision shapes: yes/no, choice from options, and ordinal scale. The backbone is a
compact encoder (mmBERT-base, 307M), so it runs on CPU and can be deployed inside a
closed perimeter where sending data to an external API is not allowed.

Every example is reduced to a context, a list of candidate options and the index of
the correct one. The model scores each (context, option) pair with a linear head over
the first-token representation, and a softmax across the options gives the decision
distribution. One objective therefore covers yes/no, multiple choice and ordinal
scales at once. After training, a single temperature is fitted on the validation set,
which is what makes the returned confidence usable as a routing threshold.

**Not affiliated with TypeSafe AI. Not trained on Jev outputs.** Ru-Jev is trained
independently. The scoring head is trained from random initialisation and the
mmBERT-base backbone is fine-tuned along with it, on open, human-labelled
datasets only. The name is a play on the category, not a claim of compatibility
or endorsement.

### Results

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
the trajectory table of the model card and in the paper; the log it comes from is
in `results/training/`. 0.8208 is a separate scoring pass over the same rows: it
differs by 72 decisions out of 26,746, because the trainer pads every candidate
of a batch of sixteen to the longest sequence in it while the scoring pass pads
within one decision, and two percent of these rows have less than 0.1 between
their top two candidates. 0.8234 is the half held out from temperature fitting,
and it is the figure the coverage table is built on, because the temperature has
to come from rows the curve is not scored on.

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

Per task, v1.3 on its own validation split:

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

### Per-option biases

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
v1.1 rises to 0.673 under the same fit, so v1.3 stays about three points
behind on this set.

The same lean shows on the probe vocabularies: under several unfamiliar
wordings v1.1 almost never answers the `qa` label and v1.3 answers it freely.
`results/vocab_bias/README.md` shows how much of each vocabulary difference
this accounts for.

If your options are codes whose meaning you know, pass the full name as the
option text and keep the code as your ID. In a check posted on the Hugging Face
forum by John6666, the full name alone did as well as code plus name.

### What a cheap baseline gets

A number means nothing without something to compare it to. `scripts/baselines.py`
trains the two things a practitioner reaches for first, on the same training
rows, scored on the same validation rows. The Ru-Jev column here is v1.1, which
is the release this table was produced for.

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

### Withholding an answer

A decision is only useful if you know when not to trust it. Pass `threshold`
and the service withholds any answer below that confidence and lists the
question under `providerMetadata.ru_jev.review`. Answers below the threshold
carry `"abstain": true`. Omit `threshold` and nothing is withheld, which is how
the service behaved before.

What a threshold buys, on v1.3, measured on the half of the validation split
held out from temperature fitting (13,373 decisions, T = 1.8640, split seed 17).
Reproduce with `python scripts/coverage.py results/validation/val_v13_trainsplit.jsonl`.

| Share answered | Threshold | Accuracy on answered | Errors per 1000 |
| --- | --- | --- | --- |
| 30% | 0.98 | 0.9928 | 7 |
| 40% | 0.93 | 0.9847 | 15 |
| 50% | 0.85 | 0.9666 | 33 |
| 60% | 0.77 | 0.9470 | 53 |
| 70% | 0.69 | 0.9214 | 79 |
| 80% | 0.61 | 0.8918 | 108 |
| 90% | 0.54 | 0.8592 | 141 |
| 100% | 0.23 | 0.8234 | 177 |

How much of each decision shape can go through while accuracy on what went
through stays at 0.95 or better, same rows and same release:

| Decision shape | n | Share answered | Threshold |
| --- | --- | --- | --- |
| binary | 7164 | 49.5% | 0.816 |
| choice of 3 to 5 | 4704 | 53.0% | 0.760 |
| choice of many | 1505 | 100.0% | 0.225 |

Naming one option out of many is the shape that clears completely, because the
alternatives carry a lot of information. Binary questions are the shape that
does not: half of them still need a person at this accuracy bar. An earlier
edition of this file gave the same two tables for v1.1.

## Dataset information

Open datasets only, all permitting commercial use. Non-commercial sets are
excluded, and nothing is trained on the outputs of a closed decision API.
`DATASETS.md` carries the licence of each source, the pinned revision every
figure comes from, the caveats and the reasoning behind the exclusions.
`CODEBOOK.md` describes every file in `results/` field by field and glosses
every Russian value that appears in the data.

Eight sources go into the training mixture, which holds 143,991 training and
26,746 validation decisions over thirteen tasks:

| Source | Hugging Face id | Licence | What is used | Decision shape |
| --- | --- | --- | --- | --- |
| Russian SuperGLUE | `RussianNLP/russian_super_glue` | MIT | terra, danetqa, russe, rwsd (yes/no), parus (choice of 2), rcb (3 classes) | boolean, choice |
| RuReviews (MTEB) | `mteb/RuReviewsClassification` | Apache-2.0 | sentiment 0/1/2 | scale |
| RuDetoxifier data | `d0rj/rudetoxifier_data` | MIT | toxicity 0/1 | boolean |
| RuCoLA | `RussianNLP/rucola` | Apache-2.0 (labels and code) | acceptability 0/1 | boolean |
| Georeview | `mteb/GeoreviewClassification` | MIT | review rating, 5 levels | scale |
| MASSIVE scenario | `mteb/amazon_massive_scenario` (ru) | Apache-2.0 on the mirror, CC-BY-4.0 upstream | 18 scenarios | choice |
| MASSIVE intent | `mteb/amazon_massive_intent` (ru) | Apache-2.0 on the mirror, CC-BY-4.0 upstream | 60 intents | choice |
| MERA | `ai-forever/MERA` (ruopenbookqa, ruworldtree) | MIT | 4-way reasoning questions, train split only | choice |

Two further datasets are used for evaluation only. No release trained on them,
and by its own model card neither did the comparison model:

| Source | Hugging Face id | Licence in metadata | What is used | Decision shape |
| --- | --- | --- | --- | --- |
| Headline topics | `ai-forever/headline-classification` | MIT | 6 news topics, test split, 1,000 sampled rows | choice |
| CEDR emotions | `sagteam/cedr_v1` | Apache-2.0 | 5 emotions, rows with exactly one label, 1,000 sampled rows | choice |

For both of these the licence is declared only by the `license` field of the
repository's README metadata. Neither repository ships a licence file. That is a
statement by the repository owner, not a legal opinion, and
`results/unseen/FINDINGS.md` says so in the same words.

The label vocabularies used as probes are data in their own right and are in the
repository: `results/fix18/vocab.json` for the six-vocabulary wording bench,
`results/vocab2/vocab.json` for the code-identifier and abbreviation styles, and
`results/vocab3/vocab.json` for the independent slang and opaque-code probe,
which was supplied verbatim by the author before any figure was seen.

## Code information

Every script has one job. The ones marked with a star are the ones a reader of
the paper will want first.

| Script | What it does |
| --- | --- |
| `scripts/build_data.py` | builds the unified decision dataset into `data/*.jsonl`, including the label wording styles that distinguish the releases |
| `scripts/train.py` | trains the scoring head and the backbone, keeps a checkpoint per epoch, fits the temperature |
| `scripts/eval.py` | per-task accuracy and ECE for one checkpoint, applying the calibration file |
| `scripts/probe.py` | capability probe on realistic requests, compares two checkpoints on the same cases |
| `scripts/robustness.py` * | the wording bench: the same questions under six label vocabularies |
| `scripts/score_rujev.py` * | scores a checkpoint over a file of decisions and writes raw logits |
| `scripts/score_laya.py` | the same for the comparison model, in its own environment |
| `scripts/calib.py` * | temperature, per-option biases and ECE over half-splits with a recorded seed; the shared implementation everything else calls |
| `scripts/fit_temperature.py` * | fits a temperature on your own decisions, and with `--biases` one additive bias per option |
| `scripts/test_calibration.py` | checks the calibration fit on synthetic scores, with no weights and no network |
| `scripts/verify_temperature.py` | refits the released temperature and checks it against `calibration.json` |
| `scripts/coverage.py` * | coverage against accuracy, thresholded on confidence |
| `scripts/latency.py` | latency per decision at 2, 5 and 18 candidates |
| `scripts/overlap.py` * | audits a probe vocabulary against the training mixtures, by exact match and by prefix in both directions |
| `scripts/split_overlap.py` | how much of a rebuilt validation split had been in the earlier training split |
| `scripts/build_unseen.py` | builds the rows of the two datasets no model trained on |
| `scripts/build_fix18.py` | builds the bench rows with all 18 candidates per question |
| `scripts/build_vocab2.py` | builds the code-identifier and abbreviation probe |
| `scripts/build_vocab3.py` | builds the independent slang and opaque-code probe |
| `scripts/build_comparable.py` | builds the rows on which two releases can be compared, where both the text and the option wording coincide |
| `scripts/compare_rows.py` | per-row comparison of two releases, printing the weight hashes alongside the counts |
| `scripts/aggregate.py` | the five-candidate wording bench as the model card reports it |
| `scripts/unseen_table.py` | the summary of the two datasets no model trained on |
| `scripts/table_fix18.py`, `scripts/table_vocab2.py`, `scripts/table_vocab3.py` | the summary table of each probe |
| `scripts/bylabel.py` | the per-label breakdown of a probe |
| `scripts/fit_and_enrich.py` | fits a temperature over a score dump and writes `p_pick` back into it |
| `scripts/baselines.py` | majority class and TF-IDF with logistic regression on the same rows |
| `scripts/run_pipeline.sh` | build, train, evaluate, probe, end to end |
| `results/analysis/final.py` * | every cell of Table 2 of the paper, with McNemar, Holm and the bootstrap |
| `results/analysis/unseen_bias.py` * | per-option biases on the two unseen datasets, out of fold |
| `serve/app.py` | the serving API with a typed-decision endpoint |

During the work two scripts carried other names, which appear in the logs and in
earlier notes: `score_one.py` was `scripts/score_rujev.py` as it ran on the rented
machine, and `score_rows.py` was the same file after it was moved here. There is
one implementation and it is `scripts/score_rujev.py`.

Supporting files:

```
configs/base.yaml      v1: mmBERT-base, 3 epochs
configs/v1_1.yaml      v1.1: same backbone, broader data mix
configs/v1_3.yaml      v1.3: released; label wordings varied per example, 4 epochs
configs/v2.yaml        v2: xlm-roberta-large, 4 epochs (trained, not published)
configs/seed_*.yaml    the seed replicates of the v1.1 and v1.3 recipes
results/               per-row scores and tables of every experiment in the paper
REPRODUCE.md           which command produces which table
ENVIRONMENT.md         hardware, package versions, GPU time
DATASETS.md            sources, licences, pinned revisions and what is excluded
CODEBOOK.md            every result file, every field, every Russian value
SHA256SUMS             hashes of the files here; SHA256SUMS.weights for the weights
```

## Usage instructions

### Deciding with the released model

```python
import json, torch, torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

REPO = "besktech/ru-jev"
tok = AutoTokenizer.from_pretrained(REPO)
backbone = AutoModel.from_pretrained(REPO).eval()
head = nn.Linear(backbone.config.hidden_size, 1)
head.load_state_dict(load_file(hf_hub_download(REPO, "head.safetensors")))
cal = json.load(open(hf_hub_download(REPO, "calibration.json")))

@torch.no_grad()
def decide(context, options):
    enc = tok([context] * len(options), [str(o) for o in options], truncation=True,
              max_length=cal["max_len"], padding=True, return_tensors="pt")
    scores = head(backbone(**enc).last_hidden_state[:, 0]).squeeze(-1)
    return torch.softmax(scores / cal["temperature"], dim=0).tolist()
```

The standard `transformers` pipelines do not work on this checkpoint. It is a
backbone plus a separate scoring head, and every stock pipeline loads it through
`AutoModelForSequenceClassification`, which silently initialises a fresh head.
Use the snippet above.

### Serving

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

`GET /health` reports the loaded model directory, device, max length, temperature
and the per-option biases if the calibration file carries any.

### Training

Run on a GPU box with python 3.11. Torch has no 3.15 wheels at the time of
writing. Apple silicon works too: the scripts pick up MPS when no CUDA device is
present.

```bash
CONFIG=configs/v1_3.yaml bash scripts/run_pipeline.sh    # the released recipe
CONFIG=configs/v1_1.yaml bash scripts/run_pipeline.sh    # the baseline recipe
CONFIG=configs/base.yaml bash scripts/run_pipeline.sh    # v1
```

That creates a virtualenv, builds `data/train.jsonl` and `data/val.jsonl` from the
Hugging Face sources, trains, and writes the checkpoints plus `calibration.json`
into the `out_dir` of the config, which is `out/ru-jev-v1.3` for the released
recipe. The released run took about 2.8 GPU hours on a rented RTX 4090;
`ENVIRONMENT.md` has the exact figures. `CONFIG` defaults to `configs/v1_1.yaml`
when it is not set. Each data source is wrapped so that a schema change on Hugging
Face is retried and then stops the build rather than silently skipping a task.

To run the steps separately:

```bash
python scripts/build_data.py --cap 30000
python scripts/train.py --config configs/v1_3.yaml
python scripts/eval.py --model out/ru-jev-v1.3
python scripts/probe.py --model out/ru-jev-v1.3 --model-b model/ru-jev-v1.1
```

`eval.py` reports accuracy per task and the calibration error on the held-out
split. `probe.py` is a different question: it runs the kinds of request a caller
of the serving API actually sends, such as routing a ticket to a department or
asking a free-form yes/no question, and compares two checkpoints on the same
cases. The dev splits can look healthy while those requests do not work.

Russian SuperGLUE and RuCoLA are script-based datasets and stopped opening in
`datasets` 4.0. The builder falls back to their auto-converted parquet branch, so
the build runs on both old and current versions of the library.

### Reproducing the paper

`REPRODUCE.md` says which command produces which table. The per-row scores of
every experiment are in `results/`, so most tables can be rebuilt without the
weights and without a GPU.

## Requirements

- Python 3.11 or newer. Torch has no 3.15 wheels at the time of writing.
- `pip install -r requirements.txt` for everything except serving.
- `pip install -r serve/requirements.txt` for the serving API.
- A GPU is needed only for training. Every measurement in the paper except the
  training trajectory runs on CPU.
- About 1.5 GB of disk for the weights and 40 MB for the per-row scores here.

`ENVIRONMENT.md` records the exact versions, the hardware of every measurement
and the GPU hours. Where `requirements.txt` gives a floor and `ENVIRONMENT.md`
gives a number, the number is what the published figures were produced with:
`datasets` is pinned at a floor of 2.20 in the requirements and the measurements
ran on 5.1.0. An earlier edition of `requirements.txt` carried an upper bound of
`<5`, which excluded the version actually used; the bound was wrong and has been
removed, because the builder handles the removal of dataset scripts in 4.x and
5.x through the parquet branch.

The comparison model pins versions of its own and has to live in a separate
environment:

```bash
python -m venv peer && peer/bin/pip install laya
```

## Methodology

### The model

A context and its candidate options are fed to the encoder as pairs. Each pair
gets its own forward pass, a linear head over the first-token representation
produces one score, and a softmax across the options of that decision gives the
distribution. Because each option is scored independently, the result does not
depend on the order the options are given in, which is what makes the probe
vocabularies below a fair test. The cost is linear in the number of candidates,
which `results/fix18/AUDIT.md` measures.

A single temperature is fitted after training by minimising the negative log
likelihood on the validation split. It does not change which option is picked.
One additive bias per option does, and `scripts/fit_temperature.py --biases`
fits both together by maximum likelihood with L-BFGS.

### Evaluation

The wording bench. The same 1,000 questions from `massive_scenario` are asked
under six label vocabularies: the English key, the Russian word, a Russian
phrase, anglicism jargon, department names and a descriptive sentence. The
published bench offers five candidates out of the 18 categories, so chance is
0.200; `results/fix18/` repeats it with all 18 offered, where chance is 0.056.
Two of the six vocabularies are training styles and serve as controls, which
`results/fix18/AUDIT.md` sets out label by label.

The probes. Two further vocabularies were built to test an unfamiliar kind of
form rather than unfamiliar words: `results/vocab2/` uses a code identifier and
a three-letter abbreviation, and `results/vocab3/` uses everyday slang and an
opaque code, supplied verbatim by the author. For vocab3 the rule for reading
the result was written down with a timestamp before the figures were seen, and
the Russian original of that rule is kept in `results/vocab3/README.md`.

The overlap audit. Every probe label is compared with every option string of
both training mixtures, by exact match and by prefix in both directions, for the
correct label and for the distractors separately. A label that matches exactly
is excluded from the headline figure and reported separately rather than
replaced. `scripts/overlap.py` produces the report.

The statistics. The difference between the two releases is tested with the exact
McNemar test over the paired decisions, since both releases answer the same
rows. Over the 23 held-out cells the p-values carry a Holm correction.
Confidence intervals come from a two-stage bootstrap that resamples labels and
then questions within a label, so that a result cannot rest on one lucky label;
`results/analysis/final.py` computes all of it with the bootstrap seed 20261105.
For the per-option biases the intervals are computed both over rows and over
labels, and `results/unseen/bias.md` gives both.

What is deliberately not claimed. Accuracy on the two unseen datasets does not
favour this model, and `results/unseen/FINDINGS.md` says so. The latency
comparison runs against it as well. The withdrawn figures and the four hazards
that produced them are written up in `results/fix18/AUDIT.md`,
`results/fix18/half_split_withdrawn.md`, `results/seeds/FINDINGS.md` and
`results/validation/split_overlap.md`.

## Citations

The preprint is [arXiv:2611.08439](https://arxiv.org/abs/2611.08439). This
repository is archived at Zenodo under
[10.5281/zenodo.23986846](https://doi.org/10.5281/zenodo.23986846), the concept
DOI, which always resolves to the latest version.

Which version to cite depends on what for:

| Cite | For |
|---|---|
| `10.5281/zenodo.23986847` | the tables of the paper as first released, tag `v1.3.0` |
| `10.5281/zenodo.23986846` | the latest version, whatever it is at the time of reading |

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

`CITATION.cff` at the root carries the same information in machine-readable
form, and GitHub builds its "Cite this repository" box from it.

The datasets this work is built on are cited in `DATASETS.md`, with the pinned
revision of each.

## License and contribution guidelines

MIT, for the code in this repository and for the released weights. Author:
Vladimir Beskorovainyi, Besk Tech (besk.tech).

The training data keep the licences of their sources, which are listed per
source in `DATASETS.md`. None of the ten datasets ships a licence file, so every
licence recorded there is the `license` field of the repository's README
metadata.

Contributions are welcome through the GitHub repository,
https://github.com/beskvladimir-create/ru-jev. Open an issue for a bug, a
figure that does not reproduce, or a question about the method, and say which
command you ran and what you got. Pull requests are accepted for code and
documentation; please keep the per-row result files in `results/` as they are,
since the published figures are computed from them and `SHA256SUMS` records
their hashes. If a change affects a number in `results/`, say so in the pull
request and regenerate `SHA256SUMS` with it.
