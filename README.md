---
license: mit
language:
  - ru
  - en
  - multilingual
pipeline_tag: zero-shot-classification
tags:
  - text-classification
  - zero-shot-classification
  - typed-decision
  - russian
  - jev
  - calibrated
library_name: transformers
base_model: jhu-clsp/mmBERT-base
---

<!-- ЧЕРНОВИК КАРТОЧКИ. Не публиковать без подтверждения Владимира. Цифры бенчмарка
     подставляются после обучения, сейчас плейсхолдеры. -->

# Ружев (Ru-Jev) — open Russian typed-decision model

Ружев (Ru-Jev) is an open, Russian-first typed-decision model. It does not generate
text. Given a context and a set of candidate options, it returns a calibrated
probability over the options and picks one. It handles the same decision shapes
as closed decision models: yes/no, choice from options, and ordinal scale.

It runs on CPU, so it can be deployed inside a closed perimeter where sending
data to an external API is not allowed.

**Not affiliated with TypeSafe AI. Not trained on Jev outputs.** Ружев is
trained only on open, human-labelled datasets (see Training data). The name is a
play on the category, not a claim of compatibility or endorsement.

## Why calibrated probabilities matter

The model returns a confidence, not just a label. That enables selective
routing: confident decisions go through automatically, uncertain ones go to a
human. In applied classification the bottleneck is usually the review step, not
raw accuracy, and a calibrated confidence is what lets you cut that review load
without losing control.

## Languages

Russian-first: the benchmark and the headline results are Russian. The backbone
(mmBERT) is multilingual, so English and other languages work as well.

## Usage

```python
# placeholder, finalized at release
from transformers import AutoTokenizer, AutoModel
# load backbone + scoring head + temperature from this repo
```

## Training data

Open datasets only, all permitting commercial use. See `DATASETS.md`.

- Russian SuperGLUE (MIT): terra, danetqa, russe, rwsd, parus, rcb
- RuReviews (Apache-2.0): sentiment scale
- RuDetoxifier data (MIT): toxicity
- RuCoLA (Apache-2.0 labels): acceptability

Non-commercial datasets are excluded.

## Results

| Task | Accuracy | ECE |
|---|---|---|
| (filled after training) | | |

## License

MIT. Author: Vladimir Beskorovainyi, Besk Tech (besk.tech).
