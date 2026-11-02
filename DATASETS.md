# Data sources and licences

Verified on 2026-11-01 against the Hugging Face dataset cards. Only sources that
permit commercial use go into the build. Non-commercial sets are excluded.

| Dataset | HF id | Licence | Commercial | What is used | Decision type |
|---|---|---|---|---|---|
| Russian SuperGLUE | `RussianNLP/russian_super_glue` | MIT | yes | terra, danetqa, russe, rwsd (yes/no), parus (choice of 2), rcb (3 classes) | boolean, choice |
| RuReviews (MTEB) | `mteb/RuReviewsClassification` | Apache-2.0 | yes | sentiment 0/1/2 | scale |
| RuDetoxifier data | `d0rj/rudetoxifier_data` | MIT | yes | toxicity 0/1 | boolean |
| RuCoLA | `RussianNLP/rucola` | Apache-2.0 (labels and code) | yes, with a caveat | acceptability 0/1 | boolean |
| Georeview | `mteb/GeoreviewClassification` | MIT | yes | review rating, 5 levels | scale |
| MASSIVE scenario | `mteb/amazon_massive_scenario` (ru) | Apache-2.0 on the mirror, CC-BY-4.0 upstream | yes | 18 scenarios | choice |
| MASSIVE intent | `mteb/amazon_massive_intent` (ru) | Apache-2.0 on the mirror, CC-BY-4.0 upstream | yes | 60 intents | choice |
| MERA | `ai-forever/MERA` (ruopenbookqa, ruworldtree) | MIT | yes | 4-way reasoning questions, train split only | choice |

## Caveats

- **RuCoLA.** Apache-2.0 covers the labels and the code. The underlying sentences are
  drawn from linguistic publications and carry their own authorship. The model is
  trained on them, the corpus itself is not redistributed. Controlled by the
  `include_rucola` flag, on by default, switch it off with `--no-rucola` for the
  cleanest possible licence position.
- **RuReviews.** The MTEB package is Apache-2.0, the original source is
  `github.com/sismetanin/rureviews`. Training on the labels is sufficient for the
  release.
- **MASSIVE.** The corpus is Amazon's MASSIVE, released under CC-BY-4.0, which
  requires attribution: FitzGerald et al., MASSIVE, Amazon Science. The MTEB
  mirror used here carries an Apache-2.0 tag. Both permit commercial use. The
  upstream `AmazonScience/massive` repository is a loading script and no longer
  opens with current `datasets`, which is why the mirror is used.
- **MERA.** MIT. Only the labelled `train` splits are used. The benchmark also
  ships parus, rcb and rwsd, which are already present through Russian
  SuperGLUE, so those configs are left out to avoid contaminating the dev splits.
- **Loading.** Russian SuperGLUE and RuCoLA are script-based datasets and stopped
  opening in `datasets` 4.0. The builder falls back to their auto-converted
  parquet branch, so the build works on both old and current versions.

## Excluded, non-commercial licence

- `Mikimi/RuAllToxic`, cc-by-nc-sa-4.0
- `mteb/SensitiveTopicsClassification` and `NiGuLa/Russian_Sensitive_Topics`,
  cc-by-nc-sa and cc
- other toxicity sets without an explicit commercial licence

## Excluded on principle

Nothing here is trained on the outputs of a closed decision API. Distillation from
such an API is forbidden by its terms, and every source above is an open dataset with
human annotation that has no connection to it.
