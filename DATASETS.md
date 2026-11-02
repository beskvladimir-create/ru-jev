# Data sources and licences

Verified on 2026-11-01 against the Hugging Face dataset cards. Only sources that
permit commercial use go into the build. Non-commercial sets are excluded.

| Dataset | HF id | Licence | Commercial | What is used | Decision type |
|---|---|---|---|---|---|
| Russian SuperGLUE | `RussianNLP/russian_super_glue` | MIT | yes | terra, danetqa, russe, rwsd (yes/no), parus (choice of 2), rcb (3 classes) | boolean, choice |
| RuReviews (MTEB) | `mteb/RuReviewsClassification` | Apache-2.0 | yes | sentiment 0/1/2 | scale |
| RuDetoxifier data | `d0rj/rudetoxifier_data` | MIT | yes | toxicity 0/1 | boolean |
| RuCoLA | `RussianNLP/rucola` | Apache-2.0 (labels and code) | yes, with a caveat | acceptability 0/1 | boolean |

## Caveats

- **RuCoLA.** Apache-2.0 covers the labels and the code. The underlying sentences are
  drawn from linguistic publications and carry their own authorship. The model is
  trained on them, the corpus itself is not redistributed. Controlled by the
  `include_rucola` flag, on by default, switch it off with `--no-rucola` for the
  cleanest possible licence position.
- **RuReviews.** The MTEB package is Apache-2.0, the original source is
  `github.com/sismetanin/rureviews`. Training on the labels is sufficient for the
  release.

## Excluded, non-commercial licence

- `Mikimi/RuAllToxic`, cc-by-nc-sa-4.0
- `mteb/SensitiveTopicsClassification` and `NiGuLa/Russian_Sensitive_Topics`,
  cc-by-nc-sa and cc
- other toxicity sets without an explicit commercial licence

## Excluded on principle

Nothing here is trained on the outputs of a closed decision API. Distillation from
such an API is forbidden by its terms, and every source above is an open dataset with
human annotation that has no connection to it.
