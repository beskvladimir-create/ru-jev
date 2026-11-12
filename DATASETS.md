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

## Evaluation-only datasets

Neither release trained on these. They are used in the paper for the
unseen-dataset comparison and are not part of any training mixture.

| Dataset | HF id | Licence in metadata | Commercial | What is used | Decision type |
|---|---|---|---|---|---|
| Headline topics | `ai-forever/headline-classification` | MIT | yes | 6 news topics, test split, 1,000 sampled rows | choice |
| CEDR emotions | `sagteam/cedr_v1` | Apache-2.0 | yes | 5 emotions, rows with exactly one label, 1,000 sampled rows | choice |
| GRNTI scientific rubrics | `ai-forever/ru-scibench-grnti-classification` | MIT | yes | 10 most frequent rubrics of the test split, 990 rows, 5 candidates per question | choice |

The same corpora are mirrored by MTEB as `mteb/HeadlineClassification` and
`mteb/CEDRClassification`; the figures in the paper come from the two sources
above. Category names are taken from the source, from the `label_text` field for
headlines and from `ClassLabel(names=...)` for CEDR, because the model reads the
option text and an invented name would corrupt the measurement. Only
single-label CEDR rows are used: multi-label rows do not reduce to choosing one
candidate, and an empty label set means absence of emotion, a class that is not
among the five.

**Licence caveat for both.** The licence is declared only in the `license` field
of the repository's README metadata. Neither repository, nor either MTEB mirror,
ships a LICENSE file. That is a statement by the repository owner rather than a
licence document, and it is not a legal opinion. Checked 2026-11-05.

## Pinned revisions

Every figure in the paper comes from these exact revisions. Verified
2026-11-05 against the Hugging Face API.

| Repository | Kind | Revision |
|---|---|---|
| `RussianNLP/russian_super_glue` | dataset | `53cb9d7f38b3` |
| `mteb/RuReviewsClassification` | dataset | `e6a7fcaca9cd` |
| `d0rj/rudetoxifier_data` | dataset | `7509275056f8` |
| `RussianNLP/rucola` | dataset | `00e61d446ceb` |
| `mteb/GeoreviewClassification` | dataset | `fb29d6137c54` |
| `mteb/amazon_massive_scenario` | dataset | `58871793b91a` |
| `mteb/amazon_massive_intent` | dataset | `940fd47a81ea` |
| `ai-forever/MERA` | dataset | `e9628afed9c1` |
| `ai-forever/headline-classification` | dataset | `1e9d989d65f6` |
| `sagteam/cedr_v1` | dataset | `abafbe63cf92` |
| `ai-forever/ru-scibench-grnti-classification` | dataset | `a984f8a65b2e` |
| `jhu-clsp/mmBERT-base` | model, backbone | `c5955035435e` |
| `convaiinnovations/laya-multilingual` | model, comparison | `1720e3e3357c` |

None of the ten datasets ships a LICENSE file in its Hugging Face repository;
every licence in the tables above is the `license` field of the repository's
README metadata, except Georeview, where the upstream Yandex repository on
GitHub does ship an MIT `LICENSE.md` as noted below.

## Caveats

- **RuCoLA.** Apache-2.0 covers the labels and the code. The underlying sentences are
  drawn from linguistic publications and carry their own authorship. The model is
  trained on them, the corpus itself is not redistributed. Controlled by the
  `include_rucola` flag, on by default, switch it off with `--no-rucola` for the
  cleanest possible licence position.
- **Georeview.** The MTEB card is MIT and the upstream Yandex repository ships an
  unmodified MIT `LICENSE.md`, which grants the right to use, sell and
  sublicense without restriction. The upstream README separately asks readers to
  "use it for academic and research purposes". That sentence sits in the README,
  not in the licence, and does not narrow the grant, so the set stays in the
  build. Stated as what the two files say, not as a legal opinion. Checked
  2026-11-04.
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
