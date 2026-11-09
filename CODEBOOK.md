# Codebook

Every file in `results/`, every field in it, and every Russian value that
appears as data. The Russian is the object of study, not an oversight: the
model decides between Russian label strings, and the experiments are about what
happens when those strings are rewritten. Translating them inside the data would
destroy the measurement, so they are kept as they are and glossed here.

The question texts are kept in Russian for the same reason. They are the
utterances the model reads, taken verbatim from the source datasets listed in
`DATASETS.md`; a translation would be a different input and would score
differently. Where a text is quoted in prose in this repository, an English
rendering is given beside it.

## 1. Conventions used in every file

| Field | Type | Meaning |
|---|---|---|
| `model` | string | which model produced the row: `v1.1`, `v1.3`, `ru-jev-v1.1`, `ru-jev-v1.3`, `laya`, or a seed run name |
| `text` | string | the question, in Russian, exactly as the source dataset has it |
| `input` | string | the same thing, in the task validation format |
| `options` | list of strings | the candidates offered for this decision, in the order they were shown |
| `answer` | integer | the index in `options` of the correct candidate |
| `pick` | integer | the index the model chose, the argmax of its scores |
| `logits` | list of floats | the raw score of each candidate, one per option, with no temperature applied |
| `probs` | list of floats | probabilities as the comparison model returns them, already normalised |
| `p_pick` | float | the probability of the chosen candidate |
| `scores` | list of floats | the same as `logits`, under the name the validation dump uses |
| `task` | string | which of the thirteen training tasks a row comes from |
| `decision_type` | string | `boolean`, `choice` or `scale` |
| `row_id` | integer | a stable row number, used to line up two releases on the same rows |
| `vocab` | string | which label vocabulary this row was rendered in, see section 3 |
| `set` | string | which dataset or probe the row belongs to, see section 2 |
| `gold` | string | the correct label as a string, for the unseen datasets |
| `gold_label` | string | the correct scenario key, for the bench and the probes |
| `gold_clean` | boolean | true when the correct label has no exact match in the training mixture |
| `seen_from` | string | the first release whose training mixture contains this vocabulary, or `нигде` |
| `licence` | string | the licence of the source dataset, from its README metadata |
| `instr_id` | string | which instruction the comparison model was given, `A` or `B` |
| `truncated` | boolean | whether the input hit the length limit |

Three values of `model` need care. `v1.1` and `ru-jev-v1.1` are the same
release under two spellings, because the dumps were written by two different
scripts; the same holds for v1.3. The tag matters when a script keys a
temperature off it, which `scripts/unseen_table.py` does.

`seen_from` takes the values `v1`, `v1.1`, `v1.2` and `нигде`. The Russian word
`нигде` means "nowhere" and marks a vocabulary that is in no training mixture,
which is what makes it a held-out test.

## 2. Values of `set`

| Value | English | What it is |
|---|---|---|
| `рубрики новостей` | news rubrics | the headline topic dataset, 6 candidates, evaluation only |
| `эмоции` | emotions | the CEDR emotion dataset, 5 candidates, evaluation only |
| `стиль_A` | style A | the first of the two probe styles, see section 4 |
| `стиль_B` | style B | the second of the two probe styles |
| `fix18` | | the wording bench recomputed with all 18 candidates per question |

### Labels of the news rubrics

| Russian | English |
|---|---|
| `политика` | politics |
| `экономика` | economy |
| `спорт` | sport |
| `наука` | science |
| `культура` | culture |
| `происшествия` | incidents |

### Labels of the emotions

| Russian | English | Source class |
|---|---|---|
| `радость` | joy | `joy` |
| `грусть` | sadness | `sadness` |
| `удивление` | surprise | `surprise` |
| `страх` | fear | `fear` |
| `злость` | anger | `anger` |

## 3. The six vocabularies of the wording bench

The values of `vocab` in `results/bench_*.jsonl`, `results/fix18/*.jsonl` and
`results/seeds/*.jsonl`.

| Value | English | What the style is |
|---|---|---|
| `английский ключ` | English key | the MASSIVE scenario key itself, a training style |
| `русское слово` | Russian word | one Russian word per scenario, a training style |
| `русская фраза` | Russian phrase | a short phrase, in v1.3 training only |
| `жаргон` | anglicism jargon | industry slang, held out from both releases |
| `названия служб` | department name | the name of a team that would handle it, held out |
| `описание запроса` | descriptive sentence | a sentence describing the request, held out |

All eighteen labels of each, next to the English scenario key they render:

| Scenario key | Meaning | русское слово | русская фраза | жаргон | названия служб | описание запроса |
|---|---|---|---|---|---|---|
| `alarm` | alarms and reminders | будильник | вопрос про будильники | алармы | служба напоминаний | человек спрашивает про будильник |
| `audio` | sound control | звук | управление звуком | аудио | отдел звука | человек спрашивает про звук |
| `calendar` | calendar events | календарь | работа с календарём | календарь-ивенты | отдел планирования | человек спрашивает про календарь |
| `cooking` | cooking and recipes | кулинария | готовка и рецепты | фудтех | кулинарная редакция | человек спрашивает про кулинария |
| `datetime` | date and time | дата и время | вопрос про дату и время | таймзоны | служба времени | человек спрашивает про дата и время |
| `email` | email | почта | работа с почтой | мейлы | почтовая служба | человек спрашивает про почта |
| `general` | small talk and general questions | общий вопрос | болтовня и общие вопросы | смолток | общая линия | человек спрашивает про общий вопрос |
| `iot` | smart home | умный дом | управление умным домом | ай-оу-ти | инженерная служба | человек спрашивает про умный дом |
| `lists` | lists | списки | работа со списками | листы | отдел списков | человек спрашивает про списки |
| `music` | music and player | музыка | музыка и плеер | плейлисты | музыкальная редакция | человек спрашивает про музыка |
| `news` | news | новости | новости | ньюс | новостная редакция | человек спрашивает про новости |
| `play` | play something | воспроизведение | включить что-нибудь | плеер | служба воспроизведения | человек спрашивает про воспроизведение |
| `qa` | factual question | справка | справочный вопрос | факт-чек | справочная служба | человек спрашивает про справка |
| `recommendation` | recommendations | рекомендации | посоветовать что-нибудь | реко | отдел рекомендаций | человек спрашивает про рекомендации |
| `social` | social networks | соцсети | социальные сети | сошиал | отдел соцсетей | человек спрашивает про соцсети |
| `takeaway` | food delivery | доставка еды | заказ и доставка еды | фуд-доставка | служба доставки | человек спрашивает про доставка еды |
| `transport` | transport | транспорт | поездки и транспорт | логистика | транспортная служба | человек спрашивает про транспорт |
| `weather` | weather | погода | прогноз погоды | вэзер | метеослужба | человек спрашивает про погода |

Three jargon labels coincide with the output of the training rule
`jargon_translit` and are excluded from the clean figure: `аудио`, `ньюс` and
`вэзер`. Fifteen remain. `results/fix18/AUDIT.md` checks that count.

## 4. The two probe vocabularies

### vocab2, a code identifier and a three-letter abbreviation

Built by a rule from the existing bench vocabularies, with no manual selection.
Style A is the English scenario key in capitals with the suffix `_SCN`, as an
enum constant would be written. Style B is the first three letters of the
Russian scenario word in capital Cyrillic, as an internal abbreviation.

| Scenario key | Style A | Style B | Style B is the start of |
|---|---|---|---|
| `alarm` | `ALARM_SCN` | `БУД` | `будильник` |
| `audio` | `AUDIO_SCN` | `ЗВУ` | `звук` |
| `calendar` | `CALENDAR_SCN` | `КАЛ` | `календарь` |
| `cooking` | `COOKING_SCN` | `КУЛ` | `кулинария` |
| `datetime` | `DATETIME_SCN` | `ДАТ` | `дата` |
| `email` | `EMAIL_SCN` | `ПОЧ` | `почта` |
| `general` | `GENERAL_SCN` | `ОБЩ` | `общий` |
| `iot` | `IOT_SCN` | `УМН` | `умный дом` |
| `lists` | `LISTS_SCN` | `СПИ` | `списки` |
| `music` | `MUSIC_SCN` | `МУЗ` | `музыка` |
| `news` | `NEWS_SCN` | `НОВ` | `новости` |
| `play` | `PLAY_SCN` | `ВОС` | `воспроизведение` |
| `qa` | `QA_SCN` | `СПР` | `справка` |
| `recommendation` | `RECOMMENDATION_SCN` | `РЕК` | `рекомендации` |
| `social` | `SOCIAL_SCN` | `СОЦ` | `соцсети` |
| `takeaway` | `TAKEAWAY_SCN` | `ДОС` | `доставка` |
| `transport` | `TRANSPORT_SCN` | `ТРА` | `транспорт` |
| `weather` | `WEATHER_SCN` | `ПОГ` | `погода` |

`РЕК` is excluded from the clean figure for style B: it occurs literally among
the training options of both mixtures.

### vocab3, everyday slang and an opaque code

Supplied verbatim by the author before any figure was seen, and not changed
afterwards. Style B is slang a person might actually use, style A is an opaque
three-letter code that carries no visible relation to the English key. This is
the probe the paper reports as the independent one.

| Scenario key | Style A, code | Style B, slang | Style B in English | Where the code comes from |
|---|---|---|---|---|
| `alarm` | `БДК` | `побудка` | wake-up call | from будильник, alarm clock |
| `audio` | `ГРМ` | `крутилка звука` | volume knob | from громкость, volume |
| `calendar` | `КЛН` | `ежедневник` | day planner | from календарь, calendar |
| `cooking` | `КУХ` | `поварёнок` | kitchen boy | from кухня, kitchen |
| `datetime` | `ВРМ` | `который час` | what's the time | from время, time |
| `email` | `ЭП` | `ящик` | mailbox | from электронная почта, email |
| `general` | `ОБЩ` | `болталка` | chit-chat | from общий, general |
| `iot` | `УД` | `розетки и лампы` | sockets and lamps | from умный дом, smart home |
| `lists` | `СПК` | `перечни` | itemised lists | from списки, lists |
| `music` | `МУЗ` | `пластинки` | records | from музыка, music |
| `news` | `НОВ` | `сводка` | bulletin | from новости, news |
| `play` | `МЕД` | `развлекалово` | entertainment stuff | from медиа, media |
| `qa` | `СПР` | `справочное бюро` | enquiry desk | from справка, reference |
| `recommendation` | `СОВ` | `что посоветуешь` | what do you recommend | from совет, advice |
| `social` | `СОЦ` | `лента друзей` | friends' feed | from соцсети, social networks |
| `takeaway` | `ДОСТ` | `курьерка` | courier run | from доставка, delivery |
| `transport` | `ТР` | `извозчик` | cabman | from транспорт, transport |
| `weather` | `ПГД` | `синоптик` | weather forecaster | from прогноз, forecast |

Style B has no exact match with training at all. In style A, `МЕД` matches
exactly and is excluded from the clean figure. The strict slices, clean by both
the exact and the prefix criterion, hold 9 labels in style A and 14 in style B;
`results/vocab3/README.md` lists which.

## 5. The keys of the overlap reports

`results/fix18/overlap_bench.json`, `results/vocab2/overlap.json` and
`results/vocab3/overlap.json` are produced by `scripts/overlap.py`. Their keys
are in Russian because they are the field names the script writes.

| Key | English | Meaning |
|---|---|---|
| `описание` | description | a free-text description of the vocabulary |
| `метки` | labels | the map from scenario key to the label string |
| `метка` | label | the label string of this scenario |
| `совпадения` | matches | per training mixture, the kinds of relation found |
| `чистая` | clean | true when the label has no relation to training of any kind |

The kinds of relation inside `совпадения`:

| Value | English | Meaning |
|---|---|---|
| `точное` | exact | the label is literally among the training options, ignoring case |
| `обуч_с_метки` | training-starts-with-label | some training option begins with the label; the risk for short abbreviations, since `КАЛ` is the start of `календарь` |
| `метка_с_обуч` | label-starts-with-training | the label begins with a training option; the risk for codes, since `ALARM_SCN` begins with `alarm` |

The headline "clean" figure of every probe uses `точное` alone, which is the
criterion the model card documents. The two prefix relations are reported
separately and never silently folded in.

One example row, from `results/fix18/overlap_bench.json`:

```json
{"audio": {"метка": "аудио",
           "совпадения": {"v1.3": ["точное", "обуч_с_метки"]},
           "чистая": false}}
```

## 6. File by file

### At the root of `results/`

| File | Rows | Contents |
|---|---|---|
| `bench_rujev_v11.jsonl`, `bench_rujev_v13.jsonl` | 6,000 | the wording bench at five candidates, raw logits, fields `model, vocab, text, gold_clean, answer, pick, p_pick, probs, logits` |
| `bench_laya_instrA.jsonl`, `bench_laya_instrB.jsonl` | 6,000 | the same rows scored by the comparison model under two instructions; `probs` instead of `logits`, plus `instr_id` and `truncated` |
| `table.md` | | the summary of the four files above |
| `README.md` | | how the bench was built and what cannot be read off it |

### `results/fix18/`, the bench with all 18 candidates

| File | Rows | Contents |
|---|---|---|
| `rows.jsonl` | 6,000 | the questions as asked, fields `text, options, answer, vocab, gold_label, set, gold_clean` |
| `scores_v11.jsonl`, `scores_v13.jsonl` | 6,000 | raw logits of each release |
| `scores_laya_instrA.jsonl` | 6,000 | the comparison model on the same rows |
| `vocab.json` | | the six vocabularies, section 3 |
| `overlap_bench.json` | | the audit of all six against both training mixtures, section 5 |
| `comparable_rows.jsonl` | 10,157 | the rows on which the two releases can be compared, where both the text and the option wording coincide |
| `comparable_v11.jsonl`, `comparable_v13.jsonl` | 10,157 | each release on those rows |
| `comparable_byrow.jsonl` | 10,157 | the per-row verdict, fields `row_id, task, answer, pick_v11, pick_v13, ok_v11, ok_v13` |
| `comparable.md` | | the counts, with the weight hashes |
| `calib_bench.json`, `calib_unseen.json` | | temperature and ECE over 25 half-splits, keyed `<set or vocab>|<model>` |
| `calib.md` | | the same as a table |
| `latency_rujev.json`, `latency_laya.json` | | median latency per decision at 2, 5 and 18 candidates, under `by_k` |
| `table.md` | | the full summary with a per-label breakdown |
| `AUDIT.md` | | the audit of the bench against training, the dates, toxicity and calibration |
| `half_split_withdrawn.md` | | where the withdrawn 0.7948 against 0.7920 came from |

### `results/vocab2/` and `results/vocab3/`, the probes

Both hold the same shape: `vocab.json` the vocabulary, `rows.jsonl` the
questions, `scores_v11.jsonl` and `scores_v13.jsonl` the raw logits,
`scores_laya_instrA.jsonl` the comparison model, `overlap.json` the audit
against training, `table.md` the summary, `FINDINGS.md` the write-up. vocab3
additionally holds `README.md`, which carries the reading rule and the timestamp
at which it was recorded.

### `results/unseen/`, the two datasets no model trained on

| File | Rows | Contents |
|---|---|---|
| `unseen_rows.jsonl` | 2,000 | the questions, fields `set, licence, text, gold, options, answer` |
| `unseen_v11.jsonl`, `unseen_v13.jsonl` | 2,000 | raw logits of each release |
| `unseen_laya_headline.jsonl`, `unseen_laya_cedr.jsonl` | 1,000 each | the comparison model on each dataset |
| `table.md` | | accuracy, confidence and calibration per dataset |
| `bias.md` | | what one additive bias per option does, out of fold |
| `ladder/<checkpoint>.jsonl` | 1,000 each | the raw logits of eleven checkpoints on the news rubrics, same fields as `unseen_v13.jsonl`; the `model` field names the checkpoint |
| `ladder.md` | | where between v1.1 and v1.3 the economy and politics boundary moves |
| `FINDINGS.md` | | the write-up, including what cannot be claimed |

### `results/validation/`, the released validation split

| File | Rows | Contents |
|---|---|---|
| `val_v13_trainsplit.jsonl` | 26,746 | v1.3 on the split it was measured on, fields `task, answer, scores` |
| `split_overlap_v11.json`, `split_overlap_v13.json` | | how much of a rebuilt split had been in the earlier training split |
| `split_overlap.md` | | both episodes side by side, with the hashes of every build |
| `README.md` | | the three accuracy figures and why a rebuild does not recover the split |

### `results/seeds/`, the seed replicates

| File | Rows | Contents |
|---|---|---|
| `seed-base-s2_bench.jsonl`, `seed-base-s3_bench.jsonl` | 6,000 | replicates of the v1.1 recipe on other seeds |
| `seed-aug-s2_bench.jsonl`, `seed-aug-s2-ep3_bench.jsonl`, `seed-aug-s3_bench.jsonl` | 6,000 | runs that trained on a stale mixture and are not replicates of v1.3 |
| `README.md` | | the per-vocabulary table and which run is which |
| `FINDINGS.md` | | the mistake that made the augmented runs void |

### `results/training/` and `results/analysis/`

`results/training/` holds the standard output of the released training run and
the per-epoch trajectory read off it. `results/analysis/` holds the scripts that
compute the paper's tables, their output in `final.txt`, and the per-label
tables copied from the probes.

## 7. Values that look like data and are not

`нигде` in `seen_from` means "nowhere" and is a marker, not a label.

`стиль_A` and `стиль_B` in `set` mean "style A" and "style B" and index the two
halves of a probe, not two datasets.

In `model`, `laya` is the comparison model
`convaiinnovations/laya-multilingual`, pinned in `DATASETS.md`. `instr_id` `A`
and `B` are its two instructions, which are given in full in
`results/README.md`.

Composite keys. Several JSON reports key a figure by more than one thing at
once, and they do it by joining the parts with a vertical bar, `|`, with no
spaces: `<vocabulary or set>|<model>`. The left part is a value of `vocab` or of
`set` as section 2 and section 3 list them, the right part is a value of
`model`. So `жаргон|ru-jev-v1.3` is the jargon vocabulary as scored by v1.3, and
`рубрики новостей|ru-jev-v1.1` is the news rubrics as scored by v1.1. Neither is
a label. `results/fix18/calib_bench.json` and `calib_unseen.json` use this form,
and `scripts/calib.py` writes it. A key whose left part is `all` means the
figure was computed over every row rather than one slice.

