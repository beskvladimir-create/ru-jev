# vocab3: the reading rule, fixed before the figures

**The rule was recorded at 2026-11-05 08:53:11 +04.**

**An honest note about the order.** The requirement was that the rule be written
down before the measurement was started. The measurement was started earlier,
around 08:45, before that requirement arrived. At the moment the rule was
recorded no score file had been opened and no vocab3 accuracy figure had been
seen: `scores_v11.jsonl` finished at 08:52, `scores_v13.jsonl` was still being
written, and Laya had not been started. The letter of the requirement was
therefore broken and its purpose was kept: the rule was fixed before the result
was known. This can be checked against the session journal and the file
timestamps.

The vocabulary was fixed earlier, in `vocab.json`, sha256
`ae8b9463ad009d2812b57505b1cd672f0007c9f3587d494cedf2bb4bfa656c1b`, supplied
verbatim by the author. The rows: `rows.jsonl`, sha256
`419ab5189b8753aeef37fdfdfb35a39241ef355b625461c28b1f0f9b06229044`.

## Setting

The same 1,000 bench questions, but with **all 18 categories offered in every
question** rather than 5. Chance is 0.056, not 0.200. The figures here are NOT
comparable with the model card or with vocab2, where a question offers 5 options
out of the 18 categories. The option order comes from the crc32 of the text and
is the same for both styles and all models.

Models: `model/ru-jev-v1.1` (its root directory is what was published; v1.1 has
no `ship` subdirectory), `model/ru-jev-v1.3/ship`, and Laya with instruction A,
"К какой категории относится запрос пользователя?" ("Which category does the
user request belong to?").

## Three slices

| Slice | Style A | Style B |
|---|---|---|
| all 18 | 18 labels | 18 labels |
| clean by exact match | 17 (without `МЕД`) | 18 |
| strict, clean by both criteria | 9 | 14 |

Strict slice of style A: `БДК`, `ГРМ`, `КЛН`, `ВРМ`, `ЭП`, `УД`, `СПК`, `ТР`,
`ПГД`. Strict slice of style B: everything except `который час`, `розетки и
лампы`, `ящик`, `синоптик`. The labels are listed in `CODEBOOK.md` with their
English glosses.

## Statistics

The difference of v1.3 minus v1.1 on each slice is tested with a paired McNemar
test over questions: both releases answer the same rows, so the comparison is
paired. The counts are `b` (v1.1 right, v1.3 wrong) and `c` (v1.1 wrong, v1.3
right), with the p-value from the exact two-sided binomial test on `b` and `c`.
An exact test rather than chi-squared, because on the strict slices the number
of disagreements can be small.

## The reading rule

**The main slice is style B, strict, 14 labels.**

- v1.3 above v1.1 and p < 0.05 means the robustness carries over to slang.
- p >= 0.05 means it does not carry over and the difference is noise.
- v1.3 below v1.1 and p < 0.05 means the augmentation hurts on this kind of
  surface form.

**Style A is read by the same rule, but as a lower bound**, because code-like
abbreviations are closer to the trained forms than slang is.

For each slice the gap between mean confidence and accuracy is shown separately,
together with the number of confident errors at p_pick > 0.8, per 1,000
questions. Each model uses its own temperature: for Ru-Jev the released values,
2.2754 and 1.8640, and for Laya one.

This measurement did not change the model card, the landing page or the text of
the paper.

### The reading rule, original, as recorded

The paper refers to the time at which this rule was written down, so the Russian
original is kept here verbatim, exactly as it was recorded at 08:53:11 +04 on
2026-11-05. The English above is a translation of it.

```text
## Правило прочтения

**Главный срез: стиль B, строгий (14 меток).**

- v1.3 выше v1.1 и p < 0.05 → устойчивость переносится на сленг.
- p >= 0.05 → переноса нет, разница в шуме.
- v1.3 ниже v1.1 и p < 0.05 → расширение вредит на этом типе формы.

**Стиль A читается по тому же правилу, но как нижняя граница**, потому что
кодовые сокращения ближе к обученным формам, чем сленг.

Отдельно по каждому срезу показываются зазор уверенности минус точность и число
уверенных ошибок, при p_pick > 0.8, на 1000 вопросов. Температура у каждой
модели своя: у Ру-Жев из релиза, 2.2754 и 1.8640, у Laya единица.
```
