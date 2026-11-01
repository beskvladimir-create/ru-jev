# Источники данных и лицензии

Проверено 2026-11-01 по карточкам на Hugging Face. В сборку идут только наборы, разрешающие коммерческое использование. NC-наборы исключены.

| Набор | HF id | Лицензия | Коммерция | Что берём | Тип решения |
|---|---|---|---|---|---|
| Russian SuperGLUE | `RussianNLP/russian_super_glue` | MIT | да | terra, danetqa, russe, rwsd (да/нет), parus (выбор из 2), rcb (3 класса) | boolean, choice |
| RuReviews (MTEB) | `mteb/RuReviewsClassification` | Apache-2.0 | да | тональность 0/1/2 | scale |
| RuDetoxifier data | `d0rj/rudetoxifier_data` | MIT | да | токсичность 0/1 | boolean |
| RuCoLA | `RussianNLP/rucola` | Apache-2.0 (метки и код) | да, с оговоркой | приемлемость 0/1 | boolean |

## Оговорки
- **RuCoLA**: Apache-2.0 покрывает метки и код. У исходных предложений из лингвистических публикаций своё авторство. Мы обучаем на них свою модель, но не редистрибутируем сам корпус. Включается флагом `include_rucola` (по умолчанию включён, можно выключить для максимально чистой лицензии).
- **RuReviews**: пакет MTEB под Apache-2.0, исходник `github.com/sismetanin/rureviews`. Для релиза достаточно обучения на метках.

## Исключено (лицензия non-commercial, в коммерческий релиз НЕ идут)
- `Mikimi/RuAllToxic` — cc-by-nc-sa-4.0
- `mteb/SensitiveTopicsClassification`, `NiGuLa/Russian_Sensitive_Topics` — cc-by-nc-sa / cc
- прочие токсичные наборы без явной коммерческой лицензии

## Запрещено отдельно
- Не обучать на ответах API Jev (запрет дистилляции в условиях TypeSafe AI). Все источники выше это открытые датасеты с разметкой людьми, к Jev отношения не имеют.
