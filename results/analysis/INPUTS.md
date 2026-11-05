# Какие файлы нужны, чтобы это пересчитать

`final.py` читает построчные выгрузки оценок: по строке на решение, с сырыми
логитами без температуры у Ру-Жев и с вероятностями у Laya. Эти выгрузки здесь
не лежат: их около 38 МБ, и в репозиторий модели им не место.

Что в этом каталоге есть и чего достаточно для проверки чисел без пересчёта:

| файл | что это |
|---|---|
| `final.txt` | все ячейки таблицы 2 препринта: точность обеих версий, разница, интервал по меткам, счётчики Макнемара `b` и `c`, сырое и поправленное по Холму p, уверенные ошибки на 1000 |
| `per-label-fix18.md` | доля верных по каждой из 18 метк, по шести словарям основного стенда |
| `per-label-vocab2.md` | то же для кодовых идентификаторов и трёхбуквенных сокращений |
| `per-label-vocab3.md` | то же для сленга и непрозрачных кодов, со срезами |

Чтобы пересчитать с нуля, выгрузки собираются скриптами из `../../scripts/`:
`score_rows.py` для Ру-Жев, `score_laya.py` для Laya. Строки стендов строятся
`build_unseen.py` и скриптами сборки в каталогах замеров. Ожидаемое дерево
входных данных, относительно каталога, который передаётся `final.py` первым
аргументом:

    bench_rujev_v11.jsonl, bench_rujev_v13.jsonl, bench_laya_instrA.jsonl
    fix18/{rows,scores_v11,scores_v13,scores_laya_instrA}.jsonl
    fix18/{vocab,overlap_bench}.json
    vocab2/{rows,scores_v11,scores_v13,scores_laya_instrA}.jsonl
    vocab3/{rows,scores_v11,scores_v13,scores_laya_instrA}.jsonl, vocab3/vocab.json
    unseen/{unseen_rows,unseen_v11,unseen_v13}.jsonl
    unseen/{unseen_laya_headline,unseen_laya_cedr}.jsonl

Нужны `numpy` и `scipy`, они есть в `requirements.txt` в корне.
