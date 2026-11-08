v1.1 backbone: sha256 8aca6840cad16019ea7039efd24ab9a9974090207dd1c011bdb16d4d61794c5d  (1227771776 bytes)  model/ru-jev-v1.1/model.safetensors
v1.3 backbone: sha256 7e07664cf8bb677a9285a034c73920cc96853c0aebdc885e10e3b98812e3596d  (1227771776 bytes)  model/ru-jev-v1.3/ship/model.safetensors
v1.1 head: sha256 f5632bb843eefd055a9e62460e3b247b90d83843efcba5cac05054bffd6b6740  (3244 bytes)  model/ru-jev-v1.1/head.safetensors
v1.3 head: sha256 724740455841c6c4c14ff53489b8c9e3956952d2cbf32ab1884b1f0855508bcf  (3244 bytes)  model/ru-jev-v1.3/ship/head.safetensors

rows 10157
both correct: 7280
only v11 correct: 840
only v13 correct: 840
neither correct: 1197
total correct: v11 8120, v13 8120
disagreements: 1680

| Task | both correct | only v11 | only v13 | neither | n |
|---|---|---|---|---|---|
| russe | 6341 | 650 | 667 | 847 | 8505 |
| danetqa | 551 | 58 | 67 | 145 | 821 |
| terra | 199 | 39 | 26 | 43 | 307 |
| parus | 46 | 22 | 13 | 19 | 100 |
| rcb | 76 | 23 | 36 | 85 | 220 |
| rwsd | 67 | 48 | 31 | 58 | 204 |

per-row dump: `comparable_byrow.jsonl` in this directory

## Calibration on the 10,157 comparable rows, SPLIT_SEED = 20261105

| Quantity | v1.1 | v1.3 |
|---|---|---|
| Fitted T, mean | 2.0054 | 1.5593 |
| T, range over the 25 splits | 1.942 to 2.098 | 1.500 to 1.604 |
| ECE after its own T | 0.0190 | 0.0191 |

The temperature of v1.3 is lower in 25 of the 25 splits. The ranges do NOT overlap.
ECE difference, v1.3 minus v1.1: mean +0.0001, from -0.0063 to +0.0109, in favour of v1.3 in 14 of the 25 splits.
