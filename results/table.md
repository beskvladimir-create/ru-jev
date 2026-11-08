## Wording bench: the same 1,000 questions, 18 categories, five candidates per question

| Vocabulary | ru-jev-v1.1 | ru-jev-v1.3 | laya instr A | laya instr B |
|---|---|---|---|---|
| русское слово | 0.956 | 0.960 | 0.696 | 0.681 |
| английский ключ | 0.957 | 0.962 | 0.655 | 0.649 |
| описание запроса | 0.939 | 0.958 | 0.652 | 0.649 |
| русская фраза | 0.881 | 0.957 | 0.668 | 0.673 |
| названия служб | 0.835 | 0.869 | 0.557 | 0.552 |
| жаргон | 0.633 | 0.855 | 0.394 | 0.419 |

### Anglicism jargon in detail

| Model | 18 labels | 15 clean | Mean confidence | Confidence minus accuracy | Confident errors per 1000 |
|---|---|---|---|---|---|
| ru-jev-v1.1 | 0.633 | 0.624 | 0.754 | +0.121 | 85 |
| ru-jev-v1.3 | 0.855 | 0.838 | 0.885 | +0.030 | 44 |
| laya instr A | 0.394 | 0.422 | 0.630 | +0.236 | 110 |
| laya instr B | 0.419 | 0.449 | 0.650 | +0.231 | 125 |

### Calibration on all 6,000 bench rows

| Model | ECE as is | Fitted T | ECE after its own T |
|---|---|---|---|
| ru-jev-v1.1 | 0.0867 | 3.009 | 0.0171 |
| ru-jev-v1.3 | 0.0473 | 2.474 | 0.0103 |
| laya instr A | 0.0950 | 1.515 | 0.0268 |
| laya instr B | 0.1061 | 1.569 | 0.0269 | (rows left out of the fit: 2)
