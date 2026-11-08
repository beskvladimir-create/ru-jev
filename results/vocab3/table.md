
## style A

### all 18: 18 labels, 1000 questions

| Model | Accuracy | Mean confidence | Gap | Confident errors per 1000 |
|---|---|---|---|---|
| ru-jev-v1.1 | 0.104 | 0.361 | +0.257 | 60 |
| ru-jev-v1.3 | 0.146 | 0.411 | +0.265 | 39 |
| laya instr A | 0.087 | 0.444 | +0.357 | 83 |

McNemar, v1.3 against v1.1: difference +0.042, only v1.1 correct 35, only v1.3 correct 77, disagreements 112, p = 8.98e-05

### clean by exact match: 17 labels, 865 questions

| Model | Accuracy | Mean confidence | Gap | Confident errors per 1000 |
|---|---|---|---|---|
| ru-jev-v1.1 | 0.120 | 0.364 | +0.244 | 60 |
| ru-jev-v1.3 | 0.169 | 0.412 | +0.243 | 43 |
| laya instr A | 0.097 | 0.445 | +0.348 | 81 |

McNemar, v1.3 against v1.1: difference +0.049, only v1.1 correct 35, only v1.3 correct 77, disagreements 112, p = 8.98e-05

### strict, clean by both criteria: 9 labels, 512 questions

| Model | Accuracy | Mean confidence | Gap | Confident errors per 1000 |
|---|---|---|---|---|
| ru-jev-v1.1 | 0.018 | 0.353 | +0.335 | 84 |
| ru-jev-v1.3 | 0.061 | 0.327 | +0.266 | 31 |
| laya instr A | 0.088 | 0.429 | +0.341 | 88 |

McNemar, v1.3 against v1.1: difference +0.043, only v1.1 correct 6, only v1.3 correct 28, disagreements 34, p = 0.0001951

Candidates per question 18, chance 0.056.

## style B

### all 18: 18 labels, 1000 questions

| Model | Accuracy | Mean confidence | Gap | Confident errors per 1000 |
|---|---|---|---|---|
| ru-jev-v1.1 | 0.428 | 0.652 | +0.224 | 130 |
| ru-jev-v1.3 | 0.476 | 0.683 | +0.207 | 154 |
| laya instr A | 0.229 | 0.590 | +0.361 | 181 |

McNemar, v1.3 against v1.1: difference +0.048, only v1.1 correct 141, only v1.3 correct 189, disagreements 330, p = 0.009569

### clean by exact match: 18 labels, 1000 questions

| Model | Accuracy | Mean confidence | Gap | Confident errors per 1000 |
|---|---|---|---|---|
| ru-jev-v1.1 | 0.428 | 0.652 | +0.224 | 130 |
| ru-jev-v1.3 | 0.476 | 0.683 | +0.207 | 154 |
| laya instr A | 0.229 | 0.590 | +0.361 | 181 |

McNemar, v1.3 against v1.1: difference +0.048, only v1.1 correct 141, only v1.3 correct 189, disagreements 330, p = 0.009569

### strict, clean by both criteria: 14 labels, 773 questions

| Model | Accuracy | Mean confidence | Gap | Confident errors per 1000 |
|---|---|---|---|---|
| ru-jev-v1.1 | 0.369 | 0.624 | +0.255 | 151 |
| ru-jev-v1.3 | 0.419 | 0.694 | +0.275 | 191 |
| laya instr A | 0.171 | 0.588 | +0.417 | 203 |

McNemar, v1.3 against v1.1: difference +0.050, only v1.1 correct 106, only v1.3 correct 145, disagreements 251, p = 0.01629

Candidates per question 18, chance 0.056.

### style A: accuracy per label

| Label | surface form | n | v1.1 | v1.3 | laya | v1.3 minus v1.1 |
|---|---|---|---|---|---|---|
| music | МУЗ | 20 | 0.150 | 0.000 | 0.100 | -0.150 |
| cooking | КУХ | 22 | 0.864 | 0.727 | 0.591 | -0.136 |
| qa | СПР | 106 | 0.104 | 0.000 | 0.047 | -0.104 |
| recommendation | СОВ | 41 | 0.024 | 0.000 | 0.024 | -0.024 |
| email | ЭП | 86 | 0.035 | 0.012 | 0.093 | -0.023 |
| alarm | БДК | 31 | 0.000 | 0.000 | 0.032 | +0.000 |
| audio | ГРМ | 22 | 0.000 | 0.000 | 0.318 | +0.000 |
| play (exact match in training) | МЕД | 135 | 0.000 | 0.000 | 0.022 | +0.000 |
| weather | ПГД | 57 | 0.000 | 0.000 | 0.035 | +0.000 |
| calendar | КЛН | 127 | 0.008 | 0.047 | 0.110 | +0.039 |
| transport | ТР | 47 | 0.043 | 0.106 | 0.000 | +0.064 |
| lists | СПК | 58 | 0.017 | 0.121 | 0.190 | +0.103 |
| datetime | ВРМ | 35 | 0.057 | 0.171 | 0.000 | +0.114 |
| general | ОБЩ | 68 | 0.397 | 0.515 | 0.029 | +0.118 |
| iot | УД | 49 | 0.000 | 0.122 | 0.041 | +0.122 |
| takeaway | ДОСТ | 24 | 0.417 | 0.625 | 0.125 | +0.208 |
| news | НОВ | 39 | 0.359 | 0.692 | 0.103 | +0.333 |
| social | СОЦ | 33 | 0.303 | 0.667 | 0.273 | +0.364 |

Where v1.3's errors go: news 119, qa 93, general 90, recommendation 65, datetime 64, weather 61

### style B: accuracy per label

| Label | surface form | n | v1.1 | v1.3 | laya | v1.3 minus v1.1 |
|---|---|---|---|---|---|---|
| lists | перечни | 58 | 0.534 | 0.086 | 0.052 | -0.448 |
| news | сводка | 39 | 0.846 | 0.436 | 0.103 | -0.410 |
| email | ящик | 86 | 0.849 | 0.616 | 0.337 | -0.233 |
| music | пластинки | 20 | 0.550 | 0.350 | 0.200 | -0.200 |
| play | развлекалово | 135 | 0.281 | 0.148 | 0.081 | -0.133 |
| recommendation | что посоветуешь | 41 | 0.927 | 0.805 | 0.220 | -0.122 |
| datetime | который час | 35 | 0.857 | 0.743 | 0.657 | -0.114 |
| transport | извозчик | 47 | 0.872 | 0.787 | 0.574 | -0.085 |
| audio | крутилка звука | 22 | 0.955 | 0.955 | 0.591 | +0.000 |
| cooking | поварёнок | 22 | 0.227 | 0.227 | 0.091 | +0.000 |
| social | лента друзей | 33 | 0.727 | 0.727 | 0.455 | +0.000 |
| takeaway | курьерка | 24 | 0.042 | 0.083 | 0.417 | +0.042 |
| iot | розетки и лампы | 49 | 0.653 | 0.735 | 0.694 | +0.082 |
| alarm | побудка | 31 | 0.000 | 0.161 | 0.000 | +0.161 |
| calendar | ежедневник | 127 | 0.291 | 0.488 | 0.157 | +0.197 |
| general | болталка | 68 | 0.074 | 0.353 | 0.029 | +0.279 |
| weather | синоптик | 57 | 0.140 | 0.649 | 0.193 | +0.509 |
| qa | справочное бюро | 106 | 0.000 | 0.585 | 0.113 | +0.585 |

Where v1.3's errors go: lists 116, email 89, news 41, calendar 38, weather 37, transport 28
