# Calibration with a recorded split seed

## The whole bench, 6,000 rows

SPLIT_SEED = 20261105, splits 25, ECE bins 15

| Model | n | Fitted T | T min | T max | ECE as is | ECE after its own T |
|---|---|---|---|---|---|---|
| ru-jev-v1.1 | 6000 | 2.9955 | 2.904 | 3.138 | 0.0867 | 0.0182 |
| ru-jev-v1.3 | 6000 | 2.4870 | 2.358 | 2.618 | 0.0473 | 0.0111 |
| laya instr A | 6000 | 1.5156 | 1.474 | 1.578 | 0.0950 | 0.0267 |


## The unseen datasets

SPLIT_SEED = 20261105, splits 25, ECE bins 15

## рубрики новостей (news rubrics)

| Model | n | Fitted T | T min | T max | ECE as is | ECE after its own T |
|---|---|---|---|---|---|---|
| ru-jev-v1.1 | 1000 | 2.7594 | 2.540 | 2.930 | 0.2037 | 0.0674 |
| ru-jev-v1.3 | 1000 | 2.4620 | 2.280 | 2.592 | 0.2452 | 0.0599 |

## эмоции (CEDR emotions)

| Model | n | Fitted T | T min | T max | ECE as is | ECE after its own T |
|---|---|---|---|---|---|---|
| ru-jev-v1.1 | 1000 | 3.4302 | 3.242 | 3.710 | 0.2868 | 0.0564 |
| ru-jev-v1.3 | 1000 | 1.5458 | 1.422 | 1.630 | 0.0815 | 0.0587 |

figures: `calib_unseen.json` in this directory
