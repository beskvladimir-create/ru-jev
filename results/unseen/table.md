
## рубрики новостей (news rubrics)

| Model | Accuracy | Mean confidence | Gap | ECE as is | Fitted T | ECE after its own T |
|---|---|---|---|---|---|---|
| v1.1 | 0.652 | 0.680 | +0.028 | 0.2037 | 2.759 | 0.0674 |
| v1.3 | 0.568 | 0.663 | +0.095 | 0.2452 | 2.462 | 0.0599 |
| laya | 0.610 | 0.852 | +0.242 | 0.2422 | 2.237 | 0.0545 |

Rows 1000, candidates 6, chance 0.167.

## эмоции (CEDR emotions)

| Model | Accuracy | Mean confidence | Gap | ECE as is | Fitted T | ECE after its own T |
|---|---|---|---|---|---|---|
| v1.1 | 0.455 | 0.540 | +0.085 | 0.2868 | 3.430 | 0.0564 |
| v1.3 | 0.511 | 0.447 | -0.064 | 0.0815 | 1.546 | 0.0587 |
| laya | 0.605 | 0.661 | +0.056 | 0.0708 | 1.381 | 0.0766 |

Rows 1000, candidates 5, chance 0.200.

Rebuilt with the current `scripts/unseen_table.py`, which takes its half-splits
from `calib.py` with the recorded `SPLIT_SEED = 20261105`. Accuracy, mean
confidence, the gap and the raw ECE are unchanged from the earlier edition of
this file, row for row. The fitted temperature and the residual ECE moved,
because the earlier edition was produced before the split seed was recorded:
2.754 to 2.759, 2.467 to 2.462, 3.391 to 3.430, 1.547 to 1.546 for the two
releases, and 2.182 to 2.237 and 1.364 to 1.381 for the comparison model. The
figures here now agree with `../fix18/calib.md`, which is what the paper cites.

Reproduce with:

    python scripts/unseen_table.py \
        "v1.1=results/unseen/unseen_v11.jsonl" \
        "v1.3=results/unseen/unseen_v13.jsonl" \
        "laya=<unseen_laya_headline.jsonl and unseen_laya_cedr.jsonl, concatenated>"
