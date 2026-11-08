# The withdrawn half-split summary: 0.7948 against 0.7920

The paper lists as its third hazard a summary row computed on one random half of
the comparable rows, 0.7948 against 0.7920 on 5,078 decisions, printed beside a
per-task table counting all 10,157, with a calibration advantage of 0.0172
against 0.0160 read off that same single split. Both figures reached the
published model card and were withdrawn. This file records where they came from
and what can be checked about them.

## Where the figures came from

They were produced by a one-off comparison script on 2026-11-03 at 15:18 UTC,
which printed:

```
Шесть задач, где и строки, и формулировки вариантов совпадают у обеих версий.
Подбор температуры на одной половине, замер на другой.

         точность      ECE       T
v1.1      0.7948   0.0172   1.942
v1.2      0.7727   0.0190   2.020
v1.3      0.7920   0.0160   1.500

замер на 5078 решениях; разница точности -0.0028, это 0.3 сигмы
```

In English: six tasks where both the rows and the wording of the options
coincide for both releases; the temperature fitted on one half and the
measurement taken on the other; measured on 5,078 decisions; the difference in
accuracy is -0.0028, which is 0.3 sigma.

The script itself was not kept. `scripts/compare_rows.py` and `scripts/calib.py`
are the published successors of it, and they report over all rows rather than
over one half.

## What is wrong with them

Three things, and the paper names all three.

The accuracy pair comes from one half of the rows while the per-task table
printed next to it counts all 10,157, so a reader compares a number measured on
5,078 decisions with a table built on twice that.

On all the rows the two releases are exactly level, 8,120 correct each; the
difference of 0.0028 exists only on that half.

The calibration advantage of 0.0172 against 0.0160 was read off one split. Over
25 recorded splits the difference in residual ECE is +0.0001 on average, runs
from -0.0063 to +0.0109, and favours v1.3 in 14 of the 25. It changes sign
across splits, so no advantage can be claimed from it. The figures are in
`comparable.md` and in `calib.md`.

## What can be reproduced and what cannot

The figures on all the rows reproduce exactly, from
`comparable_v11.jsonl` and `comparable_v13.jsonl`:

    python scripts/compare_rows.py --rows results/fix18/comparable_rows.jsonl \
        --a "v11=results/fix18/comparable_v11.jsonl" \
        --b "v13=results/fix18/comparable_v13.jsonl"

giving 8,120 correct for each release over 10,157 rows, an overall accuracy of
0.7994 for both.

The withdrawn half-split pair does not reproduce. Four hundred candidate halves
were tried against the published dumps, 200 seeds with a fresh shuffle and 200
steps of the chained in-place shuffle that the old code used, and none gives
0.7948 for v1.1 and 0.7920 for v1.3 at the same time. Either the row set used
then differed slightly from the published `comparable_rows.jsonl`, or the split
came from an ordering that was not recorded. That is part of why the figures
were withdrawn rather than corrected: a number that cannot be recomputed from
the published artefacts does not belong in a paper.
