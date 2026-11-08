# Summary analysis for the preprint

`final.py` computes every cell of Table 2 of the preprint, in both settings, 5
candidates and 18, together with the exact McNemar test, the Holm correction
over 23 cells, the label-level bootstrap interval and the count of confident
errors per 1,000 decisions. The five-candidate values for the vocab3 and fix18
vocabularies are obtained from the eighteen-candidate logits by renormalising
over the same five options the main bench uses. For Ru-Jev this is exact,
because each option is scored on its own. For Laya it is not valid, and its
five-candidate column is not used in the output.

Run from the root of this repository:

    python3 -I results/analysis/final.py results > results/analysis/final.txt

The bootstrap seed is 20261105. `analyze.py` and `fiveway.py` are intermediate
versions, kept so that the history of the figures can be reproduced.
