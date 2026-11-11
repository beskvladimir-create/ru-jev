# Biases, per-label contributions and seed spread on the probe vocabularies

Reproduce with:

    python3 -I results/analysis/vocab_bias.py

Everything here is computed from the per-row score files already in this
repository. No weights, no network, no new measurement.

## What is asked

The wording bench and the two probes are read three ways that the published
tables do not read them.

**How much of a difference is one label.** `results/unseen/bias.md` showed that
on the news rubrics almost all of what separates the releases is a lean towards
one rubric. The same question is put to every vocabulary and slice here, by
giving each label's contribution to the difference in questions and by
recomputing the difference with that label left out.

**What survives a per-option correction.** Each release is allowed its own
additive bias per option, fitted out of fold with the machinery of
`results/analysis/unseen_bias.py`: 25 half-splits seeded 0 to 24, each used
twice, bootstrap from seed 20261105 with 5,000 draws, nothing fitted and scored
on the same rows.

**How the difference compares with the spread between seeds.** Three runs of
the v1.1 recipe are published, the release and two replicates. A difference
smaller than the gap between them is not evidence about a recipe.

## Two things to know before reading

**The five-candidate option strings of the bench are recovered, not stored.**
The dump that `robustness.py --dump` writes was a working file and is not in the
repository. `results/vocab2/rows.jsonl` keeps, per question, the five scenario
keys of that question shuffled once more by the crc32 of the text; inverting
that shuffle gives the bench's own order back. The check is that the position of
the correct answer in the recovered order equals the `answer` field stored in
every bench score file, for all 1,000 questions and all six vocabularies. It
passes, and the script stops if it ever does not. Rows are matched by position
inside a vocabulary block and never by text, because seven of the thousand
questions repeat verbatim.

**The biases of a slice are fitted on the whole vocabulary.** A slice keeps the
questions whose correct label is in it and leaves the option lists alone, so on
a slice some options are never the right answer. Fitting on the slice itself
then deletes them: on the strict jargon slice seven of the eighteen options
never occur as the answer and their fitted biases run from -20 to the bound at
-50, while the eleven that do occur stay within plus or minus 2.5. That is the
slice definition leaking into the correction, and a caller could not reproduce
it, because a caller does not know which labels are impossible. The headline
figure for every slice is therefore fitted on all 1,000 questions of that
vocabulary and only then read on the slice. The naive figure is printed beside
it, marked, and it is the larger one.

Per-option biases, per-label contributions and seed spread on the vocabularies
Out of fold: 25 half-splits, seeds 0 to 24, both halves scored. Bootstrap 5000 draws, seed 20261105.


## английский ключ, 5 candidates

### all: 18 labels, 1000 questions

| Run | accuracy |
|---|---|
| v1.3 released | 0.962 |
| seed-base-s3 | 0.960 |
| seed-aug-s3 | 0.959 |
| v1.1 released | 0.957 |
| seed-aug-s2-ep3 | 0.955 |
| seed-base-s2 | 0.951 |
| seed-aug-s2 | 0.947 |

| v1.3 minus v1.1 | difference | 95% over rows | 95% over labels |
|---|---|---|---|
| as published | +0.005 | [-0.006, +0.017] | [-0.007, +0.018] |
| after biases and T | +0.002 | [-0.008, +0.013] | [-0.007, +0.014] |
| v1.1 with biases | 0.950 | | |
| v1.3 with biases | 0.952 | | |

| Label | n | net questions, v1.3 minus v1.1 | difference without it |
|---|---|---|---|
| news | 39 | +3 | +0.002 |
| play | 135 | +3 | +0.002 |
| weather | 57 | +3 | +0.002 |
| qa | 106 | -2 | +0.008 |

Leave one label out: lowest +0.002 without `news`, highest +0.008 without `qa`, against +0.005 on every label.
By label v1.3 is ahead on 6 and behind on 5 of 18.


## английский ключ, 18 candidates

### all: 18 labels, 1000 questions

| Run | accuracy |
|---|---|
| v1.3 released | 0.900 |
| v1.1 released | 0.894 |

| v1.3 minus v1.1 | difference | 95% over rows | 95% over labels |
|---|---|---|---|
| as published | +0.006 | [-0.010, +0.022] | [-0.011, +0.025] |
| after biases and T | -0.000 | [-0.014, +0.014] | [-0.011, +0.013] |
| v1.1 with biases | 0.891 | | |
| v1.3 with biases | 0.891 | | |

| Label | n | net questions, v1.3 minus v1.1 | difference without it |
|---|---|---|---|
| general | 68 | +5 | +0.001 |
| qa | 106 | -4 | +0.011 |
| weather | 57 | +4 | +0.002 |

Leave one label out: lowest +0.001 without `general`, highest +0.011 without `qa`, against +0.006 on every label.
By label v1.3 is ahead on 5 and behind on 6 of 18.


## русское слово, 5 candidates

### all: 18 labels, 1000 questions

| Run | accuracy |
|---|---|
| v1.3 released | 0.960 |
| seed-aug-s3 | 0.959 |
| seed-aug-s2 | 0.958 |
| seed-aug-s2-ep3 | 0.958 |
| seed-base-s3 | 0.957 |
| v1.1 released | 0.956 |
| seed-base-s2 | 0.953 |

| v1.3 minus v1.1 | difference | 95% over rows | 95% over labels |
|---|---|---|---|
| as published | +0.004 | [-0.008, +0.016] | [-0.009, +0.014] |
| after biases and T | +0.004 | [-0.005, +0.014] | [-0.002, +0.013] |
| v1.1 with biases | 0.951 | | |
| v1.3 with biases | 0.955 | | |

| Label | n | net questions, v1.3 minus v1.1 | difference without it |
|---|---|---|---|
| qa | 106 | +3 | +0.001 |
| email | 86 | +2 | +0.002 |
| iot | 49 | +2 | +0.002 |

Leave one label out: lowest +0.001 without `qa`, highest +0.006 without `social`, against +0.004 on every label.
By label v1.3 is ahead on 6 and behind on 7 of 18.


## русское слово, 18 candidates

### all: 18 labels, 1000 questions

| Run | accuracy |
|---|---|
| v1.3 released | 0.892 |
| v1.1 released | 0.883 |

| v1.3 minus v1.1 | difference | 95% over rows | 95% over labels |
|---|---|---|---|
| as published | +0.009 | [-0.007, +0.025] | [-0.017, +0.038] |
| after biases and T | +0.007 | [-0.007, +0.021] | [-0.011, +0.030] |
| v1.1 with biases | 0.881 | | |
| v1.3 with biases | 0.887 | | |

| Label | n | net questions, v1.3 minus v1.1 | difference without it |
|---|---|---|---|
| general | 68 | +11 | -0.002 |
| datetime | 35 | -4 | +0.013 |
| music | 20 | -4 | +0.013 |
| qa | 106 | -2 | +0.012 |

Leave one label out: lowest -0.002 without `general`, highest +0.013 without `datetime`, against +0.009 on every label.
By label v1.3 is ahead on 8 and behind on 5 of 18.


## русская фраза, 5 candidates

### all: 18 labels, 1000 questions

| Run | accuracy |
|---|---|
| v1.3 released | 0.957 |
| seed-aug-s2 | 0.957 |
| seed-aug-s2-ep3 | 0.950 |
| seed-aug-s3 | 0.950 |
| seed-base-s2 | 0.902 |
| seed-base-s3 | 0.893 |
| v1.1 released | 0.881 |

| v1.3 minus v1.1 | difference | 95% over rows | 95% over labels |
|---|---|---|---|
| as published | +0.076 | [+0.056, +0.096] | [+0.022, +0.133] |
| after biases and T | +0.052 | [+0.037, +0.067] | [+0.031, +0.078] |
| v1.1 with biases | 0.906 | | |
| v1.3 with biases | 0.958 | | |

| Label | n | net questions, v1.3 minus v1.1 | difference without it |
|---|---|---|---|
| play | 135 | +27 | +0.057 |
| general | 68 | +23 | +0.057 |
| iot | 49 | +9 | +0.070 |
| qa | 106 | +7 | +0.077 |

Leave one label out: lowest +0.057 without `play`, highest +0.087 without `calendar`, against +0.076 on every label.
By label v1.3 is ahead on 10 and behind on 4 of 18.


## русская фраза, 18 candidates

### all: 18 labels, 1000 questions

| Run | accuracy |
|---|---|
| v1.3 released | 0.885 |
| v1.1 released | 0.703 |

| v1.3 minus v1.1 | difference | 95% over rows | 95% over labels |
|---|---|---|---|
| as published | +0.182 | [+0.153, +0.211] | [+0.041, +0.343] |
| after biases and T | +0.086 | [+0.065, +0.107] | [+0.044, +0.133] |
| v1.1 with biases | 0.795 | | |
| v1.3 with biases | 0.880 | | |

| Label | n | net questions, v1.3 minus v1.1 | difference without it |
|---|---|---|---|
| play | 135 | +92 | +0.104 |
| general | 68 | +31 | +0.162 |
| iot | 49 | +24 | +0.166 |
| qa | 106 | +22 | +0.179 |

Leave one label out: lowest +0.104 without `play`, highest +0.212 without `calendar`, against +0.182 on every label.
By label v1.3 is ahead on 7 and behind on 5 of 18.


## жаргон, 5 candidates

### all: 18 labels, 1000 questions

| Run | accuracy |
|---|---|
| v1.3 released | 0.855 |
| seed-base-s2 | 0.658 |
| v1.1 released | 0.633 |
| seed-base-s3 | 0.622 |
| seed-aug-s2 | 0.619 |
| seed-aug-s2-ep3 | 0.611 |
| seed-aug-s3 | 0.588 |

| v1.3 minus v1.1 | difference | 95% over rows | 95% over labels |
|---|---|---|---|
| as published | +0.222 | [+0.192, +0.252] | [+0.096, +0.374] |
| after biases and T | +0.182 | [+0.155, +0.211] | [+0.092, +0.311] |
| v1.1 with biases | 0.708 | | |
| v1.3 with biases | 0.890 | | |

| Label | n | net questions, v1.3 minus v1.1 | difference without it |
|---|---|---|---|
| qa | 106 | +74 | +0.166 |
| weather | 57 | +30 | +0.204 |
| iot | 49 | +28 | +0.204 |

Leave one label out: lowest +0.166 without `qa`, highest +0.249 without `calendar`, against +0.222 on every label.
By label v1.3 is ahead on 14 and behind on 3 of 18.

### clean: 15 labels, 882 questions

| Run | accuracy |
|---|---|
| v1.3 released | 0.838 |
| seed-base-s2 | 0.645 |
| v1.1 released | 0.624 |
| seed-aug-s2 | 0.610 |
| seed-base-s3 | 0.599 |
| seed-aug-s2-ep3 | 0.595 |
| seed-aug-s3 | 0.574 |

| v1.3 minus v1.1 | difference | 95% over rows | 95% over labels |
|---|---|---|---|
| as published | +0.214 | [+0.181, +0.247] | [+0.079, +0.391] |
| after biases and T | +0.160 | [+0.131, +0.188] | [+0.080, +0.287] |
| v1.1 with biases | 0.721 | | |
| v1.3 with biases | 0.881 | | |
| after biases fitted on the slice itself | +0.140 | | |

The biases behind the headline row were fitted on all 1000 questions of this vocabulary and then read on the slice. Fitting them on the slice itself gives 0.886 against 0.747, which is larger because on this slice some options are never the answer and the fit removes them.

| Label | n | net questions, v1.3 minus v1.1 | difference without it |
|---|---|---|---|
| qa | 106 | +74 | +0.148 |
| iot | 49 | +28 | +0.193 |
| social | 33 | +23 | +0.196 |

Leave one label out: lowest +0.148 without `qa`, highest +0.244 without `email`, against +0.214 on every label.
By label v1.3 is ahead on 11 and behind on 3 of 15.

### strict: 11 labels, 625 questions

| Run | accuracy |
|---|---|
| v1.3 released | 0.786 |
| seed-base-s2 | 0.648 |
| seed-aug-s3 | 0.586 |
| seed-aug-s2 | 0.578 |
| seed-aug-s2-ep3 | 0.565 |
| seed-base-s3 | 0.563 |
| v1.1 released | 0.550 |

| v1.3 minus v1.1 | difference | 95% over rows | 95% over labels |
|---|---|---|---|
| as published | +0.235 | [+0.194, +0.275] | [+0.051, +0.447] |
| after biases and T | +0.145 | [+0.109, +0.180] | [+0.056, +0.275] |
| v1.1 with biases | 0.712 | | |
| v1.3 with biases | 0.858 | | |
| after biases fitted on the slice itself | +0.089 | | |

The biases behind the headline row were fitted on all 1000 questions of this vocabulary and then read on the slice. Fitting them on the slice itself gives 0.885 against 0.797, which is larger because on this slice some options are never the answer and the fit removes them.

| Label | n | net questions, v1.3 minus v1.1 | difference without it |
|---|---|---|---|
| qa | 106 | +74 | +0.141 |
| iot | 49 | +28 | +0.207 |
| social | 33 | +23 | +0.209 |

Leave one label out: lowest +0.141 without `qa`, highest +0.282 without `email`, against +0.235 on every label.
By label v1.3 is ahead on 7 and behind on 3 of 11.


## жаргон, 18 candidates

### all: 18 labels, 1000 questions

| Run | accuracy |
|---|---|
| v1.3 released | 0.640 |
| v1.1 released | 0.431 |

| v1.3 minus v1.1 | difference | 95% over rows | 95% over labels |
|---|---|---|---|
| as published | +0.209 | [+0.175, +0.244] | [+0.056, +0.390] |
| after biases and T | +0.217 | [+0.186, +0.250] | [+0.088, +0.387] |
| v1.1 with biases | 0.538 | | |
| v1.3 with biases | 0.755 | | |

| Label | n | net questions, v1.3 minus v1.1 | difference without it |
|---|---|---|---|
| qa | 106 | +55 | +0.172 |
| weather | 57 | +55 | +0.163 |
| iot | 49 | +33 | +0.185 |

Leave one label out: lowest +0.163 without `weather`, highest +0.255 without `play`, against +0.209 on every label.
By label v1.3 is ahead on 13 and behind on 4 of 18.

### clean: 15 labels, 882 questions

| Run | accuracy |
|---|---|
| v1.3 released | 0.595 |
| v1.1 released | 0.427 |

| v1.3 minus v1.1 | difference | 95% over rows | 95% over labels |
|---|---|---|---|
| as published | +0.168 | [+0.133, +0.203] | [+0.028, +0.344] |
| after biases and T | +0.176 | [+0.144, +0.208] | [+0.065, +0.333] |
| v1.1 with biases | 0.557 | | |
| v1.3 with biases | 0.733 | | |
| after biases fitted on the slice itself | +0.157 | | |

The biases behind the headline row were fitted on all 1000 questions of this vocabulary and then read on the slice. Fitting them on the slice itself gives 0.736 against 0.579, which is larger because on this slice some options are never the answer and the fit removes them.

| Label | n | net questions, v1.3 minus v1.1 | difference without it |
|---|---|---|---|
| qa | 106 | +55 | +0.120 |
| iot | 49 | +33 | +0.138 |
| alarm | 31 | +24 | +0.146 |

Leave one label out: lowest +0.120 without `qa`, highest +0.214 without `play`, against +0.168 on every label.
By label v1.3 is ahead on 10 and behind on 4 of 15.

### strict: 11 labels, 625 questions

| Run | accuracy |
|---|---|
| v1.3 released | 0.470 |
| v1.1 released | 0.328 |

| v1.3 minus v1.1 | difference | 95% over rows | 95% over labels |
|---|---|---|---|
| as published | +0.142 | [+0.098, +0.186] | [-0.040, +0.357] |
| after biases and T | +0.151 | [+0.113, +0.191] | [+0.023, +0.301] |
| v1.1 with biases | 0.519 | | |
| v1.3 with biases | 0.670 | | |
| after biases fitted on the slice itself | +0.101 | | |

The biases behind the headline row were fitted on all 1000 questions of this vocabulary and then read on the slice. Fitting them on the slice itself gives 0.691 against 0.590, which is larger because on this slice some options are never the answer and the fit removes them.

| Label | n | net questions, v1.3 minus v1.1 | difference without it |
|---|---|---|---|
| qa | 106 | +55 | +0.066 |
| iot | 49 | +33 | +0.097 |
| social | 33 | +22 | +0.113 |

Leave one label out: lowest +0.066 without `qa`, highest +0.206 without `play`, against +0.142 on every label.
By label v1.3 is ahead on 6 and behind on 4 of 11.


## названия служб, 5 candidates

### all: 18 labels, 1000 questions

| Run | accuracy |
|---|---|
| seed-aug-s3 | 0.885 |
| seed-aug-s2 | 0.883 |
| seed-aug-s2-ep3 | 0.879 |
| v1.3 released | 0.869 |
| v1.1 released | 0.835 |
| seed-base-s2 | 0.828 |
| seed-base-s3 | 0.828 |

| v1.3 minus v1.1 | difference | 95% over rows | 95% over labels |
|---|---|---|---|
| as published | +0.034 | [+0.014, +0.054] | [-0.031, +0.115] |
| after biases and T | +0.044 | [+0.027, +0.061] | [+0.021, +0.073] |
| v1.1 with biases | 0.849 | | |
| v1.3 with biases | 0.893 | | |

| Label | n | net questions, v1.3 minus v1.1 | difference without it |
|---|---|---|---|
| qa | 106 | +36 | -0.002 |
| general | 68 | -14 | +0.052 |
| iot | 49 | +7 | +0.028 |

Leave one label out: lowest -0.002 without `qa`, highest +0.052 without `general`, against +0.034 on every label.
By label v1.3 is ahead on 6 and behind on 5 of 18.

### strict: 16 labels, 933 questions

| Run | accuracy |
|---|---|
| seed-aug-s3 | 0.879 |
| seed-aug-s2 | 0.876 |
| seed-aug-s2-ep3 | 0.872 |
| v1.3 released | 0.863 |
| v1.1 released | 0.827 |
| seed-base-s2 | 0.820 |
| seed-base-s3 | 0.817 |

| v1.3 minus v1.1 | difference | 95% over rows | 95% over labels |
|---|---|---|---|
| as published | +0.035 | [+0.015, +0.057] | [-0.035, +0.121] |
| after biases and T | +0.044 | [+0.026, +0.062] | [+0.018, +0.077] |
| v1.1 with biases | 0.847 | | |
| v1.3 with biases | 0.891 | | |
| after biases fitted on the slice itself | +0.043 | | |

The biases behind the headline row were fitted on all 1000 questions of this vocabulary and then read on the slice. Fitting them on the slice itself gives 0.893 against 0.851, which is larger because on this slice some options are never the answer and the fit removes them.

| Label | n | net questions, v1.3 minus v1.1 | difference without it |
|---|---|---|---|
| qa | 106 | +36 | -0.004 |
| general | 68 | -14 | +0.054 |
| iot | 49 | +7 | +0.029 |

Leave one label out: lowest -0.004 without `qa`, highest +0.054 without `general`, against +0.035 on every label.
By label v1.3 is ahead on 5 and behind on 5 of 16.


## названия служб, 18 candidates

### all: 18 labels, 1000 questions

| Run | accuracy |
|---|---|
| v1.3 released | 0.733 |
| v1.1 released | 0.632 |

| v1.3 minus v1.1 | difference | 95% over rows | 95% over labels |
|---|---|---|---|
| as published | +0.101 | [+0.074, +0.127] | [+0.000, +0.235] |
| after biases and T | +0.069 | [+0.047, +0.092] | [+0.028, +0.118] |
| v1.1 with biases | 0.677 | | |
| v1.3 with biases | 0.746 | | |

| Label | n | net questions, v1.3 minus v1.1 | difference without it |
|---|---|---|---|
| qa | 106 | +68 | +0.037 |
| calendar | 127 | +17 | +0.096 |
| general | 68 | -12 | +0.121 |

Leave one label out: lowest +0.037 without `qa`, highest +0.121 without `general`, against +0.101 on every label.
By label v1.3 is ahead on 10 and behind on 3 of 18.

### strict: 16 labels, 933 questions

| Run | accuracy |
|---|---|
| v1.3 released | 0.726 |
| v1.1 released | 0.623 |

| v1.3 minus v1.1 | difference | 95% over rows | 95% over labels |
|---|---|---|---|
| as published | +0.103 | [+0.075, +0.131] | [-0.006, +0.246] |
| after biases and T | +0.070 | [+0.046, +0.092] | [+0.026, +0.125] |
| v1.1 with biases | 0.670 | | |
| v1.3 with biases | 0.740 | | |
| after biases fitted on the slice itself | +0.067 | | |

The biases behind the headline row were fitted on all 1000 questions of this vocabulary and then read on the slice. Fitting them on the slice itself gives 0.743 against 0.675, which is larger because on this slice some options are never the answer and the fit removes them.

| Label | n | net questions, v1.3 minus v1.1 | difference without it |
|---|---|---|---|
| qa | 106 | +68 | +0.034 |
| calendar | 127 | +17 | +0.098 |
| general | 68 | -12 | +0.125 |

Leave one label out: lowest +0.034 without `qa`, highest +0.125 without `general`, against +0.103 on every label.
By label v1.3 is ahead on 9 and behind on 3 of 16.


## описание запроса, 5 candidates

### all: 18 labels, 1000 questions

| Run | accuracy |
|---|---|
| v1.3 released | 0.958 |
| seed-aug-s2-ep3 | 0.954 |
| seed-aug-s2 | 0.952 |
| seed-aug-s3 | 0.952 |
| seed-base-s3 | 0.951 |
| v1.1 released | 0.939 |
| seed-base-s2 | 0.939 |

| v1.3 minus v1.1 | difference | 95% over rows | 95% over labels |
|---|---|---|---|
| as published | +0.019 | [+0.005, +0.033] | [+0.005, +0.031] |
| after biases and T | +0.009 | [-0.002, +0.021] | [+0.001, +0.016] |
| v1.1 with biases | 0.943 | | |
| v1.3 with biases | 0.952 | | |

| Label | n | net questions, v1.3 minus v1.1 | difference without it |
|---|---|---|---|
| email | 86 | +6 | +0.014 |
| general | 68 | +4 | +0.016 |
| calendar | 127 | +3 | +0.018 |
| qa | 106 | +3 | +0.018 |

Leave one label out: lowest +0.014 without `email`, highest +0.022 without `recommendation`, against +0.019 on every label.
By label v1.3 is ahead on 9 and behind on 2 of 18.


## описание запроса, 18 candidates

### all: 18 labels, 1000 questions

| Run | accuracy |
|---|---|
| v1.3 released | 0.878 |
| v1.1 released | 0.848 |

| v1.3 minus v1.1 | difference | 95% over rows | 95% over labels |
|---|---|---|---|
| as published | +0.030 | [+0.011, +0.049] | [+0.007, +0.050] |
| after biases and T | +0.024 | [+0.009, +0.039] | [+0.011, +0.037] |
| v1.1 with biases | 0.857 | | |
| v1.3 with biases | 0.880 | | |

| Label | n | net questions, v1.3 minus v1.1 | difference without it |
|---|---|---|---|
| email | 86 | +8 | +0.024 |
| general | 68 | +6 | +0.026 |
| lists | 58 | +5 | +0.027 |
| qa | 106 | +5 | +0.028 |

Leave one label out: lowest +0.024 without `email`, highest +0.034 without `recommendation`, against +0.030 on every label.
By label v1.3 is ahead on 10 and behind on 5 of 18.


## коды _SCN, 5 candidates

### all: 18 labels, 1000 questions

| Run | accuracy |
|---|---|
| v1.3 released | 0.889 |
| v1.1 released | 0.856 |

| v1.3 minus v1.1 | difference | 95% over rows | 95% over labels |
|---|---|---|---|
| as published | +0.033 | [+0.012, +0.054] | [+0.001, +0.082] |
| after biases and T | +0.024 | [+0.008, +0.040] | [+0.010, +0.044] |
| v1.1 with biases | 0.880 | | |
| v1.3 with biases | 0.904 | | |

| Label | n | net questions, v1.3 minus v1.1 | difference without it |
|---|---|---|---|
| iot | 49 | +18 | +0.016 |
| takeaway | 24 | +7 | +0.027 |
| play | 135 | +5 | +0.032 |
| qa | 106 | -1 | +0.038 |

Leave one label out: lowest +0.016 without `iot`, highest +0.039 without `calendar`, against +0.033 on every label.
By label v1.3 is ahead on 8 and behind on 6 of 18.

### strict: 1 labels, skipped


## аббревиатуры, 5 candidates

### all: 18 labels, 1000 questions

| Run | accuracy |
|---|---|
| v1.1 released | 0.403 |
| v1.3 released | 0.379 |

| v1.3 minus v1.1 | difference | 95% over rows | 95% over labels |
|---|---|---|---|
| as published | -0.024 | [-0.061, +0.013] | [-0.179, +0.170] |
| after biases and T | +0.015 | [-0.014, +0.046] | [-0.064, +0.129] |
| v1.1 with biases | 0.451 | | |
| v1.3 with biases | 0.467 | | |

| Label | n | net questions, v1.3 minus v1.1 | difference without it |
|---|---|---|---|
| qa | 106 | -52 | +0.031 |
| recommendation | 41 | +34 | -0.060 |
| lists | 58 | -31 | +0.007 |

Leave one label out: lowest -0.060 without `recommendation`, highest +0.031 without `qa`, against -0.024 on every label.
By label v1.3 is ahead on 11 and behind on 6 of 18.

### clean: 17 labels, 959 questions

| Run | accuracy |
|---|---|
| v1.1 released | 0.420 |
| v1.3 released | 0.360 |

| v1.3 minus v1.1 | difference | 95% over rows | 95% over labels |
|---|---|---|---|
| as published | -0.060 | [-0.098, -0.024] | [-0.206, +0.115] |
| after biases and T | +0.000 | [-0.030, +0.031] | [-0.076, +0.110] |
| v1.1 with biases | 0.467 | | |
| v1.3 with biases | 0.467 | | |
| after biases fitted on the slice itself | -0.002 | | |

The biases behind the headline row were fitted on all 1000 questions of this vocabulary and then read on the slice. Fitting them on the slice itself gives 0.476 against 0.478, which is larger because on this slice some options are never the answer and the fit removes them.

| Label | n | net questions, v1.3 minus v1.1 | difference without it |
|---|---|---|---|
| qa | 106 | -52 | -0.007 |
| lists | 58 | -31 | -0.030 |
| calendar | 127 | -29 | -0.035 |

Leave one label out: lowest -0.092 without `alarm`, highest -0.007 without `qa`, against -0.060 on every label.
By label v1.3 is ahead on 10 and behind on 6 of 17.


## коды, 5 candidates

### all: 18 labels, 1000 questions

| Run | accuracy |
|---|---|
| v1.3 released | 0.270 |
| v1.1 released | 0.265 |

| v1.3 minus v1.1 | difference | 95% over rows | 95% over labels |
|---|---|---|---|
| as published | +0.005 | [-0.025, +0.034] | [-0.078, +0.080] |
| after biases and T | -0.013 | [-0.030, +0.005] | [-0.047, +0.030] |
| v1.1 with biases | 0.373 | | |
| v1.3 with biases | 0.360 | | |

| Label | n | net questions, v1.3 minus v1.1 | difference without it |
|---|---|---|---|
| qa | 106 | -28 | +0.037 |
| iot | 49 | +13 | -0.008 |
| lists | 58 | +12 | -0.007 |

Leave one label out: lowest -0.008 without `iot`, highest +0.037 without `qa`, against +0.005 on every label.
By label v1.3 is ahead on 9 and behind on 9 of 18.

### clean: 17 labels, 865 questions

| Run | accuracy |
|---|---|
| v1.1 released | 0.298 |
| v1.3 released | 0.298 |

| v1.3 minus v1.1 | difference | 95% over rows | 95% over labels |
|---|---|---|---|
| as published | +0.000 | [-0.032, +0.031] | [-0.093, +0.088] |
| after biases and T | -0.007 | [-0.025, +0.012] | [-0.046, +0.039] |
| v1.1 with biases | 0.321 | | |
| v1.3 with biases | 0.315 | | |
| after biases fitted on the slice itself | +0.002 | | |

The biases behind the headline row were fitted on all 1000 questions of this vocabulary and then read on the slice. Fitting them on the slice itself gives 0.406 against 0.404, which is larger because on this slice some options are never the answer and the fit removes them.

| Label | n | net questions, v1.3 minus v1.1 | difference without it |
|---|---|---|---|
| qa | 106 | -28 | +0.037 |
| iot | 49 | +13 | -0.016 |
| lists | 58 | +12 | -0.015 |

Leave one label out: lowest -0.016 without `iot`, highest +0.037 without `qa`, against +0.000 on every label.
By label v1.3 is ahead on 8 and behind on 9 of 17.

### strict: 9 labels, 512 questions

| Run | accuracy |
|---|---|
| v1.3 released | 0.168 |
| v1.1 released | 0.135 |

| v1.3 minus v1.1 | difference | 95% over rows | 95% over labels |
|---|---|---|---|
| as published | +0.033 | [-0.002, +0.068] | [-0.036, +0.117] |
| after biases and T | -0.001 | [-0.024, +0.022] | [-0.043, +0.043] |
| v1.1 with biases | 0.319 | | |
| v1.3 with biases | 0.318 | | |
| after biases fitted on the slice itself | +0.002 | | |

The biases behind the headline row were fitted on all 1000 questions of this vocabulary and then read on the slice. Fitting them on the slice itself gives 0.552 against 0.550, which is larger because on this slice some options are never the answer and the fit removes them.

| Label | n | net questions, v1.3 minus v1.1 | difference without it |
|---|---|---|---|
| iot | 49 | +13 | +0.009 |
| lists | 58 | +12 | +0.011 |
| transport | 47 | -7 | +0.052 |

Leave one label out: lowest +0.009 without `iot`, highest +0.052 without `transport`, against +0.033 on every label.


## коды, 18 candidates

### all: 18 labels, 1000 questions

| Run | accuracy |
|---|---|
| v1.3 released | 0.146 |
| v1.1 released | 0.104 |

| v1.3 minus v1.1 | difference | 95% over rows | 95% over labels |
|---|---|---|---|
| as published | +0.042 | [+0.022, +0.063] | [-0.005, +0.100] |
| after biases and T | -0.024 | [-0.036, -0.012] | [-0.060, +0.011] |
| v1.1 with biases | 0.063 | | |
| v1.3 with biases | 0.039 | | |

| Label | n | net questions, v1.3 minus v1.1 | difference without it |
|---|---|---|---|
| news | 39 | +13 | +0.030 |
| social | 33 | +12 | +0.031 |
| qa | 106 | -11 | +0.059 |

Leave one label out: lowest +0.030 without `news`, highest +0.059 without `qa`, against +0.042 on every label.
By label v1.3 is ahead on 9 and behind on 5 of 18.

### clean: 17 labels, 865 questions

| Run | accuracy |
|---|---|
| v1.3 released | 0.169 |
| v1.1 released | 0.120 |

| v1.3 minus v1.1 | difference | 95% over rows | 95% over labels |
|---|---|---|---|
| as published | +0.049 | [+0.024, +0.072] | [-0.006, +0.115] |
| after biases and T | -0.007 | [-0.019, +0.005] | [-0.035, +0.017] |
| v1.1 with biases | 0.049 | | |
| v1.3 with biases | 0.042 | | |
| after biases fitted on the slice itself | -0.008 | | |

The biases behind the headline row were fitted on all 1000 questions of this vocabulary and then read on the slice. Fitting them on the slice itself gives 0.140 against 0.148, which is larger because on this slice some options are never the answer and the fit removes them.

| Label | n | net questions, v1.3 minus v1.1 | difference without it |
|---|---|---|---|
| news | 39 | +13 | +0.035 |
| social | 33 | +12 | +0.036 |
| qa | 106 | -11 | +0.070 |

Leave one label out: lowest +0.035 without `news`, highest +0.070 without `qa`, against +0.049 on every label.
By label v1.3 is ahead on 9 and behind on 5 of 17.

### strict: 9 labels, 512 questions

| Run | accuracy |
|---|---|
| v1.3 released | 0.061 |
| v1.1 released | 0.018 |

| v1.3 minus v1.1 | difference | 95% over rows | 95% over labels |
|---|---|---|---|
| as published | +0.043 | [+0.021, +0.066] | [+0.010, +0.081] |
| after biases and T | +0.008 | [-0.005, +0.021] | [-0.000, +0.020] |
| v1.1 with biases | 0.037 | | |
| v1.3 with biases | 0.045 | | |
| after biases fitted on the slice itself | -0.011 | | |

The biases behind the headline row were fitted on all 1000 questions of this vocabulary and then read on the slice. Fitting them on the slice itself gives 0.237 against 0.248, which is larger because on this slice some options are never the answer and the fit removes them.

| Label | n | net questions, v1.3 minus v1.1 | difference without it |
|---|---|---|---|
| iot | 49 | +6 | +0.035 |
| lists | 58 | +6 | +0.035 |
| calendar | 127 | +5 | +0.044 |

Leave one label out: lowest +0.035 without `iot`, highest +0.056 without `email`, against +0.043 on every label.


## сленг, 5 candidates

### all: 18 labels, 1000 questions

| Run | accuracy |
|---|---|
| v1.3 released | 0.780 |
| v1.1 released | 0.648 |

| v1.3 minus v1.1 | difference | 95% over rows | 95% over labels |
|---|---|---|---|
| as published | +0.132 | [+0.099, +0.164] | [+0.015, +0.260] |
| after biases and T | +0.066 | [+0.038, +0.094] | [+0.006, +0.137] |
| v1.1 with biases | 0.729 | | |
| v1.3 with biases | 0.795 | | |

| Label | n | net questions, v1.3 minus v1.1 | difference without it |
|---|---|---|---|
| qa | 106 | +63 | +0.077 |
| general | 68 | +26 | +0.114 |
| play | 135 | +21 | +0.128 |

Leave one label out: lowest +0.077 without `qa`, highest +0.156 without `email`, against +0.132 on every label.
By label v1.3 is ahead on 9 and behind on 6 of 18.

### strict: 14 labels, 773 questions

| Run | accuracy |
|---|---|
| v1.3 released | 0.761 |
| v1.1 released | 0.595 |

| v1.3 minus v1.1 | difference | 95% over rows | 95% over labels |
|---|---|---|---|
| as published | +0.166 | [+0.129, +0.203] | [+0.025, +0.309] |
| after biases and T | +0.069 | [+0.037, +0.102] | [+0.002, +0.153] |
| v1.1 with biases | 0.712 | | |
| v1.3 with biases | 0.780 | | |
| after biases fitted on the slice itself | +0.084 | | |

The biases behind the headline row were fitted on all 1000 questions of this vocabulary and then read on the slice. Fitting them on the slice itself gives 0.824 against 0.740, which is larger because on this slice some options are never the answer and the fit removes them.

| Label | n | net questions, v1.3 minus v1.1 | difference without it |
|---|---|---|---|
| qa | 106 | +63 | +0.097 |
| general | 68 | +26 | +0.145 |
| play | 135 | +21 | +0.168 |

Leave one label out: lowest +0.097 without `qa`, highest +0.192 without `lists`, against +0.166 on every label.
By label v1.3 is ahead on 8 and behind on 5 of 14.


## сленг, 18 candidates

### all: 18 labels, 1000 questions

| Run | accuracy |
|---|---|
| v1.3 released | 0.476 |
| v1.1 released | 0.428 |

| v1.3 minus v1.1 | difference | 95% over rows | 95% over labels |
|---|---|---|---|
| as published | +0.048 | [+0.012, +0.084] | [-0.114, +0.213] |
| after biases and T | +0.037 | [+0.006, +0.067] | [-0.070, +0.141] |
| v1.1 with biases | 0.478 | | |
| v1.3 with biases | 0.515 | | |

| Label | n | net questions, v1.3 minus v1.1 | difference without it |
|---|---|---|---|
| qa | 106 | +62 | -0.016 |
| weather | 57 | +29 | +0.020 |
| lists | 58 | -26 | +0.079 |

Leave one label out: lowest -0.016 without `qa`, highest +0.079 without `lists`, against +0.048 on every label.
By label v1.3 is ahead on 7 and behind on 8 of 18.

### strict: 14 labels, 773 questions

| Run | accuracy |
|---|---|
| v1.3 released | 0.419 |
| v1.1 released | 0.369 |

| v1.3 minus v1.1 | difference | 95% over rows | 95% over labels |
|---|---|---|---|
| as published | +0.050 | [+0.012, +0.091] | [-0.140, +0.234] |
| after biases and T | +0.064 | [+0.032, +0.097] | [-0.041, +0.173] |
| v1.1 with biases | 0.433 | | |
| v1.3 with biases | 0.497 | | |
| after biases fitted on the slice itself | +0.082 | | |

The biases behind the headline row were fitted on all 1000 questions of this vocabulary and then read on the slice. Fitting them on the slice itself gives 0.553 against 0.472, which is larger because on this slice some options are never the answer and the fit removes them.

| Label | n | net questions, v1.3 minus v1.1 | difference without it |
|---|---|---|---|
| qa | 106 | +62 | -0.034 |
| lists | 58 | -26 | +0.091 |
| calendar | 127 | +25 | +0.022 |

Leave one label out: lowest -0.034 without `qa`, highest +0.091 without `lists`, against +0.050 on every label.
By label v1.3 is ahead on 5 and behind on 6 of 14.


## How often each release answers `qa`, 18 candidates

| Vocabulary | label for qa | questions that are qa | v1.1 picks it | v1.3 picks it |
|---|---|---|---|---|
| английский ключ | qa | 106 | 106 | 99 |
| русское слово | справка | 106 | 103 | 99 |
| русская фраза | справочный вопрос | 106 | 81 | 99 |
| жаргон | факт-чек | 106 | 7 | 73 |
| названия служб | справочная служба | 106 | 21 | 113 |
| описание запроса | человек спрашивает про справка | 106 | 100 | 101 |
| коды | СПР | 106 | 209 | 93 |
| сленг | справочное бюро | 106 | 4 | 79 |

## The v1.1 recipe across seeds, five candidates

| Vocabulary | slice | v1.1 released | seed-base-s2 | seed-base-s3 | spread | v1.3 | v1.3 minus best | v1.3 minus mean |
|---|---|---|---|---|---|---|---|---|
| английский ключ | all | 0.957 | 0.951 | 0.960 | 0.009 | 0.962 | +0.002 | +0.006 |
| русское слово | all | 0.956 | 0.953 | 0.957 | 0.004 | 0.960 | +0.003 | +0.005 |
| русская фраза | all | 0.881 | 0.902 | 0.893 | 0.021 | 0.957 | +0.055 | +0.065 |
| жаргон | all | 0.633 | 0.658 | 0.622 | 0.036 | 0.855 | +0.197 | +0.217 |
| жаргон | clean | 0.624 | 0.645 | 0.599 | 0.046 | 0.838 | +0.193 | +0.215 |
| жаргон | strict | 0.550 | 0.648 | 0.563 | 0.098 | 0.786 | +0.138 | +0.198 |
| названия служб | all | 0.835 | 0.828 | 0.828 | 0.007 | 0.869 | +0.034 | +0.039 |
| названия служб | strict | 0.827 | 0.820 | 0.817 | 0.011 | 0.863 | +0.035 | +0.041 |
| описание запроса | all | 0.939 | 0.939 | 0.951 | 0.012 | 0.958 | +0.007 | +0.015 |

The same runs after per-option biases, fitted on all 1,000 questions of the vocabulary and read on the slice:

| Vocabulary | slice | v1.1 released | seed-base-s2 | seed-base-s3 | spread | v1.3 |
|---|---|---|---|---|---|---|
| английский ключ | all | 0.950 | 0.947 | 0.953 | 0.007 | 0.952 |
| русское слово | all | 0.951 | 0.953 | 0.950 | 0.003 | 0.955 |
| русская фраза | all | 0.906 | 0.921 | 0.912 | 0.015 | 0.958 |
| жаргон | all | 0.708 | 0.707 | 0.668 | 0.040 | 0.890 |
| жаргон | clean | 0.721 | 0.717 | 0.675 | 0.046 | 0.881 |
| жаргон | strict | 0.712 | 0.711 | 0.671 | 0.041 | 0.858 |
| названия служб | all | 0.849 | 0.851 | 0.839 | 0.012 | 0.893 |
| названия служб | strict | 0.847 | 0.845 | 0.835 | 0.012 | 0.891 |
| описание запроса | all | 0.943 | 0.940 | 0.951 | 0.012 | 0.952 |

Exact McNemar, `seed-base-s2` against the released v1.1, five candidates:

| Vocabulary | slice | b (v1.1 only) | c (seed only) | p |
|---|---|---|---|---|
| английский ключ | all | 18 | 12 | 0.362 |
| русское слово | all | 19 | 16 | 0.736 |
| русская фраза | all | 33 | 54 | 0.0314 |
| жаргон | all | 104 | 129 | 0.116 |
| жаргон | strict | 42 | 103 | 4.29e-07 |
| названия служб | all | 52 | 45 | 0.543 |
| названия служб | strict | 49 | 42 | 0.53 |
| описание запроса | all | 20 | 20 | 1 |

## Cells where the difference between the releases is no larger than the seed spread

| Vocabulary | slice | v1.3 minus v1.1 | seed spread |
|---|---|---|---|
| английский ключ | all | +0.005 | 0.009 |
| русское слово | all | +0.004 | 0.004 |

## After the biases, every cell

| Vocabulary | candidates | slice | n | as published | after biases | after biases fitted on the slice |
|---|---|---|---|---|---|---|
| английский ключ | 5 | all | 1000 | +0.005 | +0.002 | same |
| английский ключ | 18 | all | 1000 | +0.006 | -0.000 | same |
| русское слово | 5 | all | 1000 | +0.004 | +0.004 | same |
| русское слово | 18 | all | 1000 | +0.009 | +0.007 | same |
| русская фраза | 5 | all | 1000 | +0.076 | +0.052 | same |
| русская фраза | 18 | all | 1000 | +0.182 | +0.086 | same |
| жаргон | 5 | all | 1000 | +0.222 | +0.182 | same |
| жаргон | 5 | clean | 882 | +0.214 | +0.160 | +0.140 |
| жаргон | 5 | strict | 625 | +0.235 | +0.145 | +0.089 |
| жаргон | 18 | all | 1000 | +0.209 | +0.217 | same |
| жаргон | 18 | clean | 882 | +0.168 | +0.176 | +0.157 |
| жаргон | 18 | strict | 625 | +0.142 | +0.151 | +0.101 |
| названия служб | 5 | all | 1000 | +0.034 | +0.044 | same |
| названия служб | 5 | strict | 933 | +0.035 | +0.044 | +0.043 |
| названия служб | 18 | all | 1000 | +0.101 | +0.069 | same |
| названия служб | 18 | strict | 933 | +0.103 | +0.070 | +0.067 |
| описание запроса | 5 | all | 1000 | +0.019 | +0.009 | same |
| описание запроса | 18 | all | 1000 | +0.030 | +0.024 | same |
| коды _SCN | 5 | all | 1000 | +0.033 | +0.024 | same |
| аббревиатуры | 5 | all | 1000 | -0.024 | +0.015 | same |
| аббревиатуры | 5 | clean | 959 | -0.060 | +0.000 | -0.002 |
| коды | 5 | all | 1000 | +0.005 | -0.013 | same |
| коды | 5 | clean | 865 | +0.000 | -0.007 | +0.002 |
| коды | 5 | strict | 512 | +0.033 | -0.001 | +0.002 |
| коды | 18 | all | 1000 | +0.042 | -0.024 | same |
| коды | 18 | clean | 865 | +0.049 | -0.007 | -0.008 |
| коды | 18 | strict | 512 | +0.043 | +0.008 | -0.011 |
| сленг | 5 | all | 1000 | +0.132 | +0.066 | same |
| сленг | 5 | strict | 773 | +0.166 | +0.069 | +0.084 |
| сленг | 18 | all | 1000 | +0.048 | +0.037 | same |
| сленг | 18 | strict | 773 | +0.050 | +0.064 | +0.082 |

## What this says

**One label carries most of several headline differences.** On the
abbreviations, the clean slice difference of -0.060 is -52 questions on `qa` out
of -58 in total; without `qa` it is -0.007. On the slang probe at 18 candidates,
+0.048 is +62 questions on `qa`; without `qa` it is -0.016, that is, the sign
turns over. On the department names at 5 candidates, +0.034 is +36 on `qa`;
without it, -0.002. The same label, in both directions, on three different
vocabularies.

**What `qa` is doing is a lean, and it is visible directly.** At 18 candidates
106 questions of the thousand are `qa`. On jargon v1.1 answers `факт-чек` 7
times and v1.3 answers it 73; on department names, `справочная служба` 21 times
against 113; on slang, `справочное бюро` 4 times against 79. v1.1 almost never
reaches for that label and v1.3 reaches for it freely. On the opaque codes the
picture reverses, 209 against 93. This is the same shape as `политика` on the
news rubrics, and it is why a per-option bias changes these numbers so much.

**The per-option correction removes between a third and all of the gap.** On
the slang probe at 5 candidates, +0.132 becomes +0.066; on its strict slice,
+0.166 becomes +0.069. On department names at 18, +0.101 becomes +0.069. On the
opaque codes at 18 the sign turns over, +0.042 becomes -0.024. The jargon
vocabulary is the one that survives: +0.222 becomes +0.182 at 5 candidates and
+0.209 becomes +0.217 at 18. Whatever v1.3 learned about jargon is not a lean
towards one label.

**Two of the six bench vocabularies are inside the seed spread.** On the English
key the difference is +0.005 against a spread of 0.009 between the three runs of
the v1.1 recipe, and on the Russian word +0.004 against 0.004. Both are training
styles, so this is expected, and it is also a measurement of how little a
difference of that size means anywhere else.

**The strict jargon slice is where the seeds disagree most, and the release is
the weakest of the three.** As published v1.1 is 0.550 there, `seed-base-s2`
0.648 and `seed-base-s3` 0.563, a spread of 0.098. An exact McNemar test of
`seed-base-s2` against the release gives b = 42, c = 103 and p = 4.3e-07, so the
two runs of one recipe differ significantly from each other on that slice. After
the biases the three sit at 0.712, 0.711 and 0.671, a spread of 0.041. Any claim
about the v1.1 recipe on this slice has to be read against that, and the
released checkpoint is not representative of its own recipe there.

## What this does not say

Nothing here is a new measurement. Every figure comes from score files that were
already published, and the only thing added is a different way of reading them.

The slices are defined by the overlap audits already in the repository, exact
match for `clean` and exact match plus prefix in either direction for `strict`.
A different definition would give different slices.

The bias fit is a correction a caller could make with labelled data of their
own. It is not something the model does, and the corrected accuracies are not
the accuracy of a released artefact. They answer the question of how much of a
difference is a lean, and nothing else.
