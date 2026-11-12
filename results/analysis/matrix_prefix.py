"""Whether v1.3 stays ahead of the second v1.1 seed on the scientific rubrics
when the rubrics whose names share a prefix with a training option are left out,
and how often the separation clears zero over random sets of ten.

Added after the reading rule in results/unseen3/README.md, in answer to a
question on the Hugging Face thread. It reuses the construction and the bias
fit of matrix_sensitivity.py and needs only the score matrix, no weights.

Run: python3 -I results/analysis/matrix_prefix.py
"""
import os
import random
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts"))
import matrix_sensitivity as ms                                    # noqa: E402
import unseen_bias as ub                                           # noqa: E402

# The four rubrics results/unseen3/README.md lists as starting with a training option.
PREFIX = {"Биология", "Геология", "Горное дело", "Математика"}
PAIRS = (("v1.1", "v1.3 minus v1.1"), ("seed-base-s2", "v1.3 minus seed-base-s2"))


def compare(rows, labels, got, ten, drop=frozenset()):
    built = ms.build_ten(rows, labels, ten)
    oof = {t: ub.out_of_fold(ms.samples_for(built, labels, got[t]), True)
           for t in ("v1.1", "seed-base-s2", "v1.3-ep3of4")}
    gold = [b[3] for b in built]
    keep = [i for i, g in enumerate(gold) if g not in drop]
    out = {}
    for ref, _ in PAIRS:
        d = [oof["v1.3-ep3of4"][i] - oof[ref][i] for i in keep]
        lab = [gold[i] for i in keep]
        out[ref] = (ub.mean(d), ub.ci_rows(d, ub.BOOT_SEED), ub.ci_labels(d, lab, ub.BOOT_SEED))
    return out


def fmt(t):
    m, r, l = t
    return "%+.3f | [%+.3f, %+.3f] | [%+.3f, %+.3f]" % (m, r[0], r[1], l[0], l[1])


def table(title, results):
    print("\n## %s\n" % title)
    print("| Set | prefix rubrics in it | v1.3 minus seed-base-s2 | 95% over rows | 95% over labels |")
    print("|---|---|---|---|---|")
    for name, n4, res in results:
        print("| %s | %s | %s |" % (name, n4, fmt(res["seed-base-s2"])))
    xs = [res["seed-base-s2"] for _, _, res in results]
    print("\nMedian %+.3f, lowest %+.3f, highest %+.3f. Row interval above zero in %d of %d, "
          "label interval above zero in %d of %d."
          % (statistics.median(x[0] for x in xs), min(x[0] for x in xs), max(x[0] for x in xs),
             sum(x[1][0] > 0 for x in xs), len(xs), sum(x[2][0] > 0 for x in xs), len(xs)))


def main():
    rows, labels, got, missing = ms.load_matrix()
    if rows is None:
        print("The matrix is not complete yet: %s" % missing)
        return
    print("# Prefix rubrics and the second v1.1 seed, added after the reading rule\n")
    print("Biases and temperature fitted out of fold over %d half-splits, intervals with seed %d "
          "and %d resamples, as in unseen_bias.py." % (ub.SPLITS, ub.BOOT_SEED, ub.BOOT))
    alpha = sorted(labels)[:ms.TEN_SIZE]
    print("\n## The alphabetical ten\n")
    print("| Rows | comparison | difference | 95% over rows | 95% over labels |")
    print("|---|---|---|---|---|")
    for drop, name in ((frozenset(), "all ten rubrics"), (PREFIX, "without the four prefix rubrics")):
        res = compare(rows, labels, got, alpha, drop)
        for ref, title in PAIRS:
            print("| %s | %s | %s |" % (name, title, fmt(res[ref])))
    rnd = []
    for seed in range(ms.TENS):
        ten = random.Random(seed).sample(sorted(labels), ms.TEN_SIZE)
        rnd.append(("seed %d" % seed, len(PREFIX & set(ten)), compare(rows, labels, got, ten)))
    table("Twenty random tens of all 28 rubrics, seeds 0 to 19", rnd)
    rest = [l for l in sorted(labels) if l not in PREFIX]
    clean = []
    for seed in range(ms.TENS):
        ten = random.Random(seed).sample(rest, ms.TEN_SIZE)
        clean.append(("seed %d" % seed, 0, compare(rows, labels, got, ten)))
    table("Twenty random tens of the 24 rubrics without a prefix match, seeds 0 to 19", clean)


if __name__ == "__main__":
    main()
