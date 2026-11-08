#!/usr/bin/env python3
"""Checks on the calibration fit. Standard library only, no network, no weights.

Run: python3 -I scripts/test_calibration.py

Two things are worth checking and neither needs the model.

One. On scores built with a known lean the fit has to find that lean back: the
same sign on every option and the same order from the most favoured to the
least. The sizes are checked loosely, because a bias and a temperature trade
against each other and the likelihood is entitled to split the difference
differently from the way the data were made.

Two. With the biases switched off the fit has to agree with the grid search
that produced every published temperature. They are different algorithms on the
same likelihood, so they are compared to within half a step of that grid.
"""
import math
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import calib

FAIL = []


def check(name, ok, detail=""):
    print(("  ok   " if ok else "  FAIL ") + name + (("  " + detail) if detail else ""))
    if not ok:
        FAIL.append(name)


def make(n, lean, T=2.0, seed=7, spread=1.5):
    """Rows whose correct answer is drawn from softmax(clean scores).

    The scores handed back are the clean ones multiplied by T and pushed away
    from each label by its lean, which is exactly what a model with a taste for
    one label looks like from outside. Recovering `lean` is the job.
    """
    rng = random.Random(seed)
    names = sorted(lean)
    rows = []
    for _ in range(n):
        opts = names[:]
        rng.shuffle(opts)
        clean = [rng.gauss(0, spread) for _ in opts]
        top = max(clean)
        e = [math.exp(v - top) for v in clean]
        z = sum(e)
        u, acc, ans = rng.random(), 0.0, len(opts) - 1
        for i, ei in enumerate(e):
            acc += ei / z
            if u <= acc:
                ans = i
                break
        rows.append(([T * (c - lean[o]) for c, o in zip(clean, opts)], opts, ans))
    return rows


# ---------------------------------------------------------------- the lean
LEAN = {"a": 0.0, "b": -1.2, "c": 0.8, "d": 0.3}
rows = make(20000, LEAN)
got = calib.fit_calibration(rows)
b = got["biases"]

check("every option got a bias", sorted(b) == sorted(LEAN), repr(sorted(b)))
check("the held option is at zero", b[got["held_at_zero"]] == 0.0, got["held_at_zero"])
check("signs match",
      all((b[k] > 0) == (v > 0) and (b[k] < 0) == (v < 0) for k, v in LEAN.items()),
      str({k: round(v, 3) for k, v in b.items()}))
check("order matches",
      sorted(b, key=lambda k: b[k]) == sorted(LEAN, key=lambda k: LEAN[k]),
      " < ".join(sorted(b, key=lambda k: b[k])))
check("sizes are close", max(abs(b[k] - v) for k, v in LEAN.items()) < 0.1,
      "largest miss %.3f" % max(abs(b[k] - v) for k, v in LEAN.items()))
check("temperature recovered", abs(got["temperature"] - 2.0) < 0.1,
      "%.4f against 2.0" % got["temperature"])

# A lean that is really there should be worth accuracy. Fit on one half, answer
# the other, so nothing is scored by a fit that saw it.
half = len(rows) // 2
f = calib.fit_calibration(rows[:half])
raw = sum(lg.index(max(lg)) == ans for lg, _o, ans in rows[half:]) / (len(rows) - half)
fixed = sum(
    max(range(len(p)), key=lambda k: p[k]) == ans
    for p, ans in ((calib.apply_calibration(lg, o, f["temperature"], f["biases"], warn=False), ans)
                   for lg, o, ans in rows[half:])) / (len(rows) - half)
check("accuracy rises out of fold", fixed > raw, "%.4f from %.4f" % (fixed, raw))

# ------------------------------------------------------- against the grid
# calib.fitT is the search every published temperature came out of. Its grid
# runs from 0.2 to 8.0 in 300 steps, so a step is 0.026 and two methods that
# agree are within half of one.
pairs = [{"logits": lg, "answer": ans} for lg, _o, ans in rows]
grid = calib.fitT(pairs)
lbfgs_T = calib.fit_T_only(rows)
check("temperature agrees with the grid search", abs(grid - lbfgs_T) < 0.013,
      "grid %.4f, lbfgs %.4f, apart by %.4f" % (grid, lbfgs_T, abs(grid - lbfgs_T)))

# and the likelihood at the fitted point is not worse than at the grid point
data = [(lg, o, ans) for lg, o, ans in rows]
nll_grid = calib.nll(pairs, grid)
nll_lbfgs = calib.nll(pairs, lbfgs_T)
check("the fitted temperature is at least as likely", nll_lbfgs <= nll_grid + 1e-9,
      "nll %.6f against %.6f" % (nll_lbfgs, nll_grid))

# ------------------------------------------- no lean means nothing to find
flat = make(8000, {"a": 0.0, "b": 0.0, "c": 0.0}, T=1.5, seed=11)
none = calib.fit_calibration(flat)
check("no lean, no biases", max(abs(v) for v in none["biases"].values()) < 0.12,
      str({k: round(v, 3) for k, v in none["biases"].items()}))

# ------------------------------------------------------------- applying it
p = calib.apply_calibration([1.0, 2.0, 3.0], ["a", "b", "c"], 1.0, {})
check("no biases is a plain softmax",
      abs(p[2] - math.exp(3) / (math.exp(1) + math.exp(2) + math.exp(3))) < 1e-12)
check("probabilities sum to one", abs(sum(p) - 1.0) < 1e-12)

p2 = calib.apply_calibration([1.0, 2.0, 3.0], ["a", "b", "c"], 1.0, {"a": 0.0, "b": 5.0, "c": 0.0})
check("a bias moves the answer", max(range(3), key=lambda k: p2[k]) == 1,
      "picked %d" % max(range(3), key=lambda k: p2[k]))

p3 = calib.apply_calibration([1.0, 2.0], ["a", "unknown"], 1.0, {"a": 0.0, "b": 1.0})
check("an option with no bias is scored at zero",
      abs(p3[1] - math.exp(2) / (math.exp(1) + math.exp(2))) < 1e-12)

# the temperature on its own cannot change the answer, the biases can
for T in (0.3, 1.0, 7.0):
    q = calib.apply_calibration([1.0, 2.0, 1.5], ["a", "b", "c"], T, {})
    check("temperature %.1f leaves the answer alone" % T,
          max(range(3), key=lambda k: q[k]) == 1)

print()
if FAIL:
    print("%d check(s) failed: %s" % (len(FAIL), ", ".join(FAIL)))
    sys.exit(1)
print("all checks passed")
